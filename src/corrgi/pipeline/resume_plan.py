"""Utility to hold the pipeline execution plan."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from hipscat.io import FilePointer, file_io
from hipscat.pixel_math import HealpixPixel
from hipscat_import.pipeline_resume_plan import PipelineResumePlan

from corrgi.alignment import read_alignment, write_alignment
from corrgi.pipeline.arguments import CorrgiArguments
from corrgi.pipeline.utils import (
    filter_auto_pixel_keys,
    filter_cross_pixel_keys,
    get_auto_file_alignment,
    get_auto_pixel_keys,
    get_cross_file_alignment,
    get_cross_pixel_keys,
    get_pixel_key,
)


@dataclass
class CorrgiResumePlan(PipelineResumePlan):
    """Container class for holding the state of each file in the pipeline plan."""

    auto_pixels: list[HealpixPixel] = field(default_factory=list)
    """The list of partitions to compute auto counts on"""
    cross_pixels: dict[HealpixPixel, list[HealpixPixel]] = field(default_factory=dict)
    """The list of partitions to compute cross counts on"""

    MAPPING_STAGE = "mapping"
    MAPPING_STAGE_AUTO = "mapping_auto"
    MAPPING_STAGE_CROSS = "mapping_cross"
    REDUCING_STAGE = "reducing"

    def __init__(self, args: CorrgiArguments):
        if not args.tmp_path:  # pragma: no cover (not reachable, but required for mypy)
            raise ValueError("tmp_path is required")
        super().__init__(
            resume=args.resume,
            progress_bar=args.progress_bar,
            simple_progress_bar=args.simple_progress_bar,
            tmp_path=args.tmp_path,
            delete_resume_log_files=False,
        )
        self.output_artifact_path = file_io.append_paths_to_pointer(
            args.output_path, f"{args.output_artifact_name}.npy"
        )
        self._gather_plan(args)

    def _gather_plan(self, args: CorrgiArguments):
        """Initialize the plan."""
        with self.print_progress(total=4, stage_name="Planning") as step_progress:
            super().safe_to_resume()

            mapping_done = self.is_mapping_done()
            reducing_done = self.is_reducing_done()
            if reducing_done and (not mapping_done):
                raise ValueError("mapping must be complete before reducing")

            step_progress.update(1)

            self.check_original_input_paths([args.left_catalog_path, args.right_catalog_path])

            step_progress.update(1)

            # Read the HiPSCat catalog's information
            self.get_alignment(args)

            step_progress.update(1)

            # Create the directories for the mapping stage
            for pixel in args.left_hc_catalog.get_healpix_pixels():
                file_io.make_directory(
                    file_io.append_paths_to_pointer(self.tmp_path, self.MAPPING_STAGE, get_pixel_key(pixel)),
                    exist_ok=True,
                )

            step_progress.update(1)

    def is_mapping_done(self) -> bool:
        """Are there sources left to count?"""
        return self.is_mapping_auto_done() and self.is_mapping_cross_done()

    def is_mapping_auto_done(self) -> bool:
        """Are there partitions for which the mapping of self-counts hasn't concluded?"""
        return self.done_file_exists(self.MAPPING_STAGE_AUTO)

    def is_mapping_cross_done(self) -> bool:
        """Are there pairs of partitions for which the mapping of cross-counts hasn't concluded?"""
        return self.done_file_exists(self.MAPPING_STAGE_CROSS)

    @classmethod
    def mapping_key_done(cls, tmp_path, mapping_key: str):
        """Mark a single mapping task as done

        Args:
            tmp_path (str): where to write intermediate resume files.
            mapping_key (str): unique string for each mapping task (e.g. "map_1_24")
        """
        cls.touch_key_done_file(tmp_path, cls.MAPPING_STAGE, mapping_key)

    def get_remaining_map_auto_keys(self) -> dict[HealpixPixel, str]:
        """What are the pixels for each auto_counts still needs to run on"""
        done_keys = set(self.read_done_mapping_keys())
        pixel_keys = get_auto_pixel_keys(self.auto_pixels)
        return filter_auto_pixel_keys(pixel_keys, done_keys)

    def get_remaining_map_cross_keys(self) -> dict[HealpixPixel, list[tuple[HealpixPixel, str]]]:
        """What are the pairs of pixels for each cross_counts still needs to run on"""
        done_keys = set(self.read_done_mapping_keys())
        pixel_keys = get_cross_pixel_keys(self.cross_pixels)
        return filter_cross_pixel_keys(pixel_keys, done_keys)

    def wait_for_auto_mapping(self, futures):
        """Runs the auto_counts for all remaining partitions"""
        self.wait_for_futures(futures, self.MAPPING_STAGE_AUTO)
        remaining_pixels_to_map = self.get_remaining_map_auto_keys()
        if len(remaining_pixels_to_map) > 0:
            raise RuntimeError(
                f"{len(remaining_pixels_to_map)} auto mapping stages did not complete successfully."
            )
        self.touch_stage_done_file(self.MAPPING_STAGE_AUTO)

    def wait_for_cross_mapping(self, futures):
        """Runs the cross_counts for all remaining pairs of partitions"""
        self.wait_for_futures(futures, self.MAPPING_STAGE_CROSS)
        remaining_pixels_to_map = self.get_remaining_map_cross_keys()
        if len(remaining_pixels_to_map) > 0:
            raise RuntimeError(
                f"{len(remaining_pixels_to_map)} cross mapping stages did not complete successfully."
            )
        self.touch_stage_done_file(self.MAPPING_STAGE_CROSS)

    def is_reducing_done(self) -> bool:
        """Are there partitions left to reduce?"""
        return self.done_file_exists(self.REDUCING_STAGE)

    @classmethod
    def reducing_key_done(cls, tmp_path, reducing_key: str):
        """Mark a single reducing task as done

        Args:
            tmp_path (str): where to write intermediate resume files.
            reducing_key (str): unique string for each reducing task (e.g. "3_57")
        """
        cls.touch_key_done_file(tmp_path, cls.REDUCING_STAGE, reducing_key)

    def get_reducing_keys(self):
        """Fetch a tuple for each object catalog pixel to reduce."""
        mapping_dir = Path(self.tmp_path, self.MAPPING_STAGE)
        return list(mapping_dir.rglob("*.npy"))

    def wait_for_reducing(self, future):
        """Wait for reducing stage futures to complete."""
        self.wait_for_futures([future], self.REDUCING_STAGE)
        if not file_io.is_regular_file(self.output_artifact_path):
            raise RuntimeError("The reducing stage did not complete successfully.")
        self.touch_stage_done_file(self.REDUCING_STAGE)

    @classmethod
    def get_histogram_filepath(cls, tmp_path: FilePointer, mapping_key: str):
        """File name for writing a histogram file to a special intermediate directory."""
        file_io.make_directory(file_io.append_paths_to_pointer(tmp_path, cls.MAPPING_STAGE), exist_ok=True)
        return file_io.append_paths_to_pointer(tmp_path, cls.MAPPING_STAGE, f"{mapping_key}.npy")

    def read_done_mapping_keys(self):
        """Inspect the stage's directory of done files, fetching the keys from done file names.

        Return:
            List[str] - all keys found in done directory
        """
        done_keys = []
        stage_dir = file_io.append_paths_to_pointer(self.tmp_path, self.MAPPING_STAGE)
        mapping_dirs = [
            content
            for content in file_io.get_directory_contents(stage_dir)
            if not file_io.is_regular_file(content)
        ]
        for directory in mapping_dirs:
            done_prefixes = self.get_keys_from_file_names(directory, "_done")
            pixel_dir = file_io.get_basename_from_filepointer(directory)
            keys = [f"{pixel_dir}/{key}" for key in done_prefixes]
            done_keys.extend(keys)
        return done_keys

    def get_alignment(self, args):
        """Read alignment from disk if it exists, or calculate it on the fly"""
        auto_alignment_path = f"{args.tmp_path}/auto_alignment"
        cross_alignment_path = f"{args.tmp_path}/cross_alignment"
        if os.path.exists(auto_alignment_path) and os.path.exists(cross_alignment_path):
            self.auto_pixels = read_alignment(auto_alignment_path)
            self.cross_pixels = read_alignment(cross_alignment_path)
        else:
            self.auto_pixels, self.cross_pixels = (
                get_auto_file_alignment(args.left_hc_catalog)
                if args.left_catalog_path == args.right_catalog_path
                else get_cross_file_alignment(args.left_hc_catalog, args.right_hc_catalog)
            )
            write_alignment(auto_alignment_path, self.auto_pixels)
            write_alignment(cross_alignment_path, self.cross_pixels)
