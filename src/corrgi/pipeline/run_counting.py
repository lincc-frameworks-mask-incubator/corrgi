"""Compute correlation using dask for parallelization"""

import numpy as np
from hipscat.io import paths

import corrgi.pipeline.map_reduce as mr
from corrgi.pipeline.resume_plan import CorrgiResumePlan


def run_counting(args, client):
    """Run counting of pairs in a map-reduce pipeline.

    This pipeline is divided into two procedures:
    - `auto_counts`: computes counts with partitions against themselves.
    - `cross_counts`: computes counts with partitions against every other
    partition of the catalog (the same catalog if computing the auto-correlation,
    a different catalog if computing the cross-correlation)."""
    resume_plan = CorrgiResumePlan(args)
    # Compute counts for `auto_counts`
    if not resume_plan.is_mapping_auto_done():
        auto_futures = get_auto_futures(args, resume_plan, client)
        resume_plan.wait_for_auto_mapping(auto_futures)
    # Compute counts for `cross_counts`
    if not resume_plan.is_mapping_cross_done():
        cross_futures = get_cross_futures(args, resume_plan, client)
        resume_plan.wait_for_cross_mapping(cross_futures)
    # Merge all partial histograms into a single one
    if not resume_plan.is_reducing_done():
        reducing_future = get_reducing_future(resume_plan, client)
        resume_plan.wait_for_reducing(reducing_future)
    # Return the final count for the correlation
    return np.load(resume_plan.output_artifact_path)


def get_auto_futures(args, resume_plan, client):
    """Generates the features for the `auto_count` procedure. Each worker
    is assigned a partition which it will call `count_auto_pairs` with."""
    auto_futures = []
    correlation_future = client.scatter(args.correlation)
    for pixel, mapping_key in resume_plan.get_remaining_map_auto_keys().items():
        partition_file = paths.pixel_catalog_file(args.left_catalog_path, pixel.order, pixel.pixel)
        auto_futures.append(
            client.submit(
                mr.map_pixel_auto_counts,
                partition_file=partition_file,
                ra_column=args.left_hc_catalog.catalog_info.ra_column,
                dec_column=args.left_hc_catalog.catalog_info.dec_column,
                correlation=correlation_future,
                mapping_key=mapping_key,
                resume_path=resume_plan.tmp_path,
            )
        )
    return auto_futures


def get_cross_futures(args, resume_plan, client):
    """Generates the features for the `cross_count` procedure (catalog A) x (catalog B).
    Each worker is assigned a partition of the left catalog (A) and a list of partitions
    of right catalog (B) which it will call `count_cross_pairs` with."""
    cross_futures = []
    correlation_future = client.scatter(args.correlation)
    for left_pixel, (right_pixels, mapping_keys) in resume_plan.get_remaining_map_cross_keys().items():
        left_partition_file = paths.pixel_catalog_file(
            args.left_catalog_path, left_pixel.order, left_pixel.pixel
        )
        right_partition_files = paths.pixel_catalog_files(args.right_catalog_path, right_pixels)
        cross_futures.append(
            client.submit(
                mr.map_pixel_cross_counts,
                left_partition_file=left_partition_file,
                right_partition_files=right_partition_files,
                left_ra_column=args.left_hc_catalog.catalog_info.ra_column,
                left_dec_column=args.left_hc_catalog.catalog_info.dec_column,
                right_ra_column=args.right_hc_catalog.catalog_info.ra_column,
                right_dec_column=args.right_hc_catalog.catalog_info.dec_column,
                correlation=correlation_future,
                mapping_keys=mapping_keys,
                resume_path=resume_plan.tmp_path,
            )
        )
    return cross_futures


def get_reducing_future(resume_plan, client):
    """Generates a future which will collect all the partial histograms from
    `auto_counts` and `cross_counts` and merge them into a final count histogram.
    The result of this step is the result of the correlation."""
    return client.submit(
        mr.reduce_pixel_counts,
        reducing_keys=resume_plan.get_reducing_keys(),
        output_artifact_path=resume_plan.output_artifact_path,
    )
