import numpy as np
import pandas as pd
from hipscat.io import FilePointer, file_io
from hipscat_import.pipeline_resume_plan import print_task_failure

from corrgi.correlation.correlation import Correlation
from corrgi.pipeline.resume_plan import CorrgiResumePlan


def map_pixel_auto_counts(
    partition_file: FilePointer,
    ra_column: str,
    dec_column: str,
    correlation: Correlation,
    mapping_key: str,
    resume_path: FilePointer,
):
    """Computes counts in partitions for points within themselves"""
    try:
        left_df = pd.read_parquet(partition_file, dtype_backend="pyarrow", memory_map=True)
        hist = correlation.count_auto_pairs(left_df, ra_column, dec_column)
        filename = CorrgiResumePlan.get_histogram_filepath(tmp_path=resume_path, mapping_key=mapping_key)
        np.save(filename, hist)
        CorrgiResumePlan.mapping_key_done(tmp_path=resume_path, mapping_key=mapping_key)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed MAPPING auto stage for file {partition_file}", exception)
        raise exception


def map_pixel_cross_counts(
    left_partition_file: FilePointer,
    right_partition_files: list[FilePointer],
    left_ra_column: str,
    left_dec_column: str,
    right_ra_column: str,
    right_dec_column: str,
    correlation: Correlation,
    mapping_keys: list[str],
    resume_path: FilePointer,
):
    """Computes counts for points in different partitions"""
    try:
        left_df = file_io.read_parquet_file_to_pandas(left_partition_file)
        for right_partition, mapping_key in zip(right_partition_files, mapping_keys):
            right_df = file_io.read_parquet_file_to_pandas(right_partition)
            hist = correlation.count_cross_pairs(
                left_df, right_df, left_ra_column, left_dec_column, right_ra_column, right_dec_column
            )
            filename = CorrgiResumePlan.get_histogram_filepath(tmp_path=resume_path, mapping_key=mapping_key)
            np.save(filename, hist)
            CorrgiResumePlan.mapping_key_done(tmp_path=resume_path, mapping_key=mapping_key)
            del right_df
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed cross MAPPING stage for file {left_partition_file}", exception)
        raise exception


def reduce_pixel_counts(reducing_keys: list[str], output_artifact_path: str):
    """Sums all the intermediate counts to a final histogram"""
    try:
        histogram = None
        for path in reducing_keys:
            partial_histogram = np.load(path)
            histogram = histogram + partial_histogram if histogram is not None else partial_histogram
            del partial_histogram
        np.save(output_artifact_path, histogram)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure("Failed REDUCING stage", exception)
        raise exception
