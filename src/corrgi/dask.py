import dask
import numpy as np
import pandas as pd
from dask.distributed import print as dask_print
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math import HealpixPixel
from lsdb import Catalog
from lsdb.dask.merge_catalog_functions import align_and_apply, get_healpix_pixels_from_alignment

from corrgi.alignment import autocorrelation_alignment, crosscorrelation_alignment
from corrgi.correlation.correlation import Correlation
from corrgi.utils import join_count_histograms


def perform_auto_counts(catalog: Catalog, *args) -> np.ndarray:
    """Aligns the pixel of a single catalog and performs the pairs counting.

    Args:
        catalog (Catalog): The catalog.
        *args: The arguments to pass to the counting methods.

    Returns:
        The histogram with the sample distance counts.
    """
    # Get counts between points of different partitions
    alignment = autocorrelation_alignment(catalog.hc_structure)
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    cross_partials = align_and_apply(
        [(catalog, left_pixels), (catalog, right_pixels)], count_cross_pairs, *args
    )
    # Get counts between points of the same partition
    auto_partials = [
        count_auto_pairs(partition, catalog.hc_structure.catalog_info, *args)
        for partition in catalog._ddf.to_delayed()
    ]
    all_partials = [*cross_partials, *auto_partials]
    return join_count_histograms(all_partials)


def perform_cross_counts(left: Catalog, right: Catalog, *args) -> np.ndarray:
    """Aligns the pixel of two catalogs and performs the pairs counting.

    Args:
        left (Catalog): The left catalog.
        right (Catalog): The right catalog.
        *args: The arguments to pass to the count_pairs method.

    Returns:
        The histogram with the sample distance counts.
    """
    alignment = crosscorrelation_alignment(left.hc_structure, right.hc_structure)
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    cross_partials = align_and_apply([(left, left_pixels), (right, right_pixels)], count_cross_pairs, *args)
    return join_count_histograms(cross_partials)


@dask.delayed
def count_auto_pairs(
    df: pd.DataFrame,
    catalog_info: CatalogInfo,
    correlation: Correlation,
) -> np.ndarray:
    """Calls the fortran routine to compute the counts for pairs of
    partitions belonging to the same catalog.

    Args:
       df (pd.DataFrame): The partition dataframe.
       catalog_info (CatalogInfo): The catalog metadata.
       correlation (Correlation): The correlation instance.

    Returns:
       The count histogram for the partition pair.
    """
    try:
        return correlation.count_auto_pairs(df, catalog_info)
    except Exception as exception:
        dask_print(exception)
        raise exception


@dask.delayed
def count_cross_pairs(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_pix: HealpixPixel,
    right_pix: HealpixPixel,
    left_catalog_info: CatalogInfo,
    right_catalog_info: CatalogInfo,
    correlation: Correlation,
) -> np.ndarray:
    """Calls the fortran routine to compute the counts for pairs of
    partitions belonging to two different catalogs.

    Args:
       left_df (pd.DataFrame): The left partition dataframe.
       right_df (pd.DataFrame): The right partition dataframe.
       left_pix (HealpixPixel): The pixel corresponding to `left_df`.
       right_pix (HealpixPixel): The pixel corresponding to `right_df`.
       left_catalog_info (CatalogInfo): The left catalog metadata.
       right_catalog_info (CatalogInfo): The right catalog metadata.
       correlation (Correlation): The correlation instance.

    Returns:
       The count histogram for the partition pair.
    """
    try:
        return correlation.count_cross_pairs(
            left_df,
            right_df,
            left_catalog_info,
            right_catalog_info,
        )
    except Exception as exception:
        dask_print(exception)
        raise exception
