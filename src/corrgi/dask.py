import dask
import gundam.cflibfor as cff
import numpy as np
import pandas as pd
from dask.distributed import print as dask_print
from gundam import gundam
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math import HealpixPixel
from lsdb import Catalog
from lsdb.dask.merge_catalog_functions import align_and_apply, get_healpix_pixels_from_alignment
from munch import Munch

from corrgi.alignment import autocorrelation_alignment, crosscorrelation_alignment
from corrgi.parameters import generate_dd_rr_params
from corrgi.utils import join_count_histograms, project_coordinates


def compute_autocorrelation_counts(catalog: Catalog, random: Catalog, params: Munch) -> np.ndarray:
    """Computes the auto-correlation counts for a catalog.

    Args:
        catalog (Catalog): The catalog with galaxy samples.
        random (Catalog): The catalog with random samples.
        params (dict): The gundam parameters for the Fortran subroutine.

    Returns:
        The histogram counts to calculate the auto-correlation.
    """
    # Calculate the angular separation bins
    bins, _ = gundam.makebins(params.nsept, params.septmin, params.dsept, params.logsept)
    params_dd, params_rr = generate_dd_rr_params(params)
    # Generate the histograms with counts for each catalog
    counts_dd = perform_auto_counts(catalog, bins, params_dd)
    counts_rr = perform_auto_counts(random, bins, params_rr)
    # Actually compute the results
    return dask.compute(*[counts_dd, counts_rr])


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
    bins: np.ndarray,
    params: Munch,
) -> np.ndarray:
    """Calls the fortran routine to compute the counts for pairs of
    partitions belonging to the same catalog.

    Args:
       df (pd.DataFrame): The partition dataframe.
       catalog_info (CatalogInfo): The catalog metadata.
       bins (np.ndarray): The separation bins, in angular space.
       params (Munch): The gundam subroutine parameters.

    Returns:
       The count histogram for the partition pair.
    """
    try:
        return _count_auto_pairs(df, catalog_info, bins, params)
    except Exception as exception:
        dask_print(exception)


@dask.delayed
def count_cross_pairs(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_pix: HealpixPixel,
    right_pix: HealpixPixel,
    left_catalog_info: CatalogInfo,
    right_catalog_info: CatalogInfo,
    bins: np.ndarray,
    params: Munch,
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
       bins (np.ndarray): The separation bins, in angular space.
       params (Munch): The gundam subroutine parameters.

    Returns:
       The count histogram for the partition pair.
    """
    try:
        return _count_cross_pairs(
            left_df,
            right_df,
            left_catalog_info,
            right_catalog_info,
            bins,
            params,
        )
    except Exception as exception:
        dask_print(exception)


def _count_auto_pairs(
    df: pd.DataFrame,
    catalog_info: CatalogInfo,
    bins: np.ndarray,
    params: Munch,
) -> np.ndarray:
    x, y, z = project_coordinates(
        ra=df[catalog_info.ra_column].to_numpy(),
        dec=df[catalog_info.dec_column].to_numpy(),
    )
    args = [
        len(df),  # number of particles
        x,
        y,
        z,  # X,Y,Z coordinates of particles
        params.nsept,  # number of angular separation bins
        bins,  # bins in angular separation [deg]
    ]
    counts = cff.mod.th_A_naiveway(*args)  # fast unweighted counting
    return counts


def _count_cross_pairs(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_catalog_info: CatalogInfo,
    right_catalog_info: CatalogInfo,
    bins: np.ndarray,
    params: Munch,
) -> np.ndarray:
    left_x, left_y, left_z = project_coordinates(
        ra=left_df[left_catalog_info.ra_column].to_numpy(),
        dec=left_df[left_catalog_info.dec_column].to_numpy(),
    )
    right_x, right_y, right_z = project_coordinates(
        ra=right_df[right_catalog_info.ra_column].to_numpy(),
        dec=right_df[right_catalog_info.dec_column].to_numpy(),
    )
    args = [
        1,  # number of threads OpenMP
        len(left_df),  # number of particles of the left partition
        left_df[left_catalog_info.ra_column].to_numpy(),  # RA of particles [deg]
        left_df[left_catalog_info.dec_column].to_numpy(),  # DEC of particles [deg]
        left_x,
        left_y,
        left_z,  # X,Y,Z coordinates of particles
        len(right_df),  # number of particles of the right partition
        right_x,
        right_y,
        right_z,  # X,Y,Z coordinates of particles
        params.nsept,  # number of angular separation bins
        bins,  # bins in angular separation [deg]
        params.sbound,
        params.mxh1,
        params.mxh2,
        params.cntid,
        params.logf,
        params.sk1,
        np.zeros(len(right_df)),
        params.grid,
    ]
    # TODO: Create gundam th_C_naive_way that accepts only the necessary arguments
    counts = cff.mod.th_C(*args)  # fast unweighted counting
    return counts
