import dask
import gundam.cflibfor as cff
import numpy as np
import pandas as pd
from dask.delayed import Delayed
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


def perform_counts(left: Catalog, right: Catalog, *args) -> Delayed:
    """Aligns the pixel of two catalogs and performs the pairs counting.

    Args:
        left (Catalog): The left catalog.
        right (Catalog): The right catalog.
        *args: The arguments to pass to the count_pairs method.

    Returns:
        The histogram with the sample distance counts.
    """
    alignment = (
        crosscorrelation_alignment(left.hc_structure, right.hc_structure)
        if left != right
        else autocorrelation_alignment(left.hc_structure)
    )
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    partials = align_and_apply([(left, left_pixels), (right, right_pixels)], count_pairs, *args)
    return join_count_histograms(partials)


def perform_auto_counts(catalog: Catalog, *args) -> Delayed:
    """Aligns the pixel of a single catalog and performs the pairs counting.

    Args:
        catalog (Catalog): The catalog.
        *args: The arguments to pass to the counting methods.

    Returns:
        The histogram with the sample distance counts.
    """
    alignment = autocorrelation_alignment(catalog.hc_structure)
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    partials = align_and_apply([(catalog, left_pixels), (catalog, right_pixels)], count_pairs, *args)
    print(partials)
    cross_hist = join_count_histograms(partials)

    partitions = catalog._ddf.to_delayed()
    partials = [
        count_auto_pairs(partition, catalog.hc_structure.catalog_info, *args) for partition in partitions
    ]
    print(partials)
    auto_hist = join_count_histograms(partials)
    return join_count_histograms([cross_hist, auto_hist])


@dask.delayed
def count_pairs(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_pix: HealpixPixel,
    right_pix: HealpixPixel,
    left_catalog_info: CatalogInfo,
    right_catalog_info: CatalogInfo,
    bins: np.ndarray,
    params: Munch,
) -> np.ndarray:
    """Calls the fortran routine to compute the counts for each partition pair.

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
       The delayed count histogram for the partition pair.
    """
    try:
        left_x, left_y, left_z = project_coordinates(
            ra=left_df[left_catalog_info.ra_column].to_numpy(),
            dec=left_df[left_catalog_info.dec_column].to_numpy(),
        )
        right_x, right_y, right_z = project_coordinates(
            ra=right_df[right_catalog_info.ra_column].to_numpy(),
            dec=right_df[right_catalog_info.dec_column].to_numpy(),
        )
        # Pack arguments to the th_C subroutine
        args = [
            1,  # number of threads OpenMP
            len(left_df),  # number of particles
            left_df[left_catalog_info.ra_column].to_numpy(),  # RA of particles [deg]
            left_df[left_catalog_info.dec_column].to_numpy(),  # DEC of particles [deg]
            left_x,
            left_y,
            left_z,  # X,Y,Z coordinates of particles (see radec2xyz())
            len(right_df),  # number of particles
            right_x,
            right_y,
            right_z,  # X,Y,Z coordinates of particles (see radec2xyz())
            params.nsept,  # Number of angular separation bins
            bins,  # Bins in angular separation [deg]
            params.sbound,
            params.mxh1,
            params.mxh2,
            params.cntid,
            params.logf,
            params.sk1,
            np.zeros(len(right_df)),  # ll1
            params.grid,
        ]
        return cff.mod.th_C(*args)  # fast unweighted counting
    except Exception as exception:
        dask_print(exception)


@dask.delayed
def count_auto_pairs(
    partition: pd.DataFrame,
    catalog_info: CatalogInfo,
    bins: np.ndarray,
    params: Munch,
) -> np.ndarray:
    """Calls the fortran routine to compute the counts for each partition pair.

    Args:
       partition (pd.DataFrame): The partition dataframe.
       catalog_info (CatalogInfo): The catalog metadata.
       bins (np.ndarray): The separation bins, in angular space.
       params (Munch): The gundam subroutine parameters.

    Returns:
       The delayed count histogram for the partition pair.
    """
    try:
        # Distance must be converted to cartesian space
        cart_x, cart_y, cart_z = project_coordinates(
            ra=partition[catalog_info.ra_column].to_numpy(),
            dec=partition[catalog_info.dec_column].to_numpy(),
        )
        # Pack arguments to the th_A subroutine
        args = [
            1,  # number of threads OpenMP
            len(partition),  # number of particles
            partition[catalog_info.dec_column].to_numpy(),  # DEC of particles [deg]
            cart_x,
            cart_y,
            cart_z,  # X,Y,Z coordinates of particles (see radec2xyz())
            params.nsept,  # Number of angular separation bins
            bins,  # Bins in angular separation [deg]
            params.sbound,
            params.mxh1,
            params.mxh2,
            params.cntid,
            "/home/delucchi/git/gundam/FORTRAN.log",
            params.sk1,
            np.zeros(len(partition)),  # ll
        ]
        return cff.mod.th_A(*args)  # fast unweighted counting
    except Exception as exception:
        dask_print(exception)
