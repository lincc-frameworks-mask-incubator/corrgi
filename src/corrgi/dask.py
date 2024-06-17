import dask
import gundam.cflibfor as cff
import numpy as np
import pandas as pd
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math import HealpixPixel
from munch import Munch

from corrgi.utils import project_coordinates


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
    # Distance must be converted to cartesian space
    left_xyz = project_coordinates(
        ra=left_df[left_catalog_info.ra_column].to_numpy(),
        dec=left_df[left_catalog_info.dec_column].to_numpy(),
    )
    right_xyz = project_coordinates(
        ra=right_df[right_catalog_info.ra_column].to_numpy(),
        dec=right_df[right_catalog_info.dec_column].to_numpy(),
    )
    # Pack arguments to the th_Cb subroutine
    args = [
        1,  # number of threads OpenMP
        len(left_df),  # number of particles
        left_df[left_catalog_info.ra_column].to_numpy(),  # RA of particles [deg]
        left_df[left_catalog_info.dec_column].to_numpy(),  # DEC of particles [deg]
        left_xyz,  # X,Y,Z coordinates of particles (see radec2xyz())
        len(right_df),  # number of particles
        right_xyz,  # X,Y,Z coordinates of particles (see radec2xyz())
        params.nsept,  # Number of angular separation bins
        bins,  # Bins in angular separation [deg]
        params.sbound,
        params.mxh1,
        params.mxh2,
        params.nbts,
        params.bseed,
        params.cntid,
        # logff,
        # sk1,
        # ll1,
    ]
    return cff.mod.th_Cb(*args)  # fast unweighted counting
