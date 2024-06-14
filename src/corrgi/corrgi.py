from copy import deepcopy

import dask
import dask.array as da
import gundam
import gundam.cflibfor as cff
import numpy as np
from dask.delayed import Delayed
from gundam import tpcf
from hipscat.pixel_tree import PixelAlignment
from lsdb import Catalog
from lsdb.dask.merge_catalog_functions import align_and_apply, get_healpix_pixels_from_alignment
from munch import Munch
from numpy import deg2rad


def compute_autocorrelation(galaxies: Catalog, random: Catalog) -> float:
    """Computes the auto-correlation for a catalog.

    Args:
        galaxies (Catalog): The catalog with galaxy samples.
        random (Catalog): The catalog with random samples.

    Returns:
        The natural estimate for the auto-correlation function.
    """
    par_dd, par_rr = generate_gundam_params()
    left_len = galaxies.hc_structure.catalog_info.total_rows
    right_len = random.hc_structure.catalog_info.total_rows

    # Calculate the bins and project them into cartesian space
    sept, _ = gundam.makebins(par_dd.nsept, par_dd.septmin, par_dd.dsept, par_dd.logsept)
    # bins = project_bins(sept)

    # Generate the partial histograms with counts for each catalog
    counts_dd = perform_counts(galaxies, galaxies, sept, par_dd)
    counts_rr = perform_counts(random, random, sept, par_rr)

    # Actually compute the counts
    counts_dd = counts_dd.compute()
    counts_rr = counts_rr.compute()

    # Finally, compute the correlation estimate with the natural estimator
    return compute_natural_estimate(counts_dd, counts_rr, left_len, right_len)


def perform_counts(left: Catalog, right: Catalog, sept: np.ndarray, par: Munch) -> Delayed:
    """Aligns the pixel of two catalogs and performs the pairs counting.

    Args:
        left (Catalog): The left catalog.
        right (Catalog): The right catalog.
        sept (np.ndarray): The bins, in angular space.
        par (Munch): The gundam parameters.

    Returns:
        The histogram with the counts for the
    """
    alignment = get_pixel_alignment(left, right)
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    partials = align_and_apply([(left, left_pixels)], pairs_auto, sept, par)
    return join_count_histograms(partials)


def compute_natural_estimate(
    counts_dd: np.ndarray,
    counts_rr: np.ndarray,
    num_galaxies: int,
    num_random: int,
) -> float:
    """Compute the auto-correlation function for a given estimator.

    Args:
        counts_dd (np.ndarray): The counts for the galaxy samples.
        counts_rr (np.ndarray): The counts for the random samples.
        num_galaxies (int): The number of galaxy samples.
        num_random (int): The number of random samples.

    Returns:
        The natural correlation function estimate.
    """
    dr = 0  # We do not use DR counts for the natural estimator
    bdd = np.zeros([len(counts_dd), 0])  # We do not compute the bootstrap counts
    wth, _ = tpcf(num_galaxies, num_random, counts_dd, bdd, counts_rr, dr, estimator="NAT")
    return wth


def generate_gundam_params() -> tuple[Munch, Munch]:
    """Generates the parameters for the galaxy and random samples.

    Returns:
        A tuple containing the Munch dictionaries of parameters for
        the galaxy samples and the random samples.
    """
    par = gundam.packpars(kind="acf", write=False)
    par_dd = deepcopy(par)
    par_dd.kind = "thA"
    par_dd.cntid = "DD"
    par_rr = deepcopy(par)
    par_rr.kind = "thA"
    par_rr.cntid = "RR"
    par_rr.wfib = False  # don't do fiber corrections in random counts
    par_rr.doboot = False  # don't do bootstraping in random counts
    par_dr = deepcopy(par)
    par_dr.kind = "thC"
    par_dr.cntid = "DR"
    par_dr.wfib = False  # don't do fiber corrections in crounts counts
    par_dr.doboot = False  # don't do bootstraping in cross counts
    return par_dd, par_rr


def project_bins(sep: np.ndarray) -> np.ndarray:
    """Projects the angular bins to cartesian space according
    to a sphere of radius=0.5.

    Args:
        sep (np.ndarray): The bins, in angular space.

    Returns:
        The distance bins in the projected cartesian space.
    """
    return (np.sin(0.5 * sep * deg2rad)) ** 2


@dask.delayed
def pairs_auto(df, catalog_info, sept, par) -> np.ndarray:
    """Calls the fortran routine to compute the counts for a partition-pair."""
    # Distance must be converted to cartesian space
    left_xyz = project_coordinates(
        ra=df[catalog_info.ra_column].to_numpy(),
        dec=df[catalog_info.dec_column].to_numpy(),
    )
    # Pack arguments to the th_A subroutine
    args = [
        -1,  # number of threads OpenMP
        len(df),  # number of particles
        df[catalog_info.dec_column],  # DEC of particles
        left_xyz,  # X,Y,Z coordinates of particles
        par.nsept,
        sept,  # The separation bins
        par.sbound,
        par.mxh1,
        par.mxh2,
        par.cntid,
        # logff,
        # sk,
        # ll,
    ]
    return cff.mod.th_A(*args)


def project_coordinates(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Project spherical coordinates (ra, dec) to cartesian space
    on a sphere of radius=0.5.

    Args:
        ra (np.ndarray): Right ascension, in radians.
        dec (np.ndarray): Declination, in radians.

    Returns:
        The coordinates, in cartesian space, on a sphere of radius=0.5.
    """
    ra = np.radians(ra)
    dec = np.radians(dec)
    x = 0.5 * np.cos(ra) * np.cos(dec)
    y = 0.5 * np.cos(ra) * np.sin(dec)
    z = 0.5 * np.sin(ra)
    return np.stack([x, y, z], axis=1)


def join_count_histograms(partial_histograms: list[Delayed]) -> Delayed:
    """Joins the partial count histograms (lazily).

    Args:
        partial_histograms (list[np.ndarray]): The list of delayed count
            histograms, generated for each partition pair.

    Returns:
        The delayed numpy array that results from combining all partial histograms.
    """
    if len(partial_histograms) == 0:
        raise ValueError("No partial histograms provided!")
    stacked_arrays = da.stack(partial_histograms)
    count_histogram = da.sum(stacked_arrays, axis=0)
    return count_histogram


def get_pixel_alignment(left: Catalog, right: Catalog) -> PixelAlignment:
    """Computes the pixel alignment between two catalogs."""
    raise NotImplementedError()
