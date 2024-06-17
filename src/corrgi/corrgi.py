import gundam
import numpy as np
from dask.delayed import Delayed
from lsdb import Catalog
from lsdb.dask.merge_catalog_functions import align_and_apply, get_healpix_pixels_from_alignment
from munch import Munch

from corrgi.alignment import get_pixel_alignment
from corrgi.dask import count_pairs
from corrgi.estimators import compute_natural_estimate
from corrgi.parameters import generate_gundam_params
from corrgi.utils import join_count_histograms


def compute_autocorrelation(catalog: Catalog, random: Catalog) -> np.ndarray:
    """Computes the auto-correlation for a catalog.

    Args:
        catalog (Catalog): The catalog with galaxy samples.
        random (Catalog): The catalog with random samples.

    Returns:
        The natural estimate for the auto-correlation function.
    """
    params_dd, params_rr = generate_gundam_params()

    left_len = catalog.hc_structure.catalog_info.total_rows
    right_len = random.hc_structure.catalog_info.total_rows

    # Calculate the angular separation bins
    bins, _ = gundam.makebins(params_dd.nsept, params_dd.septmin, params_dd.dsept, params_dd.logsept)

    # Generate the histograms with counts for each catalog
    counts_dd = perform_counts(catalog, catalog, bins, params_dd)
    counts_rr = perform_counts(random, random, bins, params_rr)

    counts_dd = counts_dd.compute()
    counts_rr = counts_rr.compute()

    # Compute the auto-correlation using the natural estimator
    return compute_natural_estimate(counts_dd, counts_rr, left_len, right_len)


def compute_crosscorrelation(left: Catalog, right: Catalog, random: Catalog) -> np.ndarray:
    """Computes the cross-correlation between two catalogs.

    Args:
        left (Catalog): A catalog with galaxy samples.
        right (Catalog): A catalog with galaxy samples.
        random (Catalog): A catalog with random samples.

    Returns:
        The natural estimate for the cross-correlation function.
    """
    raise NotImplementedError()


def perform_counts(left: Catalog, right: Catalog, bins: np.ndarray, params: Munch) -> Delayed:
    """Aligns the pixel of two catalogs and performs the pairs counting.

    Args:
        left (Catalog): The left catalog.
        right (Catalog): The right catalog.
        bins (np.ndarray): The bins, in angular space.
        params (Munch): The gundam parameters.

    Returns:
        The histogram with the counts for the
    """
    alignment = get_pixel_alignment(left, right)
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    partials = align_and_apply([(left, left_pixels), (right, right_pixels)], count_pairs, bins, params)
    return join_count_histograms(partials)
