import dask.array
import gundam
import numpy as np
from dask.delayed import Delayed
from lsdb import Catalog
from lsdb.dask.merge_catalog_functions import align_and_apply, get_healpix_pixels_from_alignment

from corrgi.alignment import autocorrelation_alignment, crosscorrelation_alignment
from corrgi.dask import count_pairs
from corrgi.estimators import compute_natural_estimate
from corrgi.parameters import create_gundam_params
from corrgi.utils import join_count_histograms


def compute_autocorrelation(catalog: Catalog, random: Catalog) -> np.ndarray:
    """Computes the auto-correlation for a catalog.

    Args:
        catalog (Catalog): The catalog with galaxy samples.
        random (Catalog): The catalog with random samples.

    Returns:
        The natural estimate for the auto-correlation function.
    """
    params = gundam.packpars(kind="acf1", write=False)

    params.dsept = 0.10
    params.nsept = 33
    params.septmin = 0.01

    # Calculate the angular separation bins
    bins, _ = gundam.makebins(params.nsept, params.septmin, params.dsept, params.logsept)
    params_dd, params_rr = create_gundam_params(params)

    left_len = catalog.hc_structure.catalog_info.total_rows
    right_len = random.hc_structure.catalog_info.total_rows

    # Generate the histograms with counts for each catalog
    counts_dd = perform_counts(catalog, catalog, bins, params_dd)
    counts_rr = perform_counts(random, random, bins, params_rr)

    # Actually compute the results
    counts_dd, counts_rr = dask.compute(*[counts_dd, counts_rr])

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
