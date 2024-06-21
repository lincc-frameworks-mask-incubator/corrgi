import dask.array
import gundam
import numpy as np
from dask.delayed import Delayed
from lsdb import Catalog
from lsdb.dask.merge_catalog_functions import align_and_apply, get_healpix_pixels_from_alignment

from corrgi.alignment import autocorrelation_alignment, crosscorrelation_alignment
from corrgi.dask import count_auto_pairs, count_pairs
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
    params = gundam.packpars(kind="acf", write=False)

    params.dsept = 0.10
    params.nsept = 33
    params.septmin = 0.01

    # Calculate the angular separation bins
    bins, _ = gundam.makebins(params.nsept, params.septmin, params.dsept, params.logsept)
    print(bins)
    params_dd, params_rr = create_gundam_params(params)

    num_galaxies = catalog.hc_structure.catalog_info.total_rows
    num_rand = random.hc_structure.catalog_info.total_rows

    # Generate the histograms with counts for each catalog
    counts_dd = perform_auto_counts(catalog, bins, params_dd)
    counts_rr = perform_auto_counts(random, bins, params_rr)

    # Actually compute the results
    counts_dd, counts_rr = dask.compute(*[counts_dd, counts_rr])
    print(counts_dd)

    # Compute the auto-correlation using the natural estimator
    return compute_natural_estimate(counts_dd, counts_rr, num_galaxies, num_rand)


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
