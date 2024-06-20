import numpy as np
from lsdb import Catalog
from munch import Munch

from corrgi.dask import compute_autocorrelation_counts
from corrgi.estimators import calculate_natural_estimate


def compute_autocorrelation(catalog: Catalog, random: Catalog, params: Munch) -> np.ndarray:
    """Calculates the auto-correlation natural estimate."""
    num_galaxies = catalog.hc_structure.catalog_info.total_rows
    num_random = random.hc_structure.catalog_info.total_rows
    counts_dd, counts_rr = compute_autocorrelation_counts(catalog, random, params)
    return calculate_natural_estimate(counts_dd, counts_rr, num_galaxies, num_random)


def compute_crosscorrelation(left: Catalog, right: Catalog, random: Catalog) -> np.ndarray:
    """Computes the cross-correlation between two catalogs."""
    raise NotImplementedError()
