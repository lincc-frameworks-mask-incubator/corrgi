import numpy as np
from lsdb import Catalog
from munch import Munch

from corrgi.correlation.correlation import Correlation
from corrgi.dask import compute_autocorrelation_counts
from corrgi.estimators import calculate_natural_estimate


def compute_autocorrelation(
    corr_type: type[Correlation],
    catalog: Catalog,
    random: Catalog,
    params: Munch,
) -> np.ndarray:
    """Calculates the auto-correlation for a catalog.

    Args:
        corr_type (type[Correlation]): The corrgi class corresponding to the type of
            correlation (AngularCorrelation, RedshiftCorrelation, or ProjectedCorrelation).
        catalog (Catalog): The catalog.
        random (Catalog): A random samples catalog.
        params (Munch): The parameters dictionary to run gundam with.

    Returns:
        A numpy array with the result of the auto-correlation, using the natural estimator.
    """
    num_galaxies = catalog.hc_structure.catalog_info.total_rows
    num_random = random.hc_structure.catalog_info.total_rows
    counts_dd, counts_rr = compute_autocorrelation_counts(corr_type, catalog, random, params)
    return calculate_natural_estimate(counts_dd, counts_rr, num_galaxies, num_random)


def compute_crosscorrelation(left: Catalog, right: Catalog, random: Catalog, params: Munch) -> np.ndarray:
    """Computes the cross-correlation between two catalogs.

    Args:
        left (Catalog): Left catalog for the cross-correlation.
        right (Catalog): Right catalog for the cross-correlation.
        random (Catalog): A random samples catalog.
        params (Munch): The parameters dictionary to run gundam with.

    Returns:
        A numpy array with the result of the cross-correlation, using the natural estimator.
    """
    raise NotImplementedError()
