import numpy as np
from lsdb import Catalog
from munch import Munch

from corrgi.correlation.correlation import Correlation
from corrgi.dask import compute_autocorrelation_counts
from corrgi.estimators import calculate_natural_estimate
from corrgi.utils import compute_catalog_size


def compute_autocorrelation(
    catalog: Catalog, random: Catalog, corr_type: type[Correlation], **kwargs
) -> np.ndarray:
    """Calculates the auto-correlation for a catalog.

    Args:
        catalog (Catalog): The catalog.
        random (Catalog): A random samples catalog.
        corr_type (type[Correlation]): The corrgi class corresponding to the type of
            correlation (AngularCorrelation, RedshiftCorrelation, or ProjectedCorrelation).
        **kwargs (dict): The arguments for the creation of the correlation instance.

    Returns:
        A numpy array with the result of the auto-correlation, using the natural estimator.
    """
    correlation = corr_type(**kwargs)
    correlation.validate([catalog, random])
    num_galaxies = compute_catalog_size(catalog)
    num_random = compute_catalog_size(random)
    counts_dd, counts_rr = compute_autocorrelation_counts(catalog, random, correlation)
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
