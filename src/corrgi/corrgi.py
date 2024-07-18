import numpy as np
from lsdb import Catalog
from munch import Munch

from corrgi.correlation.correlation import Correlation
from corrgi.estimators.estimator_factory import get_estimator_for_correlation


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
    estimator = get_estimator_for_correlation(correlation)
    return estimator.compute_auto_estimate(catalog, random)


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
