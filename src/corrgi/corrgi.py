import numpy as np
from lsdb import Catalog

from corrgi.correlation.correlation import Correlation
from corrgi.estimators.estimator_factory import get_estimator_for_correlation


def compute_autocorrelation(
    catalog: Catalog, random: Catalog, corr_type: type[Correlation], **kwargs
) -> np.ndarray:
    """Calculates the auto-correlation for a catalog.

    Args:
        catalog (Catalog): The galaxies catalog (D).
        random (Catalog): A random samples catalog (R).
        corr_type (type[Correlation]): The corrgi class corresponding to the type of
            correlation (AngularCorrelation, RedshiftCorrelation, or ProjectedCorrelation).
        **kwargs (dict): The arguments for the creation of the correlation instance.

    Returns:
        A numpy array with the result of the auto-correlation, according to the estimator
        provided in the correlation kwargs. More information on how to set up the input parameters
        in https://gundam.readthedocs.io/en/latest/introduction.html#set-up-input-parameters.
    """
    correlation = corr_type(**kwargs)
    correlation.validate([catalog, random])
    estimator = get_estimator_for_correlation(correlation)
    return estimator.compute_auto_estimate(catalog, random)


def compute_crosscorrelation(
    left: Catalog, right: Catalog, random: Catalog, corr_type: type[Correlation], **kwargs
) -> np.ndarray:
    """Computes the cross-correlation between two catalogs.

    Args:
        left (Catalog): Left catalog for the cross-correlation (D).
        right (Catalog): Right catalog for the cross-correlation (C).
        random (Catalog): A random samples catalog (R).
        corr_type (type[Correlation]): The corrgi class corresponding to the type of
            correlation (AngularCorrelation, RedshiftCorrelation, or ProjectedCorrelation).
        **kwargs (dict): The arguments for the creation of the correlation instance.

    Returns:
        A numpy array with the result of the cross-correlation, according to the estimator
        provided in the correlation kwargs. More information on how to set up the input parameters
        in https://gundam.readthedocs.io/en/latest/introduction.html#set-up-input-parameters.
    """
    correlation = corr_type(**kwargs)
    correlation.validate([left, right, random])
    estimator = get_estimator_for_correlation(correlation)
    return estimator.compute_cross_estimate(left, right, random)
