import numpy as np
from distributed import Client
from hipscat.io import FilePointer

from corrgi.correlation.correlation import Correlation
from corrgi.estimators.estimator_factory import get_estimator_for_correlation


def compute_autocorrelation(
    catalog_path: FilePointer,
    random_catalog_path: FilePointer,
    *,
    output_dir: FilePointer,
    client: Client,
    corr_type: type[Correlation],
    **kwargs,
) -> np.ndarray:
    """Calculates the auto-correlation for a catalog.

    Args:
        client (Client): The distributed client instance.
        output_dir (str): The path to the directory where we save the intermediate (per-pixel)
            counts as well as the final correlation result (as numpy arrays).
        catalog_path (str): The path to the galaxies catalog (D).
        random_catalog_path (str): The path to the random samples catalog (R).
        corr_type (type[Correlation]): The corrgi class corresponding to the type of
            correlation (AngularCorrelation or ProjectedCorrelation).
        **kwargs (dict): Arguments for the creation of the correlation instance.

    Returns:
        A numpy array with the result of the auto-correlation, according to the estimator
        provided in the correlation kwargs. More information on how to set up the input parameters
        in https://gundam.readthedocs.io/en/latest/introduction.html#set-up-input-parameters.
    """
    correlation = corr_type(**kwargs)
    estimator = get_estimator_for_correlation(correlation)
    return estimator.compute_auto_estimate(
        catalog_path, random_catalog_path, output_dir=output_dir, client=client
    )


def compute_crosscorrelation(
    left_catalog_path: FilePointer,
    right_catalog_path: FilePointer,
    random_catalog_path: FilePointer,
    *,
    output_dir: FilePointer,
    client: Client,
    corr_type: type[Correlation],
    **kwargs,
) -> np.ndarray:
    """Computes the cross-correlation between two catalogs.

    Args:
        left_catalog_path (str): Path to the left catalog for the cross-correlation (D).
        right_catalog_path (str): Path to the right catalog for the cross-correlation (C).
        random_catalog_path (str): Path to the random samples catalog (R).
        output_dir (str): The path to the directory where we save the intermediate (per-pixel)
            counts as well as the final correlation result (as numpy arrays).
        client (Client): The distributed client instance.
        corr_type (type[Correlation]): The corrgi class corresponding to the type of
            correlation (AngularCorrelation or ProjectedCorrelation).
        **kwargs (dict): Arguments for the creation of the correlation instance.

    Returns:
        A numpy array with the result of the cross-correlation, according to the estimator
        provided in the correlation kwargs. More information on how to set up the input parameters
        in https://gundam.readthedocs.io/en/latest/introduction.html#set-up-input-parameters.
    """
    correlation = corr_type(**kwargs)
    estimator = get_estimator_for_correlation(correlation)
    return estimator.compute_cross_estimate(
        left_catalog_path, right_catalog_path, random_catalog_path, output_dir=output_dir, client=client
    )
