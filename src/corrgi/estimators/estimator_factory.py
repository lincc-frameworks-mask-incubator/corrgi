from corrgi.correlation.correlation import Correlation
from corrgi.estimators.davis_peebles_estimator import DavisPeeblesEstimator
from corrgi.estimators.estimator import Estimator
from corrgi.estimators.natural_estimator import NaturalEstimator

estimator_class_for_type: dict[str, type[Estimator]] = {"NAT": NaturalEstimator, "DP": DavisPeeblesEstimator}


def get_estimator_for_correlation(correlation: Correlation) -> Estimator:
    """Constructs an Estimator instance for the specified correlation.

    Args:
        correlation (Correlation): The correlation instance. The type of
            "estimator" to use is specified in its parameters.

    Returns:
        An initialized Estimator object wrapping the correlation to compute.
    """
    type_to_use = correlation.params.estimator
    if type_to_use not in estimator_class_for_type:
        raise ValueError(f"Cannot load estimator type: {str(type_to_use)}")
    estimator_class = estimator_class_for_type[type_to_use]
    return estimator_class(correlation)
