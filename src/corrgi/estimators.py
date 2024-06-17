import numpy as np
from gundam import tpcf


def compute_natural_estimate(
    counts_dd: np.ndarray,
    counts_rr: np.ndarray,
    num_galaxies: int,
    num_random: int,
) -> np.ndarray:
    """Compute the auto-correlation function for a given estimator.

    Args:
        counts_dd (np.ndarray): The counts for the galaxy samples.
        counts_rr (np.ndarray): The counts for the random samples.
        num_galaxies (int): The number of galaxy samples.
        num_random (int): The number of random samples.

    Returns:
        The natural correlation function estimate.
    """
    dr = 0  # We do not use DR counts for the natural estimator
    bdd = np.zeros([len(counts_dd), 0])  # We do not compute the bootstrap counts
    wth, _ = tpcf(num_galaxies, num_random, counts_dd, bdd, counts_rr, dr, estimator="NAT")
    return wth
