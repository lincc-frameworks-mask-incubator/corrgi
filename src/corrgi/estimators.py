import numpy as np
from gundam import tpcf


def calculate_tpcf(
    counts_dd: np.ndarray,
    counts_rr: np.ndarray,
    num_particles: int,
    num_random: int,
) -> np.ndarray:
    """Calculates the auto-correlation value for the natural estimator.

    Evaluation given data (D), and random (R) samples.

    Args:
        counts_dd (np.ndarray): The counts for the galaxy samples.
        counts_rr (np.ndarray): The counts for the random samples.
        num_particles (int): The number of particles in data (D).
        num_random (int): The number of random samples.

    Returns:
        The natural correlation function estimate.
    """
    dr = 0  # We do not use DR counts for the natural estimator
    bdd = np.zeros([len(counts_dd), 0])  # We do not compute the bootstrap counts
    wth, _ = tpcf(num_particles, num_random, counts_dd, bdd, counts_rr, dr, estimator="NAT")
    return wth
