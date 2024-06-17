import dask.array as da
import numpy as np
from dask.delayed import Delayed
from numpy import deg2rad


def project_coordinates(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Project spherical coordinates (ra, dec) to cartesian space
    on a sphere of radius=0.5.

    Args:
        ra (np.ndarray): Right ascension, in radians.
        dec (np.ndarray): Declination, in radians.

    Returns:
        The coordinates, in cartesian space, on a sphere of radius=0.5.
    """
    ra = np.radians(ra)
    dec = np.radians(dec)
    x = 0.5 * np.cos(ra) * np.cos(dec)
    y = 0.5 * np.cos(ra) * np.sin(dec)
    z = 0.5 * np.sin(ra)
    return np.stack([x, y, z], axis=1)


def project_bins(sep: np.ndarray) -> np.ndarray:
    """Projects the angular bins to cartesian space according
    to a sphere of radius=0.5.

    Args:
        sep (np.ndarray): The bins, in angular space.

    Returns:
        The distance bins in the projected cartesian space.
    """
    return (np.sin(0.5 * sep * deg2rad)) ** 2


def join_count_histograms(partial_histograms: list[Delayed]) -> Delayed:
    """Joins the partial count histograms (lazily).

    Args:
        partial_histograms (list[np.ndarray]): The list of delayed count
            histograms, generated for each partition pair.

    Returns:
        The delayed numpy array that results from combining all partial histograms.
    """
    if len(partial_histograms) == 0:
        raise ValueError("No partial histograms provided!")
    stacked_arrays = da.stack(partial_histograms)
    count_histogram = da.sum(stacked_arrays, axis=0)
    return count_histogram
