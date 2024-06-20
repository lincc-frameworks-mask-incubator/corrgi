import numpy as np
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
    return x, y, z


def project_bins(sep: np.ndarray) -> np.ndarray:
    """Projects the angular bins to cartesian space according
    to a sphere of radius=0.5.

    Args:
        sep (np.ndarray): The bins, in angular space.

    Returns:
        The distance bins in the projected cartesian space.
    """
    return (np.sin(0.5 * sep * deg2rad)) ** 2


def join_count_histograms(partial_histograms: list[np.ndarray]) -> np.ndarray:
    """Joins the partial count histograms.

    Args:
        partial_histograms (list[np.ndarray]): The list of count histograms
            generated for each pair of partitions.

    Returns:
        The numpy array that results from combining all partial histograms.
    """
    return np.sum(np.stack(partial_histograms), axis=0)
