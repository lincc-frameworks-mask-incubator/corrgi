import numpy as np
from gundam import gundam
from numpy import deg2rad


def project_coordinates(ra: np.ndarray, dec: np.ndarray, radius: float = 0.5) -> tuple[float, float, float]:
    """Project spherical coordinates (ra, dec) to cartesian space
    on a sphere of radius=0.5.

    Args:
        ra (np.ndarray): Right ascension, in radians.
        dec (np.ndarray): Declination, in radians.
        radius (float): The radius of the sphere. Defaults to 0.5.

    Returns:
        The coordinates, in cartesian space, on a sphere of radius 0.5.
    """
    return gundam.radec2xyz(ra * np.pi / 180.0, dec * np.pi / 180.0, r=radius)


def project_bins(angular_bins: np.ndarray) -> np.ndarray:
    """Project the angular bins to cartesian space according
    to a sphere of radius=0.5.

    Args:
        angular_bins (np.ndarray): The bins, in angular space.

    Returns:
        The distance bins in the projected cartesian space.
    """
    return (np.sin(0.5 * angular_bins * deg2rad)) ** 2


def join_count_histograms(partial_histograms: list[np.ndarray]) -> np.ndarray:
    """Stack all partial histograms and sum their counts.

    Args:
        partial_histograms (list[np.ndarray]): The list of count histograms
            generated for each pair of partitions.

    Returns:
        The numpy array with the total counts for the partial histograms.
    """
    return np.sum(np.stack(partial_histograms), axis=0)
