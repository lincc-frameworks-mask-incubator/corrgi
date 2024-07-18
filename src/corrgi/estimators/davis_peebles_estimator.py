from __future__ import annotations

import dask
import numpy as np
from lsdb import Catalog

from corrgi.dask import perform_cross_counts
from corrgi.estimators.estimator import Estimator


class DavisPeeblesEstimator(Estimator):
    """Davis-Peebles Estimator"""

    def compute_autocorrelation_counts(
        self, catalog: Catalog, random: Catalog
    ) -> list[np.ndarray, np.ndarray, np.ndarray | int]:
        """Computes the auto-correlation counts for the provided catalog (`DD/RR - 1`)"""
        raise NotImplementedError()

    def compute_crosscorrelation_counts(
        self, left: Catalog, right: Catalog, random: Catalog
    ) -> list[np.ndarray, np.ndarray, np.ndarray | int]:
        """Computes the cross-correlation counts for the provided catalog.

        Args:
            left (Catalog): A left galaxy samples catalog.
            right (Catalog): A right galaxy samples catalog.
            random (Catalog): A random samples catalog.

        Returns:
            The CD and CR counts for the DP estimator.
        """
        counts_cd = perform_cross_counts(right, left, self.correlation)
        counts_cr = perform_cross_counts(right, random, self.correlation)
        counts_cd_cr = dask.compute(*[counts_cd, counts_cr])
        return self.correlation.transform_counts(counts_cd_cr)
