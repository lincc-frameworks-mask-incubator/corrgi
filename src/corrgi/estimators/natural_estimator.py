from __future__ import annotations

import dask
import numpy as np
from lsdb import Catalog

from corrgi.dask import perform_auto_counts
from corrgi.estimators.estimator import Estimator


class NaturalEstimator(Estimator):
    """Natural Estimator"""

    def compute_autocorrelation_counts(
        self, catalog: Catalog, random: Catalog
    ) -> list[np.ndarray, np.ndarray, np.ndarray | int]:
        """Computes the auto-correlation counts for the provided catalog (`DD/RR - 1`).

        Args:
            catalog (Catalog): A galaxy samples catalog (D).
            random (Catalog): A random samples catalog (R).

        Returns:
            The DD, RR and DR counts for the natural estimator.
        """
        counts_dd = perform_auto_counts(catalog, self.correlation)
        counts_rr = perform_auto_counts(random, self.correlation)
        counts_dr = 0  # The natural estimator does not use DR counts
        counts_dd_rr = dask.compute(*[counts_dd, counts_rr])
        counts_dd_rr = self.correlation.transform_counts(counts_dd_rr)
        return [*counts_dd_rr, counts_dr]

    def compute_crosscorrelation_counts(
        self, left: Catalog, right: Catalog, random: Catalog
    ) -> list[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the cross-correlation counts for the provided catalog"""
        raise NotImplementedError()
