import dask
import numpy as np
from lsdb import Catalog

from corrgi.dask import perform_auto_counts, perform_cross_counts
from corrgi.estimators.estimator import Estimator


class DavisPeeblesEstimator(Estimator):
    """Davis-Peebles Estimator (`DD/DR - 1`)"""

    def _get_auto_args(self, catalog: Catalog, random: Catalog) -> list:
        """Returns the args for the auto-correlation estimator subroutine"""
        raise NotImplementedError()

    def compute_autocorrelation_counts(self, catalog: Catalog, random: Catalog) -> list[np.ndarray]:
        """Computes the auto-correlation counts for a catalog"""
        counts_dd = perform_auto_counts(catalog, self.correlation)
        counts_dr = perform_cross_counts(catalog, random, self.correlation)
        return dask.compute(*[counts_dd, counts_dr])
