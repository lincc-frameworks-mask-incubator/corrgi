import dask
import numpy as np
from lsdb import Catalog

from corrgi.correlation.projected_correlation import ProjectedCorrelation
from corrgi.dask import perform_auto_counts
from corrgi.estimators.estimator import Estimator


class NaturalEstimator(Estimator):
    """Natural Estimator (`DD/RR - 1`)"""

    def compute_autocorrelation_counts(self, catalog: Catalog, random: Catalog) -> list[np.ndarray]:
        """Computes the auto-correlation counts for the provided catalog"""
        counts_dd = perform_auto_counts(catalog, self.correlation)
        counts_rr = perform_auto_counts(random, self.correlation)
        counts = dask.compute(*[counts_dd, counts_rr])
        counts_dd, counts_rr = self.correlation.transform_counts(counts)
        return [counts_dd, counts_rr]

    def _get_auto_args(
        self, counts_dd: np.ndarray, counts_rr: np.ndarray, num_galaxies: int, num_random: int
    ) -> list:
        """Returns the args for the auto-correlation estimator subroutine"""
        bdd = self.correlation.get_bdd_counts()
        counts_dr = 0  # The natural estimator does not use DR counts
        args = [
            num_galaxies,
            num_random,
            counts_dd,
            bdd,
            counts_rr,
            counts_dr,
            self.correlation.params.estimator,
        ]
        if isinstance(self.correlation, ProjectedCorrelation):
            args = [*args[:6], self.correlation.params.dsepv, args[-1]]
        return args
