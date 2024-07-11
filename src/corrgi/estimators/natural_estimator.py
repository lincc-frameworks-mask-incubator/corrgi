import dask
import numpy as np
from lsdb import Catalog

from corrgi.correlation.projected_correlation import ProjectedCorrelation
from corrgi.dask import perform_auto_counts
from corrgi.estimators.estimator import Estimator


class NaturalEstimator(Estimator):
    """Natural Estimator (`DD/RR - 1`)"""

    def _get_auto_args(self, catalog: Catalog, random: Catalog) -> list:
        """Returns the args for the auto-correlation estimator subroutine"""
        num_galaxies = catalog.hc_structure.catalog_info.total_rows
        num_random = random.hc_structure.catalog_info.total_rows
        counts_dd, counts_rr = self.compute_autocorrelation_counts(catalog, random)
        counts_dr = 0
        bdd = np.zeros([len(counts_dd), 0])
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

    def compute_autocorrelation_counts(self, catalog: Catalog, random: Catalog) -> list[np.ndarray]:
        """Computes the auto-correlation counts for the provided catalog"""
        counts_dd = perform_auto_counts(catalog, self.correlation)
        counts_rr = perform_auto_counts(random, self.correlation)
        return dask.compute(*[counts_dd, counts_rr])
