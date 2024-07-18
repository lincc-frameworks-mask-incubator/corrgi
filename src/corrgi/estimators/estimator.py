from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from gundam.gundam import tpccf, tpccf_wrp, tpcf, tpcf_wrp
from lsdb import Catalog

from corrgi.correlation.correlation import Correlation
from corrgi.correlation.projected_correlation import ProjectedCorrelation
from corrgi.utils import compute_catalog_size


class Estimator(ABC):
    """Estimator base class"""

    def __init__(self, correlation: Correlation):
        self.correlation = correlation

    def compute_auto_estimate(self, catalog: Catalog, random: Catalog) -> np.ndarray:
        """Computes the auto-correlation for this estimator.

        Args:
            catalog (Catalog): The catalog of galaxy samples (D).
            random (Catalog): The catalog of random samples (R).

        Returns:
            The statistical estimate of the auto-correlation function, as a numpy array.
        """
        num_galaxies = compute_catalog_size(catalog)
        num_random = compute_catalog_size(random)
        dd, rr, dr = self.compute_autocorrelation_counts(catalog, random)
        args = self._get_auto_args(num_galaxies, num_random, dd, rr, dr)
        estimate, _ = self._get_auto_subroutine()(*args)
        return estimate

    def compute_cross_estimate(self, left: Catalog, right: Catalog, random: Catalog) -> np.ndarray:
        """Computes the cross-correlation for this estimator.

        Args:
            left (Catalog): The left catalog of galaxy samples (D).
            right (Catalog): The right catalog of galaxy samples (C).
            random (Catalog): The catalog of random samples (R).

        Returns:
            The statistical estimate of the cross-correlation function, as a numpy array.
        """
        num_galaxies = compute_catalog_size(left)
        num_random = compute_catalog_size(random)
        cd, cr = self.compute_crosscorrelation_counts(left, right, random)
        args = self._get_cross_args(num_galaxies, num_random, cd, cr)
        estimate, _ = self._get_cross_subroutine()(*args)
        return estimate

    @abstractmethod
    def compute_autocorrelation_counts(
        self, catalog: Catalog, random: Catalog
    ) -> list[np.ndarray, np.ndarray, np.ndarray | int]:
        """Computes the auto-correlation counts (DD, RR, DR). These counts are
        represented as numpy arrays but DR may be 0 if it isn't used (e.g. with
        the natural estimator)."""
        raise NotImplementedError()

    @abstractmethod
    def compute_crosscorrelation_counts(
        self, left: Catalog, right: Catalog, random: Catalog
    ) -> list[np.ndarray, np.ndarray]:
        """Computes the cross-correlation counts (CD, CR)."""
        raise NotImplementedError()

    def _get_auto_subroutine(self) -> Callable:
        """Returns the Fortran routine to calculate the auto-correlation estimate"""
        return tpcf_wrp if isinstance(self.correlation, ProjectedCorrelation) else tpcf

    def _get_auto_args(
        self,
        num_galaxies: int,
        num_random: int,
        counts_dd: np.ndarray,
        counts_rr: np.ndarray,
        counts_dr: np.ndarray,
    ) -> list:
        """Returns the args for the auto-correlation estimator routine"""
        counts_bdd = self.correlation.get_bdd_counts()
        args = [num_galaxies, num_random, counts_dd, counts_bdd, counts_rr, counts_dr]
        if isinstance(self.correlation, ProjectedCorrelation):
            # The projected routines require an additional parameter
            args.append(self.correlation.params.dsepv)
        args.append(self.correlation.params.estimator)
        return args

    def _get_cross_subroutine(self) -> Callable:
        """Returns the Fortran routine to calculate the cross-correlation estimate"""
        return tpccf_wrp if isinstance(self.correlation, ProjectedCorrelation) else tpccf

    def _get_cross_args(
        self,
        num_galaxies: int,
        num_random: int,
        counts_cd: np.ndarray,
        counts_cr: np.ndarray,
    ) -> list:
        """Returns the args for the cross-correlation estimator routine"""
        counts_bdd = self.correlation.get_bdd_counts()
        args = [num_galaxies, num_random, counts_cd, counts_bdd, counts_cr]
        if isinstance(self.correlation, ProjectedCorrelation):
            # The projected routines require an additional parameter
            args.append(self.correlation.params.dsepv)
        args.append(self.correlation.params.estimator)
        return args
