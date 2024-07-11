import abc
from abc import abstractmethod
from typing import Callable

import numpy as np
from gundam.gundam import tpcf, tpcf_wrp
from lsdb import Catalog

from corrgi.correlation.correlation import Correlation
from corrgi.correlation.projected_correlation import ProjectedCorrelation


class Estimator(abc.ABC):
    """Estimator base class"""

    def __init__(self, correlation: Correlation):
        self.correlation = correlation

    def compute_auto_estimate(self, catalog: Catalog, random: Catalog) -> np.ndarray:
        """Computes the auto-correlation estimate"""
        subroutine_args = self._get_auto_args(catalog, random)
        wth, _ = self._get_auto_subroutine()(*subroutine_args)
        return wth

    def _get_auto_subroutine(self) -> Callable:
        """Returns the Fortran routine to calculate the correlation estimate"""
        return tpcf_wrp if isinstance(self.correlation, ProjectedCorrelation) else tpcf

    @abstractmethod
    def _get_auto_args(self, catalog: Catalog, random: Catalog) -> list:
        """Returns the args for the auto-correlation estimator subroutine"""
        raise NotImplementedError()

    @abstractmethod
    def compute_autocorrelation_counts(self, catalog: Catalog, random: Catalog) -> list[np.ndarray]:
        """Computes the auto-correlation counts for the provided catalog"""
        raise NotImplementedError()
