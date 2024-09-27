from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from distributed import Client
from gundam.gundam import tpccf, tpccf_wrp, tpcf, tpcf_wrp
from hipscat.io import FilePointer

from corrgi.correlation.correlation import Correlation
from corrgi.correlation.projected_correlation import ProjectedCorrelation
from corrgi.utils import read_catalog_total_rows


class Estimator(ABC):
    """Estimator base class"""

    def __init__(self, correlation: Correlation):
        self.correlation = correlation

    def compute_auto_estimate(
        self,
        catalog_path: FilePointer,
        random_catalog_path: FilePointer,
        *,
        output_dir: str,
        client: Client,
    ) -> np.ndarray:
        """Computes the auto-correlation for this estimator.

        Args:
            catalog_path (str): The catalog of galaxy samples (D).
            random_catalog_path (str): The catalog of random samples (R).
            output_dir (str): The path to the directory where we save the intermediate (per-pixel)
                counts as well as the final correlation result (as numpy arrays).
            client (Client): The distributed client instance.

        Returns:
            The statistical estimate of the auto-correlation function, as a numpy array.
        """
        num_galaxies = read_catalog_total_rows(catalog_path)
        num_random = read_catalog_total_rows(random_catalog_path)
        dd, rr, dr = self.compute_autocorrelation_counts(
            catalog_path, random_catalog_path, output_dir=output_dir, client=client
        )
        args = self._get_auto_args(num_galaxies, num_random, dd, rr, dr)
        estimate, _ = self._get_auto_subroutine()(*args)
        return estimate

    def compute_cross_estimate(
        self,
        left_catalog_path: FilePointer,
        right_catalog_path: FilePointer,
        random_catalog_path: FilePointer,
        *,
        output_dir: str,
        client: Client,
    ) -> np.ndarray:
        """Computes the cross-correlation for this estimator.

        Args:
            left_catalog_path (str): The left catalog of galaxy samples (D).
            right_catalog_path (str): The right catalog of galaxy samples (C).
            random_catalog_path (str): The catalog of random samples (R).
            output_dir (str): The path to the directory where we save the intermediate (per-pixel)
                counts as well as the final correlation result (as numpy arrays).
            client (Client): The distributed client instance.

        Returns:
            The statistical estimate of the cross-correlation function, as a numpy array.
        """
        num_galaxies = read_catalog_total_rows(left_catalog_path)
        num_random = read_catalog_total_rows(random_catalog_path)
        cd, cr = self.compute_crosscorrelation_counts(
            left_catalog_path, right_catalog_path, random_catalog_path, output_dir=output_dir, client=client
        )
        args = self._get_cross_args(num_galaxies, num_random, cd, cr)
        estimate, _ = self._get_cross_subroutine()(*args)
        return estimate

    @abstractmethod
    def compute_autocorrelation_counts(
        self,
        catalog_path: FilePointer,
        random_catalog_path: FilePointer,
        *,
        output_dir: str,
        client: Client,
    ) -> list[np.ndarray, np.ndarray, np.ndarray | int]:
        """Computes the auto-correlation counts (DD, RR, DR). These counts are
        represented as numpy arrays but DR may be 0 if it isn't used (e.g. with
        the natural estimator)."""
        raise NotImplementedError()

    @abstractmethod
    def compute_crosscorrelation_counts(
        self,
        left_catalog_path: FilePointer,
        right_catalog_path: FilePointer,
        random_catalog_path: FilePointer,
        *,
        output_dir: str,
        client: Client,
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
