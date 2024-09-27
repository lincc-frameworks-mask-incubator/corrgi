from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import hipscat as hc
import numpy as np
import pandas as pd

from corrgi.utils import project_coordinates


class Correlation(ABC):
    """Correlation base class."""

    def __init__(
        self,
        params: dict,
        weight_column: str = "wei",
        use_weights: bool = False,
    ):
        self.params = params
        self.weight_column = weight_column
        self.use_weights = use_weights

    def validate(self, hc_catalogs: list[hc.catalog.Catalog]):
        """Validates the catalogs. Makes sure that if we're using weights, the
        specified column exists in all catalogs."""
        if not self.use_weights:
            return
        for catalog in hc_catalogs:
            if catalog.schema.field_by_name(self.weight_column) is None:
                raise ValueError(
                    f"Weight column '{self.weight_column}' does not"
                    + f" exist in {catalog.catalog_info.catalog_name}"
                )

    def count_auto_pairs(self, df: pd.DataFrame, ra_column: str, dec_column: str) -> np.ndarray:
        """Computes the counts for pairs of the same partition"""
        args = self._construct_auto_args(df, ra_column, dec_column)
        return self._get_auto_method()(*args)

    def count_cross_pairs(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_ra_column: str,
        left_dec_column: str,
        right_ra_column: str,
        right_dec_column: str,
    ) -> np.ndarray:
        """Computes the counts for pairs of different partitions"""
        args = self._construct_cross_args(
            left_df, right_df, left_ra_column, left_dec_column, right_ra_column, right_dec_column
        )
        return self._get_cross_method()(*args)

    @abstractmethod
    def make_bins(self):
        """Generate bins for the correlation"""
        raise NotImplementedError()

    @abstractmethod
    def _get_auto_method(self) -> Callable:
        """Reference to Fortran routine to be called on auto pairing"""
        raise NotImplementedError()

    @abstractmethod
    def _construct_auto_args(self, df: pd.DataFrame, ra_column: str, dec_column: str) -> list:
        """Generate the arguments required for the auto pairing method"""
        raise NotImplementedError()

    @abstractmethod
    def _get_cross_method(self) -> Callable:
        """Reference to Fortran routine to be called on cross pairing"""
        raise NotImplementedError()

    @abstractmethod
    def _construct_cross_args(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_ra_column: str,
        left_dec_column: str,
        right_ra_column: str,
        right_dec_column: str,
    ) -> list:
        """Generate the arguments required for the cross pairing method"""
        raise NotImplementedError()

    @abstractmethod
    def get_bdd_counts(self) -> np.ndarray:
        """Returns the boostrap counts for the correlation"""
        raise NotImplementedError()

    def transform_counts(self, counts: list[np.ndarray]) -> list[np.ndarray]:
        """Applies final transformations to the correlation counts"""
        return counts

    @staticmethod
    def get_coords(df: pd.DataFrame, ra_column: str, dec_column: str) -> tuple[float, float, float]:
        """Calculate the cartesian coordinates for the points in the partition"""
        return project_coordinates(ra=df[ra_column].to_numpy(), dec=df[dec_column].to_numpy())
