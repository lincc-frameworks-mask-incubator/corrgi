from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from hipscat.catalog.catalog_info import CatalogInfo
from lsdb import Catalog
from munch import Munch

from corrgi.utils import project_coordinates


class Correlation(ABC):
    """Correlation base class."""

    def __init__(
        self,
        params: Munch,
        weight_column: str = "wei",
        use_weights: bool = False,
    ):
        self.params = params
        self.weight_column = weight_column
        self.use_weights = use_weights

    def validate(self, catalogs: list[Catalog]):
        """Validate that the correlation args/data are valid"""
        if not self.use_weights:
            return
        for catalog in catalogs:
            if self.weight_column not in catalog.columns:
                raise ValueError(
                    f"Weight column '{self.weight_column}' does not exist"
                    + f" in {catalog.hc_structure.catalog_info.catalog_name}"
                )

    def count_auto_pairs(self, df: pd.DataFrame, catalog_info: CatalogInfo) -> np.ndarray:
        """Computes the counts for pairs of the same partition"""
        args = self._construct_auto_args(df, catalog_info)
        return self._get_auto_method()(*args)

    def count_cross_pairs(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_catalog_info: CatalogInfo,
        right_catalog_info: CatalogInfo,
    ) -> np.ndarray:
        """Computes the counts for pairs of different partitions"""
        args = self._construct_cross_args(left_df, right_df, left_catalog_info, right_catalog_info)
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
    def _construct_auto_args(self, df: pd.DataFrame, catalog_info: CatalogInfo) -> list:
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
        left_catalog_info: CatalogInfo,
        right_catalog_info: CatalogInfo,
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
    def get_coords(df: pd.DataFrame, catalog_info: CatalogInfo) -> tuple[float, float, float]:
        """Calculate the cartesian coordinates for the points in the partition"""
        return project_coordinates(
            ra=df[catalog_info.ra_column].to_numpy(), dec=df[catalog_info.dec_column].to_numpy()
        )
