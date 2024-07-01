from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from hipscat.catalog.catalog_info import CatalogInfo
from munch import Munch

from corrgi.utils import project_coordinates


class Correlation(ABC):
    """Correlation base class."""

    def __init__(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame | None,
        left_catalog_info: CatalogInfo,
        right_catalog_info: CatalogInfo | None,
        bins: np.ndarray,
        params: Munch,
        use_weights: bool = False,
    ):
        self.left_df = left_df
        self.right_df = right_df
        self.left_catalog_info = left_catalog_info
        self.right_catalog_info = right_catalog_info
        self.bins = bins
        self.params = params
        self.use_weights = use_weights

    def get_left_coords(self) -> tuple[float, float, float]:
        """Calculate the cartesian coordinates for the left partition"""
        return project_coordinates(
            ra=self.left_df[self.left_catalog_info.ra_column].to_numpy(),
            dec=self.left_df[self.left_catalog_info.dec_column].to_numpy(),
        )

    def get_right_coords(self) -> tuple[float, float, float]:
        """Calculate the cartesian coordinates for the right partition"""
        return project_coordinates(
            ra=self.right_df[self.right_catalog_info.ra_column].to_numpy(),
            dec=self.right_df[self.right_catalog_info.dec_column].to_numpy(),
        )

    def count_auto_pairs(self) -> np.ndarray:
        """Computes the counts for pairs of the same partition"""
        auto_method = self._get_auto_method()
        return auto_method(*self._construct_auto_args())

    def count_cross_pairs(self) -> np.ndarray:
        """Computes the counts for pairs of different partitions"""
        cross_method = self._get_cross_method()
        return cross_method(*self._construct_cross_args())

    @abstractmethod
    def _get_auto_method(self) -> Callable:
        """Reference to Fortran routine to be called on auto pairing"""
        raise NotImplementedError()

    @abstractmethod
    def _construct_auto_args(self) -> list:
        """Generate the arguments required for the auto pairing method"""
        raise NotImplementedError()

    @abstractmethod
    def _get_cross_method(self) -> Callable:
        """Reference to Fortran routine to be called on cross pairing"""
        raise NotImplementedError()

    @abstractmethod
    def _construct_cross_args(self) -> list:
        """Generate the arguments required for the cross pairing method"""
        raise NotImplementedError()
