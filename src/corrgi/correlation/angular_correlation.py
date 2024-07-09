from typing import Callable

import gundam.cflibfor as cff
import pandas as pd
from gundam import gundam
from hipscat.catalog.catalog_info import CatalogInfo
from munch import Munch

from corrgi.correlation.correlation import Correlation


class AngularCorrelation(Correlation):
    """The angular correlation utilities."""

    def __init__(self, params: Munch, weight_column: str = "wei", use_weights: bool = False):
        super().__init__(params, weight_column, use_weights)
        self.sept = self.make_bins()

    def make_bins(self) -> list:
        """Generate the angular separation bins"""
        bins, _ = gundam.makebins(
            self.params.nsept, self.params.septmin, self.params.dsept, self.params.logsept
        )
        return bins

    def _get_auto_method(self):
        return cff.mod.th_A_wg_naiveway if self.use_weights else cff.mod.th_A_naiveway

    def _get_cross_method(self) -> Callable:
        return cff.mod.th_C_wg_naiveway if self.use_weights else cff.mod.th_C_naiveway

    def _construct_auto_args(self, df: pd.DataFrame, catalog_info: CatalogInfo) -> list:
        args = [
            len(df),
            *self.get_coords(df, catalog_info),  # cartesian coordinates
            self.params.nsept,  # number of angular separation bins
            self.sept,  # bins in angular separation [deg]
        ]
        if self.use_weights:
            args = [args[0], df[self.weight_column].to_numpy(), *args[1:]]
        return args

    def _construct_cross_args(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_catalog_info: CatalogInfo,
        right_catalog_info: CatalogInfo,
    ) -> list:
        args = [
            len(left_df),  # number of particles of the left partition
            *self.get_coords(left_df, left_catalog_info),  # X,Y,Z coordinates of particles
            len(right_df),  # number of particles of the right partition
            *self.get_coords(right_df, right_catalog_info),  # X,Y,Z coordinates of particles
            self.params.nsept,  # number of angular separation bins
            self.sept,  # bins in angular separation [deg]
        ]
        if self.use_weights:
            args = [
                args[0],
                left_df[self.weight_column].to_numpy(),
                *args[1:5],
                right_df[self.weight_column].to_numpy(),
                *args[5:],
            ]
        return args
