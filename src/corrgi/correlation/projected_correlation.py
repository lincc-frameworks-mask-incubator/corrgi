from typing import Callable

import gundam.cflibfor as cff
import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from gundam import gundam
from hipscat.catalog.catalog_info import CatalogInfo
from munch import Munch

from corrgi.correlation.correlation import Correlation


class ProjectedCorrelation(Correlation):
    """The projected correlation utilities."""

    def __init__(self, params: Munch, use_weights: bool = False):
        super().__init__(params, use_weights)
        self.cosmo = LambdaCDM(H0=params.h0, Om0=params.omegam, Ode0=params.omegal)
        self.sepp, self.sepv = self.make_bins()

    def make_bins(self) -> tuple[list]:
        """Generate bins of projected separation and LOS for the correlation"""
        sepp, _ = gundam.makebins(
            self.params.nsepp, self.params.seppmin, self.params.dsepp, self.params.logsepp
        )
        sepv, _ = gundam.makebins(self.params.nsepv, 0.0, self.params.dsepv, False)
        return sepp, sepv

    def calculate_comoving_distances(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate the comoving distances from the redshift of each particle"""
        return self.cosmo.comoving_distance(df["z"].to_numpy()).value

    def _get_auto_method(self) -> Callable:
        return cff.mod.rppi_A_wg_naiveway if self.use_weights else cff.mod.rppi_A_naiveway

    def _construct_auto_args(self, df: pd.DataFrame, catalog_info: CatalogInfo) -> list:
        args = [
            len(df),
            self.calculate_comoving_distances(df),
            *self.get_coords(df, catalog_info),  # cartesian coordinates
            self.params.nsepp,  # number of bins of projected separation rp
            self.sepp,  # Bins in projected separation rp
            self.params.nsepv,  # number of bins of LOS separation pi
            self.sepv,  # Bins in radial separation
        ]
        if self.use_weights:
            args = [*args[:2], df["wei"].to_numpy(), *args[2:]]
        return args

    def _get_cross_method(self) -> Callable:
        return cff.mod.rppi_C_wg_naiveway if self.use_weights else cff.mod.rppi_C_naiveway

    def _construct_cross_args(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_catalog_info: CatalogInfo,
        right_catalog_info: CatalogInfo,
    ) -> list:
        args = [
            len(left_df),
            self.calculate_comoving_distances(left_df),
            *self.get_coords(left_df, left_catalog_info),
            len(right_df),
            self.calculate_comoving_distances(right_df),
            *self.get_coords(right_df, left_catalog_info),
            self.params.nsepp,
            self.sepp,
            self.params.nsepv,
            self.sepv,
        ]
        if self.use_weights:
            args = [*args[:2], left_df["wei"], *args[2:5], right_df["wei"], *args[5:]]
        return args
