from typing import Callable

import gundam.cflibfor as cff
import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from gundam import gundam
from hipscat.catalog.catalog_info import CatalogInfo
from lsdb import Catalog
from munch import Munch

from corrgi.correlation.correlation import Correlation


class ProjectedCorrelation(Correlation):
    """The projected correlation utilities."""

    def __init__(
        self,
        params: Munch,
        weight_column: str = "wei",
        redshift_column: str = "z",
        use_weights: bool = False,
    ):
        super().__init__(params, weight_column, use_weights)
        self.redshift_column = redshift_column
        self.sepp, self.sepv = self.make_bins()
        self.cosmo = LambdaCDM(H0=params.h0, Om0=params.omegam, Ode0=params.omegal)

    def validate(self, catalogs: list[Catalog]):
        """Validate that the correlation args/data are valid"""
        super().validate(catalogs)
        for catalog in catalogs:
            if self.redshift_column not in catalog.columns:
                raise ValueError(f"Redshift column {self.redshift_column} not found in {catalog}")

    def make_bins(self) -> tuple[list]:
        """Generate bins of projected separation and LOS for the correlation"""
        sepp, _ = gundam.makebins(
            self.params.nsepp, self.params.seppmin, self.params.dsepp, self.params.logsepp
        )
        sepv, _ = gundam.makebins(self.params.nsepv, 0.0, self.params.dsepv, False)
        return sepp, sepv

    def calculate_comoving_distances(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate the comoving distances from the redshift of each particle"""
        return self.cosmo.comoving_distance(df[self.redshift_column].to_numpy()).value

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
            args = [*args[:2], df[self.weight_column].to_numpy(), *args[2:]]
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
            *self.get_coords(right_df, right_catalog_info),
            self.params.nsepp,
            self.sepp,
            self.params.nsepv,
            self.sepv,
        ]
        if self.use_weights:
            args = [
                *args[:2],
                left_df[self.weight_column].to_numpy(),
                *args[2:7],
                right_df[self.weight_column].to_numpy(),
                *args[7:],
            ]
        return args

    def transform_counts(self, counts: list[np.ndarray]) -> list[np.ndarray]:
        """The projected counts need to be transposed before being sent to Fortran"""
        return [c.transpose([1, 0]) for c in counts]

    def get_bdd_counts(self) -> np.ndarray:
        """Returns the boostrap counts for the projected correlation"""
        return np.zeros([self.params.nsepp, self.params.nsepv, 0])
