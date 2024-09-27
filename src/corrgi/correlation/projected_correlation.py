from typing import Callable

import gundam.cflibfor as cff
import hipscat as hc
import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from gundam import gundam
from munch import Munch

from corrgi.correlation.correlation import Correlation


class ProjectedCorrelation(Correlation):
    """The projected correlation utilities."""

    def __init__(
        self,
        params: Munch,
        weight_column: str = "wei",
        use_weights: bool = False,
        redshift_column: str = "z",
    ):
        super().__init__(params, weight_column, use_weights)
        self.redshift_column = redshift_column
        self.sepp, self.sepv = self.make_bins()
        self.cosmo = LambdaCDM(H0=params.h0, Om0=params.omegam, Ode0=params.omegal)

    def validate(self, hc_catalogs: list[hc.catalog.Catalog]):
        """Validate that the correlation args/data are valid"""
        super().validate(hc_catalogs)
        for catalog in hc_catalogs:
            if catalog.schema.field_by_name(self.redshift_column) is None:
                raise ValueError(
                    f"Redshift column {self.redshift_column} does not"
                    + f" exist in {catalog.catalog_info.catalog_name}"
                )

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

    def _construct_auto_args(self, df: pd.DataFrame, ra_column: str, dec_column: str) -> list:
        args = [
            len(df),
            self.calculate_comoving_distances(df),
            *self.get_coords(df, ra_column, dec_column),  # cartesian coordinates
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
        left_ra_column: str,
        left_dec_column: str,
        right_ra_column: str,
        right_dec_column: str,
    ) -> list:
        args = [
            len(left_df),
            self.calculate_comoving_distances(left_df),
            *self.get_coords(left_df, left_ra_column, left_dec_column),
            len(right_df),
            self.calculate_comoving_distances(right_df),
            *self.get_coords(right_df, right_ra_column, right_dec_column),
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
