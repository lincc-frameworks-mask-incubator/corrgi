from typing import Callable

import gundam.cflibfor as cff
import numpy as np

from corrgi.correlation.correlation import Correlation


class AngularCorrelation(Correlation):
    """The angular correlation utilities."""

    def _get_auto_method(self):
        return cff.mod.th_A_wg if self.use_weights else cff.mod.th_A_naiveway

    def _get_cross_method(self) -> Callable:
        return cff.mod.th_C_wg if self.use_weights else cff.mod.th_C

    def _construct_auto_args(self):
        return [
            len(self.left_df),  # number of particles
            *self.get_left_coords(),  # cartesian coordinates
            self.params.nsept,  # number of angular separation bins
            self.bins,  # bins in angular separation [deg]
        ]

    def _construct_cross_args(self):
        return [
            1,  # number of threads OpenMP
            len(self.left_df),  # number of particles of the left partition
            self.left_df[self.left_catalog_info.ra_column].to_numpy(),  # RA of particles [deg]
            self.left_df[self.left_catalog_info.dec_column].to_numpy(),  # DEC of particles [deg]
            *self.get_left_coords(),  # X,Y,Z coordinates of particles
            len(self.right_df),  # number of particles of the right partition
            *self.get_right_coords(),  # X,Y,Z coordinates of particles
            self.params.nsept,  # number of angular separation bins
            self.bins,  # bins in angular separation [deg]
            self.params.sbound,
            self.params.mxh1,
            self.params.mxh2,
            self.params.cntid,
            self.params.logf,
            self.params.sk1,
            np.zeros(len(self.right_df)),
            self.params.grid,
        ]
