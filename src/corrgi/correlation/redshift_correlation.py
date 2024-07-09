from typing import Callable

import pandas as pd
from hipscat.catalog.catalog_info import CatalogInfo

from corrgi.correlation.correlation import Correlation
from lsdb import Catalog


class RedshiftCorrelation(Correlation):
    """The redshift correlation utilities."""

    def validate(self, catalogs: list[Catalog]):
        super().validate(catalogs)

    def make_bins(self):
        """Generate bins for the correlation"""
        raise NotImplementedError()

    def _get_auto_method(self) -> Callable:
        raise NotImplementedError()

    def _construct_auto_args(self, df: pd.DataFrame, catalog_info: CatalogInfo) -> list:
        raise NotImplementedError()

    def _get_cross_method(self) -> Callable:
        raise NotImplementedError()

    def _construct_cross_args(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_catalog_info: CatalogInfo,
        right_catalog_info: CatalogInfo,
    ) -> list:
        raise NotImplementedError()
