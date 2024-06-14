import itertools

import pandas as pd
from hipscat.catalog import Catalog
from hipscat.pixel_tree.pixel_alignment import PixelAlignment
from hipscat.pixel_tree.pixel_alignment_types import PixelAlignmentType

column_names = [
    PixelAlignment.PRIMARY_ORDER_COLUMN_NAME,
    PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME,
    PixelAlignment.JOIN_ORDER_COLUMN_NAME,
    PixelAlignment.JOIN_PIXEL_COLUMN_NAME,
    PixelAlignment.ALIGNED_ORDER_COLUMN_NAME,
    PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME,
]


def autocorrelation_alignment(catalog: Catalog) -> PixelAlignment:
    """TODO"""
    upper_triangle = [
        [left.order, left.pixel, right.order, right.pixel, left.order, left.pixel]
        for (left, right) in itertools.combinations(catalog.get_healpix_pixels(), 2)
    ]
    upper_triangle = pd.DataFrame(upper_triangle, columns=column_names)
    diagonal = pd.DataFrame(
        [
            [pix.order, pix.pixel, pix.order, pix.pixel, pix.order, pix.pixel]
            for pix in catalog.get_healpix_pixels()
        ]
    )
    result_mapping = pd.concat([upper_triangle, diagonal])
    return PixelAlignment(catalog.pixel_tree, result_mapping, PixelAlignmentType.OUTER)


def crosscorrelation_alignment(catalog_left: Catalog, catalog_right: Catalog) -> PixelAlignment:
    """TODO"""
    pass
