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
    """Determine all pairs of partitions that should be correlated within the same catalog.

    This considers all combinations, without duplicates between the "primary" and "join"
    pixels in the alignment. It does not include the combinations of partitions with themselves.

    Args:
        catalog (Catalog): catalog for auto-correlation.

    Returns:
        The alignment object where the `aligned` columns simply match the left pixel.
    """
    upper_triangle = [
        [left.order, left.pixel, right.order, right.pixel, left.order, left.pixel]
        for (left, right) in itertools.combinations(catalog.get_healpix_pixels(), 2)
    ]
    upper_triangle = pd.DataFrame(upper_triangle, columns=column_names)
    return PixelAlignment(catalog.pixel_tree, upper_triangle, PixelAlignmentType.OUTER)


def crosscorrelation_alignment(catalog_left: Catalog, catalog_right: Catalog) -> PixelAlignment:
    """Determine all pairs of partitions that should be correlated between two catalogs.

    This considers the full cross-product of pixels.

    Args:
        catalog_left (Catalog): left side of the cross-correlation.
        catalog_right (Catalog): right side of the cross-correlation.

    Returns:
        The alignment object where the `aligned` columns simply match the left pixel.
    """
    full_product = [
        [left.order, left.pixel, right.order, right.pixel, left.order, left.pixel]
        for (left, right) in itertools.product(
            catalog_left.get_healpix_pixels(), catalog_right.get_healpix_pixels()
        )
    ]
    result_mapping = pd.DataFrame(full_product, columns=column_names)
    return PixelAlignment(catalog_left.pixel_tree, result_mapping, PixelAlignmentType.OUTER)
