import itertools

import pandas as pd
from hipscat.catalog import Catalog
from hipscat.pixel_tree.pixel_alignment import PixelAlignment
from hipscat.pixel_tree.pixel_alignment_types import PixelAlignmentType
import hipscat.pixel_math.healpix_shim as hp

from corrgi.utils import project_coordinates
import gundam.cflibfor as cff
import numpy as np

column_names = [
    PixelAlignment.PRIMARY_ORDER_COLUMN_NAME,
    PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME,
    PixelAlignment.JOIN_ORDER_COLUMN_NAME,
    PixelAlignment.JOIN_PIXEL_COLUMN_NAME,
    PixelAlignment.ALIGNED_ORDER_COLUMN_NAME,
    PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME,
]


def autocorrelation_alignment(catalog: Catalog, bins=None) -> PixelAlignment:
    """Determine all pairs of partitions that should be correlated within the same catalog.

    This considers all combinations, without duplicates between the "primary" and "join"
    pixels in the alignment. It does not include the combinations of partitions with themselves.

    Args:
        catalog (Catalog): catalog for auto-correlation.

    Returns:
        The alignment object where the `aligned` columns simply match the left pixel.
    """
    if bins is None or len(bins) == 0:
        upper_triangle = [
            [left.order, left.pixel, right.order, right.pixel, left.order, left.pixel]
            for (left, right) in itertools.combinations(catalog.get_healpix_pixels(), 2)
        ]
        upper_triangle = pd.DataFrame(upper_triangle, columns=column_names)
        return PixelAlignment(catalog.pixel_tree, upper_triangle, PixelAlignmentType.OUTER)

    all_pixels = catalog.get_healpix_pixels()
    num_partitions = len(all_pixels)
    all_radec = [
        hp.vec2dir(hp.boundaries(2**part.order, part.pixel, step=64, nest=True), lonlat=True)
        for part in all_pixels
    ]
    all_xyz = [project_coordinates(ra=bounds[0], dec=bounds[1]) for bounds in all_radec]

    pix_a_order = []
    pix_a_pixel = []
    pix_b_order = []
    pix_b_pixel = []
    for a in range(0, num_partitions):
        for b in range(a+1, num_partitions):
            a_x, a_y, a_z = all_xyz[a]
            b_x, b_y, b_z = all_xyz[b]
            args = [
                len(a_x),  # number of particles of the left partition
                a_x,
                a_y,
                a_z,  # X,Y,Z coordinates of particles
                len(b_x),  # number of particles of the right partition
                b_x,
                b_y,
                b_z,  # X,Y,Z coordinates of particles
                len(bins) - 1,  # number of angular separation bins
                bins,  # bins in angular separation [deg]
            ]
            bins_populated = cff.mod.th_C_naiveway(*args)
            populated_bins = np.flatnonzero(bins_populated)
            if len(populated_bins) > 0:
                pix_a_order.append(all_pixels[a].order)
                pix_a_pixel.append(all_pixels[a].pixel)
                pix_b_order.append(all_pixels[b].order)
                pix_b_pixel.append(all_pixels[b].pixel)

    if len(pix_a_order) == 0:
        raise ValueError("no valid overlaps")

    constrained_upper_triangle = pd.DataFrame(
        {
            PixelAlignment.PRIMARY_ORDER_COLUMN_NAME: pix_a_order,
            PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME: pix_a_pixel,
            PixelAlignment.JOIN_ORDER_COLUMN_NAME: pix_b_order,
            PixelAlignment.JOIN_PIXEL_COLUMN_NAME: pix_b_pixel,
            PixelAlignment.ALIGNED_ORDER_COLUMN_NAME: pix_a_order,
            PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME: pix_a_pixel,
        }
    )

    return PixelAlignment(catalog.pixel_tree, constrained_upper_triangle, PixelAlignmentType.OUTER)


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
