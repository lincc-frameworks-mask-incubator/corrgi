from hipscat.pixel_tree import PixelAlignment
from lsdb import Catalog


def get_pixel_alignment(left: Catalog, right: Catalog) -> PixelAlignment:
    """Computes the pixel alignment between two catalogs."""
    raise NotImplementedError()
