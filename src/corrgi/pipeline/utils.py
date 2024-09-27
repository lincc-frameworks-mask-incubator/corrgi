from collections import defaultdict

import hipscat as hc
from hipscat.pixel_math import HealpixPixel
from lsdb.dask.merge_catalog_functions import get_healpix_pixels_from_alignment

from corrgi.alignment import autocorrelation_alignment, crosscorrelation_alignment


def get_auto_file_alignment(catalog: hc.catalog.Catalog):
    """Returns the auto and cross pairs for a single catalog"""
    alignment = autocorrelation_alignment(catalog)
    auto_pixels = [pixel for pixel in catalog.get_healpix_pixels()]
    cross_pixels = get_healpix_pixels_from_alignment(alignment)
    cross_pixels = get_groups_by_left_pixel(cross_pixels)
    return auto_pixels, cross_pixels


def get_cross_file_alignment(left_hc_catalog: hc.catalog.Catalog, right_hc_catalog: hc.catalog.Catalog):
    """Returns the cross pairs for two catalogs"""
    alignment = crosscorrelation_alignment(left_hc_catalog, right_hc_catalog)
    cross_pixels = get_healpix_pixels_from_alignment(alignment)
    cross_pixels = get_groups_by_left_pixel(cross_pixels)
    return [], cross_pixels


def get_auto_pixel_keys(hp_pixels) -> dict[HealpixPixel, str]:
    """Generates the full list of mapping keys for a set of pixels"""
    auto_keys = {}
    for hp_pixel in hp_pixels:
        key = get_pixel_key(hp_pixel)
        auto_keys[hp_pixel] = "/".join([key, key])
    return auto_keys


def filter_auto_pixel_keys(pixel_keys, done_keys) -> dict[HealpixPixel, str]:
    """Filters out the done pixels for the auto counts"""
    return {pixel: mapping_key for pixel, mapping_key in pixel_keys.items() if mapping_key not in done_keys}


def get_cross_pixel_keys(cross_pixels) -> dict[HealpixPixel, tuple[list[HealpixPixel], list[str]]]:
    """Generates the full mapping of left_pixel->(right_pixels,mapping_keys)
    for a dictionary of cross pixels."""
    cross_keys = {}
    for left_pixel, right_pixels in cross_pixels.items():
        right_keys = []
        left_pixel_key = get_pixel_key(left_pixel)
        for right_pixel in right_pixels:
            right_pixel_key = get_pixel_key(right_pixel)
            pixel_key = "/".join([left_pixel_key, right_pixel_key])
            right_keys.append(pixel_key)
        cross_keys[left_pixel] = (right_pixels, right_keys)
    return cross_keys


def filter_cross_pixel_keys(cross_keys, done_cross_keys):
    """Filters out the done pairs of pixels for the cross counts"""
    remaining_cross_keys = {}
    for left_pixel, right_pixels_map in cross_keys.items():
        remaining_cross_pixels = [
            (hp_pixel, mapping_key)
            for hp_pixel, mapping_key in zip(*right_pixels_map)
            if mapping_key not in done_cross_keys
        ]
        if len(remaining_cross_pixels) > 0:
            remaining_cross_keys[left_pixel] = list(zip(*remaining_cross_pixels))
    return remaining_cross_keys


def get_groups_by_left_pixel(cross_pixels) -> dict[HealpixPixel, list[HealpixPixel]]:
    """Groups a cross-alignment by pixels on the left. This way, each worker will compute the counts
    for each left pixel relative to all its matches with pixels on the right"""
    grouped = defaultdict(list)
    for left, right in zip(*cross_pixels):
        grouped[left].append(right)
    return grouped


def get_pixel_key(hp_pixel: HealpixPixel) -> str:
    """Returns an identifier for the HEALPix pixel"""
    return f"Norder={hp_pixel.order}_Npix={hp_pixel.pixel}"
