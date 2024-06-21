import hipscat

from corrgi.alignment import autocorrelation_alignment, crosscorrelation_alignment


def test_autocorrelation_alignment(data_catalog_dir):
    data_catalog = hipscat.read_from_hipscat(data_catalog_dir)
    alignment = autocorrelation_alignment(data_catalog)
    assert len(alignment.pixel_mapping) == 21
    assert len(alignment.pixel_mapping.columns) == 6


def test_crosscorrelation_alignment(dr7_lrg_catalog_dir, dr7_lrg_rand_catalog_dir):
    dr7_catalog = hipscat.read_from_hipscat(dr7_lrg_catalog_dir)
    dr7_rand_catalog = hipscat.read_from_hipscat(dr7_lrg_rand_catalog_dir)
    alignment = crosscorrelation_alignment(dr7_catalog, dr7_rand_catalog)
    ## dr7_catalog has 12 partitions
    ## dr7_rand_catalog has 22 partitions
    ## 12*21 = 252
    assert len(alignment.pixel_mapping) == 252
    assert len(alignment.pixel_mapping.columns) == 6
