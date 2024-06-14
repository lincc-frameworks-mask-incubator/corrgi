import hipscat
from corrgi.alignment import autocorrelation_alignment


def test_autocorrelation_alignment(data_catalog_dir):
    data_catalog = hipscat.read_from_hipscat(data_catalog_dir)
    alignment = autocorrelation_alignment(data_catalog)
    assert len(alignment.pixel_mapping) == 28
