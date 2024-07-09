import hipscat

from corrgi.correlation.angular_correlation import AngularCorrelation


def test_count_auto_pairs(
    single_data_partition,
    data_catalog_dir,
    acf_corr_bins,
    acf_params,
):
    data_catalog = hipscat.read_from_hipscat(data_catalog_dir)
    partial = AngularCorrelation(acf_params).count_auto_pairs(
        single_data_partition, data_catalog.catalog_info
    )
    assert len(partial) == len(acf_corr_bins) - 1
