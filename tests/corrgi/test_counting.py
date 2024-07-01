import hipscat

from corrgi.correlation.angular_correlation import AngularCorrelation


def test_count_auto_pairs(
    single_data_partition,
    data_catalog_dir,
    corr_bins,
    autocorr_params,
):
    data_catalog = hipscat.read_from_hipscat(data_catalog_dir)
    partial = AngularCorrelation(
        single_data_partition,
        None,
        data_catalog.catalog_info,
        None,
        corr_bins,
        autocorr_params,
    ).count_auto_pairs()
    assert len(partial) == len(corr_bins) - 1
