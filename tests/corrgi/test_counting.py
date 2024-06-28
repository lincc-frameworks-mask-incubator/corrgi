import dask
import hipscat

from corrgi.dask import _count_auto_pairs


def test_count_auto_pairs(
    single_data_partition,
    data_catalog_dir,
    corr_bins,
    autocorr_params,
):
    data_catalog = hipscat.read_from_hipscat(data_catalog_dir)
    partial = _count_auto_pairs(
        single_data_partition, data_catalog.catalog_info, corr_bins, autocorr_params
    )
    assert len(partial) == len(corr_bins) - 1
