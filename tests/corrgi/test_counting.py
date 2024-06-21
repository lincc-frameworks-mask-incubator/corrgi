import dask
import hipscat
import lsdb
import numpy.testing as npt

from corrgi.corrgi import compute_autocorrelation
from corrgi.dask import count_auto_pairs, count_pairs


def test_count_auto_pairs(
    single_data_partition,
    data_catalog_dir,
    corr_bins,
    autocorr_params,
):
    data_catalog = hipscat.read_from_hipscat(data_catalog_dir)
    partial = count_auto_pairs(single_data_partition, data_catalog.catalog_info, corr_bins, autocorr_params)
    partial = dask.compute(partial)
    assert len(partial[0]) == len(corr_bins) - 1
