import lsdb
from corrgi.correlation.projected_correlation import ProjectedCorrelation
from corrgi.dask import compute_autocorrelation_counts
import numpy.testing as npt


def test_pcf_counts_are_correct(
    dask_client,
    data_catalog_dir,
    rand_catalog_dir,
    pcf_dd_counts,
    pcf_rr_counts,
    pcf_params,
):
    galaxy_catalog = lsdb.read_hipscat(data_catalog_dir)
    random_catalog = lsdb.read_hipscat(rand_catalog_dir)
    assert isinstance(galaxy_catalog, lsdb.Catalog)
    assert isinstance(random_catalog, lsdb.Catalog)
    counts_dd, counts_rr = compute_autocorrelation_counts(
        ProjectedCorrelation, galaxy_catalog, random_catalog, pcf_params
    )
    npt.assert_allclose(counts_dd.transpose([1, 0]), pcf_dd_counts, rtol=1e-3)
    npt.assert_allclose(counts_rr.transpose([1, 0]), pcf_rr_counts, rtol=2e-3)
