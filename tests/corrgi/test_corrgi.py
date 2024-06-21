import lsdb
import numpy.testing as npt

from corrgi.corrgi import compute_autocorrelation


def test_acf1_natural_estimator(dask_client, data_catalog_dir, rand_catalog_dir, w_acf_nat, w_acf_nat_true):
    galaxy_catalog = lsdb.read_hipscat(data_catalog_dir)
    random_catalog = lsdb.read_hipscat(rand_catalog_dir)
    assert isinstance(galaxy_catalog, lsdb.Catalog)
    assert isinstance(random_catalog, lsdb.Catalog)
    result = compute_autocorrelation(galaxy_catalog, random_catalog)
    npt.assert_allclose(result, w_acf_nat_true, rtol=1e-7)
