import tempfile
from pathlib import Path

from corrgi.correlation.projected_correlation import ProjectedCorrelation
from corrgi.estimators.davis_peebles_estimator import DavisPeeblesEstimator
from corrgi.estimators.natural_estimator import NaturalEstimator
from distributed import Client
from gundam import gundam

DATA_DIR_NAME = Path(__file__).parent.parent / "tests" / "data" / "hipscat"
GALS_WEIGHT_DIR = str(DATA_DIR_NAME / "pcf_gals_weight")
GALS1_WEIGHT_DIR = str(DATA_DIR_NAME / "pcf_gals1_weight")
RANS_WEIGHT_DIR = str(DATA_DIR_NAME / "pcf_rans_weight")


def generate_pcf_params():
    """Create the projected params"""
    params = gundam.packpars(kind="pcf")
    params.nsepp = 28  # Number of bins of projected separation rp
    params.seppmin = 0.02  # Minimum rp in Mpc/h
    params.dsepp = 0.12  # Bin size of rp (in log space)
    params.nsepv = 1  # Number of bins of LOS separation pi
    params.dsepv = 40.0  # Bin size of pi (in linear space)
    params.omegam = 0.25  # Omega matter
    params.omegal = 0.75  # Omega lambda
    params.h0 = 100  # Hubble constant [km/s/Mpc]
    return params


class ProjectedSuite:
    """Benchmarks for projected correlation"""

    timeout = 600  # in seconds

    def setup(self):
        """Creates client and output directory for each benchmark"""
        self.dask_client = Client(n_workers=4, threads_per_worker=1)
        self.output_dir = tempfile.TemporaryDirectory()

    def teardown(self):
        """Tears down client and output directory for each benchmark"""
        self.dask_client.close()
        self.output_dir.cleanup()

    def time_pcf_natural_estimator(self):
        """Times the Natural estimator for a projected auto-correlation"""
        estimator = NaturalEstimator(ProjectedCorrelation(params=generate_pcf_params(), use_weights=True))
        estimator.compute_autocorrelation_counts(
            GALS_WEIGHT_DIR, RANS_WEIGHT_DIR, output_dir=self.output_dir.name, client=self.dask_client
        )

    def time_pccf_davis_peebles_estimator(self):
        """Times the Davis-Peebles estimator for a projected cross-correlation"""
        estimator = DavisPeeblesEstimator(
            ProjectedCorrelation(params=generate_pcf_params(), use_weights=True)
        )
        estimator.compute_crosscorrelation_counts(
            GALS_WEIGHT_DIR,
            GALS1_WEIGHT_DIR,
            RANS_WEIGHT_DIR,
            output_dir=self.output_dir.name,
            client=self.dask_client,
        )
