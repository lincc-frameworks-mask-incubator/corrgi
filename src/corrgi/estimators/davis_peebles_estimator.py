from __future__ import annotations

import numpy as np
from distributed import Client
from hipscat.io import FilePointer

from corrgi.estimators.estimator import Estimator
from corrgi.pipeline.arguments import CorrgiArguments
from corrgi.pipeline.run_counting import run_counting


class DavisPeeblesEstimator(Estimator):
    """Davis-Peebles Estimator"""

    def compute_autocorrelation_counts(
        self,
        catalog_path: FilePointer,
        random_catalog_path: FilePointer,
        *,
        output_dir: str,
        client: Client,
    ) -> list[np.ndarray, np.ndarray, np.ndarray | int]:
        """Computes the auto-correlation counts for the provided catalog"""
        raise NotImplementedError()

    def compute_crosscorrelation_counts(
        self,
        left_catalog_path: FilePointer,
        right_catalog_path: FilePointer,
        random_catalog_path: FilePointer,
        *,
        output_dir: str,
        client: Client,
    ) -> list[np.ndarray, np.ndarray]:
        """Computes the cross-correlation counts for the provided catalog.

        Args:
            left_catalog_path (str): A left galaxy samples catalog (D).
            right_catalog_path (str): A right galaxy samples catalog (C).
            random_catalog_path (str): A random samples catalog (R).
            output_dir (str): The path to the directory where we save the intermediate
                (per-pixel) counts as well as the final correlation result (as numpy arrays).
            client (Client): The distributed client instance.

        Returns:
            The CD and CR counts for the DP estimator.
        """
        counts_cd = run_counting(
            CorrgiArguments(
                left_catalog_path=right_catalog_path,
                right_catalog_path=left_catalog_path,
                correlation=self.correlation,
                output_path=output_dir,
                output_artifact_name="cd",
            ),
            client,
        )
        counts_cr = run_counting(
            CorrgiArguments(
                left_catalog_path=right_catalog_path,
                right_catalog_path=random_catalog_path,
                correlation=self.correlation,
                output_path=output_dir,
                output_artifact_name="cr",
            ),
            client,
        )
        return self.correlation.transform_counts([counts_cd, counts_cr])
