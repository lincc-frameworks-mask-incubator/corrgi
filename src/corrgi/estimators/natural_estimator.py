from __future__ import annotations

import numpy as np
from distributed import Client
from hipscat.io import FilePointer

from corrgi.estimators.estimator import Estimator
from corrgi.pipeline.arguments import CorrgiArguments
from corrgi.pipeline.run_counting import run_counting


class NaturalEstimator(Estimator):
    """Natural Estimator"""

    def compute_autocorrelation_counts(
        self,
        catalog_path: FilePointer,
        random_catalog_path: FilePointer,
        *,
        output_dir: str,
        client: Client,
    ) -> list[np.ndarray, np.ndarray, np.ndarray | int]:
        """Computes the auto-correlation counts for the provided catalog (`DD/RR - 1`).

        Args:
            catalog_path (str): A galaxy samples catalog (D).
            random_catalog_path (str): A random samples catalog (R).
            output_dir (str): The path to the directory where we save the intermediate
                (per-pixel) counts as well as the final correlation result (as numpy arrays).
            client (Client): The distributed client instance.

        Returns:
            The DD, RR and DR counts for the natural estimator.
        """
        counts_dd = run_counting(
            CorrgiArguments(
                left_catalog_path=catalog_path,
                right_catalog_path=catalog_path,
                correlation=self.correlation,
                output_path=output_dir,
                output_artifact_name="dd",
            ),
            client,
        )
        counts_rr = run_counting(
            CorrgiArguments(
                left_catalog_path=random_catalog_path,
                right_catalog_path=random_catalog_path,
                correlation=self.correlation,
                output_path=output_dir,
                output_artifact_name="rr",
            ),
            client,
        )
        counts_dr = 0  # The natural estimator does not use DR counts
        counts_dd_rr = self.correlation.transform_counts([counts_dd, counts_rr])
        return [*counts_dd_rr, counts_dr]

    def compute_crosscorrelation_counts(
        self,
        left_catalog_path: FilePointer,
        right_catalog_path: FilePointer,
        random_catalog_path: FilePointer,
        *,
        output_dir: str,
        client: Client,
    ) -> list[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the cross-correlation counts for the provided catalog"""
        raise NotImplementedError()
