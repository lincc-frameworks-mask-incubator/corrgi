from __future__ import annotations

from dataclasses import dataclass

import hipscat as hc
from hipscat.io import FilePointer
from hipscat_import.runtime_arguments import RuntimeArguments

from corrgi.correlation.correlation import Correlation


@dataclass
class CorrgiArguments(RuntimeArguments):
    """Container for Corrgi arguments"""

    left_catalog_path: FilePointer = ""
    """the path to the left catalog"""

    right_catalog_path: FilePointer = ""
    """the path to the cross catalog"""

    correlation: Correlation | None = None
    """correlation instance, with wrappers for each counting method"""

    simple_progress_bar: bool = True
    """use plain-text progress bar"""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()

        # Make sure PosixPaths are converted to strings
        self.left_catalog_path = str(self.left_catalog_path)
        self.right_catalog_path = str(self.right_catalog_path)

        # Load catalogs and verify their metadata
        self.left_hc_catalog = hc.read_from_hipscat(self.left_catalog_path)
        self.right_hc_catalog = hc.read_from_hipscat(self.right_catalog_path)
        self.correlation.validate([self.left_hc_catalog, self.right_hc_catalog])
