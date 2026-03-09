"""Validation framework for ML model evaluation.

Provides Combinatorial Purged Cross-Validation (CPCV) for
proper time-series model validation without information leakage.
"""

from .cpcv import CPCVResult, CPCVSplit, CPCVValidator

__all__ = [
    "CPCVResult",
    "CPCVSplit",
    "CPCVValidator",
]
