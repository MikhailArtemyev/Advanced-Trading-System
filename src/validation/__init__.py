"""Validation framework for ML model evaluation.

Provides Combinatorial Purged Cross-Validation (CPCV) for proper
time-series model validation, and Deflated Sharpe Ratio (DSR) for
adjusting performance metrics under multiple testing.
"""

from .cpcv import CPCVResult, CPCVSplit, CPCVValidator
from .deflated_sharpe import (
    DSRResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe,
)

__all__ = [
    "CPCVResult",
    "CPCVSplit",
    "CPCVValidator",
    "DSRResult",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe",
]
