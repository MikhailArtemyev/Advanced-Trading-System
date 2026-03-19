"""Validation utilities for strategy evaluation.

Provides Deflated Sharpe Ratio (DSR) for adjusting performance metrics
under multiple testing.
"""

from .deflated_sharpe import (
    DSRResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe,
)

__all__ = [
    "DSRResult",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe",
]
