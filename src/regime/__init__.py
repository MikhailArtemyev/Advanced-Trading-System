"""Regime detection for market state classification.

Provides Hidden Markov Model-based regime detection that classifies
market periods as bull, bear, or sideways based on return and
volatility characteristics.
"""

from .hmm import RegimeDetector, RegimeResult

__all__ = [
    "RegimeDetector",
    "RegimeResult",
]
