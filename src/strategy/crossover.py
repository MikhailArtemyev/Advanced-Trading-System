"""Shared SMA crossover detection logic."""

import pandas as pd


def detect_sma_crossover(
    close_prices: pd.Series,
    short_window: int,
    long_window: int,
    prev_short_sma: float | None,
    prev_long_sma: float | None,
) -> tuple[str | None, float, float]:
    """Detect SMA crossover from price data.

    Args:
        close_prices: Close price series (must have at least long_window rows)
        short_window: Short SMA period
        long_window: Long SMA period
        prev_short_sma: Previous short SMA value (None if first call)
        prev_long_sma: Previous long SMA value (None if first call)

    Returns:
        Tuple of (crossover_type, short_sma, long_sma) where crossover_type
        is "bullish", "bearish", or None.
    """
    short_sma = float(close_prices.tail(short_window).mean())
    long_sma = float(close_prices.tail(long_window).mean())

    if prev_short_sma is None or prev_long_sma is None:
        return None, short_sma, long_sma

    crossover = None
    if prev_short_sma <= prev_long_sma and short_sma > long_sma:
        crossover = "bullish"
    elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
        crossover = "bearish"

    return crossover, short_sma, long_sma
