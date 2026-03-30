"""Composite Strategy — regime-adaptive strategy switching.

Detects market regime and routes to the appropriate sub-strategy:
- Trending regime (high dispersion): momentum captures big moves
- Choppy regime (low dispersion): mean reversion profits from oscillations

Regime detection uses cross-sectional return dispersion: when stocks
are moving in different directions (high std of returns), momentum works.
When they're tightly clustered, mean reversion works.
"""

from datetime import datetime
from typing import Any

import numpy as np

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from .base_strategy import Strategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy


class CompositeStrategy(Strategy):
    """Regime-adaptive: switches between momentum and mean reversion.

    Parameters (via ``parameters`` dict):
        momentum: Dict of momentum strategy parameters.
        mean_reversion: Dict of mean reversion strategy parameters.
        dispersion_lookback: Bars to compute return dispersion (default 20).
        dispersion_threshold: Std of returns above this → trending regime,
            below → choppy regime (default 0.05).
    """

    def __init__(
        self,
        symbols: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(symbols, parameters)

        mom_params = self.parameters.get("momentum", {})
        mr_params = self.parameters.get("mean_reversion", {})
        self._dispersion_lookback: int = self.parameters.get(
            "dispersion_lookback", 20
        )
        self._dispersion_threshold: float = self.parameters.get(
            "dispersion_threshold", 0.05
        )

        self._momentum = MomentumStrategy(symbols, mom_params)
        self._mean_reversion = MeanReversionStrategy(symbols, mr_params)
        self._active: str = "momentum"  # Track which strategy is active

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Detect regime, run the appropriate strategy."""
        # Sync positions to both sub-strategies
        for symbol in self.symbols:
            pos = self.current_positions.get(symbol, 0)
            self._momentum.update_position(symbol, pos)
            self._mean_reversion.update_position(symbol, pos)

        # Detect regime
        regime = self._detect_regime(data_handler)

        # If regime changed, exit all positions first
        if regime != self._active:
            exits = self._exit_all(timestamp)
            self._active = regime
            if exits:
                return exits

        # Run the active strategy
        if self._active == "momentum":
            return self._momentum.calculate_signals(timestamp, data_handler)
        else:
            return self._mean_reversion.calculate_signals(timestamp, data_handler)

    def _detect_regime(self, data_handler: DataHandler) -> str:
        """Detect trending vs choppy regime via return dispersion.

        High dispersion (stocks diverging) → momentum.
        Low dispersion (stocks clustered) → mean reversion.
        """
        n = self._dispersion_lookback
        returns: list[float] = []

        for symbol in self.symbols:
            bars = data_handler.get_latest_bars(symbol, n)
            if len(bars) < n:
                continue
            closes = bars["close"].values
            ret = (closes[-1] - closes[0]) / closes[0]
            returns.append(ret)

        if len(returns) < 5:
            return self._active  # Not enough data, keep current

        dispersion = float(np.std(returns))

        if dispersion >= self._dispersion_threshold:
            return "momentum"
        else:
            return "mean_reversion"

    def _exit_all(self, timestamp: datetime) -> list[SignalEvent]:
        """Exit all open positions on regime switch."""
        signals: list[SignalEvent] = []
        for symbol in self.symbols:
            pos = self.current_positions.get(symbol, 0)
            if pos != 0:
                signals.append(
                    self._create_signal(
                        timestamp, symbol, SignalType.EXIT, strength=1.0
                    )
                )
        return signals
