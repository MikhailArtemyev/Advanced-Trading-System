"""Walk-forward testing framework for detecting overfitting.

Splits data into rolling train/test windows, runs backtests on each
window independently, and aggregates out-of-sample results.

Walk-Forward Efficiency (WFE) measures how much of the in-sample
performance persists out-of-sample:
    - WFE > 50% suggests the strategy has a genuine edge
    - WFE < 30% suggests overfitting
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass
class WalkForwardWindow:
    """A single train/test window in walk-forward analysis.

    Attributes:
        window_id: Sequential identifier for this window
        train_start: Start date of the training (in-sample) period
        train_end: End date of the training period
        test_start: Start date of the testing (out-of-sample) period
        test_end: End date of the testing period
        train_metrics: Performance metrics from the training period backtest
        test_metrics: Performance metrics from the testing period backtest
    """

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward analysis.

    Attributes:
        windows: List of all walk-forward windows with their metrics
        aggregate_metrics: Averaged out-of-sample metrics across all windows
        walk_forward_efficiency: WFE = avg_oos_return / avg_is_return * 100
        parameter_stability: Coefficient of variation of OOS Sharpe ratios
            (lower = more stable)
    """

    windows: list[WalkForwardWindow] = field(default_factory=list)
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    walk_forward_efficiency: float = 0.0
    parameter_stability: float = 0.0


class WalkForwardRunner:
    """Walk-forward testing framework.

    Splits a date range into rolling train/test windows, runs a backtest
    on each window via a user-provided engine factory, and aggregates
    out-of-sample results to detect overfitting.

    Args:
        train_days: Number of calendar days for each training window
        test_days: Number of calendar days for each testing window
        step_days: Number of calendar days to advance between windows
        start_date: Overall start date (YYYY-MM-DD)
        end_date: Overall end date (YYYY-MM-DD)

    Raises:
        ValueError: If any days parameter is <= 0 or dates are invalid
    """

    def __init__(
        self,
        train_days: int,
        test_days: int,
        step_days: int,
        start_date: str,
        end_date: str,
    ) -> None:
        if train_days <= 0:
            raise ValueError(f"train_days must be > 0, got {train_days}")
        if test_days <= 0:
            raise ValueError(f"test_days must be > 0, got {test_days}")
        if step_days <= 0:
            raise ValueError(f"step_days must be > 0, got {step_days}")

        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

    def generate_windows(self) -> list[WalkForwardWindow]:
        """Generate train/test window pairs from the date range.

        Returns:
            List of WalkForwardWindow objects with dates populated
            but metrics empty. Empty list if the date range is too
            short for even a single window.
        """
        windows: list[WalkForwardWindow] = []
        window_id = 0
        current_start = self.start_date

        while True:
            train_start = current_start
            train_end = current_start + timedelta(days=self.train_days - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_days - 1)

            # Stop if test window would exceed the overall end date
            if test_end > self.end_date:
                # Try clamping — only if test_start is still before end_date
                if test_start <= self.end_date:
                    test_end = self.end_date
                else:
                    break

            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            window_id += 1
            current_start += timedelta(days=self.step_days)

        return windows

    def run(
        self,
        engine_factory: Callable[[str, str], Any],
    ) -> WalkForwardResult:
        """Run walk-forward analysis.

        For each window, creates train and test engines via the factory,
        runs them, and collects metrics. Aggregates results across all windows.

        Args:
            engine_factory: Function that takes (start_date, end_date) as
                YYYY-MM-DD strings and returns a BacktestEngine (or any object
                with a run() method returning a dict with an optional "metrics" key).

        Returns:
            WalkForwardResult with per-window and aggregate metrics
        """
        windows = self.generate_windows()

        if not windows:
            return WalkForwardResult()

        total = len(windows)
        for window in windows:
            train_start_str = window.train_start.strftime("%Y-%m-%d")
            train_end_str = window.train_end.strftime("%Y-%m-%d")
            test_start_str = window.test_start.strftime("%Y-%m-%d")
            test_end_str = window.test_end.strftime("%Y-%m-%d")

            print(
                f"Window {window.window_id + 1}/{total}: "
                f"Train {train_start_str} to {train_end_str}, "
                f"Test {test_start_str} to {test_end_str}"
            )

            # Train period
            train_engine = engine_factory(train_start_str, train_end_str)
            train_results = train_engine.run()
            window.train_metrics = train_results.get("metrics", {})

            # Test period
            test_engine = engine_factory(test_start_str, test_end_str)
            test_results = test_engine.run()
            window.test_metrics = test_results.get("metrics", {})

        return self._aggregate_results(windows)

    def _aggregate_results(self, windows: list[WalkForwardWindow]) -> WalkForwardResult:
        """Aggregate walk-forward results across all windows.

        Computes Walk-Forward Efficiency and parameter stability from
        in-sample and out-of-sample metrics.

        Args:
            windows: List of windows with populated metrics

        Returns:
            WalkForwardResult with aggregate statistics
        """
        if not windows:
            return WalkForwardResult()

        # Collect IS and OOS metrics
        is_returns = [
            w.train_metrics.get("annualized_return_pct", 0.0) for w in windows
        ]
        oos_returns = [
            w.test_metrics.get("annualized_return_pct", 0.0) for w in windows
        ]
        oos_sharpes = [w.test_metrics.get("sharpe_ratio", 0.0) for w in windows]

        # Walk-Forward Efficiency
        avg_is = float(np.mean(is_returns)) if is_returns else 0.0
        avg_oos = float(np.mean(oos_returns)) if oos_returns else 0.0
        wfe = (avg_oos / avg_is * 100) if avg_is != 0.0 else 0.0

        # Parameter stability (CV of OOS Sharpe)
        mean_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        if len(oos_sharpes) > 1 and mean_sharpe != 0.0:
            param_stability = float(np.std(oos_sharpes) / abs(mean_sharpe))
        else:
            param_stability = 0.0

        # Aggregate OOS metrics
        metric_keys = [
            "annualized_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "volatility_pct",
        ]
        aggregate: dict[str, Any] = {"num_windows": len(windows)}

        for key in metric_keys:
            values = [
                float(w.test_metrics[key]) for w in windows if key in w.test_metrics
            ]
            if values:
                aggregate[f"avg_oos_{key}"] = float(np.mean(values))

        # Additional aggregate stats
        profitable = sum(1 for r in oos_returns if r > 0)
        aggregate["pct_profitable_windows"] = (
            profitable / len(oos_returns) * 100 if oos_returns else 0.0
        )

        return WalkForwardResult(
            windows=windows,
            aggregate_metrics=aggregate,
            walk_forward_efficiency=wfe,
            parameter_stability=param_stability,
        )
