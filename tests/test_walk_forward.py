"""Unit tests for walk-forward testing framework."""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

from src.backtest.walk_forward import (
    WalkForwardResult,
    WalkForwardRunner,
    WalkForwardWindow,
)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


class MockEngine:
    """Engine returning predetermined metrics for testing aggregation."""

    def __init__(self, metrics: dict[str, Any]) -> None:
        self._metrics = metrics

    def run(self) -> dict[str, Any]:
        return {"metrics": self._metrics}


class MockEngineNoTracker:
    """Engine that doesn't include metrics in results."""

    def run(self) -> dict[str, Any]:
        return {"bars_processed": 100, "final_equity": 100_000.0}


# ──────────────────────────────────────────────────────────────
# Dataclass tests
# ──────────────────────────────────────────────────────────────


class TestWalkForwardWindow:
    def test_creation(self):
        w = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2022, 1, 1),
            train_end=datetime(2022, 6, 30),
            test_start=datetime(2022, 7, 1),
            test_end=datetime(2022, 12, 31),
        )
        assert w.window_id == 0
        assert w.train_start == datetime(2022, 1, 1)
        assert w.test_end == datetime(2022, 12, 31)

    def test_default_metrics_empty(self):
        w = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2022, 1, 1),
            train_end=datetime(2022, 6, 30),
            test_start=datetime(2022, 7, 1),
            test_end=datetime(2022, 12, 31),
        )
        assert w.train_metrics == {}
        assert w.test_metrics == {}


class TestWalkForwardResult:
    def test_creation(self):
        r = WalkForwardResult(
            windows=[],
            aggregate_metrics={"foo": 1},
            walk_forward_efficiency=50.0,
            parameter_stability=0.3,
        )
        assert r.walk_forward_efficiency == 50.0
        assert r.parameter_stability == 0.3

    def test_default_values(self):
        r = WalkForwardResult()
        assert r.windows == []
        assert r.aggregate_metrics == {}
        assert r.walk_forward_efficiency == 0.0
        assert r.parameter_stability == 0.0


# ──────────────────────────────────────────────────────────────
# Init tests
# ──────────────────────────────────────────────────────────────


class TestWalkForwardRunnerInit:
    def test_valid_initialization(self):
        runner = WalkForwardRunner(
            train_days=180,
            test_days=90,
            step_days=60,
            start_date="2020-01-01",
            end_date="2023-12-31",
        )
        assert runner.train_days == 180
        assert runner.test_days == 90
        assert runner.step_days == 60

    def test_invalid_train_days(self):
        with pytest.raises(ValueError, match="train_days must be > 0"):
            WalkForwardRunner(
                train_days=0,
                test_days=90,
                step_days=60,
                start_date="2020-01-01",
                end_date="2023-12-31",
            )

    def test_invalid_test_days(self):
        with pytest.raises(ValueError, match="test_days must be > 0"):
            WalkForwardRunner(
                train_days=180,
                test_days=-1,
                step_days=60,
                start_date="2020-01-01",
                end_date="2023-12-31",
            )

    def test_invalid_step_days(self):
        with pytest.raises(ValueError, match="step_days must be > 0"):
            WalkForwardRunner(
                train_days=180,
                test_days=90,
                step_days=0,
                start_date="2020-01-01",
                end_date="2023-12-31",
            )

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            WalkForwardRunner(
                train_days=30,
                test_days=30,
                step_days=30,
                start_date="2023-01-01",
                end_date="2020-01-01",
            )


# ──────────────────────────────────────────────────────────────
# Window generation tests
# ──────────────────────────────────────────────────────────────


class TestGenerateWindows:
    def test_generates_correct_windows(self):
        """With a 1-year range, 180 train + 90 test + 90 step -> multiple windows."""
        runner = WalkForwardRunner(
            train_days=180,
            test_days=90,
            step_days=90,
            start_date="2022-01-01",
            end_date="2022-12-31",
        )
        windows = runner.generate_windows()
        assert len(windows) >= 2

        # First window
        w0 = windows[0]
        assert w0.window_id == 0
        assert w0.train_start == datetime(2022, 1, 1)
        assert w0.train_end == datetime(2022, 1, 1) + timedelta(days=179)

        # Second window starts 90 days after the first
        w1 = windows[1]
        assert w1.window_id == 1
        assert w1.train_start == datetime(2022, 1, 1) + timedelta(days=90)

    def test_no_windows_short_range(self):
        """Date range too short for even one window -> empty list."""
        runner = WalkForwardRunner(
            train_days=365,
            test_days=180,
            step_days=90,
            start_date="2022-01-01",
            end_date="2022-06-30",
        )
        windows = runner.generate_windows()
        assert windows == []

    def test_windows_dont_exceed_end(self):
        """No test_end should exceed the overall end_date."""
        runner = WalkForwardRunner(
            train_days=60,
            test_days=30,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-06-30",
        )
        windows = runner.generate_windows()
        end = datetime(2022, 6, 30)
        for w in windows:
            assert w.test_end <= end

    def test_single_window(self):
        """Exact range for one window produces exactly 1."""
        # 30 train + 30 test = 60 days needed
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=90,  # step so large only 1 fits
            start_date="2022-01-01",
            end_date="2022-03-01",
        )
        windows = runner.generate_windows()
        assert len(windows) == 1
        assert windows[0].window_id == 0

    def test_overlapping_test_periods(self):
        """step_days < test_days creates overlapping test windows."""
        runner = WalkForwardRunner(
            train_days=60,
            test_days=60,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-12-31",
        )
        windows = runner.generate_windows()
        assert len(windows) >= 3

        # Check overlap: window 1 test starts before window 0 test ends
        if len(windows) >= 2:
            w0_test_end = windows[0].test_end
            w1_test_start = windows[1].test_start
            assert w1_test_start < w0_test_end

    def test_window_ids_sequential(self):
        runner = WalkForwardRunner(
            train_days=60,
            test_days=30,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-12-31",
        )
        windows = runner.generate_windows()
        ids = [w.window_id for w in windows]
        assert ids == list(range(len(windows)))


# ──────────────────────────────────────────────────────────────
# Run & aggregation tests
# ──────────────────────────────────────────────────────────────


class TestWalkForwardRun:
    def test_run_returns_result(self):
        """Run with mock engines returns a WalkForwardResult."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-06-30",
        )
        metrics = {
            "annualized_return_pct": 10.0,
            "sharpe_ratio": 1.0,
            "max_drawdown_pct": -5.0,
            "volatility_pct": 15.0,
        }

        def factory(start: str, end: str) -> MockEngine:
            return MockEngine(metrics)

        result = runner.run(factory)
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0

    def test_train_and_test_metrics_populated(self):
        """Each window should have non-empty metrics."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=60,
            start_date="2022-01-01",
            end_date="2022-06-30",
        )
        metrics = {"annualized_return_pct": 5.0, "sharpe_ratio": 0.5}

        def factory(start: str, end: str) -> MockEngine:
            return MockEngine(metrics)

        result = runner.run(factory)
        for w in result.windows:
            assert w.train_metrics == metrics
            assert w.test_metrics == metrics

    def test_wfe_calculation(self):
        """Verify WFE with known IS and OOS returns."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-04-01",
        )

        # IS returns 20%, OOS returns 10% -> WFE = 10/20 * 100 = 50%
        call_count = [0]

        def factory(start: str, end: str) -> MockEngine:
            call_count[0] += 1
            # Odd calls = train, even calls = test
            if call_count[0] % 2 == 1:
                return MockEngine({"annualized_return_pct": 20.0, "sharpe_ratio": 1.5})
            else:
                return MockEngine({"annualized_return_pct": 10.0, "sharpe_ratio": 0.8})

        result = runner.run(factory)
        assert pytest.approx(result.walk_forward_efficiency) == 50.0

    def test_parameter_stability(self):
        """Verify parameter stability with known OOS Sharpes."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-04-01",
        )

        sharpe_values = [1.0, 2.0]  # OOS Sharpes
        call_idx = [0]

        def factory(start: str, end: str) -> MockEngine:
            call_idx[0] += 1
            if call_idx[0] % 2 == 1:
                return MockEngine({"annualized_return_pct": 10.0, "sharpe_ratio": 1.0})
            else:
                idx = (call_idx[0] // 2 - 1) % len(sharpe_values)
                return MockEngine(
                    {
                        "annualized_return_pct": 10.0,
                        "sharpe_ratio": sharpe_values[idx],
                    }
                )

        result = runner.run(factory)
        windows = result.windows
        if len(windows) >= 2:
            oos_sharpes = [w.test_metrics.get("sharpe_ratio", 0.0) for w in windows]
            expected_cv = float(np.std(oos_sharpes) / abs(np.mean(oos_sharpes)))
            assert pytest.approx(result.parameter_stability, abs=1e-6) == expected_cv

    def test_aggregate_metrics_keys(self):
        """Aggregate metrics should contain expected keys."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=60,
            start_date="2022-01-01",
            end_date="2022-06-30",
        )
        metrics = {
            "annualized_return_pct": 10.0,
            "sharpe_ratio": 1.0,
            "max_drawdown_pct": -5.0,
            "volatility_pct": 15.0,
        }

        def factory(start: str, end: str) -> MockEngine:
            return MockEngine(metrics)

        result = runner.run(factory)
        agg = result.aggregate_metrics
        assert "num_windows" in agg
        assert "avg_oos_annualized_return_pct" in agg
        assert "avg_oos_sharpe_ratio" in agg
        assert "pct_profitable_windows" in agg

    def test_no_performance_tracker(self):
        """Engine without metrics key -> metrics are {}."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=60,
            start_date="2022-01-01",
            end_date="2022-06-30",
        )

        def factory(start: str, end: str) -> MockEngineNoTracker:
            return MockEngineNoTracker()

        result = runner.run(factory)
        for w in result.windows:
            assert w.train_metrics == {}
            assert w.test_metrics == {}
        # WFE should be 0 with no metrics
        assert result.walk_forward_efficiency == 0.0

    def test_empty_windows_returns_default(self):
        """Short date range -> no windows -> default result."""
        runner = WalkForwardRunner(
            train_days=365,
            test_days=180,
            step_days=90,
            start_date="2022-01-01",
            end_date="2022-06-30",
        )

        def factory(start: str, end: str) -> MockEngine:
            return MockEngine({})

        result = runner.run(factory)
        assert result.windows == []
        assert result.walk_forward_efficiency == 0.0
        assert result.parameter_stability == 0.0


# ──────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────


class TestWalkForwardEdgeCases:
    def test_zero_is_return_wfe(self):
        """avg IS return == 0 -> WFE = 0.0 (no division by zero)."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-04-01",
        )
        call_count = [0]

        def factory(start: str, end: str) -> MockEngine:
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return MockEngine({"annualized_return_pct": 0.0, "sharpe_ratio": 0.0})
            else:
                return MockEngine({"annualized_return_pct": 5.0, "sharpe_ratio": 0.5})

        result = runner.run(factory)
        assert result.walk_forward_efficiency == 0.0

    def test_zero_mean_sharpe_stability(self):
        """Mean OOS Sharpe == 0 -> parameter_stability = 0.0."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-04-01",
        )

        def factory(start: str, end: str) -> MockEngine:
            return MockEngine({"annualized_return_pct": 5.0, "sharpe_ratio": 0.0})

        result = runner.run(factory)
        assert result.parameter_stability == 0.0

    def test_single_window_stability(self):
        """Only 1 window -> parameter_stability = 0.0."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=999,
            start_date="2022-01-01",
            end_date="2022-03-15",
        )

        def factory(start: str, end: str) -> MockEngine:
            return MockEngine({"annualized_return_pct": 10.0, "sharpe_ratio": 1.5})

        result = runner.run(factory)
        assert len(result.windows) == 1
        assert result.parameter_stability == 0.0

    def test_negative_wfe(self):
        """Negative OOS return with positive IS return -> negative WFE."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-04-01",
        )
        call_count = [0]

        def factory(start: str, end: str) -> MockEngine:
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return MockEngine({"annualized_return_pct": 20.0, "sharpe_ratio": 1.0})
            else:
                return MockEngine(
                    {"annualized_return_pct": -10.0, "sharpe_ratio": -0.5}
                )

        result = runner.run(factory)
        assert result.walk_forward_efficiency < 0

    def test_pct_profitable_windows(self):
        """Verify pct_profitable_windows calculation."""
        runner = WalkForwardRunner(
            train_days=30,
            test_days=30,
            step_days=30,
            start_date="2022-01-01",
            end_date="2022-04-01",
        )
        call_count = [0]
        # 2 windows: first OOS positive, second OOS negative
        oos_returns = [10.0, -5.0]

        def factory(start: str, end: str) -> MockEngine:
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return MockEngine({"annualized_return_pct": 15.0})
            else:
                idx = (call_count[0] // 2 - 1) % len(oos_returns)
                return MockEngine({"annualized_return_pct": oos_returns[idx]})

        result = runner.run(factory)
        if len(result.windows) == 2:
            assert result.aggregate_metrics["pct_profitable_windows"] == 50.0
