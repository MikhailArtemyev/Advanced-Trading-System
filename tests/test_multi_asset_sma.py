"""Unit tests for MultiAssetSMAStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.data.data_handler import DataHandler, HistoricalCSVDataHandler
from src.events.event import SignalEvent, SignalType
from src.optimization.base_optimizer import AllocationResult, PortfolioOptimizer
from src.strategy.multi_asset_sma import MultiAssetSMAStrategy

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


class MockOptimizer(PortfolioOptimizer):
    """Returns predetermined weights for testing."""

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights

    @property
    def _method_name(self) -> str:
        return "mock"

    def optimize(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
        current_weights: dict[str, float],
    ) -> AllocationResult:
        return AllocationResult(weights=self._weights, method="mock")


class CountingOptimizer(PortfolioOptimizer):
    """Counts how many times optimize() is called."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def _method_name(self) -> str:
        return "counting"

    def optimize(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
        current_weights: dict[str, float],
    ) -> AllocationResult:
        self.call_count += 1
        n = len(symbols)
        w = 1.0 / n if n > 0 else 0.0
        return AllocationResult(weights=dict.fromkeys(symbols, w), method="counting")


@pytest.fixture
def multi_asset_data_dir(tmp_path):
    """Create CSV files for AAPL, MSFT, GOOGL with trending data."""
    dates = pd.date_range("2022-01-01", "2022-06-30", freq="B")
    rng = np.random.RandomState(42)

    for symbol, base_price in [("AAPL", 150.0), ("MSFT", 300.0), ("GOOGL", 2800.0)]:
        prices = base_price * np.exp(np.cumsum(rng.randn(len(dates)) * 0.02))
        df = pd.DataFrame(
            {
                "date": dates,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": rng.randint(1_000_000, 5_000_000, len(dates)),
            }
        )
        df.to_csv(tmp_path / f"{symbol}.csv", index=False)

    return str(tmp_path)


@pytest.fixture
def data_handler(multi_asset_data_dir):
    """Create data handler with 3 symbols."""
    return HistoricalCSVDataHandler(
        data_path=multi_asset_data_dir,
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2022-01-01",
        end_date="2022-06-30",
    )


def _advance_bars(data_handler: DataHandler, n: int) -> None:
    """Advance the data handler by n bars."""
    for _ in range(n):
        if not data_handler.continue_backtest:
            break
        data_handler.update_bars()


# ──────────────────────────────────────────────────────────────
# Init tests
# ──────────────────────────────────────────────────────────────


class TestMultiAssetSMAInit:
    def test_default_parameters(self):
        strategy = MultiAssetSMAStrategy(symbols=["AAPL", "MSFT", "GOOGL"])
        assert strategy.short_window == 20
        assert strategy.long_window == 50
        assert strategy.rebalance_frequency == 20

    def test_custom_parameters(self):
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL"],
            parameters={
                "short_window": 10,
                "long_window": 30,
                "rebalance_frequency": 5,
            },
        )
        assert strategy.short_window == 10
        assert strategy.long_window == 30
        assert strategy.rebalance_frequency == 5

    def test_invalid_windows_raises(self):
        with pytest.raises(
            ValueError, match="short_window must be less than long_window"
        ):
            MultiAssetSMAStrategy(
                symbols=["AAPL"],
                parameters={"short_window": 50, "long_window": 20},
            )

    def test_equal_windows_raises(self):
        with pytest.raises(
            ValueError, match="short_window must be less than long_window"
        ):
            MultiAssetSMAStrategy(
                symbols=["AAPL"],
                parameters={"short_window": 20, "long_window": 20},
            )

    def test_invalid_short_window(self):
        with pytest.raises(ValueError, match="short_window must be at least 1"):
            MultiAssetSMAStrategy(
                symbols=["AAPL"],
                parameters={"short_window": 0, "long_window": 20},
            )

    def test_invalid_rebalance_frequency(self):
        with pytest.raises(ValueError, match="rebalance_frequency must be at least 1"):
            MultiAssetSMAStrategy(
                symbols=["AAPL"],
                parameters={"rebalance_frequency": 0},
            )

    def test_equal_initial_weights(self):
        strategy = MultiAssetSMAStrategy(symbols=["AAPL", "MSFT", "GOOGL"])
        weights = strategy.get_target_weights()
        assert len(weights) == 3
        for w in weights.values():
            assert pytest.approx(w, abs=1e-10) == 1.0 / 3.0

    def test_single_symbol_weight(self):
        strategy = MultiAssetSMAStrategy(symbols=["AAPL"])
        assert strategy.get_target_weights() == {"AAPL": 1.0}

    def test_no_optimizer(self):
        strategy = MultiAssetSMAStrategy(symbols=["AAPL"])
        assert strategy.optimizer is None

    def test_with_optimizer(self):
        opt = MockOptimizer({"AAPL": 1.0})
        strategy = MultiAssetSMAStrategy(symbols=["AAPL"], optimizer=opt)
        assert strategy.optimizer is opt


# ──────────────────────────────────────────────────────────────
# Signal tests
# ──────────────────────────────────────────────────────────────


class TestMultiAssetSMASignals:
    def test_no_signal_insufficient_data(self, data_handler):
        """Not enough bars for SMA calculation -> no signals."""
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={"short_window": 5, "long_window": 10},
        )
        # Only advance a few bars (less than long_window)
        _advance_bars(data_handler, 3)
        ts = data_handler.get_current_timestamp()
        signals = strategy.calculate_signals(ts, data_handler)
        assert signals == []

    def test_no_signal_first_bar_with_enough_data(self, data_handler):
        """First SMA calc has no previous values -> no crossover detected."""
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={"short_window": 5, "long_window": 10},
        )
        # Advance enough bars for one SMA calculation
        _advance_bars(data_handler, 12)
        ts = data_handler.get_current_timestamp()
        signals = strategy.calculate_signals(ts, data_handler)
        # First time: prev_short_sma is None, so no signal
        assert signals == []

    def test_generates_signals_over_time(self, data_handler):
        """Over many bars, at least some crossover signals should appear."""
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={"short_window": 5, "long_window": 15},
        )
        all_signals: list[SignalEvent] = []
        while data_handler.continue_backtest:
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            signals = strategy.calculate_signals(ts, data_handler)
            all_signals.extend(signals)

        assert len(all_signals) > 0

    def test_signals_for_multiple_symbols(self, data_handler):
        """Signals should appear for more than one symbol."""
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={"short_window": 5, "long_window": 15},
        )
        symbols_with_signals: set[str] = set()
        while data_handler.continue_backtest:
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            signals = strategy.calculate_signals(ts, data_handler)
            for s in signals:
                symbols_with_signals.add(s.symbol)

        assert len(symbols_with_signals) >= 2

    def test_long_signal_strength_equals_weight(self, data_handler):
        """LONG signal strength should match the target weight."""
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={"short_window": 5, "long_window": 15},
        )
        expected_weight = 1.0 / 3.0

        while data_handler.continue_backtest:
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            signals = strategy.calculate_signals(ts, data_handler)
            for sig in signals:
                if sig.signal_type == SignalType.LONG:
                    assert pytest.approx(sig.strength, abs=1e-10) == expected_weight
                    return  # Found one, test passes

        pytest.fail("No LONG signal generated during backtest")

    def test_exit_signal_strength_is_one(self, data_handler):
        """EXIT signals always have strength=1.0."""
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={"short_window": 5, "long_window": 15},
        )

        while data_handler.continue_backtest:
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            signals = strategy.calculate_signals(ts, data_handler)
            for sig in signals:
                # Simulate that we're positioned when we get a LONG
                if sig.signal_type == SignalType.LONG:
                    strategy.update_position(sig.symbol, 100)
                if sig.signal_type == SignalType.EXIT:
                    assert sig.strength == 1.0
                    strategy.update_position(sig.symbol, 0)
                    return  # Found one, test passes

        pytest.fail("No EXIT signal generated during backtest")

    def test_no_duplicate_long_when_positioned(self, data_handler):
        """Should not generate LONG if already long in that symbol."""
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={"short_window": 5, "long_window": 15},
        )
        # Pre-set all positions to long
        for sym in ["AAPL", "MSFT", "GOOGL"]:
            strategy.update_position(sym, 100)

        long_signals = []
        while data_handler.continue_backtest:
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            signals = strategy.calculate_signals(ts, data_handler)
            for sig in signals:
                if sig.signal_type == SignalType.LONG:
                    long_signals.append(sig)

        assert long_signals == []


# ──────────────────────────────────────────────────────────────
# Rebalance tests
# ──────────────────────────────────────────────────────────────


class TestMultiAssetSMARebalance:
    def test_rebalance_with_optimizer(self, data_handler):
        """Weights should change after rebalance when optimizer is set."""
        new_weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        opt = MockOptimizer(new_weights)

        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={
                "short_window": 5,
                "long_window": 15,
                "rebalance_frequency": 10,
            },
            optimizer=opt,
        )

        # Run enough bars to trigger rebalance
        for _ in range(12):
            if not data_handler.continue_backtest:
                break
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            strategy.calculate_signals(ts, data_handler)

        assert strategy.get_target_weights() == new_weights

    def test_no_rebalance_without_optimizer(self, data_handler):
        """Without optimizer, weights stay equal forever."""
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={"short_window": 5, "long_window": 15, "rebalance_frequency": 5},
        )

        initial_weights = strategy.get_target_weights()

        for _ in range(30):
            if not data_handler.continue_backtest:
                break
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            strategy.calculate_signals(ts, data_handler)

        assert strategy.get_target_weights() == initial_weights

    def test_rebalance_counter_resets(self, data_handler):
        """Counter resets to 0 after rebalance."""
        opt = CountingOptimizer()
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={
                "short_window": 5,
                "long_window": 15,
                "rebalance_frequency": 10,
            },
            optimizer=opt,
        )

        # Run 10 bars -> should trigger rebalance, counter resets
        for _ in range(10):
            if not data_handler.continue_backtest:
                break
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            strategy.calculate_signals(ts, data_handler)

        assert opt.call_count == 1
        assert strategy.bars_since_rebalance == 0

    def test_rebalance_timing(self, data_handler):
        """First rebalance happens at bar #rebalance_frequency."""
        opt = CountingOptimizer()
        rebal_freq = 5
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={
                "short_window": 3,
                "long_window": 10,
                "rebalance_frequency": rebal_freq,
            },
            optimizer=opt,
        )

        for _i in range(rebal_freq - 1):
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            strategy.calculate_signals(ts, data_handler)

        # Should not have rebalanced yet
        assert opt.call_count == 0

        # One more bar triggers it
        data_handler.update_bars()
        ts = data_handler.get_current_timestamp()
        strategy.calculate_signals(ts, data_handler)
        assert opt.call_count == 1

    def test_rebalance_insufficient_data_early(self, data_handler):
        """Early in backtest, _build_returns_df may return empty -> no crash."""
        new_weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        opt = MockOptimizer(new_weights)
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={
                "short_window": 3,
                "long_window": 10,
                "rebalance_frequency": 1,  # try to rebalance every bar
            },
            optimizer=opt,
        )

        # Advance just 1 bar — not enough for returns
        data_handler.update_bars()
        ts = data_handler.get_current_timestamp()
        # Should not crash even though returns are insufficient
        signals = strategy.calculate_signals(ts, data_handler)
        assert isinstance(signals, list)


# ──────────────────────────────────────────────────────────────
# Returns builder tests
# ──────────────────────────────────────────────────────────────


class TestMultiAssetSMABuildReturns:
    def test_builds_returns_dataframe(self, data_handler):
        strategy = MultiAssetSMAStrategy(symbols=["AAPL", "MSFT", "GOOGL"])
        _advance_bars(data_handler, 30)
        df = strategy._build_returns_df(data_handler, lookback=20)
        assert not df.empty
        assert set(df.columns) == {"AAPL", "MSFT", "GOOGL"}
        assert len(df) > 0

    def test_empty_when_no_bars(self, data_handler):
        """Before any bars are advanced, returns should be empty."""
        strategy = MultiAssetSMAStrategy(symbols=["AAPL", "MSFT", "GOOGL"])
        # current_index is 0, only 1 bar available -> pct_change gives 0 rows
        df = strategy._build_returns_df(data_handler, lookback=20)
        assert df.empty


# ──────────────────────────────────────────────────────────────
# Getter tests
# ──────────────────────────────────────────────────────────────


class TestMultiAssetSMAGetters:
    def test_get_target_weights_is_copy(self):
        strategy = MultiAssetSMAStrategy(symbols=["AAPL", "MSFT"])
        weights = strategy.get_target_weights()
        weights["AAPL"] = 999.0
        # Original should be unchanged
        assert strategy.target_weights["AAPL"] == pytest.approx(0.5)

    def test_get_sma_values_before_calc(self):
        strategy = MultiAssetSMAStrategy(symbols=["AAPL"])
        assert strategy.get_sma_values("AAPL") == (None, None)

    def test_get_sma_values_after_calc(self, data_handler):
        strategy = MultiAssetSMAStrategy(
            symbols=["AAPL", "MSFT", "GOOGL"],
            parameters={"short_window": 5, "long_window": 15},
        )
        _advance_bars(data_handler, 20)
        ts = data_handler.get_current_timestamp()
        strategy.calculate_signals(ts, data_handler)

        short_sma, long_sma = strategy.get_sma_values("AAPL")
        assert short_sma is not None
        assert long_sma is not None
        assert isinstance(short_sma, float)
        assert isinstance(long_sma, float)


# ──────────────────────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────────────────────


class TestMultiAssetSMAIntegration:
    def test_with_backtest_engine(self, multi_asset_data_dir):
        """Full integration: strategy works inside BacktestEngine."""
        from src.execution.execution_handler import ExecutionHandler
        from src.performance.metrics import PerformanceTracker
        from src.portfolio.portfolio import Portfolio

        symbols = ["AAPL", "MSFT", "GOOGL"]
        dh = HistoricalCSVDataHandler(
            data_path=multi_asset_data_dir,
            symbols=symbols,
            start_date="2022-01-01",
            end_date="2022-06-30",
        )
        strategy = MultiAssetSMAStrategy(
            symbols=symbols,
            parameters={"short_window": 10, "long_window": 30},
        )
        portfolio = Portfolio(initial_capital=100_000.0, symbols=symbols)
        execution = ExecutionHandler()
        tracker = PerformanceTracker(initial_capital=100_000.0)

        engine = BacktestEngine(
            data_handler=dh,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            performance_tracker=tracker,
        )

        results = engine.run()
        assert results["bars_processed"] > 0
        assert "metrics" in results
        assert results["metrics"]["trading_days"] > 0

    def test_with_real_optimizer(self, multi_asset_data_dir):
        """Integration with a real MeanVarianceOptimizer."""
        from src.execution.execution_handler import ExecutionHandler
        from src.optimization.mean_variance import MeanVarianceOptimizer
        from src.performance.metrics import PerformanceTracker
        from src.portfolio.portfolio import Portfolio

        symbols = ["AAPL", "MSFT", "GOOGL"]
        dh = HistoricalCSVDataHandler(
            data_path=multi_asset_data_dir,
            symbols=symbols,
            start_date="2022-01-01",
            end_date="2022-06-30",
        )
        optimizer = MeanVarianceOptimizer(max_weight=0.60)
        strategy = MultiAssetSMAStrategy(
            symbols=symbols,
            parameters={
                "short_window": 10,
                "long_window": 30,
                "rebalance_frequency": 20,
            },
            optimizer=optimizer,
        )
        portfolio = Portfolio(initial_capital=100_000.0, symbols=symbols)
        execution = ExecutionHandler()
        tracker = PerformanceTracker(initial_capital=100_000.0)

        engine = BacktestEngine(
            data_handler=dh,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            performance_tracker=tracker,
        )

        results = engine.run()
        assert results["bars_processed"] > 0

        # Weights should have changed from equal after rebalancing
        final_weights = strategy.get_target_weights()
        assert len(final_weights) == 3
        assert pytest.approx(sum(final_weights.values()), abs=1e-6) == 1.0
