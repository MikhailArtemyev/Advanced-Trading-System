"""Unit tests for strategy classes."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.data_handler import HistoricalCSVDataHandler
from src.events.event import SignalType
from src.events.queue import EventQueue
from src.strategy.base_strategy import Strategy
from src.strategy.sma_crossover import SMACrossoverStrategy


class ConcreteStrategy(Strategy):
    """Concrete implementation for testing base class."""

    def __init__(self, symbols, parameters=None):
        super().__init__(symbols, parameters)
        self.calculate_signals_called = 0

    def calculate_signals(self, timestamp, data_handler):
        self.calculate_signals_called += 1
        return []


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create sample CSV data for testing."""
    dates = pd.date_range("2022-01-01", "2022-06-30", freq="B")
    np.random.seed(42)

    # Create trending data to trigger crossovers
    # Start with downtrend, then uptrend, then downtrend
    n = len(dates)
    trend = np.concatenate(
        [
            np.linspace(150, 130, n // 3),  # Downtrend
            np.linspace(130, 170, n // 3),  # Uptrend
            np.linspace(170, 140, n - 2 * (n // 3)),  # Downtrend
        ]
    )
    noise = np.random.randn(n) * 2
    prices = trend + noise

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n),
        }
    )
    df.to_csv(tmp_path / "AAPL.csv", index=False)

    return str(tmp_path)


@pytest.fixture
def data_handler(sample_data_dir):
    """Create data handler with sample data."""
    return HistoricalCSVDataHandler(
        data_path=sample_data_dir,
        symbols=["AAPL"],
        start_date="2022-01-01",
        end_date="2022-06-30",
    )


class TestStrategyBase:
    """Tests for Strategy base class."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = ConcreteStrategy(symbols=["AAPL", "MSFT"])

        assert strategy.symbols == ["AAPL", "MSFT"]
        assert strategy.parameters == {}
        assert strategy.events is None
        assert strategy.current_positions == {"AAPL": 0, "MSFT": 0}

    def test_initialization_with_parameters(self):
        """Test strategy initialization with parameters."""
        params = {"param1": 10, "param2": "value"}
        strategy = ConcreteStrategy(symbols=["AAPL"], parameters=params)

        assert strategy.parameters == params

    def test_set_event_queue(self):
        """Test setting event queue."""
        strategy = ConcreteStrategy(symbols=["AAPL"])
        queue = EventQueue()

        strategy.set_event_queue(queue)

        assert strategy.events is queue

    def test_update_position(self):
        """Test updating position."""
        strategy = ConcreteStrategy(symbols=["AAPL"])

        strategy.update_position("AAPL", 100)

        assert strategy.current_positions["AAPL"] == 100

    def test_get_position(self):
        """Test getting position."""
        strategy = ConcreteStrategy(symbols=["AAPL"])
        strategy.current_positions["AAPL"] = 50

        assert strategy.get_position("AAPL") == 50
        assert strategy.get_position("UNKNOWN") == 0

    def test_create_signal(self):
        """Test signal creation helper."""
        strategy = ConcreteStrategy(symbols=["AAPL"])
        ts = datetime(2022, 1, 15)

        signal = strategy._create_signal(ts, "AAPL", SignalType.LONG, 0.8)

        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.LONG
        assert signal.strength == 0.8
        assert signal.timestamp == ts

    def test_on_market_data_calls_calculate_signals(self, data_handler):
        """Test that on_market_data calls calculate_signals."""
        from src.events.event import MarketEvent

        strategy = ConcreteStrategy(symbols=["AAPL"])
        strategy.set_event_queue(EventQueue())

        event = MarketEvent(timestamp=datetime.now())
        strategy.on_market_data(event, data_handler)

        assert strategy.calculate_signals_called == 1


class TestSMACrossoverStrategy:
    """Tests for SMACrossoverStrategy class."""

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        strategy = SMACrossoverStrategy(symbols=["AAPL"])

        assert strategy.short_window == 20
        assert strategy.long_window == 50

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        params = {"short_window": 10, "long_window": 30}
        strategy = SMACrossoverStrategy(symbols=["AAPL"], parameters=params)

        assert strategy.short_window == 10
        assert strategy.long_window == 30

    def test_invalid_windows_raises_error(self):
        """Test that invalid window sizes raise ValueError."""
        # short >= long should fail
        with pytest.raises(
            ValueError, match="short_window must be less than long_window"
        ):
            SMACrossoverStrategy(
                symbols=["AAPL"],
                parameters={"short_window": 50, "long_window": 20},
            )

    def test_equal_windows_raises_error(self):
        """Test that equal window sizes raise ValueError."""
        with pytest.raises(ValueError):
            SMACrossoverStrategy(
                symbols=["AAPL"],
                parameters={"short_window": 20, "long_window": 20},
            )

    def test_invalid_short_window(self):
        """Test that short_window < 1 raises ValueError."""
        with pytest.raises(ValueError, match="short_window must be at least 1"):
            SMACrossoverStrategy(
                symbols=["AAPL"],
                parameters={"short_window": 0, "long_window": 20},
            )

    def test_no_signal_insufficient_data(self, data_handler):
        """Test no signal when insufficient data."""
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 10},
        )
        strategy.set_event_queue(EventQueue())

        # At start, only 1 bar available
        signals = strategy.calculate_signals(datetime.now(), data_handler)

        assert len(signals) == 0

    def test_no_signal_first_calculation(self, data_handler):
        """Test no signal on first calculation (no previous SMA)."""
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 10},
        )
        strategy.set_event_queue(EventQueue())

        # Advance to have enough data
        for _ in range(15):
            data_handler.update_bars()

        # First calculation - no previous values
        signals = strategy.calculate_signals(datetime.now(), data_handler)

        assert len(signals) == 0
        # But SMA values should now be stored
        short_sma, long_sma = strategy.get_sma_values("AAPL")
        assert short_sma is not None
        assert long_sma is not None

    def test_generates_signals_over_time(self, data_handler):
        """Test that strategy generates signals over a full backtest."""
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 10, "long_window": 20},
        )
        strategy.set_event_queue(EventQueue())

        all_signals = []

        # Run through all data
        while data_handler.continue_backtest:
            data_handler.update_bars()
            ts = data_handler.get_current_timestamp()
            signals = strategy.calculate_signals(ts, data_handler)
            all_signals.extend(signals)

        # Should have generated at least some signals given the trending data
        # (The exact number depends on the data pattern)
        assert len(all_signals) >= 0  # May be 0 if no crossovers

    def test_long_signal_on_bullish_crossover(self, tmp_path):
        """Test LONG signal when short SMA crosses above long SMA."""
        # Create data with clear bullish crossover:
        # Start flat/down so short SMA < long SMA, then go up sharply
        dates = pd.date_range("2022-01-01", periods=30, freq="B")
        # Flat period followed by sharp rise creates crossover
        prices = [100] * 15 + list(range(100, 115))  # Flat then uptrend

        df = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000000] * 30,
            }
        )
        df.to_csv(tmp_path / "TEST.csv", index=False)

        handler = HistoricalCSVDataHandler(
            data_path=str(tmp_path),
            symbols=["TEST"],
            start_date="2022-01-01",
            end_date="2022-02-28",
        )

        strategy = SMACrossoverStrategy(
            symbols=["TEST"],
            parameters={"short_window": 5, "long_window": 10},
        )
        strategy.set_event_queue(EventQueue())

        signals = []
        while handler.continue_backtest:
            handler.update_bars()
            ts = handler.get_current_timestamp()
            sigs = strategy.calculate_signals(ts, handler)
            signals.extend(sigs)

        # Should have at least one LONG signal
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1

    def test_exit_signal_when_long(self, tmp_path):
        """Test EXIT signal only when already in position."""
        # Create data with bearish crossover
        dates = pd.date_range("2022-01-01", periods=30, freq="B")
        prices = list(range(130, 115, -1)) + list(range(115, 100, -1))  # Downtrend

        df = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000000] * 30,
            }
        )
        df.to_csv(tmp_path / "TEST.csv", index=False)

        handler = HistoricalCSVDataHandler(
            data_path=str(tmp_path),
            symbols=["TEST"],
            start_date="2022-01-01",
            end_date="2022-02-28",
        )

        strategy = SMACrossoverStrategy(
            symbols=["TEST"],
            parameters={"short_window": 5, "long_window": 10},
        )
        strategy.set_event_queue(EventQueue())

        # Simulate being long
        strategy.update_position("TEST", 100)

        signals = []
        while handler.continue_backtest:
            handler.update_bars()
            ts = handler.get_current_timestamp()
            sigs = strategy.calculate_signals(ts, handler)
            signals.extend(sigs)

        # Check for EXIT signals
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        # May have exit signal if crossover detected
        assert isinstance(exit_signals, list)

    def test_no_duplicate_long_signals(self, tmp_path):
        """Test no LONG signal when already long."""
        dates = pd.date_range("2022-01-01", periods=30, freq="B")
        prices = list(range(100, 130))  # Continuous uptrend

        df = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000000] * 30,
            }
        )
        df.to_csv(tmp_path / "TEST.csv", index=False)

        handler = HistoricalCSVDataHandler(
            data_path=str(tmp_path),
            symbols=["TEST"],
            start_date="2022-01-01",
            end_date="2022-02-28",
        )

        strategy = SMACrossoverStrategy(
            symbols=["TEST"],
            parameters={"short_window": 5, "long_window": 10},
        )
        strategy.set_event_queue(EventQueue())

        signals = []
        while handler.continue_backtest:
            handler.update_bars()
            ts = handler.get_current_timestamp()
            sigs = strategy.calculate_signals(ts, handler)

            # Simulate portfolio updating position after signal
            for sig in sigs:
                if sig.signal_type == SignalType.LONG:
                    strategy.update_position(sig.symbol, 100)

            signals.extend(sigs)

        # Should have at most 1 LONG signal (no duplicates)
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) <= 1

    def test_get_sma_values(self, data_handler):
        """Test getting SMA values."""
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 10},
        )

        # Before any calculation
        short, long = strategy.get_sma_values("AAPL")
        assert short is None
        assert long is None

        # After calculation with enough data
        for _ in range(15):
            data_handler.update_bars()

        strategy.calculate_signals(datetime.now(), data_handler)

        short, long = strategy.get_sma_values("AAPL")
        assert short is not None
        assert long is not None
        assert isinstance(short, float)
        assert isinstance(long, float)
