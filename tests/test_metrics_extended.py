"""Extended tests for PerformanceTracker metrics calculation.

Covers calculate_extended_metrics, _calculate_turnover, print_report,
calculate_trade_metrics edge cases, and metric formula edge cases.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

from src.performance.metrics import PerformanceTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(
    initial_capital: float = 100_000.0,
    days: int = 100,
    daily_return: float = 0.001,
) -> PerformanceTracker:
    """Create a tracker with a synthetic equity curve."""
    tracker = PerformanceTracker(initial_capital=initial_capital)
    equity = initial_capital
    start = datetime(2020, 1, 2)
    for i in range(days):
        ts = start + timedelta(days=i)
        tracker.update(ts, equity)
        equity *= 1 + daily_return
    return tracker


def _make_trade(side="SELL", pnl=100.0, price=150.0, quantity=10):
    return SimpleNamespace(side=side, pnl=pnl, price=price, quantity=quantity)


def _make_portfolio_history(n=50, base_equity=100_000.0):
    """Create synthetic portfolio equity history."""
    history = []
    start = datetime(2020, 1, 2)
    for i in range(n):
        eq = base_equity + i * 100
        history.append(
            {
                "timestamp": start + timedelta(days=i),
                "equity": eq,
                "cash": eq * 0.7,
                "positions_value": eq * 0.3,
                "num_positions": 2,
            }
        )
    return history


# ---------------------------------------------------------------------------
# calculate_extended_metrics
# ---------------------------------------------------------------------------


class TestCalculateExtendedMetrics:
    def test_with_full_history(self):
        tracker = _make_tracker(days=100)
        trades = [_make_trade(pnl=200), _make_trade(pnl=-50)]
        history = _make_portfolio_history(100)
        result = tracker.calculate_extended_metrics(trades, history)
        assert "risk_metrics" in result
        assert "trade_metrics" in result
        assert "avg_position_count" in result["risk_metrics"]
        assert "max_position_count" in result["risk_metrics"]
        assert "gross_exposure_avg" in result["risk_metrics"]
        assert "turnover" in result["risk_metrics"]

    def test_with_none_history(self):
        tracker = _make_tracker(days=100)
        trades = [_make_trade(pnl=100)]
        result = tracker.calculate_extended_metrics(trades, None)
        assert "risk_metrics" in result
        # Should still have turnover (computed from tracker's equity curve)
        assert "turnover" in result["risk_metrics"]
        # Should not have position stats
        assert "avg_position_count" not in result["risk_metrics"]

    def test_with_empty_trades(self):
        tracker = _make_tracker(days=100)
        result = tracker.calculate_extended_metrics([], _make_portfolio_history())
        assert "risk_metrics" in result
        assert result["risk_metrics"]["turnover"] == 0.0
        # No trade_metrics key when trades is empty list
        assert "trade_metrics" not in result

    def test_insufficient_data(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        tracker.update(datetime(2020, 1, 1), 100_000.0)
        result = tracker.calculate_extended_metrics([], None)
        assert "error" in result

    def test_avg_position_count(self):
        tracker = _make_tracker(days=50)
        history = []
        start = datetime(2020, 1, 2)
        for i in range(50):
            history.append(
                {
                    "timestamp": start + timedelta(days=i),
                    "equity": 100_000.0,
                    "cash": 50_000.0,
                    "positions_value": 50_000.0,
                    "num_positions": 3,
                }
            )
        result = tracker.calculate_extended_metrics([_make_trade()], history)
        assert result["risk_metrics"]["avg_position_count"] == 3.0
        assert result["risk_metrics"]["max_position_count"] == 3

    def test_gross_exposure_avg(self):
        tracker = _make_tracker(days=10)
        history = []
        start = datetime(2020, 1, 2)
        for i in range(10):
            history.append(
                {
                    "timestamp": start + timedelta(days=i),
                    "equity": 100_000.0,
                    "cash": 50_000.0,
                    "positions_value": 50_000.0,
                    "num_positions": 2,
                }
            )
        result = tracker.calculate_extended_metrics([_make_trade()], history)
        # exposure = |50000| / 100000 = 0.5
        assert abs(result["risk_metrics"]["gross_exposure_avg"] - 0.5) < 0.01


# ---------------------------------------------------------------------------
# _calculate_turnover
# ---------------------------------------------------------------------------


class TestCalculateTurnover:
    def test_known_values(self):
        tracker = _make_tracker(days=252)
        trades = [
            _make_trade(price=100.0, quantity=100),  # traded value = 10000
            _make_trade(price=200.0, quantity=50),  # traded value = 10000
        ]
        history = _make_portfolio_history(252, base_equity=100_000.0)
        turnover = tracker._calculate_turnover(trades, history)
        # total_traded = 10000+10000 = 20000
        # avg_equity ≈ 100_000 + 125*100 = 112_500 (rough)
        # annualized = (20000/avg_equity) * (252/252)
        assert turnover > 0.0

    def test_empty_trades(self):
        tracker = _make_tracker(days=50)
        assert tracker._calculate_turnover([], None) == 0.0

    def test_zero_equity_history(self):
        tracker = PerformanceTracker(initial_capital=0.0)
        trades = [_make_trade(price=100.0, quantity=10)]
        history = [
            {
                "timestamp": datetime(2020, 1, 1),
                "equity": 0.0,
                "cash": 0.0,
                "positions_value": 0.0,
                "num_positions": 0,
            }
        ]
        assert tracker._calculate_turnover(trades, history) == 0.0

    def test_turnover_from_tracker_equity(self):
        tracker = _make_tracker(days=50)
        trades = [_make_trade(price=100.0, quantity=50)]
        turnover = tracker._calculate_turnover(trades, None)
        assert turnover > 0.0


# ---------------------------------------------------------------------------
# calculate_trade_metrics edge cases
# ---------------------------------------------------------------------------


class TestTradeMetricsEdgeCases:
    def test_all_winners(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        trades = [_make_trade(pnl=100), _make_trade(pnl=200), _make_trade(pnl=50)]
        result = tracker.calculate_trade_metrics(trades)
        assert result["win_rate_pct"] == 100.0
        assert result["losing_trades"] == 0
        assert result["profit_factor"] == float("inf")

    def test_all_losers(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        trades = [_make_trade(pnl=-100), _make_trade(pnl=-200)]
        result = tracker.calculate_trade_metrics(trades)
        assert result["win_rate_pct"] == 0.0
        assert result["winning_trades"] == 0
        assert result["profit_factor"] == 0.0

    def test_single_sell_trade(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        trades = [_make_trade(pnl=500)]
        result = tracker.calculate_trade_metrics(trades)
        assert result["total_trades"] == 1
        assert result["largest_win"] == 500

    def test_no_sells_only_buys(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        trades = [_make_trade(side="BUY", pnl=0)]
        result = tracker.calculate_trade_metrics(trades)
        assert result["total_trades"] == 0

    def test_empty_trades(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        result = tracker.calculate_trade_metrics([])
        assert result["total_trades"] == 0
        assert result["win_rate_pct"] == 0.0

    def test_gross_profit_and_loss(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        trades = [
            _make_trade(pnl=300),
            _make_trade(pnl=-100),
            _make_trade(pnl=200),
        ]
        result = tracker.calculate_trade_metrics(trades)
        assert result["gross_profit"] == 500
        assert result["gross_loss"] == 100
        assert result["profit_factor"] == 5.0


# ---------------------------------------------------------------------------
# Metric formula edge cases
# ---------------------------------------------------------------------------


class TestMetricEdgeCases:
    def test_sharpe_with_zero_std(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        for i in range(50):
            tracker.update(datetime(2020, 1, 2) + timedelta(days=i), 100_000.0)
        metrics = tracker.calculate_metrics()
        assert metrics["sharpe_ratio"] == 0.0

    def test_sortino_no_downside(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        equity = 100_000.0
        for i in range(50):
            equity += 100  # only positive returns
            tracker.update(datetime(2020, 1, 2) + timedelta(days=i), equity)
        metrics = tracker.calculate_metrics()
        assert metrics["sortino_ratio"] == 0.0

    def test_calmar_zero_drawdown(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        equity = 100_000.0
        for i in range(50):
            equity += 100
            tracker.update(datetime(2020, 1, 2) + timedelta(days=i), equity)
        metrics = tracker.calculate_metrics()
        assert metrics["calmar_ratio"] == 0.0

    def test_volatility_single_point(self):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        tracker.update(datetime(2020, 1, 1), 100_000.0)
        metrics = tracker.calculate_metrics()
        assert "error" in metrics

    def test_reset_clears_curve(self):
        tracker = _make_tracker(days=50)
        assert len(tracker.equity_curve) == 50
        tracker.reset()
        assert len(tracker.equity_curve) == 0


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------


class TestPrintReport:
    def test_report_with_trades_and_history(self, caplog):
        tracker = _make_tracker(days=50)
        trades = [_make_trade(pnl=100), _make_trade(pnl=-30)]
        history = _make_portfolio_history(50)
        with caplog.at_level("INFO", logger="src.performance.metrics"):
            tracker.print_report(trades=trades, portfolio_equity_history=history)
        assert "BACKTEST PERFORMANCE REPORT" in caplog.text
        assert "Turnover" in caplog.text
        assert "Avg Positions" in caplog.text

    def test_report_trades_only(self, caplog):
        tracker = _make_tracker(days=50)
        trades = [_make_trade(pnl=100)]
        with caplog.at_level("INFO", logger="src.performance.metrics"):
            tracker.print_report(trades=trades)
        assert "BACKTEST PERFORMANCE REPORT" in caplog.text
        assert "Win Rate" in caplog.text

    def test_report_insufficient_data(self, caplog):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        tracker.update(datetime(2020, 1, 1), 100_000.0)
        with caplog.at_level("WARNING", logger="src.performance.metrics"):
            tracker.print_report()
        assert "Cannot generate report" in caplog.text

    def test_report_no_trades(self, caplog):
        tracker = _make_tracker(days=50)
        with caplog.at_level("INFO", logger="src.performance.metrics"):
            tracker.print_report()
        assert "BACKTEST PERFORMANCE REPORT" in caplog.text
