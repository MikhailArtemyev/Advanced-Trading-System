"""Performance metrics tracking and calculation.

Provides comprehensive performance analysis for backtesting results.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks and calculates performance metrics.

    Metrics calculated:
    - Total Return
    - Annualized Return
    - Sharpe Ratio
    - Sortino Ratio
    - Maximum Drawdown
    - Volatility
    - Calmar Ratio
    - Win Rate (from trades)
    - Profit Factor (from trades)

    Attributes:
        initial_capital: Starting portfolio value
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculation
        periods_per_year: Trading periods per year (252 for daily)
        equity_curve: List of equity snapshots over time
    """

    def __init__(
        self,
        initial_capital: float,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> None:
        """Initialize the performance tracker.

        Args:
            initial_capital: Starting portfolio value
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Trading periods per year (default 252 for daily)
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        self.equity_curve: list[dict[str, Any]] = []

    def update(self, timestamp: datetime, equity: float) -> None:
        """Record equity value at timestamp.

        Args:
            timestamp: Current simulation timestamp
            equity: Current portfolio equity value
        """
        self.equity_curve.append({"timestamp": timestamp, "equity": equity})

    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as DataFrame.

        Returns:
            DataFrame with timestamp and equity columns
        """
        return pd.DataFrame(self.equity_curve)

    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate all performance metrics.

        Returns:
            Dictionary containing all calculated metrics
        """
        if len(self.equity_curve) < 2:
            return {"error": "Insufficient data"}

        df = self.get_equity_curve()
        equity = df["equity"]

        # Calculate returns
        returns = equity.pct_change().dropna()

        metrics: dict[str, Any] = {
            "initial_capital": self.initial_capital,
            "final_equity": equity.iloc[-1],
            "total_return_pct": self._total_return(equity),
            "annualized_return_pct": self._annualized_return(returns),
            "sharpe_ratio": self._sharpe_ratio(returns),
            "sortino_ratio": self._sortino_ratio(returns),
            "max_drawdown_pct": self._max_drawdown(equity),
            "volatility_pct": self._volatility(returns),
            "calmar_ratio": self._calmar_ratio(returns, equity),
            "trading_days": len(equity),
        }

        return metrics

    def _total_return(self, equity: pd.Series) -> float:
        """Calculate total return as percentage.

        Args:
            equity: Series of equity values

        Returns:
            Total return as percentage
        """
        return float((equity.iloc[-1] / self.initial_capital - 1) * 100)

    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return as percentage.

        Args:
            returns: Series of periodic returns

        Returns:
            Annualized return as percentage
        """
        if len(returns) == 0:
            return 0.0

        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / self.periods_per_year

        if n_years <= 0:
            return 0.0

        annualized = (1 + total_return) ** (1 / n_years) - 1
        return float(annualized * 100)

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe Ratio.

        Sharpe Ratio = (Mean Excess Return) / Std Dev of Excess Returns
        Annualized by multiplying by sqrt(periods_per_year)

        Args:
            returns: Series of periodic returns

        Returns:
            Annualized Sharpe Ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.periods_per_year)
        return float(
            np.sqrt(self.periods_per_year)
            * excess_returns.mean()
            / excess_returns.std()
        )

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino Ratio (uses downside deviation).

        Sortino Ratio = (Mean Excess Return) / Downside Std Dev
        Only considers negative returns for risk calculation.

        Args:
            returns: Series of periodic returns

        Returns:
            Annualized Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.periods_per_year)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_std = downside_returns.std()
        return float(
            np.sqrt(self.periods_per_year) * excess_returns.mean() / downside_std
        )

    def _max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown as percentage.

        Max Drawdown = Maximum peak-to-trough decline

        Args:
            equity: Series of equity values

        Returns:
            Maximum drawdown as negative percentage
        """
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min() * 100)

    def _volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility as percentage.

        Args:
            returns: Series of periodic returns

        Returns:
            Annualized volatility as percentage
        """
        if len(returns) == 0:
            return 0.0
        return float(returns.std() * np.sqrt(self.periods_per_year) * 100)

    def _calmar_ratio(self, returns: pd.Series, equity: pd.Series) -> float:
        """Calculate Calmar Ratio (annual return / max drawdown).

        Args:
            returns: Series of periodic returns
            equity: Series of equity values

        Returns:
            Calmar Ratio
        """
        annual_return = self._annualized_return(returns)
        max_dd = abs(self._max_drawdown(equity))

        if max_dd == 0:
            return 0.0
        return annual_return / max_dd

    def calculate_trade_metrics(self, trades: list[Any]) -> dict[str, Any]:
        """Calculate trade-based metrics.

        Args:
            trades: List of Trade objects with pnl and side attributes

        Returns:
            Dictionary with trade metrics
        """
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        # Filter to only sell trades (where P&L is realized)
        sell_trades = [t for t in trades if t.side == "SELL"]

        if not sell_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        winners = [t for t in sell_trades if t.pnl > 0]
        losers = [t for t in sell_trades if t.pnl < 0]

        total_trades = len(sell_trades)
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = gross_profit / len(winners) if winners else 0.0
        avg_loss = gross_loss / len(losers) if losers else 0.0

        largest_win = max((t.pnl for t in winners), default=0.0)
        largest_loss = min((t.pnl for t in losers), default=0.0)

        return {
            "total_trades": total_trades,
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }

    def calculate_extended_metrics(
        self,
        trades: list[Any],
        portfolio_equity_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Calculate extended metrics including risk and trade metrics.

        Combines base metrics with position-count stats, exposure averages,
        and portfolio turnover.

        Args:
            trades: List of Trade objects with pnl, side, price, quantity attributes
            portfolio_equity_history: Portfolio's equity_history list of dicts,
                each with keys: timestamp, equity, cash, positions_value,
                num_positions. If None, position/exposure metrics are omitted.

        Returns:
            Dictionary with base metrics plus 'risk_metrics' and optionally
            'trade_metrics' sub-dicts.
        """
        result = self.calculate_metrics()

        if "error" in result:
            return result

        risk_metrics: dict[str, Any] = {}

        if portfolio_equity_history and len(portfolio_equity_history) > 0:
            positions_counts = [
                entry["num_positions"] for entry in portfolio_equity_history
            ]
            risk_metrics["avg_position_count"] = sum(positions_counts) / len(
                positions_counts
            )
            risk_metrics["max_position_count"] = max(positions_counts)

            exposures = []
            for entry in portfolio_equity_history:
                equity = entry["equity"]
                positions_val = entry["positions_value"]
                if equity > 0:
                    exposures.append(abs(positions_val) / equity)
                else:
                    exposures.append(0.0)
            risk_metrics["gross_exposure_avg"] = (
                sum(exposures) / len(exposures) if exposures else 0.0
            )

        risk_metrics["turnover"] = self._calculate_turnover(
            trades, portfolio_equity_history
        )

        result["risk_metrics"] = risk_metrics

        if trades:
            result["trade_metrics"] = self.calculate_trade_metrics(trades)

        return result

    def _calculate_turnover(
        self,
        trades: list[Any],
        portfolio_equity_history: list[dict[str, Any]] | None = None,
    ) -> float:
        """Calculate annualized portfolio turnover.

        Turnover = (total_traded_value / avg_equity) * (252 / trading_days)

        Args:
            trades: List of Trade objects with price and quantity attributes
            portfolio_equity_history: Portfolio equity history for avg_equity.
                If None, uses the tracker's own equity curve.

        Returns:
            Annualized turnover ratio. Returns 0.0 if insufficient data.
        """
        if not trades:
            return 0.0

        total_traded = float(
            sum(
                abs(getattr(t, "price", 0.0) * getattr(t, "quantity", 0))
                for t in trades
            )
        )

        if portfolio_equity_history and len(portfolio_equity_history) > 0:
            equities = [entry["equity"] for entry in portfolio_equity_history]
            avg_equity = sum(equities) / len(equities)
            days = len(equities)
        elif len(self.equity_curve) >= 2:
            equities_list = [entry["equity"] for entry in self.equity_curve]
            avg_equity = sum(equities_list) / len(equities_list)
            days = len(equities_list)
        else:
            return 0.0

        if avg_equity <= 0 or days <= 0:
            return 0.0

        return float((total_traded / avg_equity) * (252 / days))

    def print_report(
        self,
        trades: list[Any] | None = None,
        portfolio_equity_history: list[dict[str, Any]] | None = None,
    ) -> None:
        """Print formatted performance report.

        Args:
            trades: Optional list of Trade objects for trade statistics
            portfolio_equity_history: Optional portfolio equity history for
                extended risk metrics (position counts, exposure, turnover)
        """
        if trades is not None and portfolio_equity_history is not None:
            metrics = self.calculate_extended_metrics(trades, portfolio_equity_history)
        else:
            metrics = self.calculate_metrics()

        if "error" in metrics:
            logger.warning("Cannot generate report: %s", metrics["error"])
            return

        lines = [
            "",
            "=" * 60,
            "BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            "",
            "Capital:",
            f"  Initial:        ${metrics['initial_capital']:,.2f}",
            f"  Final:          ${metrics['final_equity']:,.2f}",
            f"  Total Return:   {metrics['total_return_pct']:.2f}%",
            "",
            "Risk-Adjusted Metrics:",
            f"  Annualized Return: {metrics['annualized_return_pct']:.2f}%",
            f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}",
            f"  Sortino Ratio:     {metrics['sortino_ratio']:.2f}",
            f"  Calmar Ratio:      {metrics['calmar_ratio']:.2f}",
            "",
            "Risk Metrics:",
            f"  Volatility:        {metrics['volatility_pct']:.2f}%",
            f"  Max Drawdown:      {metrics['max_drawdown_pct']:.2f}%",
        ]

        if "risk_metrics" in metrics:
            rm = metrics["risk_metrics"]
            if "avg_position_count" in rm:
                lines.append(f"  Avg Positions:     {rm['avg_position_count']:.1f}")
            if "max_position_count" in rm:
                lines.append(f"  Max Positions:     {rm['max_position_count']}")
            if "gross_exposure_avg" in rm:
                lines.append(f"  Avg Exposure:      {rm['gross_exposure_avg']:.1%}")
            if "turnover" in rm:
                lines.append(f"  Turnover (ann.):   {rm['turnover']:.2f}")

        lines.append("")
        lines.append("Trading:")
        lines.append(f"  Trading Days:      {metrics['trading_days']}")

        if "trade_metrics" in metrics:
            trade_metrics = metrics["trade_metrics"]
        elif trades:
            trade_metrics = self.calculate_trade_metrics(trades)
        else:
            trade_metrics = None

        if trade_metrics:
            lines.append(f"  Total Trades:      {trade_metrics['total_trades']}")
            lines.append(f"  Win Rate:          {trade_metrics['win_rate_pct']:.1f}%")
            lines.append(f"  Profit Factor:     {trade_metrics['profit_factor']:.2f}")

            if trade_metrics["winning_trades"] > 0:
                lines.append(f"  Avg Win:           ${trade_metrics['avg_win']:.2f}")
            if trade_metrics["losing_trades"] > 0:
                lines.append(f"  Avg Loss:          ${trade_metrics['avg_loss']:.2f}")

        lines.append("")
        lines.append("=" * 60)
        logger.info("\n".join(lines))

    def reset(self) -> None:
        """Reset the equity curve for a new backtest."""
        self.equity_curve = []
