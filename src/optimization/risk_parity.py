"""Risk Parity (Equal Risk Contribution) portfolio optimizer.

Allocates weights so that each asset contributes equally to total
portfolio risk, measured by its marginal contribution to volatility.
"""

import numpy as np
import pandas as pd
from scipy import optimize as sco

from .base_optimizer import AllocationResult, PortfolioOptimizer

_MIN_OBSERVATIONS = 30
_TRADING_DAYS = 252


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk-parity optimizer targeting equal risk contribution.

    Args:
        max_weight: Upper bound per asset weight
        min_weight: Lower bound per asset weight
    """

    def __init__(
        self,
        max_weight: float = 0.40,
        min_weight: float = 0.01,
    ) -> None:
        self.max_weight = max_weight
        self.min_weight = min_weight

    def optimize(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
        current_weights: dict[str, float],
    ) -> AllocationResult:
        """Compute weights for equal risk contribution."""
        available = [s for s in symbols if s in returns_df.columns]
        if not available:
            return self._equal_weight(symbols, "no return data available")

        returns = returns_df[available].dropna()

        if len(returns) < _MIN_OBSERVATIONS:
            return self._equal_weight(
                available,
                f"insufficient data ({len(returns)} < {_MIN_OBSERVATIONS})",
            )

        if len(available) == 1:
            return self._single_asset(available[0])

        cov = returns.cov().values * _TRADING_DAYS
        n = len(available)

        w0 = np.ones(n) / n
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = sco.minimize(
            lambda w: self._risk_parity_objective(w, cov),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            return self._equal_weight(available, f"optimizer failed: {result.message}")

        weights = result.x
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()

        mu = returns.mean().values * _TRADING_DAYS
        port_ret = float(weights @ mu)
        port_vol = float(np.sqrt(weights @ cov @ weights))

        return AllocationResult(
            weights=dict(zip(available, weights.tolist(), strict=True)),
            method="risk_parity",
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=port_ret / port_vol if port_vol > 0 else 0.0,
        )

    def _risk_parity_objective(self, w: np.ndarray, cov: np.ndarray) -> float:
        """Sum of squared deviations from equal risk contribution."""
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return 0.0

        n = len(w)
        marginal = cov @ w
        risk_contrib = w * marginal / port_vol
        target_contrib = port_vol / n

        return float(np.sum((risk_contrib - target_contrib) ** 2))

    def _equal_weight(self, symbols: list[str], reason: str) -> AllocationResult:
        n = len(symbols)
        w = 1.0 / n if n > 0 else 0.0
        return AllocationResult(
            weights=dict.fromkeys(symbols, w),
            method="risk_parity_equal_weight",
            notes=f"equal weight fallback: {reason}",
        )

    def _single_asset(self, symbol: str) -> AllocationResult:
        return AllocationResult(
            weights={symbol: 1.0},
            method="risk_parity",
            notes="single asset",
        )
