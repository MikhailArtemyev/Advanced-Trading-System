"""Mean-Variance (Markowitz) portfolio optimizer.

Supports max-Sharpe and min-variance targets with optional
Ledoit-Wolf covariance shrinkage for numerical stability.
"""

import numpy as np
import pandas as pd
from scipy import optimize as sco

from .base_optimizer import AllocationResult, PortfolioOptimizer
from .constants import MIN_OBSERVATIONS, TRADING_DAYS


class MeanVarianceOptimizer(PortfolioOptimizer):
    """Markowitz mean-variance optimizer.

    Args:
        risk_free_rate: Annualized risk-free rate for Sharpe calculation
        target: Optimization objective — "max_sharpe" or "min_variance"
        max_weight: Upper bound per asset weight
        min_weight: Lower bound per asset weight
        shrinkage: Whether to apply Ledoit-Wolf covariance shrinkage
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        target: str = "max_sharpe",
        max_weight: float = 0.30,
        min_weight: float = 0.0,
        shrinkage: bool = True,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.target = target
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.shrinkage = shrinkage

    def optimize(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
        current_weights: dict[str, float],
    ) -> AllocationResult:
        """Compute optimal weights via mean-variance optimization."""
        available = [s for s in symbols if s in returns_df.columns]
        if not available:
            return self._equal_weight(symbols, "no return data available")

        returns = returns_df[available].dropna()

        if len(returns) < MIN_OBSERVATIONS:
            return self._equal_weight(
                available,
                f"insufficient data ({len(returns)} < {MIN_OBSERVATIONS})",
            )

        if len(available) == 1:
            return self._single_asset(available[0])

        mu = returns.mean().values * TRADING_DAYS

        if self.shrinkage:
            daily_cov = returns.cov().values
            cov = _ledoit_wolf_shrinkage(returns.values, daily_cov) * TRADING_DAYS
        else:
            cov = returns.cov().values * TRADING_DAYS

        n = len(available)
        w0 = np.ones(n) / n
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if self.target == "max_sharpe":
            result = sco.minimize(
                lambda w: self._neg_sharpe(w, mu, cov),
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
        else:  # min_variance
            result = sco.minimize(
                lambda w: self._portfolio_vol(w, cov),
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

        port_ret = float(weights @ mu)
        port_vol = float(np.sqrt(weights @ cov @ weights))
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return AllocationResult(
            weights=dict(zip(available, weights.tolist(), strict=True)),
            method=f"mean_variance_{self.target}",
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
        )

    def _neg_sharpe(self, w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return 0.0
        return float(-(port_ret - self.risk_free_rate) / port_vol)

    def _portfolio_vol(self, w: np.ndarray, cov: np.ndarray) -> float:
        return float(np.sqrt(w @ cov @ w))

    @property
    def _method_name(self) -> str:
        return f"mean_variance_{self.target}"


def _ledoit_wolf_shrinkage(returns: np.ndarray, sample_cov: np.ndarray) -> np.ndarray:
    """Apply Ledoit-Wolf shrinkage to covariance matrix.

    Shrinks toward a scaled identity matrix to improve numerical stability
    when the number of assets is close to the number of observations.

    Args:
        returns: T x N array of daily returns
        sample_cov: N x N daily sample covariance matrix

    Returns:
        Shrunk daily covariance matrix
    """
    t, n = returns.shape

    if n <= 1:
        return sample_cov

    # Target: scaled identity (average variance on diagonal)
    avg_var = np.trace(sample_cov) / n
    target = avg_var * np.eye(n)

    # Compute shrinkage intensity
    # De-mean returns for computing the shrinkage intensity
    x = returns - returns.mean(axis=0)
    # Sum of squared off-diagonal elements of sample_cov
    sum_sq = np.sum(sample_cov**2)
    # Sum of fourth moments
    y = x**2
    phi = np.sum(y.T @ y) / t**2 - sum_sq / t
    gamma = np.sum((sample_cov - target) ** 2)

    if gamma == 0:
        return sample_cov

    kappa = phi / gamma
    intensity = max(0.0, min(1.0, kappa))

    result: np.ndarray = intensity * target + (1.0 - intensity) * sample_cov
    return result
