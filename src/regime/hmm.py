"""Hidden Markov Model regime detection.

Identifies market regimes (bull/bear/sideways) from return
and volatility characteristics. Useful for:
1. Regime-conditional strategy selection
2. Position sizing adjustment by regime
3. Risk management (tighter limits in bear regimes)

Uses hmmlearn's GaussianHMM to fit a hidden Markov model on
two observable features: daily returns and rolling volatility.
Regimes are automatically labeled by their mean return:
    - Highest mean return → "bull"
    - Lowest mean return → "bear"
    - Middle regimes → "sideways"
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class RegimeResult:
    """Result of regime detection.

    Attributes:
        regimes: Series mapping dates to regime labels (0, 1, 2, ...).
        regime_names: Mapping from regime number to descriptive name.
        transition_matrix: Regime transition probability matrix.
        regime_means: Mean return per regime.
        regime_vols: Volatility per regime.
        current_regime: Most recent detected regime.
    """

    regimes: pd.Series
    regime_names: dict[int, str] = field(default_factory=dict)
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_means: dict[int, float] = field(default_factory=dict)
    regime_vols: dict[int, float] = field(default_factory=dict)
    current_regime: int = 0


class RegimeDetector:
    """HMM-based market regime detection.

    Fits a Gaussian HMM to return and volatility observations,
    then labels each period with a regime.

    Regimes are automatically labeled by their mean return:
        - Highest mean → "bull"
        - Lowest mean → "bear"
        - Middle → "sideways"

    Args:
        n_regimes: Number of regimes to detect (default 3).
        vol_window: Window for rolling volatility feature (default 20).
        n_iter: Maximum EM iterations for HMM fitting (default 100).
        random_state: Random seed for reproducibility (default 42).

    Raises:
        ValueError: If n_regimes < 2 or vol_window < 2.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        vol_window: int = 20,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        if n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")
        if vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {vol_window}")

        self.n_regimes = n_regimes
        self.vol_window = vol_window
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: GaussianHMM | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    def fit_predict(self, ohlcv: pd.DataFrame) -> RegimeResult:
        """Fit HMM and predict regimes for the entire series.

        Args:
            ohlcv: OHLCV DataFrame with a 'close' column.
                Must have enough rows for vol_window + 1.

        Returns:
            RegimeResult with regime labels, statistics, and names.

        Raises:
            ValueError: If insufficient data or missing 'close' column.
        """
        if "close" not in ohlcv.columns:
            raise ValueError("ohlcv must contain a 'close' column")

        observations = self._build_observations(ohlcv)
        observations = observations.dropna()

        if len(observations) < self.n_regimes:
            raise ValueError(
                f"Insufficient data after building observations: "
                f"{len(observations)} rows, need >= {self.n_regimes}"
            )

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(observations.values)
        self._is_fitted = True

        hidden_states = self.model.predict(observations.values)
        regimes = pd.Series(hidden_states, index=observations.index, name="regime")

        # Compute per-regime statistics from returns
        returns = ohlcv["close"].pct_change().reindex(observations.index)
        regime_means: dict[int, float] = {}
        regime_vols: dict[int, float] = {}

        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.any():
                regime_means[r] = float(returns[mask].mean())
                regime_vols[r] = float(returns[mask].std())
            else:
                regime_means[r] = 0.0
                regime_vols[r] = 0.0

        # Auto-label regimes by mean return
        regime_names = self._label_regimes(regime_means)

        return RegimeResult(
            regimes=regimes,
            regime_names=regime_names,
            transition_matrix=self.model.transmat_,
            regime_means=regime_means,
            regime_vols=regime_vols,
            current_regime=int(hidden_states[-1]),
        )

    def predict_current(self, ohlcv: pd.DataFrame) -> int:
        """Predict the current regime for the latest bar.

        Requires the model to be fitted first via fit_predict().

        Args:
            ohlcv: Recent OHLCV data with a 'close' column.

        Returns:
            Regime label for the latest bar.

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If insufficient data.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        observations = self._build_observations(ohlcv).dropna()
        if observations.empty:
            raise ValueError("Insufficient data for prediction")

        states = self.model.predict(observations.values)
        return int(states[-1])

    def predict_proba(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """Get regime probabilities for the latest bar.

        Args:
            ohlcv: Recent OHLCV data with a 'close' column.

        Returns:
            Array of shape (n_regimes,) with probability for each regime.

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If insufficient data.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        observations = self._build_observations(ohlcv).dropna()
        if observations.empty:
            raise ValueError("Insufficient data for prediction")

        posteriors = self.model.predict_proba(observations.values)
        return np.array(posteriors[-1])

    def _build_observations(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Build observation matrix for HMM from OHLCV data.

        Features:
            - return: daily log return
            - volatility: rolling standard deviation of returns

        Args:
            ohlcv: OHLCV DataFrame with a 'close' column.

        Returns:
            DataFrame with 'return' and 'volatility' columns.
        """
        close = ohlcv["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()

        return pd.DataFrame({"return": returns, "volatility": vol})

    def _label_regimes(self, regime_means: dict[int, float]) -> dict[int, str]:
        """Auto-label regimes by their mean return.

        Args:
            regime_means: Mean return per regime.

        Returns:
            Mapping from regime number to name.
        """
        sorted_regimes = sorted(regime_means.keys(), key=lambda x: regime_means[x])

        names: dict[int, str] = {}
        if self.n_regimes == 2:
            names[sorted_regimes[0]] = "bear"
            names[sorted_regimes[1]] = "bull"
        elif self.n_regimes >= 3:
            names[sorted_regimes[0]] = "bear"
            names[sorted_regimes[-1]] = "bull"
            for i in range(1, len(sorted_regimes) - 1):
                names[sorted_regimes[i]] = "sideways"

        return names
