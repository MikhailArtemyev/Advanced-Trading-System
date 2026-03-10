"""MLflow experiment tracking wrapper.

Provides a clean interface for logging backtest and ML experiment
results to MLflow. Supports parameter, metric, and artifact logging.

Usage:
    tracker = ExperimentTracker("trading_system")
    tracker.start_run("baseline_backtest")
    tracker.log_params({"strategy": "sma", "short_window": 20})
    tracker.log_metrics({"sharpe_ratio": 0.68, "max_drawdown_pct": -12.5})
    tracker.log_artifact("output/equity_curve.png")
    tracker.end_run()

Or as a context manager:
    with tracker.run("baseline_backtest"):
        tracker.log_params({...})
        tracker.log_metrics({...})
"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


class ExperimentTracker:
    """MLflow experiment tracking wrapper.

    Wraps MLflow to provide a simplified interface for:
    1. Logging backtest parameters (strategy, sizing, risk limits)
    2. Logging performance metrics (Sharpe, drawdown, return)
    3. Logging ML model metrics (train/val scores, feature importance)
    4. Saving artifacts (trained models, equity curves, feature plots)

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking server URI. Defaults to local
            'mlruns' directory.
    """

    def __init__(
        self,
        experiment_name: str = "trading_system",
        tracking_uri: str = "mlruns",
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_id: str | None = None

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None) -> str:
        """Start a new MLflow run.

        Args:
            run_name: Optional human-readable name for the run.

        Returns:
            The run ID.

        Raises:
            RuntimeError: If a run is already active.
        """
        if self.run_id is not None:
            raise RuntimeError(
                f"Run already active (id={self.run_id}). " f"Call end_run() first."
            )

        active_run = mlflow.start_run(run_name=run_name)
        self.run_id = active_run.info.run_id
        return self.run_id

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        self.run_id = None

    @contextmanager
    def run(self, run_name: str | None = None) -> Generator[str, None, None]:
        """Context manager for an MLflow run.

        Args:
            run_name: Optional human-readable name for the run.

        Yields:
            The run ID.
        """
        run_id = self.start_run(run_name)
        try:
            yield run_id
        finally:
            self.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters (flattens nested dicts with dot notation).

        Args:
            params: Parameter dict. Nested dicts are flattened:
                {"strategy": {"name": "sma"}} → {"strategy.name": "sma"}
        """
        flat = _flatten_dict(params)
        for key, value in flat.items():
            # MLflow param values must be strings of <= 500 chars
            str_val = str(value)
            if len(str_val) > 500:
                str_val = str_val[:497] + "..."
            mlflow.log_param(key, str_val)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics.

        Non-numeric values are silently skipped.

        Args:
            metrics: Metric name → value dict.
            step: Optional step number for time-series metrics.
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if step is not None:
                    mlflow.log_metric(key, float(value), step=step)
                else:
                    mlflow.log_metric(key, float(value))

    def log_artifact(self, path: str | Path) -> None:
        """Log a file as an artifact.

        Args:
            path: Path to the file to log.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        mlflow.log_artifact(str(p))

    def log_backtest_result(
        self,
        config: dict[str, Any],
        metrics: dict[str, Any],
        trade_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Log a complete backtest result.

        Convenience method that logs config as parameters and
        performance/trade metrics as metrics.

        Args:
            config: Backtest configuration dict.
            metrics: Performance metrics from PerformanceTracker.
            trade_metrics: Optional trade-level metrics.
        """
        self.log_params(config)

        numeric_metrics = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        self.log_metrics(numeric_metrics)

        if trade_metrics:
            prefixed = {
                f"trade_{k}": v
                for k, v in trade_metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            self.log_metrics(prefixed)

    def log_model_result(
        self,
        model_name: str,
        train_score: float,
        val_score: float | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> None:
        """Log ML model training results.

        Args:
            model_name: Name of the model (e.g., "xgboost_clf").
            train_score: Training score (accuracy or R²).
            val_score: Optional validation score.
            feature_importance: Optional feature importance dict.
        """
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("train_score", train_score)
        if val_score is not None:
            mlflow.log_metric("val_score", val_score)
        if feature_importance:
            # Log top 20 features as metrics
            sorted_feats = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for feat_name, importance in sorted_feats[:20]:
                safe_name = feat_name.replace(" ", "_").replace("/", "_")
                mlflow.log_metric(f"feat_imp_{safe_name}", importance)


def _flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """Flatten nested dict with dot notation keys.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys (used in recursion).

    Returns:
        Flat dict with dot-separated keys.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
