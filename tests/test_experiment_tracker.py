"""Tests for MLflow experiment tracking."""

import tempfile
from pathlib import Path

import mlflow
import pytest

from src.tracking.mlflow_tracker import ExperimentTracker, _flatten_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker(tmp_path):
    """Create a tracker with a temporary MLflow directory."""
    tracking_uri = str(tmp_path / "mlruns")
    t = ExperimentTracker(
        experiment_name="test_experiment",
        tracking_uri=tracking_uri,
    )
    yield t
    # Clean up any active runs
    if t.run_id is not None:
        t.end_run()


@pytest.fixture
def active_tracker(tracker):
    """Create a tracker with an active run."""
    tracker.start_run("test_run")
    yield tracker
    if tracker.run_id is not None:
        tracker.end_run()


# ---------------------------------------------------------------------------
# TestFlattenDict
# ---------------------------------------------------------------------------


class TestFlattenDict:
    def test_flat_dict(self):
        result = _flatten_dict({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = _flatten_dict({"a": {"b": 1, "c": 2}})
        assert result == {"a.b": 1, "a.c": 2}

    def test_deeply_nested(self):
        result = _flatten_dict({"a": {"b": {"c": 3}}})
        assert result == {"a.b.c": 3}

    def test_mixed_nested_and_flat(self):
        result = _flatten_dict({"x": 1, "y": {"z": 2}})
        assert result == {"x": 1, "y.z": 2}

    def test_empty_dict(self):
        result = _flatten_dict({})
        assert result == {}

    def test_list_values_not_flattened(self):
        result = _flatten_dict({"a": [1, 2, 3]})
        assert result == {"a": [1, 2, 3]}


# ---------------------------------------------------------------------------
# TestTrackerInit
# ---------------------------------------------------------------------------


class TestTrackerInit:
    def test_initialization(self, tracker):
        assert tracker.experiment_name == "test_experiment"
        assert tracker.run_id is None

    def test_custom_experiment_name(self, tmp_path):
        t = ExperimentTracker(
            experiment_name="my_experiment",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        assert t.experiment_name == "my_experiment"


# ---------------------------------------------------------------------------
# TestStartEndRun
# ---------------------------------------------------------------------------


class TestStartEndRun:
    def test_start_run_returns_id(self, tracker):
        run_id = tracker.start_run("test")
        assert isinstance(run_id, str)
        assert len(run_id) > 0
        assert tracker.run_id == run_id
        tracker.end_run()

    def test_end_run_clears_id(self, tracker):
        tracker.start_run("test")
        tracker.end_run()
        assert tracker.run_id is None

    def test_start_while_active_raises(self, tracker):
        tracker.start_run("first")
        with pytest.raises(RuntimeError, match="already active"):
            tracker.start_run("second")
        tracker.end_run()

    def test_run_name_optional(self, tracker):
        run_id = tracker.start_run()
        assert isinstance(run_id, str)
        tracker.end_run()


# ---------------------------------------------------------------------------
# TestContextManager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager(self, tracker):
        with tracker.run("ctx_test") as run_id:
            assert isinstance(run_id, str)
            assert tracker.run_id is not None
        assert tracker.run_id is None

    def test_context_manager_cleans_up_on_error(self, tracker):
        with pytest.raises(ValueError):
            with tracker.run("ctx_error"):
                raise ValueError("test error")
        assert tracker.run_id is None


# ---------------------------------------------------------------------------
# TestLogParams
# ---------------------------------------------------------------------------


class TestLogParams:
    def test_log_simple_params(self, active_tracker):
        active_tracker.log_params({"strategy": "sma", "window": 20})
        # No exception means success
        run = mlflow.active_run()
        assert run is not None

    def test_log_nested_params(self, active_tracker):
        active_tracker.log_params(
            {"strategy": {"name": "sma", "params": {"window": 20}}}
        )

    def test_long_param_truncated(self, active_tracker):
        long_val = "x" * 600
        active_tracker.log_params({"long_param": long_val})
        # Should not raise — value is truncated to 500 chars


# ---------------------------------------------------------------------------
# TestLogMetrics
# ---------------------------------------------------------------------------


class TestLogMetrics:
    def test_log_numeric_metrics(self, active_tracker):
        active_tracker.log_metrics({"sharpe_ratio": 0.68, "max_drawdown": -12.5})

    def test_skips_non_numeric(self, active_tracker):
        # Should not raise — non-numeric values are silently skipped
        active_tracker.log_metrics(
            {"sharpe": 0.68, "name": "not_a_number"}  # type: ignore[dict-item]
        )

    def test_skips_bool(self, active_tracker):
        # bool is technically int subclass, but should be skipped
        active_tracker.log_metrics(
            {"is_good": True, "sharpe": 1.0}  # type: ignore[dict-item]
        )

    def test_log_with_step(self, active_tracker):
        active_tracker.log_metrics({"equity": 100000}, step=0)
        active_tracker.log_metrics({"equity": 105000}, step=1)

    def test_log_integers(self, active_tracker):
        active_tracker.log_metrics({"n_trades": 42, "n_wins": 20})


# ---------------------------------------------------------------------------
# TestLogArtifact
# ---------------------------------------------------------------------------


class TestLogArtifact:
    def test_log_existing_file(self, active_tracker):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test artifact")
            f.flush()
            active_tracker.log_artifact(f.name)

    def test_log_nonexistent_file_raises(self, active_tracker):
        with pytest.raises(FileNotFoundError):
            active_tracker.log_artifact("/nonexistent/path.txt")

    def test_log_path_object(self, active_tracker):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2")
            f.flush()
            active_tracker.log_artifact(Path(f.name))


# ---------------------------------------------------------------------------
# TestLogBacktestResult
# ---------------------------------------------------------------------------


class TestLogBacktestResult:
    def test_logs_config_and_metrics(self, active_tracker):
        config = {
            "strategy": "sma_crossover",
            "sizing": {"method": "fixed", "fraction": 0.1},
        }
        metrics = {
            "sharpe_ratio": 0.68,
            "total_return_pct": 15.3,
            "max_drawdown_pct": -12.5,
        }
        active_tracker.log_backtest_result(config, metrics)

    def test_with_trade_metrics(self, active_tracker):
        config = {"strategy": "sma"}
        metrics = {"sharpe_ratio": 0.5}
        trade_metrics = {
            "win_rate_pct": 42.0,
            "profit_factor": 1.2,
            "avg_trade_pnl": 150.0,
        }
        active_tracker.log_backtest_result(config, metrics, trade_metrics)

    def test_filters_non_numeric_metrics(self, active_tracker):
        config = {"strategy": "sma"}
        metrics = {
            "sharpe_ratio": 0.5,
            "equity_curve": "not_numeric",
        }
        active_tracker.log_backtest_result(config, metrics)


# ---------------------------------------------------------------------------
# TestLogModelResult
# ---------------------------------------------------------------------------


class TestLogModelResult:
    def test_basic_model_result(self, active_tracker):
        active_tracker.log_model_result("xgboost_clf", train_score=0.85)

    def test_with_val_score(self, active_tracker):
        active_tracker.log_model_result("lightgbm_reg", train_score=0.9, val_score=0.75)

    def test_with_feature_importance(self, active_tracker):
        importance = {
            "rsi_14": 0.15,
            "sma_20": 0.12,
            "volatility": 0.10,
        }
        active_tracker.log_model_result(
            "xgboost_clf",
            train_score=0.85,
            feature_importance=importance,
        )

    def test_special_chars_in_feature_names(self, active_tracker):
        importance = {"return/lag_1": 0.1, "close diff": 0.05}
        active_tracker.log_model_result(
            "model",
            train_score=0.8,
            feature_importance=importance,
        )


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    def test_full_workflow(self, tracker):
        """Start run, log everything, end run."""
        with tracker.run("integration_test") as run_id:
            tracker.log_params(
                {
                    "strategy": "sma_crossover",
                    "sizing": {"method": "volatility", "target": 0.02},
                }
            )
            tracker.log_metrics(
                {
                    "sharpe_ratio": 0.68,
                    "total_return_pct": 15.3,
                    "max_drawdown_pct": -12.5,
                }
            )
            tracker.log_model_result("xgboost", train_score=0.85, val_score=0.72)

        assert tracker.run_id is None

        # Verify the run was logged
        client = mlflow.tracking.MlflowClient(tracker.tracking_uri)
        run = client.get_run(run_id)
        assert run.data.params["strategy"] == "sma_crossover"
        assert run.data.params["sizing.method"] == "volatility"
        assert abs(run.data.metrics["sharpe_ratio"] - 0.68) < 1e-6

    def test_multiple_runs(self, tracker):
        """Multiple sequential runs should work."""
        with tracker.run("run1"):
            tracker.log_metrics({"sharpe": 0.5})
        with tracker.run("run2"):
            tracker.log_metrics({"sharpe": 0.8})
        assert tracker.run_id is None
