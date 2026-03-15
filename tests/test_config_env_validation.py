"""Tests for environment variable loading and startup validation."""

from src.config import (
    AlertConfig,
    BacktestConfig,
    BrokerConfig,
    DatabaseConfig,
    DataConfig,
    ExecutionConfig,
    LiveConfig,
    StrategyConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(**live_overrides):
    """Build a minimal valid BacktestConfig with optional LiveConfig overrides."""
    live_cfg = LiveConfig(**live_overrides) if live_overrides else LiveConfig()
    return BacktestConfig(
        data=DataConfig(
            symbols=["AAPL", "MSFT"],
            start_date="2020-01-01",
            end_date="2020-12-31",
        ),
        execution=ExecutionConfig(),
        strategy=StrategyConfig(name="sma_crossover"),
        live=live_cfg,
    )


# ===========================================================================
# Environment variable loading
# ===========================================================================


class TestEnvVarLoading:
    def test_broker_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "env_key_123")
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
        cfg = BrokerConfig(api_key="")
        assert cfg.api_key == "env_key_123"

    def test_broker_api_secret_from_env(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.setenv("ALPACA_API_SECRET", "env_secret_456")
        cfg = BrokerConfig(api_secret="")
        assert cfg.api_secret == "env_secret_456"

    def test_yaml_value_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "env_key")
        cfg = BrokerConfig(api_key="yaml_key")
        assert cfg.api_key == "yaml_key"

    def test_database_url_from_env(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///custom.db")
        cfg = DatabaseConfig()
        assert cfg.db_url == "sqlite:///custom.db"

    def test_database_yaml_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///env.db")
        cfg = DatabaseConfig(db_url="sqlite:///explicit.db")
        assert cfg.db_url == "sqlite:///explicit.db"

    def test_slack_webhook_from_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        cfg = AlertConfig(
            enabled=True,
            channels=[{"type": "slack"}],
        )
        assert cfg.channels[0]["webhook_url"] == "https://hooks.slack.com/test"

    def test_smtp_password_from_env(self, monkeypatch):
        monkeypatch.setenv("SMTP_PASSWORD", "secret123")
        cfg = AlertConfig(
            enabled=True,
            channels=[{"type": "email", "smtp_host": "smtp.example.com"}],
        )
        assert cfg.channels[0]["password"] == "secret123"

    def test_no_env_vars_keeps_defaults(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        cfg = BrokerConfig()
        assert cfg.api_key == ""
        assert cfg.api_secret == ""


# ===========================================================================
# Startup validation
# ===========================================================================


class TestValidateForLive:
    def test_paper_broker_no_errors(self):
        cfg = _base_config(broker=BrokerConfig(broker_type="paper"))
        issues = cfg.validate_for_live()
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert errors == []

    def test_alpaca_broker_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
        cfg = _base_config(
            broker=BrokerConfig(broker_type="alpaca", api_key="", api_secret="")
        )
        issues = cfg.validate_for_live()
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert len(errors) == 2
        assert any("api_key" in e for e in errors)
        assert any("api_secret" in e for e in errors)

    def test_alpaca_broker_with_credentials_passes(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
        cfg = _base_config(
            broker=BrokerConfig(
                broker_type="alpaca", api_key="key", api_secret="secret"
            )
        )
        issues = cfg.validate_for_live()
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert errors == []

    def test_empty_symbols(self):
        cfg = BacktestConfig(
            data=DataConfig(symbols=[], start_date="2020-01-01", end_date="2020-12-31"),
            execution=ExecutionConfig(),
            strategy=StrategyConfig(name="sma_crossover"),
        )
        issues = cfg.validate_for_live()
        assert any("No symbols" in i for i in issues)

    def test_alerts_enabled_no_channels(self):
        cfg = _base_config(alerts=AlertConfig(enabled=True, channels=[]))
        issues = cfg.validate_for_live()
        warnings = [i for i in issues if i.startswith("WARNING:")]
        assert any("no channels" in w for w in warnings)

    def test_slack_channel_missing_webhook(self, monkeypatch):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        cfg = _base_config(
            alerts=AlertConfig(
                enabled=True,
                channels=[{"type": "slack", "webhook_url": ""}],
            )
        )
        issues = cfg.validate_for_live()
        warnings = [i for i in issues if i.startswith("WARNING:")]
        assert any("webhook_url" in w for w in warnings)

    def test_valid_config_no_issues(self):
        cfg = _base_config()
        issues = cfg.validate_for_live()
        assert issues == []
