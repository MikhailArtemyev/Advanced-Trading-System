# Phase 5 — Week 7: Environment Variables & Startup Validation

**Status:** COMPLETE
**Tests added:** 15
**Cumulative tests:** 1527

---

## What Was Built

### Environment Variable Support (`src/config.py`)

Pydantic `model_post_init` hooks resolve secrets from environment variables when YAML values are empty:

| Model | Env Var | Condition |
|-------|---------|-----------|
| `BrokerConfig` | `ALPACA_API_KEY` | When `api_key` is empty |
| `BrokerConfig` | `ALPACA_API_SECRET` | When `api_secret` is empty |
| `DatabaseConfig` | `DATABASE_URL` | When `db_url` is the default (`sqlite:///trading.db`) |
| `AlertConfig` | `SLACK_WEBHOOK_URL` | When slack channel has no `webhook_url` |
| `AlertConfig` | `SMTP_PASSWORD` | When email channel has no `password` |

**Precedence:** YAML values always win over environment variables. Env vars are only consulted when the field is empty/default.

### Startup Validation (`BacktestConfig.validate_for_live()`)

Returns a list of issue strings prefixed with `ERROR:` or `WARNING:`.

**Checks:**
1. API credentials present when `broker_type != "paper"`
2. Symbols list non-empty and no blank entries
3. Alert channels configured when alerts are enabled
4. Slack webhook_url and email password present on configured channels

### Script Integration (`scripts/run_paper_trading.py`)

`validate_for_live()` called at startup before `asyncio.run()`. Errors → `sys.exit(1)`. Warnings logged but execution continues.

### `.env.example`

Template documenting all supported environment variables.

---

## Tests

### `tests/test_config_env_validation.py` — 15 Tests

| Test Class | Count | Coverage |
|------------|-------|---------|
| `TestEnvVarLoading` | 8 | API key/secret from env, YAML precedence, DATABASE_URL, Slack webhook, SMTP password, no-env defaults |
| `TestValidateForLive` | 7 | Paper broker OK, alpaca missing creds, alpaca with creds, empty symbols, alerts no channels, slack missing webhook, valid config |
