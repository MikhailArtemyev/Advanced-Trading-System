# Phase 5 — Week 4: Alerting & Notifications

**Status:** COMPLETE
**Tests added:** 42
**Cumulative tests:** 1446

---

## What Was Built

### `src/alerts/` — New Package

Alerting system with pluggable notification channels, level filtering, and cooldown-based deduplication.

### `src/alerts/base_alert.py` — Core Types

| Type | Description |
|------|-------------|
| `AlertLevel` | Enum: `INFO`, `WARNING`, `CRITICAL` |
| `AlertMessage` | Dataclass: level, title, body, timestamp, metadata dict |
| `AlertChannel` | ABC with `send(message) → bool` and `test_connection() → bool` |

### `src/alerts/slack_alert.py` — SlackAlert

Sends alerts via Slack incoming webhooks using aiohttp.

**Constructor:** `webhook_url`, `channel` (optional override), `timeout`

**Payload format:**
- Text: `{emoji} *{title}*\n{body}`
- Metadata rendered as Slack context block with mrkdwn fields
- Level emojis: INFO → `:information_source:`, WARNING → `:warning:`, CRITICAL → `:rotating_light:`

### `src/alerts/email_alert.py` — EmailAlert

Sends alerts via SMTP email using stdlib `smtplib`.

**Constructor:** `smtp_host`, `smtp_port`, `username`, `password`, `from_address`, `to_addresses`, `use_tls`, `timeout`

**Subject format:** `[LEVEL] title`
**Body:** message body + metadata as key-value pairs

### `src/alerts/webhook_alert.py` — WebhookAlert

Sends alerts as JSON to any HTTP endpoint via aiohttp.

**Constructor:** `url`, `headers` (dict), `method` (`POST`/`PUT`), `timeout`

**JSON payload:** `{level, title, body, timestamp, metadata}`

Accepts any 2xx status as success.

### `src/alerts/alert_manager.py` — AlertManager

Routes alerts to multiple channels with filtering and cooldown.

**Constructor:** `channels` (list), `min_level` (default WARNING), `cooldown_seconds` (default 300)

| Method | Description |
|--------|-------------|
| `send_alert(message)` | Dispatch to all channels, returns list of success booleans |
| `send_trade_alert(trade)` | Format trade dict → INFO alert |
| `send_health_alert(report)` | Format HealthReport → WARNING/CRITICAL alert |
| `send_drawdown_alert(pct)` | Format drawdown → WARNING (<10%) or CRITICAL (≥10%) |
| `on_health_report(report)` | Sync callback for `HealthMonitor.add_alert_callback()` — fire-and-forgets async send |

**Level filtering:** alerts below `min_level` are silently dropped.
**Cooldown:** alerts with the same title are suppressed for `cooldown_seconds` after the last send. Set to 0 to disable.

### `src/config.py` — Updated

Added `AlertConfig` to `LiveConfig`:

```python
class AlertConfig(BaseModel):
    enabled: bool = False
    channels: list[dict[str, Any]] = []  # channel configs
    min_level: str = "warning"           # info | warning | critical
    cooldown_seconds: float = 300.0
```

Accessible via `config.live.alerts`. Defaults to disabled — existing configs unchanged.

### `scripts/run_paper_trading.py` — Updated

- `build_alert_channels(config)` factory: reads `config.live.alerts.channels`, builds `SlackAlert`/`EmailAlert`/`WebhookAlert` instances based on `type` field
- `AlertManager` created when channels exist, wired into `HealthMonitor` via `add_alert_callback(alert_manager.on_health_report)`

---

## Configuration

Add `alerts` under `live:` in any config YAML:

```yaml
live:
  alerts:
    enabled: true
    min_level: "warning"
    cooldown_seconds: 300
    channels:
      - type: slack
        webhook_url: "https://hooks.slack.com/services/..."
        channel: "#trading-alerts"
      - type: email
        smtp_host: "smtp.gmail.com"
        smtp_port: 587
        username: "user@gmail.com"
        password: "app-password"
        from_address: "user@gmail.com"
        to_addresses: ["alerts@team.com"]
      - type: webhook
        url: "https://api.example.com/alerts"
        method: "POST"
        headers:
          Authorization: "Bearer token"
```

---

## Tests

### `tests/test_alerts.py` — 42 Tests

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestAlertLevel` | 2 | Enum values, member count |
| `TestAlertMessage` | 2 | Defaults, metadata |
| `TestSlackAlert` | 7 | Send success/failure/exception, payload building (basic, channel, metadata), test_connection |
| `TestEmailAlert` | 5 | Send success/failure, test_connection success/failure, _send_sync builds message |
| `TestWebhookAlert` | 6 | Send success/non-2xx/exception, build_payload, custom PUT method, test_connection |
| `TestAlertManager` | 14 | Multi-channel dispatch, level filtering, critical passes warning filter, cooldown (suppress/zero-disable/expire), channel failure, channel exception, trade/health/drawdown alerts, on_health_report fires task |
| `TestAlertConfig` | 5 | Defaults, custom values, invalid min_level, LiveConfig has alerts, YAML roundtrip |
| `TestModuleExports` | 1 | All 7 exports importable |

### Updated Tests

- `tests/test_phase4_integration.py` — Added `test_alerts_exports` to module export tests

---

## Design Decisions

1. **Pluggable channels via ABC** — easy to add new backends (PagerDuty, Telegram, etc.) by implementing `AlertChannel`
2. **Sync email via smtplib** — avoids adding `aiosmtplib` dependency; email sends are fast enough that blocking in the async path is acceptable for infrequent alerts
3. **Fire-and-forget for health callbacks** — `on_health_report()` creates an asyncio task so the sync `HealthMonitor` callback doesn't block
4. **Cooldown by title** — prevents alert storms during sustained degradation while still allowing different alert types to fire independently
5. **No external dependencies added** — uses `aiohttp` (already a dependency) for Slack/webhook, `smtplib` (stdlib) for email
