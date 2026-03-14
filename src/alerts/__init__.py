"""Alerting and notification system for trading events.

Provides AlertChannel ABC with Slack, email, and webhook implementations,
plus an AlertManager that routes alerts with level filtering and cooldown.
"""

from .alert_manager import AlertManager
from .base_alert import AlertChannel, AlertLevel, AlertMessage
from .email_alert import EmailAlert
from .slack_alert import SlackAlert
from .webhook_alert import WebhookAlert

__all__ = [
    "AlertChannel",
    "AlertLevel",
    "AlertManager",
    "AlertMessage",
    "EmailAlert",
    "SlackAlert",
    "WebhookAlert",
]
