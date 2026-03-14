"""Base types for the alerting system.

Defines AlertLevel, AlertMessage, and the AlertChannel ABC that
all notification backends implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AlertLevel(Enum):
    """Severity level for alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertMessage:
    """A single alert notification.

    Attributes:
        level: Severity of the alert.
        title: Short summary (suitable for subject lines / titles).
        body: Detailed description.
        timestamp: When the alert was generated.
        metadata: Arbitrary key-value pairs for channel-specific rendering.
    """

    level: AlertLevel
    title: str
    body: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertChannel(ABC):
    """Abstract base for notification backends (Slack, email, webhook, etc.)."""

    @abstractmethod
    async def send(self, message: AlertMessage) -> bool:
        """Send an alert message. Returns True on success."""

    @abstractmethod
    async def test_connection(self) -> bool:
        """Verify that the channel is reachable. Returns True on success."""
