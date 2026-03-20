"""Base types for the alerting system.

Defines AlertLevel, AlertMessage, and the AlertSubscriber ABC
that all notification backends implement (observer pattern).
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


class AlertSubscriber(ABC):
    """Observer interface — receives alert messages from the publisher."""

    @abstractmethod
    async def on_alert(self, message: AlertMessage) -> bool:
        """Handle an alert message. Returns True on success."""

    @abstractmethod
    async def test_connection(self) -> bool:
        """Verify that the subscriber's backend is reachable."""
