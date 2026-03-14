"""Generic HTTP webhook notification channel."""

import logging
from typing import Any

import aiohttp

from .base_alert import AlertChannel, AlertLevel, AlertMessage

logger = logging.getLogger(__name__)


class WebhookAlert(AlertChannel):
    """Send alerts to an arbitrary HTTP endpoint.

    Args:
        url: Webhook endpoint URL.
        headers: Extra HTTP headers (e.g. auth tokens).
        method: HTTP method — ``"POST"`` (default) or ``"PUT"``.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        method: str = "POST",
        timeout: float = 10.0,
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._method = method.upper()
        self._timeout = timeout

    async def send(self, message: AlertMessage) -> bool:
        """Send alert as JSON payload to the webhook endpoint."""
        payload = self._build_payload(message)
        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                request_method = getattr(session, self._method.lower())
                async with request_method(
                    self._url, json=payload, headers=self._headers
                ) as resp:
                    if 200 <= resp.status < 300:
                        return True
                    body = await resp.text()
                    logger.warning("Webhook returned %d: %s", resp.status, body)
                    return False
        except Exception:
            logger.exception("Failed to send webhook alert")
            return False

    async def test_connection(self) -> bool:
        """Send a health-check request to the webhook endpoint."""
        test_msg = AlertMessage(
            level=AlertLevel.INFO,
            title="Connection Test",
            body="Webhook channel is connected.",
        )
        return await self.send(test_msg)

    @staticmethod
    def _build_payload(message: AlertMessage) -> dict[str, Any]:
        """Build a JSON-serializable payload."""
        return {
            "level": message.level.value,
            "title": message.title,
            "body": message.body,
            "timestamp": message.timestamp.isoformat(),
            "metadata": message.metadata,
        }
