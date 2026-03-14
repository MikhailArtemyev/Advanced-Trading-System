"""Slack webhook notification channel."""

import logging
from typing import Any

import aiohttp

from .base_alert import AlertChannel, AlertLevel, AlertMessage

logger = logging.getLogger(__name__)

_LEVEL_EMOJI: dict[AlertLevel, str] = {
    AlertLevel.INFO: ":information_source:",
    AlertLevel.WARNING: ":warning:",
    AlertLevel.CRITICAL: ":rotating_light:",
}


class SlackAlert(AlertChannel):
    """Send alerts to a Slack channel via incoming webhook.

    Args:
        webhook_url: Slack incoming-webhook URL.
        channel: Optional channel override (only works with legacy webhooks).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str = "",
        timeout: float = 10.0,
    ) -> None:
        self._webhook_url = webhook_url
        self._channel = channel
        self._timeout = timeout

    async def send(self, message: AlertMessage) -> bool:
        """Post a formatted message to Slack."""
        payload = self._build_payload(message)
        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self._webhook_url, json=payload) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    logger.warning("Slack webhook returned %d: %s", resp.status, body)
                    return False
        except Exception:
            logger.exception("Failed to send Slack alert")
            return False

    async def test_connection(self) -> bool:
        """Send a test message to verify the webhook works."""
        test_msg = AlertMessage(
            level=AlertLevel.INFO,
            title="Connection Test",
            body="Alert channel is connected.",
        )
        return await self.send(test_msg)

    def _build_payload(self, message: AlertMessage) -> dict[str, Any]:
        """Build Slack webhook JSON payload with blocks."""
        emoji = _LEVEL_EMOJI.get(message.level, "")
        text = f"{emoji} *{message.title}*\n{message.body}"

        payload: dict[str, Any] = {"text": text}
        if self._channel:
            payload["channel"] = self._channel

        # Add metadata fields as context block
        if message.metadata:
            fields = " | ".join(f"*{k}:* {v}" for k, v in message.metadata.items())
            payload["blocks"] = [
                {"type": "section", "text": {"type": "mrkdwn", "text": text}},
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": fields}],
                },
            ]

        return payload
