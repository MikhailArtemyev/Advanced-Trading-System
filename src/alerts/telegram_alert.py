"""Telegram alert subscriber — sends notifications via Telegram Bot API.

Uses the Telegram Bot API directly via HTTPS (no third-party library).
Supports sending formatted messages with MarkdownV2.

Requires:
    - TELEGRAM_BOT_TOKEN env var (or passed directly)
    - TELEGRAM_CHAT_IDS env var (comma-separated list of allowed chat IDs)
"""

import logging
import os
from typing import Any

import aiohttp

from .base_alert import AlertLevel, AlertMessage, AlertSubscriber

logger = logging.getLogger(__name__)

_LEVEL_EMOJI: dict[AlertLevel, str] = {
    AlertLevel.INFO: "\u2139\ufe0f",  # ℹ️
    AlertLevel.WARNING: "\u26a0\ufe0f",  # ⚠️
    AlertLevel.CRITICAL: "\U0001f6a8",  # 🚨
}

_API_BASE = "https://api.telegram.org/bot{token}"


def _escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"_*[]()~`>#+-=|{}.!"
    out: list[str] = []
    for ch in text:
        if ch in special:
            out.append("\\")
        out.append(ch)
    return "".join(out)


class TelegramAlert(AlertSubscriber):
    """Send alerts to one or more Telegram chats via Bot API.

    Args:
        token: Telegram Bot API token. Falls back to TELEGRAM_BOT_TOKEN env var.
        chat_ids: List of chat IDs to send messages to.
            Falls back to TELEGRAM_CHAT_IDS env var (comma-separated).
    """

    def __init__(
        self,
        token: str = "",
        chat_ids: list[str] | None = None,
    ) -> None:
        self._token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if chat_ids:
            self._chat_ids = chat_ids
        else:
            raw = os.environ.get("TELEGRAM_CHAT_IDS", "")
            self._chat_ids = [c.strip() for c in raw.split(",") if c.strip()]

        if not self._token:
            logger.warning("TelegramAlert: no bot token configured")
        if not self._chat_ids:
            logger.warning("TelegramAlert: no chat IDs configured")

        self._base_url = _API_BASE.format(token=self._token)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def on_alert(self, message: AlertMessage) -> bool:
        """Send alert to all configured Telegram chats."""
        if not self._token or not self._chat_ids:
            return False

        text = self._format_message(message)
        session = await self._get_session()
        url = f"{self._base_url}/sendMessage"

        all_ok = True
        for chat_id in self._chat_ids:
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "MarkdownV2",
            }
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(
                            "Telegram API error (chat %s): %d %s",
                            chat_id,
                            resp.status,
                            body[:200],
                        )
                        all_ok = False
            except Exception:
                logger.exception("Failed to send Telegram alert to chat %s", chat_id)
                all_ok = False

        return all_ok

    async def test_connection(self) -> bool:
        """Verify bot token is valid by calling getMe."""
        if not self._token:
            return False
        session = await self._get_session()
        url = f"{self._base_url}/getMe"
        try:
            async with session.get(url) as resp:
                return resp.status == 200
        except Exception:
            logger.exception("Telegram connection test failed")
            return False

    async def send_text(self, text: str) -> bool:
        """Send a plain text message (no MarkdownV2 escaping).

        Useful for sending pre-formatted messages directly.
        """
        if not self._token or not self._chat_ids:
            return False

        session = await self._get_session()
        url = f"{self._base_url}/sendMessage"

        all_ok = True
        for chat_id in self._chat_ids:
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
            }
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        all_ok = False
            except Exception:
                all_ok = False
        return all_ok

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @staticmethod
    def _format_message(message: AlertMessage) -> str:
        """Format AlertMessage for Telegram MarkdownV2."""
        emoji = _LEVEL_EMOJI.get(message.level, "")
        level = _escape_md(message.level.value.upper())
        title = _escape_md(message.title)
        body = _escape_md(message.body)
        ts = _escape_md(message.timestamp.strftime("%H:%M:%S"))

        lines = [
            f"{emoji} *{level}*: {title}",
            "",
            body,
            "",
            f"_{ts}_",
        ]
        return "\n".join(lines)
