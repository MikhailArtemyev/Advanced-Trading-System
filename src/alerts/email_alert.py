"""SMTP email notification channel."""

import logging
import smtplib
from email.message import EmailMessage

from .base_alert import AlertChannel, AlertLevel, AlertMessage

logger = logging.getLogger(__name__)

_LEVEL_PREFIX: dict[AlertLevel, str] = {
    AlertLevel.INFO: "[INFO]",
    AlertLevel.WARNING: "[WARNING]",
    AlertLevel.CRITICAL: "[CRITICAL]",
}


class EmailAlert(AlertChannel):
    """Send alerts via SMTP email.

    Args:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port (587 for STARTTLS, 465 for SSL).
        username: SMTP authentication username.
        password: SMTP authentication password.
        from_address: Sender email address.
        to_addresses: List of recipient email addresses.
        use_tls: Whether to use STARTTLS (default True).
        timeout: Connection timeout in seconds.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_address: str,
        to_addresses: list[str],
        use_tls: bool = True,
        timeout: float = 10.0,
    ) -> None:
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._username = username
        self._password = password
        self._from_address = from_address
        self._to_addresses = to_addresses
        self._use_tls = use_tls
        self._timeout = timeout

    async def send(self, message: AlertMessage) -> bool:
        """Send an alert email (blocking SMTP, wrapped for async callers)."""
        try:
            self._send_sync(message)
            return True
        except Exception:
            logger.exception("Failed to send email alert")
            return False

    async def test_connection(self) -> bool:
        """Verify SMTP connectivity by logging in and quitting."""
        try:
            self._connect_and_quit()
            return True
        except Exception:
            logger.exception("Email connection test failed")
            return False

    def _send_sync(self, message: AlertMessage) -> None:
        """Synchronous email send."""
        prefix = _LEVEL_PREFIX.get(message.level, "")
        subject = f"{prefix} {message.title}"

        body_parts = [message.body]
        if message.metadata:
            body_parts.append("")
            body_parts.append("---")
            for k, v in message.metadata.items():
                body_parts.append(f"{k}: {v}")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self._from_address
        msg["To"] = ", ".join(self._to_addresses)
        msg.set_content("\n".join(body_parts))

        with smtplib.SMTP(
            self._smtp_host, self._smtp_port, timeout=self._timeout
        ) as server:
            if self._use_tls:
                server.starttls()
            server.login(self._username, self._password)
            server.send_message(msg)

    def _connect_and_quit(self) -> None:
        """Connect, authenticate, and disconnect."""
        with smtplib.SMTP(
            self._smtp_host, self._smtp_port, timeout=self._timeout
        ) as server:
            if self._use_tls:
                server.starttls()
            server.login(self._username, self._password)
