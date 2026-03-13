"""Tests for AlpacaBroker — Alpaca Trade API adapter.

All tests use mocked HTTP responses (aiohttp). No real API calls.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.broker.alpaca_broker import (
    _STATUS_MAP,
    ALPACA_LIVE_URL,
    ALPACA_PAPER_URL,
    AlpacaBroker,
)
from src.broker.base_broker import BrokerOrder, OrderStatus

# ── Helpers ──────────────────────────────────────────────────────────


def _mock_response(status: int = 200, json_data: dict | list | None = None):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=json.dumps(json_data or {}))
    # Support async context manager
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_broker(api_key="test_key", api_secret="test_secret", paper=True):
    """Create an AlpacaBroker instance."""
    return AlpacaBroker(
        api_key=api_key,
        api_secret=api_secret,
        paper_mode=paper,
    )


def _make_order(
    order_id="ord-001",
    symbol="AAPL",
    side="BUY",
    order_type="MARKET",
    quantity=10,
):
    """Create a BrokerOrder."""
    return BrokerOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestInit:
    def test_default_paper_url(self):
        broker = _make_broker(paper=True)
        assert ALPACA_PAPER_URL in broker._base_url

    def test_live_url(self):
        broker = _make_broker(paper=False)
        assert ALPACA_LIVE_URL in broker._base_url

    def test_custom_base_url(self):
        broker = AlpacaBroker(
            api_key="k", api_secret="s", base_url="https://custom.api.com/"
        )
        assert broker._base_url == "https://custom.api.com"

    def test_not_connected_initially(self):
        broker = _make_broker()
        assert broker.connected is False

    def test_paper_mode_property(self):
        broker = _make_broker(paper=True)
        assert broker.paper_mode is True


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self):
        broker = _make_broker()
        account_data = {
            "account_number": "PA123",
            "equity": "100000.00",
            "status": "ACTIVE",
        }
        mock_resp = _mock_response(200, account_data)

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock_cls.return_value = mock_session

            await broker.connect()
            assert broker.connected is True

    @pytest.mark.asyncio
    async def test_connect_invalid_credentials(self):
        broker = _make_broker()
        mock_resp = _mock_response(401, {"message": "unauthorized"})

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock_cls.return_value = mock_session

            with pytest.raises(ConnectionError, match="Invalid Alpaca API credentials"):
                await broker.connect()

    @pytest.mark.asyncio
    async def test_connect_forbidden(self):
        broker = _make_broker()
        mock_resp = _mock_response(403, {"message": "forbidden"})

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock_cls.return_value = mock_session

            with pytest.raises(ConnectionError, match="forbidden"):
                await broker.connect()


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()
        broker._session.closed = False
        broker._session.close = AsyncMock()

        await broker.disconnect()
        assert broker.connected is False


class TestSubmitOrder:
    @pytest.mark.asyncio
    async def test_submit_market_order(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        alpaca_response = {
            "id": "alpaca-123",
            "status": "accepted",
            "symbol": "AAPL",
        }
        mock_resp = _mock_response(200, alpaca_response)
        broker._session.post = MagicMock(return_value=mock_resp)

        order = _make_order()
        result = await broker.submit_order(order)

        assert result.broker_order_id == "alpaca-123"
        assert result.status == OrderStatus.ACCEPTED
        assert result.submitted_at is not None

    @pytest.mark.asyncio
    async def test_submit_limit_order(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        alpaca_response = {"id": "alpaca-456", "status": "new"}
        mock_resp = _mock_response(200, alpaca_response)
        broker._session.post = MagicMock(return_value=mock_resp)

        order = _make_order(order_type="LIMIT")
        order.limit_price = 150.0
        result = await broker.submit_order(order)

        assert result.broker_order_id == "alpaca-456"
        assert result.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_submit_rejected_forbidden(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        mock_resp = _mock_response(403, {"message": "insufficient buying power"})
        broker._session.post = MagicMock(return_value=mock_resp)

        order = _make_order()
        result = await broker.submit_order(order)

        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_submit_rejected_unprocessable(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        mock_resp = _mock_response(422, {"message": "invalid symbol"})
        broker._session.post = MagicMock(return_value=mock_resp)

        order = _make_order()
        result = await broker.submit_order(order)

        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_submit_not_connected(self):
        broker = _make_broker()
        order = _make_order()

        with pytest.raises(ConnectionError, match="Not connected"):
            await broker.submit_order(order)

    @pytest.mark.asyncio
    async def test_order_tracked_after_submit(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        mock_resp = _mock_response(200, {"id": "alpaca-789", "status": "new"})
        broker._session.post = MagicMock(return_value=mock_resp)

        order = _make_order(order_id="my-order-1")
        await broker.submit_order(order)

        assert "my-order-1" in broker._orders
        assert broker._order_map["my-order-1"] == "alpaca-789"


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_success(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        # Pre-register order
        order = _make_order(order_id="ord-cancel")
        order.broker_order_id = "alpaca-cancel"
        broker._orders["ord-cancel"] = order
        broker._order_map["ord-cancel"] = "alpaca-cancel"

        mock_resp = _mock_response(204)
        broker._session.delete = MagicMock(return_value=mock_resp)

        result = await broker.cancel_order("ord-cancel")
        assert result.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_unknown_order(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        with pytest.raises(ValueError, match="Unknown order"):
            await broker.cancel_order("nonexistent")

    @pytest.mark.asyncio
    async def test_cancel_not_connected(self):
        broker = _make_broker()

        with pytest.raises(ConnectionError):
            await broker.cancel_order("ord-1")


class TestGetOrderStatus:
    @pytest.mark.asyncio
    async def test_status_filled(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        order = _make_order(order_id="ord-status")
        order.broker_order_id = "alpaca-status"
        broker._orders["ord-status"] = order
        broker._order_map["ord-status"] = "alpaca-status"

        alpaca_data = {
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "152.50",
            "filled_at": "2025-06-01T14:30:00Z",
        }
        mock_resp = _mock_response(200, alpaca_data)
        broker._session.get = MagicMock(return_value=mock_resp)

        result = await broker.get_order_status("ord-status")
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 10
        assert result.filled_avg_price == 152.50
        assert result.filled_at is not None

    @pytest.mark.asyncio
    async def test_status_partially_filled(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        order = _make_order(order_id="ord-partial", quantity=100)
        order.broker_order_id = "alpaca-partial"
        broker._orders["ord-partial"] = order
        broker._order_map["ord-partial"] = "alpaca-partial"

        alpaca_data = {
            "status": "partially_filled",
            "filled_qty": "30",
            "filled_avg_price": "151.00",
        }
        mock_resp = _mock_response(200, alpaca_data)
        broker._session.get = MagicMock(return_value=mock_resp)

        result = await broker.get_order_status("ord-partial")
        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert result.filled_quantity == 30

    @pytest.mark.asyncio
    async def test_status_unknown_order(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        with pytest.raises(ValueError, match="Unknown order"):
            await broker.get_order_status("nope")


class TestGetPositions:
    @pytest.mark.asyncio
    async def test_positions_returned(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        alpaca_positions = [
            {
                "symbol": "AAPL",
                "qty": "50",
                "avg_entry_price": "150.00",
                "market_value": "7600.00",
            },
            {
                "symbol": "MSFT",
                "qty": "30",
                "avg_entry_price": "380.00",
                "market_value": "11700.00",
            },
        ]
        mock_resp = _mock_response(200, alpaca_positions)
        broker._session.get = MagicMock(return_value=mock_resp)

        positions = await broker.get_positions()
        assert "AAPL" in positions
        assert positions["AAPL"]["quantity"] == 50.0
        assert positions["AAPL"]["avg_cost"] == 150.0
        assert "MSFT" in positions

    @pytest.mark.asyncio
    async def test_empty_positions(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        mock_resp = _mock_response(200, [])
        broker._session.get = MagicMock(return_value=mock_resp)

        positions = await broker.get_positions()
        assert positions == {}

    @pytest.mark.asyncio
    async def test_positions_not_connected(self):
        broker = _make_broker()

        with pytest.raises(ConnectionError):
            await broker.get_positions()


class TestGetAccountInfo:
    @pytest.mark.asyncio
    async def test_account_info(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        account_data = {
            "cash": "50000.00",
            "equity": "100000.00",
            "buying_power": "150000.00",
        }
        mock_resp = _mock_response(200, account_data)
        broker._session.get = MagicMock(return_value=mock_resp)

        info = await broker.get_account_info()
        assert info["cash"] == 50000.0
        assert info["equity"] == 100000.0
        assert info["buying_power"] == 150000.0

    @pytest.mark.asyncio
    async def test_account_info_failure(self):
        broker = _make_broker()
        broker._connected = True
        broker._session = AsyncMock()

        mock_resp = _mock_response(500)
        broker._session.get = MagicMock(return_value=mock_resp)

        info = await broker.get_account_info()
        assert info["cash"] == 0.0
        assert info["equity"] == 0.0


class TestStatusMapping:
    def test_all_alpaca_statuses_mapped(self):
        alpaca_statuses = [
            "new",
            "accepted",
            "pending_new",
            "partially_filled",
            "filled",
            "done_for_day",
            "canceled",
            "expired",
            "replaced",
            "rejected",
            "suspended",
        ]
        for status in alpaca_statuses:
            assert status in _STATUS_MAP
            assert isinstance(_STATUS_MAP[status], OrderStatus)

    def test_filled_maps_to_filled(self):
        assert _STATUS_MAP["filled"] == OrderStatus.FILLED

    def test_canceled_maps_to_cancelled(self):
        assert _STATUS_MAP["canceled"] == OrderStatus.CANCELLED

    def test_rejected_maps_to_rejected(self):
        assert _STATUS_MAP["rejected"] == OrderStatus.REJECTED


class TestOrderTypeMapping:
    def test_market(self):
        assert AlpacaBroker._map_order_type("MARKET") == "market"

    def test_limit(self):
        assert AlpacaBroker._map_order_type("LIMIT") == "limit"

    def test_stop(self):
        assert AlpacaBroker._map_order_type("STOP") == "stop"

    def test_unknown_defaults_to_market(self):
        assert AlpacaBroker._map_order_type("UNKNOWN") == "market"


class TestHeaders:
    def test_headers_contain_keys(self):
        broker = _make_broker(api_key="my_key", api_secret="my_secret")
        headers = broker._build_headers()
        assert headers["APCA-API-KEY-ID"] == "my_key"
        assert headers["APCA-API-SECRET-KEY"] == "my_secret"
        assert headers["Content-Type"] == "application/json"
