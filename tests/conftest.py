"""Shared pytest fixtures."""

import pytest

from src.events.queue import EventQueue


@pytest.fixture
def event_queue() -> EventQueue:
    return EventQueue()
