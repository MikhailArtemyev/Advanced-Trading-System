"""Tests for StateManager — JSON snapshot persistence."""

import json
import time
from pathlib import Path

import pytest

from src.engine.state_manager import SNAPSHOT_VERSION, StateManager


@pytest.fixture
def state_dir(tmp_path):
    """Temporary state directory."""
    return tmp_path / "state"


@pytest.fixture
def manager(state_dir):
    """StateManager using a temporary directory."""
    return StateManager(state_dir=str(state_dir), max_snapshots=5)


@pytest.fixture
def sample_state():
    """Sample engine state for testing."""
    return {
        "cash": 95_000.0,
        "positions": {
            "AAPL": {"quantity": 50, "avg_cost": 100.0},
        },
        "statistics": {
            "bars_processed": 120,
            "fills_processed": 3,
        },
    }


class TestSaveSnapshot:
    def test_creates_json_file(self, manager, sample_state):
        path = manager.save_snapshot(sample_state)
        assert path.exists()
        assert path.suffix == ".json"

    def test_file_contains_state(self, manager, sample_state):
        path = manager.save_snapshot(sample_state)
        with open(path) as f:
            data = json.load(f)
        assert data["cash"] == 95_000.0
        assert data["positions"]["AAPL"]["quantity"] == 50

    def test_adds_metadata_timestamp(self, manager, sample_state):
        path = manager.save_snapshot(sample_state)
        with open(path) as f:
            data = json.load(f)
        assert "_snapshot_timestamp" in data

    def test_adds_metadata_version(self, manager, sample_state):
        path = manager.save_snapshot(sample_state)
        with open(path) as f:
            data = json.load(f)
        assert data["_snapshot_version"] == SNAPSHOT_VERSION

    def test_returns_path_object(self, manager, sample_state):
        result = manager.save_snapshot(sample_state)
        assert isinstance(result, Path)

    def test_multiple_snapshots_have_unique_names(self, manager, sample_state):
        p1 = manager.save_snapshot(sample_state)
        time.sleep(0.01)  # ensure different timestamp
        p2 = manager.save_snapshot(sample_state)
        assert p1.name != p2.name


class TestLoadLatestSnapshot:
    def test_returns_none_when_empty(self, manager):
        assert manager.load_latest_snapshot() is None

    def test_returns_most_recent(self, manager):
        manager.save_snapshot({"version": 1})
        time.sleep(0.01)
        manager.save_snapshot({"version": 2})
        latest = manager.load_latest_snapshot()
        assert latest is not None
        assert latest["version"] == 2

    def test_round_trip_preserves_state(self, manager, sample_state):
        manager.save_snapshot(sample_state)
        loaded = manager.load_latest_snapshot()
        assert loaded is not None
        assert loaded["cash"] == sample_state["cash"]
        assert loaded["positions"] == sample_state["positions"]
        assert loaded["statistics"] == sample_state["statistics"]


class TestLoadSnapshot:
    def test_loads_specific_file(self, manager, sample_state):
        path = manager.save_snapshot(sample_state)
        loaded = manager.load_snapshot(path)
        assert loaded["cash"] == 95_000.0

    def test_accepts_string_path(self, manager, sample_state):
        path = manager.save_snapshot(sample_state)
        loaded = manager.load_snapshot(str(path))
        assert loaded["cash"] == 95_000.0

    def test_raises_on_missing_file(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load_snapshot("/nonexistent/snapshot.json")


class TestPruning:
    def test_prunes_oldest_beyond_limit(self, state_dir):
        mgr = StateManager(state_dir=str(state_dir), max_snapshots=3)
        paths = []
        for i in range(5):
            paths.append(mgr.save_snapshot({"idx": i}))
            time.sleep(0.01)
        # Only 3 should remain
        remaining = mgr._list_snapshots()
        assert len(remaining) == 3

    def test_keeps_newest_after_prune(self, state_dir):
        mgr = StateManager(state_dir=str(state_dir), max_snapshots=2)
        for i in range(4):
            mgr.save_snapshot({"idx": i})
            time.sleep(0.01)
        latest = mgr.load_latest_snapshot()
        assert latest is not None
        assert latest["idx"] == 3

    def test_no_prune_under_limit(self, manager, sample_state):
        for _ in range(3):
            manager.save_snapshot(sample_state)
            time.sleep(0.01)
        assert len(manager._list_snapshots()) == 3


class TestListSnapshots:
    def test_empty_list_when_no_snapshots(self, manager):
        assert manager.list_snapshots() == []

    def test_returns_metadata_dicts(self, manager, sample_state):
        manager.save_snapshot(sample_state)
        items = manager.list_snapshots()
        assert len(items) == 1
        assert "path" in items[0]
        assert "filename" in items[0]
        assert "size_bytes" in items[0]
        assert "modified" in items[0]

    def test_count_matches_saved(self, manager, sample_state):
        for _ in range(3):
            manager.save_snapshot(sample_state)
            time.sleep(0.01)
        assert len(manager.list_snapshots()) == 3


class TestClearSnapshots:
    def test_deletes_all(self, manager, sample_state):
        for _ in range(3):
            manager.save_snapshot(sample_state)
            time.sleep(0.01)
        deleted = manager.clear_snapshots()
        assert deleted == 3
        assert len(manager._list_snapshots()) == 0

    def test_returns_zero_when_empty(self, manager):
        assert manager.clear_snapshots() == 0

    def test_load_latest_returns_none_after_clear(self, manager, sample_state):
        manager.save_snapshot(sample_state)
        manager.clear_snapshots()
        assert manager.load_latest_snapshot() is None


class TestDirectoryCreation:
    def test_creates_state_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested" / "state"
        mgr = StateManager(state_dir=str(new_dir))
        assert new_dir.exists()
        mgr.save_snapshot({"test": True})
        assert len(mgr._list_snapshots()) == 1

    def test_state_dir_property(self, manager, state_dir):
        assert manager.state_dir == state_dir
