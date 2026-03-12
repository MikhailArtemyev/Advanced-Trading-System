"""StateManager — JSON-based state persistence for paper trading.

Saves and loads engine snapshots (portfolio positions, cash, orders,
statistics) to disk as timestamped JSON files. Supports automatic
pruning to limit disk usage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = "1.0"


class StateManager:
    """Manages engine state persistence via JSON snapshots.

    Args:
        state_dir: Directory for snapshot files. Created if missing.
        max_snapshots: Maximum snapshots to keep. Oldest are pruned.
    """

    def __init__(
        self,
        state_dir: str = "state/",
        max_snapshots: int = 10,
    ) -> None:
        self._state_dir = Path(state_dir)
        self._max_snapshots = max_snapshots
        self._state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_dir(self) -> Path:
        """Return the state directory path."""
        return self._state_dir

    def save_snapshot(self, state: dict[str, Any]) -> Path:
        """Save a state snapshot to disk.

        Adds metadata (_snapshot_timestamp, _snapshot_version) and
        writes JSON. Prunes oldest snapshots if over the limit.

        Args:
            state: Engine state dict (positions, cash, orders, etc.)

        Returns:
            Path to the saved snapshot file.
        """
        timestamp = datetime.now()
        filename = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
        filepath = self._state_dir / filename

        snapshot = {
            "_snapshot_timestamp": timestamp.isoformat(),
            "_snapshot_version": SNAPSHOT_VERSION,
            **state,
        }

        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        logger.info("Saved snapshot: %s", filepath)
        self._prune_snapshots()
        return filepath

    def load_latest_snapshot(self) -> dict[str, Any] | None:
        """Load the most recent snapshot.

        Returns:
            State dict or None if no snapshots exist.
        """
        snapshots = self._list_snapshots()
        if not snapshots:
            return None
        return self.load_snapshot(snapshots[-1])

    def load_snapshot(self, filepath: str | Path) -> dict[str, Any]:
        """Load a specific snapshot by file path.

        Args:
            filepath: Path to the snapshot JSON file.

        Returns:
            State dict from the snapshot.

        Raises:
            FileNotFoundError: If the snapshot file doesn't exist.
        """
        path = Path(filepath)
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        logger.info("Loaded snapshot: %s", path)
        return data

    def list_snapshots(self) -> list[dict[str, Any]]:
        """Return metadata for all available snapshots.

        Returns:
            List of dicts with path, filename, size_bytes, modified.
        """
        result = []
        for path in self._list_snapshots():
            stat = path.stat()
            result.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )
        return result

    def clear_snapshots(self) -> int:
        """Delete all snapshots.

        Returns:
            Number of snapshots deleted.
        """
        snapshots = self._list_snapshots()
        count = 0
        for path in snapshots:
            path.unlink()
            count += 1
        logger.info("Cleared %d snapshots", count)
        return count

    def _list_snapshots(self) -> list[Path]:
        """List snapshot files sorted by name (oldest first)."""
        if not self._state_dir.exists():
            return []
        return sorted(self._state_dir.glob("snapshot_*.json"))

    def _prune_snapshots(self) -> None:
        """Remove oldest snapshots beyond the max_snapshots limit."""
        snapshots = self._list_snapshots()
        while len(snapshots) > self._max_snapshots:
            oldest = snapshots.pop(0)
            oldest.unlink()
            logger.info("Pruned old snapshot: %s", oldest)
