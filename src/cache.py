"""SQLite cache for TMDB JSON responses."""

import json
import sqlite3
import time
from pathlib import Path


class TMDBCache:
    """Store one JSON payload per request key, with the time it was fetched."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> tuple[dict, float] | None:
        """Return ``(payload, age_days)`` or None when the key is missing."""
        row = self._conn.execute(
            "SELECT payload, fetched_at FROM responses WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        age_days = (time.time() - row[1]) / 86400.0
        return payload, age_days

    def set(self, key: str, payload_dict: dict) -> None:
        """Insert or replace the payload for ``key`` and stamp it now."""
        self._conn.execute(
            """
            INSERT INTO responses (key, payload, fetched_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                payload = excluded.payload,
                fetched_at = excluded.fetched_at
            """,
            (key, json.dumps(payload_dict), time.time()),
        )
        self._conn.commit()
