"""SQLite-backed gallery of person embeddings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person TEXT NOT NULL,
    vector BLOB NOT NULL,
    source TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person);
"""


class GalleryDB:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "GalleryDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def add(self, person: str, vector: np.ndarray, source: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO embeddings (person, vector, source) VALUES (?, ?, ?)",
            (person, np.asarray(vector, dtype=np.float32).tobytes(), source),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def count(self, person: str | None = None) -> int:
        if person is None:
            row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE person = ?", (person,)
            ).fetchone()
        return int(row[0])

    def people(self) -> list[tuple[str, int]]:
        rows = self._conn.execute(
            "SELECT person, COUNT(*) FROM embeddings GROUP BY person ORDER BY person"
        ).fetchall()
        return [(str(p), int(c)) for p, c in rows]

    def all_vectors(self) -> list[tuple[str, np.ndarray]]:
        rows = self._conn.execute("SELECT person, vector FROM embeddings").fetchall()
        return [(str(p), np.frombuffer(v, dtype=np.float32)) for p, v in rows]

    def max_similarity_to(self, vector: np.ndarray) -> tuple[str, float] | None:
        """Highest cosine similarity between `vector` and any stored embedding."""
        best: tuple[str, float] | None = None
        for person, vec in self.all_vectors():
            denom = float(np.linalg.norm(vec) * np.linalg.norm(vector))
            sim = 0.0 if denom == 0 else float(np.dot(vec, vector) / denom)
            if best is None or sim > best[1]:
                best = (person, sim)
        return best

    def remove_person(self, person: str) -> int:
        cur = self._conn.execute("DELETE FROM embeddings WHERE person = ?", (person,))
        self._conn.commit()
        return cur.rowcount

    def clear(self) -> int:
        cur = self._conn.execute("DELETE FROM embeddings")
        self._conn.commit()
        return cur.rowcount
