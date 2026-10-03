"""SQLite persistence for live CCTV analytics and dashboard."""

from dataclasses import asdict
import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EventStore:
    """Thread-safe SQLite store for CCTV events, faces, and notifications."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA busy_timeout=5000;")
            self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY,
                camera TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL,
                updated_at REAL NOT NULL,
                status TEXT NOT NULL,
                primary_class TEXT NOT NULL,
                behaviors TEXT NOT NULL DEFAULT '[]',
                max_confidence REAL NOT NULL DEFAULT 0,
                clip_path TEXT,
                thumb_path TEXT,
                track_summaries TEXT NOT NULL DEFAULT '[]',
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_started ON events(started_at);

            CREATE TABLE IF NOT EXISTS identities(
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS faces(
                id INTEGER PRIMARY KEY,
                event_id INTEGER,
                camera TEXT NOT NULL,
                track_id INTEGER,
                captured_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                crop_path TEXT NOT NULL,
                context_path TEXT NOT NULL,
                embedding BLOB NOT NULL,
                quality REAL NOT NULL,
                det_score REAL NOT NULL,
                face_width REAL NOT NULL,
                identity_id INTEGER,
                cluster_id INTEGER,
                match_score REAL
            );
            CREATE INDEX IF NOT EXISTS idx_faces_identity ON faces(identity_id);
            CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);

            CREATE TABLE IF NOT EXISTS notifications(
                id INTEGER PRIMARY KEY,
                event_id INTEGER,
                sent_at REAL NOT NULL,
                channel TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT
            );
            """
        )
        self.conn.commit()

    # --- Events ---

    def create_event(
        self,
        camera: str,
        started_at: float,
        status: str = "open",
        primary_class: str = "person",
        behaviors: str = "[]",
        max_confidence: float = 0.0,
        clip_path: Optional[str] = None,
        thumb_path: Optional[str] = None,
        track_summaries: str = "[]",
        error: Optional[str] = None,
    ) -> int:
        now = time.time()
        with self.lock:
            cur = self.conn.execute(
                """
                INSERT INTO events (
                    camera, started_at, ended_at, updated_at, status, primary_class,
                    behaviors, max_confidence, clip_path, thumb_path, track_summaries, error
                ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera,
                    started_at,
                    now,
                    status,
                    primary_class,
                    behaviors,
                    max_confidence,
                    clip_path,
                    thumb_path,
                    track_summaries,
                    error,
                ),
            )
            self.conn.commit()
            return cur.lastrowid

    def update_event(self, event_id: int, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        keys = list(fields.keys())
        set_clause = ", ".join(f"{k} = ?" for k in keys)
        values = [fields[k] for k in keys] + [event_id]
        with self.lock:
            self.conn.execute(f"UPDATE events SET {set_clause} WHERE id = ?", values)
            self.conn.commit()

    def get_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_events(
        self,
        camera: Optional[str] = None,
        behavior: Optional[str] = None,
        identity_id: Optional[int] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        clauses = []
        params: List[Any] = []

        if camera:
            clauses.append("camera = ?")
            params.append(camera)
        if behavior:
            clauses.append("behaviors LIKE ?")
            params.append(f'%"{behavior}"%')
        if identity_id is not None:
            clauses.append("EXISTS (SELECT 1 FROM faces WHERE faces.event_id = events.id AND faces.identity_id = ?)")
            params.append(identity_id)
        if since is not None:
            clauses.append("started_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("started_at <= ?")
            params.append(until)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self.lock:
            cur = self.conn.execute(f"SELECT COUNT(*) as c FROM events {where}", params)
            total = cur.fetchone()["c"]

            query = f"SELECT * FROM events {where} ORDER BY started_at DESC LIMIT ? OFFSET ?"
            cur = self.conn.execute(query, params + [limit, offset])
            items = [dict(r) for r in cur.fetchall()]

            return total, items

    def events_updated_since(self, ts: float) -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT * FROM events WHERE updated_at > ? ORDER BY updated_at ASC",
                (ts,),
            )
            return [dict(r) for r in cur.fetchall()]

    # --- Faces ---

    def insert_face(
        self,
        event_id: Optional[int],
        camera: str,
        track_id: Optional[int],
        captured_at: float,
        crop_path: str,
        context_path: str,
        embedding: bytes,
        quality: float,
        det_score: float,
        face_width: float,
        identity_id: Optional[int] = None,
        cluster_id: Optional[int] = None,
        match_score: Optional[float] = None,
    ) -> int:
        now = time.time()
        with self.lock:
            cur = self.conn.execute(
                """
                INSERT INTO faces (
                    event_id, camera, track_id, captured_at, updated_at, crop_path,
                    context_path, embedding, quality, det_score, face_width,
                    identity_id, cluster_id, match_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    camera,
                    track_id,
                    captured_at,
                    now,
                    crop_path,
                    context_path,
                    embedding,
                    quality,
                    det_score,
                    face_width,
                    identity_id,
                    cluster_id,
                    match_score,
                ),
            )
            self.conn.commit()
            return cur.lastrowid

    def update_face(self, face_id: int, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        keys = list(fields.keys())
        set_clause = ", ".join(f"{k} = ?" for k in keys)
        values = [fields[k] for k in keys] + [face_id]
        with self.lock:
            self.conn.execute(f"UPDATE faces SET {set_clause} WHERE id = ?", values)
            self.conn.commit()

    def faces_for_event(self, event_id: int) -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(
                """
                SELECT faces.*, identities.name as identity_name
                FROM faces
                LEFT JOIN identities ON faces.identity_id = identities.id
                WHERE faces.event_id = ?
                ORDER BY faces.captured_at ASC
                """,
                (event_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def labeled_faces(self) -> List[Tuple[int, int, np.ndarray]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT id, identity_id, embedding FROM faces WHERE identity_id IS NOT NULL"
            )
            return [
                (
                    r["id"],
                    r["identity_id"],
                    np.frombuffer(r["embedding"], dtype=np.float32),
                )
                for r in cur.fetchall()
            ]

    def unknown_faces(self) -> List[Tuple[int, int, np.ndarray]]:
        with self.lock:
            cur = self.conn.execute(
                """
                SELECT id, cluster_id, embedding
                FROM faces
                WHERE identity_id IS NULL AND cluster_id IS NOT NULL
                """
            )
            return [
                (
                    r["id"],
                    r["cluster_id"],
                    np.frombuffer(r["embedding"], dtype=np.float32),
                )
                for r in cur.fetchall()
            ]

    def next_cluster_id(self) -> int:
        with self.lock:
            cur = self.conn.execute("SELECT COALESCE(MAX(cluster_id), 0) + 1 as n FROM faces")
            return cur.fetchone()["n"]

    # --- Identities ---

    def list_identities(self) -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(
                """
                SELECT i.id, i.name, i.created_at, COUNT(f.id) as face_count
                FROM identities i
                LEFT JOIN faces f ON i.id = f.identity_id
                GROUP BY i.id, i.name, i.created_at
                ORDER BY i.name ASC
                """
            )
            return [dict(r) for r in cur.fetchall()]

    def create_identity(self, name: str) -> int:
        now = time.time()
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO identities (name, created_at) VALUES (?, ?)",
                (name, now),
            )
            self.conn.commit()
            return cur.lastrowid

    def rename_identity(self, identity_id: int, name: str) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE identities SET name = ? WHERE id = ?",
                (name, identity_id),
            )
            self.conn.commit()

    def delete_identity(self, identity_id: int) -> None:
        with self.lock:
            cur = self.conn.execute("SELECT id FROM identities WHERE id = ?", (identity_id,))
            if not cur.fetchone():
                raise LookupError(f"Identity {identity_id} not found")

            # Assign all faces of this identity to one new unknown cluster
            cur_max = self.conn.execute("SELECT COALESCE(MAX(cluster_id), 0) + 1 as n FROM faces")
            next_cid = cur_max.fetchone()["n"]
            now = time.time()
            self.conn.execute(
                "UPDATE faces SET identity_id = NULL, cluster_id = ?, updated_at = ? WHERE identity_id = ?",
                (next_cid, now, identity_id),
            )
            self.conn.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
            self.conn.commit()

    def assign_cluster(self, cluster_id: int, identity_id: int) -> None:
        now = time.time()
        with self.lock:
            self.conn.execute(
                "UPDATE faces SET identity_id = ?, cluster_id = NULL, updated_at = ? WHERE cluster_id = ?",
                (identity_id, now, cluster_id),
            )
            self.conn.commit()

    def assign_face(self, face_id: int, identity_id: Optional[int]) -> None:
        now = time.time()
        with self.lock:
            if identity_id is None:
                cur_max = self.conn.execute("SELECT COALESCE(MAX(cluster_id), 0) + 1 as n FROM faces")
                next_cid = cur_max.fetchone()["n"]
                self.conn.execute(
                    "UPDATE faces SET identity_id = NULL, cluster_id = ?, updated_at = ? WHERE id = ?",
                    (next_cid, now, face_id),
                )
            else:
                self.conn.execute(
                    "UPDATE faces SET identity_id = ?, cluster_id = NULL, updated_at = ? WHERE id = ?",
                    (identity_id, now, face_id),
                )
            self.conn.commit()

    def delete_face(self, face_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute("SELECT * FROM faces WHERE id = ?", (face_id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            self.conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            self.conn.commit()
            return res

    def list_clusters(self) -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(
                """
                SELECT cluster_id, COUNT(*) as count, MAX(captured_at) as last_seen
                FROM faces
                WHERE identity_id IS NULL AND cluster_id IS NOT NULL
                GROUP BY cluster_id
                ORDER BY last_seen DESC
                """
            )
            return [dict(r) for r in cur.fetchall()]

    # --- Notifications ---

    def insert_notification(
        self,
        event_id: int,
        sent_at: float,
        channel: str,
        kind: str,
        title: str,
        body: str,
        status: str,
        error: Optional[str] = None,
    ) -> int:
        with self.lock:
            cur = self.conn.execute(
                """
                INSERT INTO notifications (
                    event_id, sent_at, channel, kind, title, body, status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, sent_at, channel, kind, title, body, status, error),
            )
            self.conn.commit()
            return cur.lastrowid

    def notifications_since(self, last_id: int, channel: str = "dashboard") -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT * FROM notifications WHERE id > ? AND channel = ? ORDER BY id ASC",
                (last_id, channel),
            )
            return [dict(r) for r in cur.fetchall()]
