"""Notification dispatcher for live CCTV events (ntfy, webhook, dashboard)."""

import os
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

import requests

from engine.live.config import NotificationsConfig
from engine.live.db import EventStore


class Notifier:
    """Dispatches notifications asynchronously across dashboard, ntfy, and webhooks."""

    def __init__(self, cfg: NotificationsConfig, store: EventStore, output_dir: str):
        self.cfg = cfg
        self.store = store
        self.output_dir = output_dir

        self.last_sent: Dict[Tuple[str, str], float] = {}
        self.cooldown_lock = threading.Lock()

        self.queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def notify(
        self,
        event_id: int,
        camera: str,
        kind: str,
        title: str,
        body: str,
        image_rel: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Enqueue notification if cooldown allows."""
        now = time.time()
        key = (camera, kind)
        with self.cooldown_lock:
            last = self.last_sent.get(key, 0.0)
            if (now - last) < self.cfg.cooldown_seconds:
                return
            self.last_sent[key] = now

        self.queue.put((event_id, camera, kind, title, body, image_rel, extra, now))

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self.queue.task_done()
                break

            event_id, camera, kind, title, body, image_rel, extra, sent_at = item
            try:
                self._dispatch(event_id, camera, kind, title, body, image_rel, extra, sent_at)
            except Exception as exc:
                # Top-level catch to avoid crashing the notifier thread
                pass
            finally:
                self.queue.task_done()

    def _dispatch(
        self,
        event_id: int,
        camera: str,
        kind: str,
        title: str,
        body: str,
        image_rel: Optional[str],
        extra: Optional[Dict[str, Any]],
        sent_at: float,
    ) -> None:
        # 1. Dashboard notification row
        self.store.insert_notification(
            event_id=event_id,
            sent_at=sent_at,
            channel="dashboard",
            kind=kind,
            title=title,
            body=body,
            status="sent",
        )

        # 2. ntfy push notification
        if self.cfg.ntfy.topic:
            ntfy_url = f"{self.cfg.ntfy.url.rstrip('/')}/{self.cfg.ntfy.topic}"
            safe_title = title.encode("latin-1", "replace").decode("latin-1")
            safe_body = body.encode("latin-1", "replace").decode("latin-1")
            tag = kind.split(":")[0]
            priority = (
                "high"
                if kind in ("person", "approaching", "unknown_face") or kind.startswith("identity:")
                else "default"
            )

            headers = {
                "Title": safe_title,
                "Message": safe_body,
                "Click": f"{self.cfg.dashboard_url}/#/events/{event_id}",
                "Tags": tag,
                "Priority": priority,
            }

            img_bytes: Optional[bytes] = None
            if image_rel:
                full_img_path = os.path.join(self.output_dir, image_rel)
                if os.path.exists(full_img_path):
                    try:
                        with open(full_img_path, "rb") as f:
                            img_bytes = f.read()
                    except Exception:
                        img_bytes = None

            if img_bytes is not None:
                headers["Filename"] = "event.jpg"
                data = img_bytes
            else:
                data = safe_body.encode("utf-8")

            if self.cfg.ntfy.token:
                headers["Authorization"] = f"Bearer {self.cfg.ntfy.token}"

            try:
                resp = requests.put(ntfy_url, data=data, headers=headers, timeout=10)
                resp.raise_for_status()
                self.store.insert_notification(
                    event_id=event_id,
                    sent_at=time.time(),
                    channel="ntfy",
                    kind=kind,
                    title=title,
                    body=body,
                    status="sent",
                )
            except Exception as exc:
                self.store.insert_notification(
                    event_id=event_id,
                    sent_at=time.time(),
                    channel="ntfy",
                    kind=kind,
                    title=title,
                    body=body,
                    status="failed",
                    error=str(exc),
                )

        # 3. Generic Webhook
        if self.cfg.webhook.url:
            payload = {
                "type": kind,
                "event_id": event_id,
                "camera": camera,
                "title": title,
                "body": body,
                "behaviors": extra.get("behaviors", []) if extra else [],
                "identity": extra.get("identity") if extra else None,
                "started_at": extra.get("started_at", sent_at) if extra else sent_at,
                "thumbnail_url": f"{self.cfg.dashboard_url}/media/output/{image_rel}" if image_rel else None,
                "event_url": f"{self.cfg.dashboard_url}/#/events/{event_id}",
            }
            try:
                resp = requests.post(self.cfg.webhook.url, json=payload, timeout=10)
                resp.raise_for_status()
                self.store.insert_notification(
                    event_id=event_id,
                    sent_at=time.time(),
                    channel="webhook",
                    kind=kind,
                    title=title,
                    body=body,
                    status="sent",
                )
            except Exception as exc:
                self.store.insert_notification(
                    event_id=event_id,
                    sent_at=time.time(),
                    channel="webhook",
                    kind=kind,
                    title=title,
                    body=body,
                    status="failed",
                    error=str(exc),
                )

    def stop(self) -> None:
        """Signal worker thread to stop and wait for completion."""
        self.stop_event.set()
        self.queue.put(None)
        self.worker_thread.join(timeout=3.0)
