"""Remote inference backend: run YOLO (and ByteTrack) on another machine.

The camera server keeps RTSP decode, event logic, recording and the dashboard; only
the per-frame forward pass leaves the box. Frames travel as JPEG over HTTP with a
logical session per camera, so the remote tracker keeps one ByteTrack state per
camera and returned track ids stay meaningful.

The client mirrors the ``ClipDetector`` surface used by the live path
(``track_frame`` / ``detect_frame``) so a camera can switch backends by config.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import httpx

from engine.detector import Detection, TrackDetection

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
DEFAULT_JPEG_QUALITY = 80
_ERROR_LOG_INTERVAL = 20.0


class RemoteDetector:
    """``ClipDetector``-compatible client for an inference server on another host."""

    def __init__(
        self,
        url: str,
        session: str,
        model_path: str = "yolov8n.pt",
        imgsz: int = 640,
        tracker_config: str = "bytetrack.yaml",
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not url:
            raise ValueError("remote inference requires a url")
        self.url = url.rstrip("/")
        self.session = session
        self.model_path = model_path
        self.imgsz = imgsz
        self.tracker_config = tracker_config
        self.jpeg_quality = int(jpeg_quality)
        self.timeout = float(timeout)

        self._client = httpx.Client(
            base_url=self.url,
            timeout=httpx.Timeout(self.timeout, connect=min(3.0, self.timeout)),
        )
        self._stats_lock = threading.Lock()
        self.calls = 0
        self.errors = 0
        self.last_ms = 0.0
        self.avg_ms = 0.0
        self.last_error = ""
        self._last_error_log = 0.0

    # --- helpers ---

    def _encode(self, frame) -> str:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise ValueError("failed to encode frame as JPEG")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _record(self, ok: bool, ms: float, error: str = "") -> None:
        with self._stats_lock:
            if ok:
                self.calls += 1
                self.last_ms = ms
                self.avg_ms = ms if self.avg_ms == 0 else (self.avg_ms * 0.8 + ms * 0.2)
            else:
                self.errors += 1
                self.last_error = error

    def _log_error(self, message: str) -> None:
        now = time.time()
        if now - self._last_error_log >= _ERROR_LOG_INTERVAL:
            self._last_error_log = now
            logger.warning("remote inference (%s): %s", self.session, message)

    def _post(self, path: str, frame, conf_threshold: float, frame_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if frame is None:
            return None
        try:
            jpeg_b64 = self._encode(frame)
        except Exception as exc:
            self._record(False, 0.0, str(exc))
            self._log_error(f"encode failed: {exc}")
            return None

        payload: Dict[str, Any] = {
            "session": self.session,
            "jpeg_b64": jpeg_b64,
            "conf": float(conf_threshold),
            "frame_idx": int(frame_idx or 0),
            "model": self.model_path,
            "imgsz": self.imgsz,
            "tracker": self.tracker_config,
        }
        started = time.perf_counter()
        try:
            response = self._client.post(path, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            body = response.json()
        except Exception as exc:
            self._record(False, 0.0, str(exc))
            self._log_error(f"inference request failed: {exc}")
            return None

        self._record(True, (time.perf_counter() - started) * 1000.0)
        return body

    # --- ClipDetector-compatible API ---

    def track_frame(self, frame, frame_idx: int, conf_threshold: float = 0.30) -> List[TrackDetection]:
        """Tracked detections for one frame (track ids come from the remote tracker)."""
        body = self._post("/track", frame, conf_threshold, frame_idx)
        if body is None:
            return []
        out: List[TrackDetection] = []
        for item in body.get("detections", []):
            try:
                x1, y1, x2, y2 = (float(v) for v in item["bbox_xyxy"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(
                TrackDetection(
                    track_id=int(item.get("track_id", -1)),
                    class_name=str(item.get("class_name", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_xywh=((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1),
                    frame_idx=int(item.get("frame_idx", frame_idx)),
                )
            )
        return out

    def detect_frame(self, frame, conf_threshold: float = 0.35) -> List[Detection]:
        """Stateless detections for one frame."""
        body = self._post("/detect", frame, conf_threshold)
        if body is None:
            return []
        out: List[Detection] = []
        for item in body.get("detections", []):
            try:
                x1, y1, x2, y2 = (float(v) for v in item["bbox_xyxy"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(
                Detection(
                    class_name=str(item.get("class_name", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_xywh=((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1),
                )
            )
        return out

    def reset(self) -> bool:
        """Drop the remote tracker state (call when the local stream restarts)."""
        try:
            response = self._client.post("/reset", json={"session": self.session})
            return response.status_code == 200
        except Exception as exc:
            self._log_error(f"reset failed: {exc}")
            return False

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                "backend": "remote",
                "url": self.url,
                "session": self.session,
                "calls": self.calls,
                "errors": self.errors,
                "last_ms": round(self.last_ms, 1),
                "avg_ms": round(self.avg_ms, 1),
                "last_error": self.last_error,
            }
