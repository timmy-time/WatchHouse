"""Inference server: serve YOLO detection/tracking to camera servers on other hosts.

Runs where the GPU is (e.g. Windows/WSL2 with an NVIDIA card), while the camera
server keeps decoding and event logic. One detector (and therefore one ByteTrack
state) is kept per session, so track ids stay consistent per camera.

    python main.py infer-server --host 0.0.0.0 --port 8099 --device 0
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FrameRequest(BaseModel):
    session: str = Field(..., description="Logical camera; tracker state is kept per session")
    jpeg_b64: str
    conf: float = 0.30
    frame_idx: int = 0
    model: str = "yolov8n.pt"
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"


class ResetRequest(BaseModel):
    session: str


def _default_factory(model: str, imgsz: int, tracker: str, device: str):
    from engine.detector import ClipDetector

    return ClipDetector(model_path=model, device=device, imgsz=imgsz, tracker_config=tracker)


class InferenceSessions:
    """Lazily built, thread-safe detectors — one per session/model/imgsz/tracker key."""

    def __init__(
        self,
        device: str = "0",
        factory: Optional[Callable[[str, int, str, str], Any]] = None,
        max_sessions: int = 8,
    ):
        self.device = str(device)
        self.max_sessions = max(1, int(max_sessions))
        self._factory = factory or _default_factory
        self._sessions: "Dict[Tuple[str, str, int, str], Any]" = {}
        self._locks: "Dict[Tuple[str, str, int, str], threading.Lock]" = {}
        self._guard = threading.RLock()

    @staticmethod
    def _key(session: str, model: str, imgsz: int, tracker: str) -> Tuple[str, str, int, str]:
        return (session, model, int(imgsz), tracker)

    def acquire(self, session: str, model: str, imgsz: int, tracker: str):
        """Return (detector, lock) for the session, building it on first use."""
        key = self._key(session, model, imgsz, tracker)
        with self._guard:
            detector = self._sessions.get(key)
            if detector is None:
                while len(self._sessions) >= self.max_sessions:
                    evicted = next(iter(self._sessions))
                    self._sessions.pop(evicted, None)
                    self._locks.pop(evicted, None)
                    logger.info("inference session evicted: %s", evicted[0])
                started = time.time()
                detector = self._factory(model, imgsz, tracker, self.device)
                self._sessions[key] = detector
                self._locks[key] = threading.Lock()
                logger.info(
                    "inference session ready: %s (%s @ %dpx, %.1fs)",
                    session, model, imgsz, time.time() - started,
                )
            return detector, self._locks[key]

    def drop(self, session: str) -> int:
        with self._guard:
            keys = [k for k in self._sessions if k[0] == session]
            for key in keys:
                self._sessions.pop(key, None)
                self._locks.pop(key, None)
            return len(keys)

    def count(self) -> int:
        with self._guard:
            return len(self._sessions)

    def sessions(self) -> List[str]:
        with self._guard:
            return sorted({k[0] for k in self._sessions})


def _decode_frame(jpeg_b64: str):
    try:
        raw = base64.b64decode(jpeg_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="jpeg_b64 is not valid base64")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="could not decode JPEG frame")
    return frame


def create_infer_app(
    device: str = "0",
    factory: Optional[Callable[[str, int, str, str], Any]] = None,
    max_sessions: int = 8,
) -> FastAPI:
    """Build the inference service. ``factory`` is injectable for tests."""
    sessions = InferenceSessions(device=device, factory=factory, max_sessions=max_sessions)
    app = FastAPI(title="CCTV Inference Server")
    app.state.sessions = sessions

    @app.get("/")
    async def index():
        return {
            "status": "online",
            "service": "WatchHouse Inference Worker",
            "device": sessions.device,
            "sessions": sessions.count(),
            "endpoints": {
                "health": "GET /health",
                "track": "POST /track",
                "detect": "POST /detect",
                "reset": "POST /reset",
            },
        }

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "device": sessions.device,
            "sessions": sessions.count(),
            "session_names": sessions.sessions(),
        }

    @app.post("/track")
    async def track(request: FrameRequest):
        frame = _decode_frame(request.jpeg_b64)
        detector, lock = sessions.acquire(request.session, request.model, request.imgsz, request.tracker)
        started = time.perf_counter()
        with lock:
            detections = detector.track_frame(
                frame, frame_idx=request.frame_idx, conf_threshold=request.conf
            )
        return {
            "detections": [
                {
                    "track_id": int(d.track_id),
                    "class_name": d.class_name,
                    "confidence": float(d.confidence),
                    "bbox_xyxy": [float(v) for v in d.bbox_xyxy],
                    "frame_idx": int(d.frame_idx),
                }
                for d in detections
            ],
            "infer_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }

    @app.post("/detect")
    async def detect(request: FrameRequest):
        frame = _decode_frame(request.jpeg_b64)
        detector, lock = sessions.acquire(request.session, request.model, request.imgsz, request.tracker)
        started = time.perf_counter()
        with lock:
            detections = detector.detect_frame(frame, conf_threshold=request.conf)
        return {
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": float(d.confidence),
                    "bbox_xyxy": [float(v) for v in d.bbox_xyxy],
                }
                for d in detections
            ],
            "infer_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }

    @app.post("/reset")
    async def reset(request: ResetRequest):
        dropped = sessions.drop(request.session)
        logger.info("inference session reset: %s (%d)", request.session, dropped)
        return {"ok": True, "dropped": dropped}

    return app
