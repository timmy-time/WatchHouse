"""Supervisor and worker thread orchestration for live camera analytics."""

import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional

from engine.detector import ClipDetector
from engine.faces import FaceEngine, FaceGallery
from engine.live.clipper import Finalizer
from engine.live.config import CameraConfig, LiveConfig, load_live_config
from engine.live.db import EventStore
from engine.live.events import EventManager
from engine.live.notify import Notifier
from engine.live.stream import CameraStream

logger = logging.getLogger(__name__)


class CameraWorker(threading.Thread):
    """Worker thread running RTSP ingest, YOLO ByteTrack, and behavior/face logic for one camera."""

    def __init__(
        self,
        cam: CameraConfig,
        cfg: LiveConfig,
        store: EventStore,
        notifier: Notifier,
        output_dir: str,
        face_engine: Optional[FaceEngine],
        gallery: FaceGallery,
    ):
        super().__init__(name=f"Worker-{cam.slug}", daemon=True)
        self.cam = cam
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.output_dir = output_dir
        self.face_engine = face_engine
        self.gallery = gallery

        self.stop_event = threading.Event()
        self.fps_analyzed = 0.0
        self.last_frame_at = 0.0

        self.stream = CameraStream(cam, cfg.recording, fps=cfg.analysis.fps)
        self.detector = ClipDetector(model_path=cfg.analysis.model, device=cam.gpu)
        self.finalizer = Finalizer(cam, cfg, store, output_dir)
        self.manager = EventManager(
            cam=cam,
            cfg=cfg,
            store=store,
            notifier=notifier,
            output_dir=output_dir,
            face_engine=face_engine,
            gallery=gallery,
            on_closed=self.finalizer.enqueue,
        )

    def run(self) -> None:
        logger.info(f"{self.cam.name}: Starting camera worker on GPU {self.cam.gpu}")
        self.stream.start()
        self.finalizer.start()

        frame_idx = 0
        fps_count = 0
        fps_start = time.time()

        while not self.stop_event.is_set():
            try:
                item = self.stream.get_frame(timeout=1.0)
                if item is None:
                    continue

                t, frame = item
                frame_h, frame_w = frame.shape[:2]

                dets = self.detector.track_frame(
                    frame,
                    frame_idx=frame_idx,
                    conf_threshold=self.cfg.analysis.confidence,
                )
                frame_idx += 1

                self.manager.process(
                    frame=frame,
                    dets=dets,
                    now=t,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )

                self.last_frame_at = t
                fps_count += 1
                now = time.time()
                if now - fps_start >= 5.0:
                    self.fps_analyzed = round(fps_count / (now - fps_start), 1)
                    fps_count = 0
                    fps_start = now

            except Exception as exc:
                logger.error(f"{self.cam.name}: Worker exception: {exc}\n{traceback.format_exc()}")
                self.stop_event.wait(5.0)
                if self.stop_event.is_set():
                    break
                # Rebuild stream and detector while keeping manager event state
                logger.info(f"{self.cam.name}: Rebuilding stream and detector...")
                try:
                    self.stream.stop()
                    self.stream = CameraStream(self.cam, self.cfg.recording, fps=self.cfg.analysis.fps)
                    self.stream.start()
                    self.detector = ClipDetector(model_path=self.cfg.analysis.model, device=self.cam.gpu)
                except Exception as rebuild_exc:
                    logger.error(f"{self.cam.name}: Rebuild failed: {rebuild_exc}")

        logger.info(f"{self.cam.name}: Worker shutting down")
        self.stream.stop()
        self.finalizer.stop()

    def stop(self) -> None:
        self.stop_event.set()


def run_live(config_path: str, output_dir: str) -> int:
    """Run live multi-camera analytics daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    )
    logger.info(f"Loading live configuration from {config_path}")
    cfg = load_live_config(config_path)

    active_cams = [c for c in cfg.cameras if c.url]
    for c in cfg.cameras:
        if not c.url:
            logger.info(f"Camera '{c.name}': url not configured, skipping")

    if not active_cams:
        print("No cameras configured (set CAM_*_URL)")
        return 1

    live_dir = os.path.join(output_dir, "live")
    os.makedirs(live_dir, exist_ok=True)
    db_path = os.path.join(live_dir, "events.db")

    logger.info(f"Initializing EventStore at {db_path}")
    store = EventStore(db_path)
    notifier = Notifier(cfg.notifications, store, output_dir)
    gallery = FaceGallery(
        store,
        match_threshold=cfg.faces.match_threshold,
        cluster_threshold=cfg.faces.cluster_threshold,
    )

    model_dir = os.environ.get("FACE_MODEL_DIR", "/opt/models")
    workers: List[CameraWorker] = []

    for cam in active_cams:
        face_engine = None
        if cfg.faces.enabled:
            try:
                face_engine = FaceEngine(
                    model_dir=model_dir,
                    min_face_px=cfg.faces.min_face_px,
                    min_det_score=cfg.faces.min_det_score,
                )
            except Exception as exc:
                logger.warning(f"{cam.name}: Could not load FaceEngine ({exc}); face recognition disabled")

        worker = CameraWorker(
            cam=cam,
            cfg=cfg,
            store=store,
            notifier=notifier,
            output_dir=output_dir,
            face_engine=face_engine,
            gallery=gallery,
        )
        workers.append(worker)
        worker.start()

    stop_event = threading.Event()

    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}; initiating graceful shutdown...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    status_path = os.path.join(live_dir, "status.json")

    logger.info("Live analytics running. Press Ctrl+C to stop.")
    try:
        while not stop_event.is_set():
            status_data = {
                "updated_at": time.time(),
                "cameras": {
                    w.cam.name: {
                        "slug": w.cam.slug,
                        **w.stream.status(),
                        "fps_analyzed": w.fps_analyzed,
                        "open_event_id": w.manager.open_event_id,
                        "last_frame_at": w.last_frame_at,
                    }
                    for w in workers
                },
            }

            tmp_status = status_path + ".tmp"
            try:
                with open(tmp_status, "w", encoding="utf-8") as f:
                    json.dump(status_data, f, indent=2)
                os.replace(tmp_status, status_path)
            except Exception as exc:
                logger.warning(f"Failed to write status.json: {exc}")

            stop_event.wait(2.0)
    finally:
        logger.info("Stopping all camera workers...")
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=5.0)
        notifier.stop()
        logger.info("Live analytics stopped cleanly")

    return 0
