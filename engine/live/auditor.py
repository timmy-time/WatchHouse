"""Secondary high-resolution auditor performing background sweeps on finalized event clips."""

import json
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from engine.detector import ClipDetector
from engine.live.db import EventStore

logger = logging.getLogger(__name__)

AUDITOR_CLASSES: Set[str] = {
    "person",
    "dog",
    "cat",
    "bear",
    "horse",
    "cow",
    "sheep",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


def _extract_clip_keyframes(clip_path: str, num_frames: int = 4) -> List[np.ndarray]:
    """Extract evenly spaced keyframes from an MP4 video clip."""
    if not os.path.exists(clip_path):
        return []

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, total_frames // (num_frames + 1))
    frames = []

    for i in range(1, num_frames + 1):
        target_idx = min(total_frames - 1, i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    return frames


class EventAuditor:
    """Asynchronous secondary auditor running high-resolution model sweeps on finalized events."""

    def __init__(
        self,
        store: EventStore,
        output_dir: str,
        model_name: str = "yolov8n.pt",
        device: int = 1,
        conf_threshold: float = 0.25,
        idle_unload_seconds: float = 600.0,
    ):
        self.store = store
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.idle_unload_seconds = idle_unload_seconds

        self.queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.detector: Optional[ClipDetector] = None
        self.last_work_time = time.time()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="EventAuditor")

    def start(self) -> None:
        """Start the background auditor worker thread."""
        self.thread.start()

    def audit_event(self, event_id: int) -> None:
        """Enqueue an event ID for background high-resolution audit sweep."""
        self.queue.put(event_id)

    def _ensure_detector(self) -> bool:
        """Lazily (re)initialize the detector model on the target GPU."""
        if self.detector is not None:
            return True
        try:
            self.detector = ClipDetector(model_path=self.model_name, device=self.device)
            logger.info(f"EventAuditor initialized on device {self.device} with {self.model_name}")
            return True
        except Exception as exc:
            logger.warning(f"EventAuditor detector init error on device {self.device}: {exc}")
            try:
                self.detector = ClipDetector(model_path=self.model_name, device=0)
                return True
            except Exception:
                return False

    def _release_detector_if_idle(self, now: float) -> None:
        """Release GPU model context after prolonged inactivity to minimize idle resources."""
        if self.detector is None:
            self.last_work_time = now
            return
        if now - self.last_work_time < self.idle_unload_seconds:
            return

        self.detector = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.last_work_time = now
        logger.info(
            f"EventAuditor idle for {self.idle_unload_seconds:.0f}s; released GPU model context"
        )

    def _worker_loop(self) -> None:
        # Load detector on target GPU inside worker thread
        self._ensure_detector()

        while not self.stop_event.is_set():
            try:
                event_id = self.queue.get(timeout=2.0)
            except queue.Empty:
                self._release_detector_if_idle(time.time())
                continue

            if event_id is None:
                self.queue.task_done()
                break

            try:
                self.last_work_time = time.time()
                self._audit_single_event(event_id)
            except Exception as exc:
                logger.error(f"EventAuditor: Error auditing event {event_id}: {exc}")
            finally:
                self.queue.task_done()

    def _audit_single_event(self, event_id: int) -> None:
        ev = self.store.get_event(event_id)
        if not ev or not ev.get("clip_path"):
            return

        clip_full = os.path.join(self.output_dir, ev["clip_path"])
        if not os.path.exists(clip_full):
            return

        keyframes = _extract_clip_keyframes(clip_full, num_frames=4)
        if not keyframes:
            return

        if not self._ensure_detector() or self.detector is None:
            return

        found_classes: Dict[str, float] = {}

        for frame in keyframes:
            try:
                dets = self.detector.track_frame(frame, 0, conf_threshold=self.conf_threshold)
                for d in dets:
                    if d.class_name in AUDITOR_CLASSES:
                        if d.class_name not in found_classes or d.confidence > found_classes[d.class_name]:
                            found_classes[d.class_name] = round(d.confidence, 2)
            except Exception as infer_exc:
                logger.debug(f"EventAuditor inference error: {infer_exc}")

        if not found_classes:
            return

        # Check existing behaviors
        try:
            behaviors = set(json.loads(ev.get("behaviors", "[]")))
        except Exception:
            behaviors = set()

        added_any = False
        for cname, conf in found_classes.items():
            tag = f"audited_{cname}"
            if tag not in behaviors:
                behaviors.add(tag)
                added_any = True

        if added_any:
            self.store.update_event(
                event_id,
                behaviors=json.dumps(sorted(behaviors)),
            )
            logger.info(
                f"EventAuditor verified event {event_id}: high-res sweep confirmed {list(found_classes.keys())}"
            )

    def stop(self) -> None:
        """Signal auditor thread to exit and wait for completion."""
        self.stop_event.set()
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
