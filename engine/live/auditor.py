"""GPU-1 detailed verifier: slow, high-resolution second opinion.

Ultralytics' ``select_device`` rewrites ``CUDA_VISIBLE_DEVICES`` process-wide, so a
single process cannot hold two GPU roles. The verifier therefore runs its inference
in a **spawned child process** with ``CUDA_VISIBLE_DEVICES`` pinned to the detailed
GPU; the parent (realtime process) is pinned to the realtime GPU. Results travel back
over a queue and the parent applies database/notification side effects.

Roles:
- Live tap: camera workers push a frame every few seconds tagged with the classes
  realtime currently sees. Classes the big model finds but realtime missed raise a
  "Detailed check" alert and tag the open event with ``verified_<class>``.
- Post-clip audit: finalized clips are re-scanned at keyframes and tagged with
  ``audited_<class>`` for anything the fast path missed.
"""

from dataclasses import dataclass, field
import json
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Set

import numpy as np

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


@dataclass
class VerifierResult:
    kind: str  # "verified" | "audited"
    camera: str = ""
    event_id: Optional[int] = None
    found: Dict[str, float] = field(default_factory=dict)
    missed: Dict[str, float] = field(default_factory=dict)


def evaluate_detections(
    dets,
    seen_classes: Set[str],
    notify_classes: Set[str],
) -> "tuple[Dict[str, float], Dict[str, float]]":
    """Split detections into (found, missed) restricted to notify_classes."""
    found: Dict[str, float] = {}
    for d in dets:
        name = getattr(d, "class_name", None)
        if name not in AUDITOR_CLASSES:
            continue
        conf = float(getattr(d, "confidence", 0.0))
        if name not in found or conf > found[name]:
            found[name] = conf

    missed = {
        cls: conf
        for cls, conf in found.items()
        if cls not in seen_classes and cls in notify_classes
    }
    return found, missed


def _extract_clip_keyframes(clip_path: str, num_frames: int = 4) -> List[np.ndarray]:
    """Extract evenly spaced keyframes from an MP4 video clip."""
    import cv2

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


class VerifierCore:
    """Inference + decision logic. Runs inside the spawned child process."""

    def __init__(
        self,
        model_name: str,
        imgsz: int,
        conf_threshold: float,
        notify_classes: Set[str],
        idle_unload_seconds: float = 600.0,
    ):
        self.model_name = model_name
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.notify_classes = notify_classes
        self.idle_unload_seconds = idle_unload_seconds
        self.detector = None
        self.last_work_time = time.time()

    # --- model lifecycle ---

    def ensure_detector(self) -> bool:
        """Load the detector on the process's (already pinned) visible GPU."""
        if self.detector is not None:
            return True
        try:
            from engine.detector import ClipDetector
            self.detector = ClipDetector(
                model_path=self.model_name, device="", imgsz=self.imgsz
            )
            logger.info(
                f"DetailedVerifier core ready: {self.model_name} @ {self.imgsz}px "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
            )
            return True
        except Exception as exc:
            logger.warning(f"DetailedVerifier model load failed: {exc}")
            return False

    def release_detector_if_idle(self, now: float) -> bool:
        """Drop the GPU model context after prolonged inactivity."""
        if self.detector is None:
            self.last_work_time = now
            return False
        if now - self.last_work_time < self.idle_unload_seconds:
            return False

        self.detector = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.last_work_time = now
        logger.info(
            f"DetailedVerifier idle for {self.idle_unload_seconds:.0f}s; released GPU model context"
        )
        return True

    # --- work items ---

    def verify_frame(self, frame: np.ndarray, seen_classes: Set[str]) -> Dict[str, float]:
        """Run the detailed model on a live frame; return the classes realtime missed."""
        if not self.ensure_detector() or self.detector is None:
            return {}
        try:
            dets = self.detector.detect_frame(frame, conf_threshold=self.conf_threshold)
        except Exception as exc:
            logger.debug(f"DetailedVerifier inference error: {exc}")
            return {}
        _, missed = evaluate_detections(dets, seen_classes, self.notify_classes)
        return missed

    def audit_clip(self, clip_path: str) -> Dict[str, float]:
        """Sweep keyframes of a finalized clip; return all detected classes."""
        if not self.ensure_detector() or self.detector is None:
            return {}
        found: Dict[str, float] = {}
        for frame in _extract_clip_keyframes(clip_path, num_frames=4):
            try:
                dets = self.detector.detect_frame(frame, conf_threshold=self.conf_threshold)
            except Exception as exc:
                logger.debug(f"DetailedVerifier clip sweep error: {exc}")
                continue
            frame_found, _ = evaluate_detections(dets, set(), set())
            for cls, conf in frame_found.items():
                if cls not in found or conf > found[cls]:
                    found[cls] = conf
        return {k: round(v, 2) for k, v in found.items()}

    # --- process entry ---

    def run(self, in_q, out_q) -> None:
        """Main loop executed in the child process."""
        self.ensure_detector()
        samples = 0
        while True:
            try:
                item = in_q.get(timeout=2.0)
            except queue.Empty:
                self.release_detector_if_idle(time.time())
                continue

            if item is None:
                break

            try:
                self.last_work_time = time.time()
                kind = item[0]
                if kind == "sample":
                    _, camera, event_id, frame, seen = item
                    missed = self.verify_frame(frame, set(seen or ()))
                    samples += 1
                    if missed:
                        out_q.put(("result", VerifierResult(
                            kind="verified", camera=camera, event_id=event_id,
                            found=dict(missed), missed=dict(missed),
                        )))
                    if samples % 20 == 0:
                        out_q.put(("stat", samples, 0, 0))
                elif kind == "clip":
                    _, event_id, clip_path = item
                    found = self.audit_clip(clip_path)
                    if found:
                        out_q.put(("result", VerifierResult(
                            kind="audited", camera="", event_id=event_id, found=found,
                        )))
                    out_q.put(("stat", samples, 1, 0))
            except Exception as exc:
                logger.error(f"DetailedVerifier child error: {exc}")
        out_q.put(("stat", samples, 0, 0))


def _child_main(cfg: dict, in_q, out_q) -> None:
    """Spawned-process entry: pin the GPU *before* any CUDA call."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["physical_device"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
    core = VerifierCore(
        model_name=cfg["model_name"],
        imgsz=cfg["imgsz"],
        conf_threshold=cfg["conf_threshold"],
        notify_classes=set(cfg["notify_classes"]),
        idle_unload_seconds=cfg["idle_unload_seconds"],
    )
    core.run(in_q, out_q)


class DetailedVerifier:
    """Parent-side manager: owns the child process, applies results (DB + alerts)."""

    def __init__(
        self,
        store,
        output_dir: str,
        model_name: str = "yolov8s.pt",
        device: int = 1,
        imgsz: int = 1280,
        conf_threshold: float = 0.25,
        notify_classes: Optional[Set[str]] = None,
        notifier=None,
        idle_unload_seconds: float = 600.0,
        queue_size: int = 8,
        # test hook: run inference in-process instead of spawning
        inject_core: Optional[VerifierCore] = None,
    ):
        self.store = store
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.notify_classes = notify_classes or {"person", "dog", "cat", "bear", "horse", "cow", "sheep"}
        self.notifier = notifier
        self.idle_unload_seconds = idle_unload_seconds
        self.queue_size = queue_size

        self._mp_ctx = mp.get_context("spawn")
        self._in_q = self._mp_ctx.Queue(maxsize=queue_size)
        self._out_q = self._mp_ctx.Queue()
        self._proc: Optional[mp.process.BaseProcess] = None
        self._consumer: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self.samples_verified = 0
        self.clips_audited = 0
        self.misses_caught = 0
        self.started_at = 0.0

        # In-process test mode (no child process, no CUDA): used by unit tests.
        self._core = inject_core

    # --- lifecycle ---

    def start(self) -> None:
        """Spawn the inference child and the result consumer."""
        self.started_at = time.time()
        if self._core is None:
            cfg = {
                "physical_device": self.device,
                "model_name": self.model_name,
                "imgsz": self.imgsz,
                "conf_threshold": self.conf_threshold,
                "notify_classes": sorted(self.notify_classes),
                "idle_unload_seconds": self.idle_unload_seconds,
            }
            try:
                self._proc = self._mp_ctx.Process(
                    target=_child_main, args=(cfg, self._in_q, self._out_q),
                    daemon=True, name="DetailedVerifier",
                )
                self._proc.start()
                logger.info(
                    f"DetailedVerifier child started (pid {self._proc.pid}) "
                    f"-> GPU {self.device} ({self.model_name} @ {self.imgsz}px)"
                )
            except Exception as exc:
                logger.error(f"DetailedVerifier failed to spawn child: {exc}")
                self._proc = None

        self._consumer = threading.Thread(target=self._consume_results, daemon=True, name="VerifierResults")
        self._consumer.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=5.0)
            if self._proc.is_alive():
                self._proc.terminate()
        if self._consumer is not None:
            self._consumer.join(timeout=3.0)

    # --- public API ---

    def submit_sample(self, camera: str, event_id: Optional[int], frame: np.ndarray, seen_classes: Set[str]) -> bool:
        """Queue a live frame for detailed verification (drops when busy)."""
        try:
            self._in_q.put_nowait(("sample", camera, event_id, np.ascontiguousarray(frame), set(seen_classes)))
            return True
        except queue.Full:
            return False
        except Exception as exc:
            logger.debug(f"DetailedVerifier submit failed: {exc}")
            return False

    def audit_event(self, event_id: int) -> None:
        """Queue a finalized clip for the keyframe audit sweep."""
        ev = self.store.get_event(event_id)
        if not ev or not ev.get("clip_path"):
            return
        clip_full = os.path.join(self.output_dir, ev["clip_path"])
        try:
            self._in_q.put_nowait(("clip", event_id, clip_full))
        except queue.Full:
            logger.debug(f"DetailedVerifier queue full; skipping audit of event {event_id}")
        except Exception:
            pass

    def status(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "model": self.model_name,
            "imgsz": self.imgsz,
            "child_alive": bool(self._proc is not None and self._proc.is_alive()),
            "queue": self._in_q.qsize() if hasattr(self._in_q, "qsize") else -1,
            "samples_verified": self.samples_verified,
            "clips_audited": self.clips_audited,
            "misses_caught": self.misses_caught,
        }

    # --- result application (parent process) ---

    def _consume_results(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._out_q.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception:
                continue

            try:
                tag = item[0]
                if tag == "result":
                    self.apply_result(item[1])
                elif tag == "stat":
                    _, samples, clips, _ = item
                    self.samples_verified = max(self.samples_verified, int(samples))
                    self.clips_audited += int(clips)
            except Exception as exc:
                logger.error(f"DetailedVerifier result handling error: {exc}")

    def apply_result(self, result: VerifierResult) -> None:
        """Apply a verifier result: tag the event and raise alerts for misses."""
        if result.kind == "audited":
            if result.event_id is not None and result.found:
                self._add_behaviors(result.event_id, [f"audited_{c}" for c in result.found])
            return

        if result.kind == "verified":
            missed = result.missed or {}
            if not missed:
                return
            self.misses_caught += len(missed)
            cls_list = ", ".join(f"{c} {v:.2f}" for c, v in sorted(missed.items()))
            logger.info(
                f"DetailedVerifier: realtime missed {cls_list} on {result.camera}; alerting"
            )
            if result.event_id is not None:
                self._add_behaviors(result.event_id, [f"verified_{c}" for c in missed])
            if self.notifier is not None:
                best_cls = max(missed, key=lambda c: missed[c])
                found_desc = ", ".join(f"{c} {v:.2f}" for c, v in sorted(missed.items()))
                self.notifier.notify(
                    event_id=result.event_id if result.event_id is not None else 0,
                    camera=result.camera,
                    kind=f"detailed_{best_cls}",
                    title=f"Detailed check: {best_cls} at {result.camera}",
                    body=(
                        f"Slow high-res model ({self.model_name} @ {self.imgsz}px) "
                        f"found {found_desc} — realtime missed it."
                    ),
                )

    def _add_behaviors(self, event_id: int, tags: List[str]) -> None:
        ev = self.store.get_event(event_id)
        if not ev:
            return
        try:
            behaviors = set(json.loads(ev.get("behaviors", "[]")))
        except Exception:
            behaviors = set()
        new_tags = [t for t in tags if t not in behaviors]
        if new_tags:
            behaviors.update(new_tags)
            self.store.update_event(event_id, behaviors=json.dumps(sorted(behaviors)))
            logger.info(f"DetailedVerifier tagged event {event_id}: {new_tags}")
