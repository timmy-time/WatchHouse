"""Supervisor and worker thread orchestration for live camera analytics."""

import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from engine.detector import ClipDetector, configure_cpu_inference
from engine.faces import FaceEngine, FaceGallery
from engine.live.clipper import Finalizer
from engine.live.config import CameraConfig, LiveConfig, load_live_config, resolve_inference_device
from engine.live.db import EventStore
from engine.live.events import EventManager
from engine.live.notify import Notifier
from engine.live.stream import CameraStream
from engine.scenery import SceneryManager
import cv2
from engine.live.auditor import DetailedVerifier
from engine.live.dynamic_fps import DynamicFpsController
from engine.live.scheduler import GpuScheduler

logger = logging.getLogger(__name__)


def _resolve_tracker_config(cfg: LiveConfig, cam: Optional[CameraConfig] = None) -> str:
    """Per-camera tracker profile (e.g. flicker-tolerant), else the global one."""
    tracker = (cam.tracker_config if cam is not None and cam.tracker_config else cfg.analysis.tracker_config)
    if tracker and os.path.exists(tracker):
        return tracker
    return "bytetrack.yaml"


def _track_velocity(track, frame_w: int, frame_h: int) -> "tuple[float, float]":
    """Normalized screen velocity (units/s) from the track's recent window.

    Uses the newest observation and the oldest one at least 0.2 s older, so the
    estimate is stable enough for client-side box prediction.
    """
    window = list(track.window)
    if len(window) < 2 or frame_w <= 0 or frame_h <= 0:
        return 0.0, 0.0

    t_last, d_last = window[-1]
    ref = None
    for t, d in reversed(window[:-1]):
        if t_last - t >= 0.2:
            ref = (t, d)
            break
    if ref is None:
        return 0.0, 0.0

    t0, d0 = ref
    dt = t_last - t0
    if dt <= 1e-3:
        return 0.0, 0.0

    cx0 = (d0.bbox_xyxy[0] + d0.bbox_xyxy[2]) / 2.0 / frame_w
    cy0 = (d0.bbox_xyxy[1] + d0.bbox_xyxy[3]) / 2.0 / frame_h
    cx1 = (d_last.bbox_xyxy[0] + d_last.bbox_xyxy[2]) / 2.0 / frame_w
    cy1 = (d_last.bbox_xyxy[1] + d_last.bbox_xyxy[3]) / 2.0 / frame_h

    vx = (cx1 - cx0) / dt
    vy = (cy1 - cy0) / dt

    # Clamp absurd values (track ID swaps) to a plausible screen speed
    limit = 3.0
    vx = max(-limit, min(limit, vx))
    vy = max(-limit, min(limit, vy))
    return round(vx, 3), round(vy, 3)


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
        scenery: Optional[SceneryManager] = None,
        verifier: Optional[DetailedVerifier] = None,
    ):
        super().__init__(name=f"Worker-{cam.slug}", daemon=True)
        self.cam = cam
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.output_dir = output_dir
        self.face_engine = face_engine
        self.gallery = gallery
        self.scenery = scenery
        self.verifier = verifier

        self.stop_event = threading.Event()
        self.fps_analyzed = 0.0
        self.last_frame_at = 0.0
        self.last_verify_push = 0.0
        self.detections_dir = os.path.join(output_dir, "live", "detections")
        os.makedirs(self.detections_dir, exist_ok=True)
        self.last_det_write = 0.0
        self.fps_controller = DynamicFpsController(cfg.analysis.dynamic_fps)
        self.current_detections: List[Dict[str, Any]] = []

        self.stream = CameraStream(cam, cfg.recording, fps=cfg.analysis.fps)
        # Per-camera detection overrides (falling back to the global analysis config)
        self.model_path = cam.model or cfg.analysis.model
        self.imgsz = cam.imgsz or cfg.analysis.imgsz
        self.conf_threshold = cam.confidence if cam.confidence is not None else cfg.analysis.confidence
        self.tracker_config = _resolve_tracker_config(cfg, cam)
        # "cpu" runs YOLO on the CPU; ffmpeg decode stays on the GPU (cam.gpu).
        self.inference_device = resolve_inference_device(cam.device, cfg.analysis.realtime_device)
        if self.inference_device == "cpu":
            configure_cpu_inference(cfg.analysis.cpu_threads)
        self.detector = ClipDetector(
            model_path=self.model_path,
            device="cpu" if self.inference_device == "cpu" else "",  # GPU pinning is done via CUDA_VISIBLE_DEVICES at process start
            imgsz=self.imgsz,
            tracker_config=self.tracker_config,
        )
        # Live-only cameras (record=False) get no clip assembly thread at all.
        self.finalizer = Finalizer(
            cam, cfg, store, output_dir,
            on_finalized=(verifier.audit_event if verifier is not None else None),
        ) if cam.record else None
        self.manager = EventManager(
            cam=cam,
            cfg=cfg,
            store=store,
            notifier=notifier,
            output_dir=output_dir,
            face_engine=face_engine,
            gallery=gallery,
            on_closed=(self.finalizer.enqueue if self.finalizer is not None else None),
            scenery=scenery,
        )

    def _write_detections_file(self, now: float, frame_w: int, frame_h: int) -> None:
        """Publish current boxes + velocity for the dashboard at ~4 Hz.

        Separate from status.json so the client can poll quickly and predict box
        positions between updates without paying the full status payload cost.
        """
        if now - self.last_det_write < 0.25:
            return
        self.last_det_write = now

        payload = {
            "camera": self.cam.name,
            "updated_at": now,
            "frame_w": frame_w,
            "frame_h": frame_h,
            "source": "sub" if (self.cam.sub_url and not self.stream.sub_failed) else "main",
            "detections": self.current_detections,
        }
        path = os.path.join(self.detections_dir, f"{self.cam.slug}.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception as exc:
            logger.debug(f"{self.cam.name}: detections write failed: {exc}")

    def run(self) -> None:
        logger.info(
            f"{self.cam.name}: Starting camera worker "
            f"(inference={self.inference_device}, decode_gpu={self.cam.gpu})"
        )
        self.stream.start()
        if self.finalizer is not None:
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
                event_active = self.manager.open_event_id is not None
                active_tracks = len([
                    tr for tr in self.manager.tracks.values()
                    if (t - tr.last_seen < 4.0) and not tr.is_anchored
                ])

                should_infer, mode, target_fps = self.fps_controller.should_infer(
                    now=t,
                    frame=frame,
                    event_active=event_active,
                    active_tracks=active_tracks,
                )

                if should_infer:
                    dets = self.detector.track_frame(
                        frame,
                        frame_idx=frame_idx,
                        conf_threshold=self.conf_threshold,
                    )
                    frame_idx += 1

                    # name -> global vehicle id (for cross-camera labelling)
                    slot_vehicle = {}
                    if self.scenery is not None:
                        for s in self.scenery.get_slots(self.cam.name):
                            if s.vehicle_id is not None:
                                slot_vehicle[s.name] = s.vehicle_id

                    self.current_detections = []
                    for d in dets:
                        track = self.manager.tracks.get(d.track_id)
                        self.current_detections.append({
                            "class_name": d.class_name,
                            "track_id": d.track_id,
                            "conf": round(d.confidence, 2),
                            "box_norm": [
                                round(d.bbox_xyxy[0] / frame_w, 3),
                                round(d.bbox_xyxy[1] / frame_h, 3),
                                round(d.bbox_xyxy[2] / frame_w, 3),
                                round(d.bbox_xyxy[3] / frame_h, 3),
                            ],
                            "vel": _track_velocity(track, frame_w, frame_h) if track else (0.0, 0.0),
                            "anchored_name": track.anchored_slot_name if track else None,
                            "vehicle_id": slot_vehicle.get(track.anchored_slot_name) if track and track.anchored_slot_name else None,
                            "is_anchored": track.is_anchored if track else False,
                        })
                    self._write_detections_file(t, frame_w, frame_h)

                    self.manager.process(
                        frame=frame,
                        dets=dets,
                        now=t,
                        frame_w=frame_w,
                        frame_h=frame_h,
                    )

                    self.last_frame_at = t
                    fps_count += 1

                # Clean preview on every delivered frame (inference optional):
                # overlays are drawn client-side, this keeps the stream smooth.
                self.manager.update_preview(frame, t)

                # Slow-path tap: hand a frame to the detailed verifier (GPU 1) every
                # `detailed_interval` seconds, tagged with what realtime already sees.
                if self.verifier is not None and (t - self.last_verify_push) >= self.cfg.analysis.detailed_interval:
                    self.last_verify_push = t
                    self.verifier.submit_sample(
                        camera=self.cam.name,
                        event_id=self.manager.open_event_id,
                        frame=frame.copy(),
                        seen_classes={d["class_name"] for d in self.current_detections},
                    )

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
                    self.detector = ClipDetector(
                        model_path=self.model_path,
                        device="cpu" if self.inference_device == "cpu" else "",
                        imgsz=self.imgsz,
                        tracker_config=self.tracker_config,
                    )
                except Exception as rebuild_exc:
                    logger.error(f"{self.cam.name}: Rebuild failed: {rebuild_exc}")

        logger.info(f"{self.cam.name}: Worker shutting down")
        self.stream.stop()
        if self.finalizer is not None:
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

    # Pin this process to the realtime GPU *before* any CUDA call. Ultralytics'
    # select_device rewrites CUDA_VISIBLE_DEVICES per call, so the only reliable
    # multi-GPU strategy is one process per role: this process = realtime GPU
    # (all workers use visible device 0), the verifier child = detailed GPU.
    # Cameras configured for CPU inference do not need the realtime GPU visible;
    # their ffmpeg children are pinned to the decode GPU explicitly.
    camera_devices = {
        cam.name: resolve_inference_device(cam.device, cfg.analysis.realtime_device)
        for cam in active_cams
    }
    if any(dev == "gpu" for dev in camera_devices.values()):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.analysis.realtime_gpu)
        logger.info(
            f"Realtime process pinned to GPU {cfg.analysis.realtime_gpu} "
            f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
        )
    else:
        logger.info("All cameras run CPU inference: realtime process left unpinned")

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
    scenery = SceneryManager(store)
    gpu_scheduler = GpuScheduler()

    # --- Dual-GPU roles (process-pinned) ---
    # This process is pinned to the realtime GPU (visible device 0). The verifier
    # child pins itself to the detailed GPU. Same index => single-GPU mode.
    realtime_device = 0  # visible index inside this process (pinned to the realtime GPU)
    detailed_setting = resolve_inference_device(cfg.analysis.detailed_device)
    detailed_physical = "cpu" if detailed_setting == "cpu" else cfg.analysis.detailed_gpu
    realtime_physical = cfg.analysis.realtime_gpu
    if detailed_setting == "gpu" and realtime_physical == cfg.analysis.detailed_gpu:
        logger.warning(
            "realtime_gpu == detailed_gpu (%s): single-GPU mode, expect contention",
            realtime_physical,
        )

    verifier = DetailedVerifier(
        store=store,
        output_dir=output_dir,
        model_name=cfg.analysis.detailed_model,
        device=detailed_physical,
        imgsz=cfg.analysis.detailed_imgsz,
        conf_threshold=cfg.analysis.detailed_conf,
        notify_classes=set(cfg.notifications.notify_classes),
        notifier=notifier,
        cpu_threads=cfg.analysis.cpu_threads,
    )
    verifier.start()
    logger.info(
        f"DetailedVerifier launched on {('CPU' if detailed_setting == 'cpu' else f'physical GPU {detailed_physical}')} "
        f"({cfg.analysis.detailed_model} @ {cfg.analysis.detailed_imgsz}px)"
    )

    model_dir = os.environ.get("FACE_MODEL_DIR", "/opt/models")
    workers: List[CameraWorker] = []

    for cam in active_cams:
        # ffmpeg decodes on the realtime GPU regardless of where inference runs; the
        # ffmpeg child gets an explicit CUDA_VISIBLE_DEVICES for it (stream._decode_env).
        cam.decode_gpu = cfg.analysis.realtime_gpu
        # All GPU inference work uses visible device 0 in this process (pinned to realtime GPU).
        cam.gpu = realtime_device
        if camera_devices[cam.name] == "gpu":
            gpu_scheduler.assign_camera(cam.name, realtime_device)

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
            scenery=scenery,
            verifier=verifier,
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
                "gpus": gpu_scheduler.poll_metrics(),
                "verifier": verifier.status(),
                "cameras": {
                    w.cam.name: {
                        "slug": w.cam.slug,
                        **w.stream.status(),
                        "gpu": w.cam.gpu,
                        "inference_device": w.inference_device,
                        "record": w.cam.record,
                        "mode": w.fps_controller.current_mode,
                        "target_fps": (
                            w.fps_controller.cfg.boost_fps
                            if w.fps_controller.current_mode == "boost"
                            else w.fps_controller.cfg.idle_fps
                        ),
                        "fps_analyzed": w.fps_analyzed,
                        "open_event_id": w.manager.open_event_id,
                        "last_frame_at": w.last_frame_at,
                        "detections": list(w.current_detections),
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
        verifier.stop()
        notifier.stop()
        logger.info("Live analytics stopped cleanly")

    return 0
