"""Configuration models and loader for live CCTV analytics."""

from dataclasses import dataclass, field
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from engine.behavior import Zone
from engine.live.dynamic_fps import DynamicFpsConfig


def _env_str(val: Any) -> str:
    """Expand environment variables; if unresolved '${VAR}' remains or None, return empty string."""
    if val is None:
        return ""
    s = str(val)
    s = os.path.expandvars(s)
    if "${" in s:
        return ""
    return s.strip()


def slugify(name: str) -> str:
    """Derive camera slug safe for filesystem and URLs."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


@dataclass
class RecordingConfig:
    root: str = "/data/recordings"
    segment_seconds: int = 10
    ring_minutes: int = 15


@dataclass
class AnalysisConfig:
    fps: int = 15
    model: str = "yolov8n.pt"
    confidence: float = 0.30
    imgsz: int = 640
    pre_roll_seconds: int = 5
    post_roll_seconds: int = 10
    max_event_seconds: int = 300
    dynamic_fps: DynamicFpsConfig = field(default_factory=DynamicFpsConfig)
    # Dual-GPU roles
    realtime_gpu: int = 0          # fast path: all camera workers
    detailed_gpu: int = 1          # slow path: detailed verifier
    detailed_model: str = "yolov8s.pt"
    detailed_imgsz: int = 1280
    detailed_conf: float = 0.25
    detailed_interval: float = 2.0  # seconds between detailed frame samples per camera
    tracker_config: str = "config/bytetrack_live.yaml"


@dataclass
class FacesConfig:
    enabled: bool = True
    min_face_px: int = 40
    min_det_score: float = 0.80
    match_threshold: float = 0.40
    cluster_threshold: float = 0.45


@dataclass
class NtfyConfig:
    url: str = ""
    topic: str = ""
    token: str = ""


@dataclass
class WebhookConfig:
    url: str = ""


@dataclass
class NotificationsConfig:
    cooldown_seconds: int = 60
    dashboard_url: str = "http://localhost:8080"
    notify_classes: List[str] = field(
        default_factory=lambda: ["person", "dog", "cat", "bear", "horse", "cow", "sheep"]
    )
    ntfy: NtfyConfig = field(default_factory=NtfyConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)


@dataclass
class CameraConfig:
    name: str
    url: str
    gpu: Optional[int] = 0
    zones: List[Zone] = field(default_factory=list)
    slug: str = ""
    sub_url: str = ""   # optional DVR substream for the realtime analysis path


@dataclass
class LiveConfig:
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    faces: FacesConfig = field(default_factory=FacesConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    cameras: List[CameraConfig] = field(default_factory=list)


def load_live_config(path: str) -> LiveConfig:
    """Load and validate live configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    rec_raw = raw.get("recording", {})
    recording = RecordingConfig(
        root=_env_str(rec_raw.get("root", "/data/recordings")),
        segment_seconds=int(rec_raw.get("segment_seconds", 10)),
        ring_minutes=int(rec_raw.get("ring_minutes", 15)),
    )

    ana_raw = raw.get("analysis", {})
    analysis = AnalysisConfig(
        fps=int(ana_raw.get("fps", 15)),
        model=str(ana_raw.get("model", "yolov8n.pt")),
        confidence=float(ana_raw.get("confidence", 0.30)),
        pre_roll_seconds=int(ana_raw.get("pre_roll_seconds", 5)),
        post_roll_seconds=int(ana_raw.get("post_roll_seconds", 10)),
        max_event_seconds=int(ana_raw.get("max_event_seconds", 300)),
        imgsz=int(ana_raw.get("imgsz", 640)),
        realtime_gpu=int(ana_raw.get("realtime_gpu", 0)),
        detailed_gpu=int(ana_raw.get("detailed_gpu", 1)),
        detailed_model=str(ana_raw.get("detailed_model", "yolov8s.pt")),
        detailed_imgsz=int(ana_raw.get("detailed_imgsz", 1280)),
        detailed_conf=float(ana_raw.get("detailed_conf", 0.25)),
        detailed_interval=float(ana_raw.get("detailed_interval", 2.0)),
        tracker_config=str(ana_raw.get("tracker_config", "config/bytetrack_live.yaml")),
        dynamic_fps=(
            DynamicFpsConfig(enabled=bool(ana_raw.get("dynamic_fps")))
            if isinstance(ana_raw.get("dynamic_fps"), bool)
            else DynamicFpsConfig(
                enabled=bool(ana_raw.get("dynamic_fps", {}).get("enabled", True)),
                idle_fps=float(ana_raw.get("dynamic_fps", {}).get("idle_fps", 3.0)),
                boost_fps=float(ana_raw.get("dynamic_fps", {}).get("boost_fps", 15.0)),
                motion_threshold=float(ana_raw.get("dynamic_fps", {}).get("motion_threshold", 0.015)),
                boost_cooldown=float(ana_raw.get("dynamic_fps", {}).get("boost_cooldown", 12.0)),
            )
            if isinstance(ana_raw.get("dynamic_fps"), dict)
            else DynamicFpsConfig()
        ),
    )

    faces_raw = raw.get("faces", {})
    faces = FacesConfig(
        enabled=bool(faces_raw.get("enabled", True)),
        min_face_px=int(faces_raw.get("min_face_px", 40)),
        min_det_score=float(faces_raw.get("min_det_score", 0.80)),
        match_threshold=float(faces_raw.get("match_threshold", 0.40)),
        cluster_threshold=float(faces_raw.get("cluster_threshold", 0.45)),
    )

    notif_raw = raw.get("notifications", {})
    ntfy_raw = notif_raw.get("ntfy", {})
    webhook_raw = notif_raw.get("webhook", {})

    ntfy = NtfyConfig(
        url=_env_str(ntfy_raw.get("url", "https://ntfy.sh")),
        topic=_env_str(ntfy_raw.get("topic", "")),
        token=_env_str(ntfy_raw.get("token", "")),
    )
    webhook = WebhookConfig(
        url=_env_str(webhook_raw.get("url", "")),
    )
    dashboard_url = _env_str(notif_raw.get("dashboard_url", "http://localhost:8080"))
    if not dashboard_url:
        dashboard_url = "http://localhost:8080"

    notifications = NotificationsConfig(
        cooldown_seconds=int(notif_raw.get("cooldown_seconds", 60)),
        dashboard_url=dashboard_url,
        notify_classes=list(notif_raw.get("notify_classes", ["person", "dog", "cat", "bear", "horse", "cow", "sheep"])),
        ntfy=ntfy,
        webhook=webhook,
    )

    cameras: List[CameraConfig] = []
    for c in raw.get("cameras", []):
        name = str(c.get("name", "Camera"))
        url = _env_str(c.get("url", ""))
        gpu_raw = c.get("gpu")
        if gpu_raw is None or str(gpu_raw).lower() in ("auto", "none"):
            gpu = None
        else:
            try:
                gpu = int(gpu_raw)
            except ValueError:
                gpu = None
        slug = slugify(name)
        zones: List[Zone] = []
        for z in c.get("zones", []):
            poly_tuples = [(float(pt[0]), float(pt[1])) for pt in z.get("polygon", [])]
            zones.append(
                Zone(
                    name=str(z.get("name", "zone")),
                    type=str(z.get("type", "entry")),
                    polygon=poly_tuples,
                )
            )
        cameras.append(CameraConfig(
            name=name, url=url, gpu=gpu, zones=zones, slug=slug,
            sub_url=_env_str(c.get("sub_url", "")),
        ))

    return LiveConfig(
        recording=recording,
        analysis=analysis,
        faces=faces,
        notifications=notifications,
        cameras=cameras,
    )
