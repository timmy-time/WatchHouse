"""Configuration models and loader for live CCTV analytics."""

from dataclasses import dataclass, field
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from engine.behavior import Zone
from engine.live.dynamic_fps import DynamicFpsConfig
from engine.live.lighting import NightModeConfig

logger = logging.getLogger(__name__)


def _env_str(val: Any, default: str = "") -> str:
    """Expand environment variables supporting ${VAR:-default}, ${VAR}, and $VAR."""
    if val is None:
        return default
    s = str(val)

    def _repl(m: Any) -> str:
        var = m.group(1)
        fallback = m.group(2) if m.group(2) is not None else ""
        return os.environ.get(var, fallback)

    s = re.sub(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}", _repl, s)
    s = os.path.expandvars(s)
    if "${" in s:
        return default
    s = s.strip()
    return s if s else default


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
    night_mode: NightModeConfig = field(default_factory=NightModeConfig)
    # Dual-GPU roles
    realtime_gpu: int = 0          # fast path: all camera workers
    detailed_gpu: int = 1          # slow path: detailed verifier
    # Inference device per role: "auto"/"gpu" = use the role's GPU above,
    # "cpu" = run that role's YOLO inference on the CPU (ffmpeg decode stays
    # on the GPU, chosen by the per-camera gpu below).
    realtime_device: str = "auto"
    detailed_device: str = "auto"
    # Remote inference (device: "remote"): run YOLO on another host, e.g. a
    # Windows/WSL2 box with a GPU. Frames go out as JPEG over HTTP.
    remote_url: str = ""
    remote_timeout: float = 10.0
    remote_jpeg_quality: int = 80
    cpu_threads: int = 0           # torch intra-op threads for CPU inference (0 = torch default)
    detailed_model: str = "yolov8s.pt"
    detailed_imgsz: int = 1280
    detailed_conf: float = 0.25
    detailed_interval: float = 2.0  # seconds between detailed frame samples per camera
    tracker_config: str = "config/bytetrack_live.yaml"
    # Analysis pipe shape. The pipe is what every camera worker reads, so its frame
    # size drives per-frame CPU (rawvideo read, motion diff, preview encode) and the
    # cost of feeding the detector. 0 = native source resolution.
    pipe_width: int = 0
    # Which stream feeds the analysis pipe: "main" or "sub". The main stream defines
    # the coordinate space the dashboard zones and scenery slots were drawn in; a DVR
    # substream is often 4:3 with a different vertical field of view (measured y-offset
    # on this DVR), so switching a camera to "sub" invalidates existing zones/slots.
    source: str = "main"


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
    record: bool = True  # False = live inference only: no ring segments, no event clips
    # Physical GPU used for ffmpeg decode (set by run_live); None = inherit the
    # process's CUDA_VISIBLE_DEVICES and use `gpu` as the visible index.
    decode_gpu: Optional[int] = None
    # Per-camera detection overrides (None = use analysis.* defaults)
    confidence: Optional[float] = None
    model: Optional[str] = None
    imgsz: Optional[int] = None
    tracker_config: Optional[str] = None
    device: Optional[str] = None   # "auto" | "gpu" | "cpu" — overrides analysis.realtime_device
    night_confidence: Optional[float] = None
    night_tracker_config: Optional[str] = None
    night_imgsz: Optional[int] = None


def resolve_inference_device(spec: Optional[str], default: str = "auto") -> str:
    """Map a config value to the actual inference device: "cpu", "gpu" or "remote".

    "auto" (or an unset value) defers to the role default; the default itself being
    "auto" keeps the pre-existing behaviour of running inference on the GPU the role
    is pinned to. Anything unrecognised falls back to the GPU with a warning.
    """
    value = str(spec).strip().lower() if spec is not None else ""
    if value in ("", "auto"):
        value = str(default or "").strip().lower()
        if value == "cpu":
            return "cpu"
        if value == "remote":
            return "remote"
        return "gpu"
    if value == "cpu":
        return "cpu"
    if value == "remote":
        return "remote"
    if value in ("gpu", "cuda"):
        return "gpu"
    logger.warning("unknown inference device %r: falling back to the GPU", spec)
    return "gpu"


@dataclass
class LiveConfig:
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    faces: FacesConfig = field(default_factory=FacesConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    cameras: List[CameraConfig] = field(default_factory=list)


def _load_dotenv_if_present() -> None:
    """Load key-value pairs from .env into os.environ if .env exists and key is unset."""
    candidates = [
        ".env",
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(__file__), "../../.env"),
    ]
    for env_path in candidates:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            os.environ.setdefault(k, v)
                break
            except Exception:
                pass

def load_live_config(path: str) -> LiveConfig:
    """Load and validate live configuration from YAML file.

    If a local override file exists alongside `path` (e.g. `live.local.yaml`),
    it is preferred over the base template, allowing full per-host customization
    without modifying tracked files or leaking private configurations.
    """
    _load_dotenv_if_present()
    base, ext = os.path.splitext(path)
    local_path = f"{base}.local{ext}"
    if os.path.exists(local_path):
        logger.info("Using local configuration override: %s", local_path)
        path = local_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    rec_raw = raw.get("recording", {})
    rec_root = _env_str(rec_raw.get("root", "/data/recordings"))
    if not os.path.exists(rec_root):
        try:
            os.makedirs(rec_root, exist_ok=True)
        except OSError:
            rec_root = os.path.abspath("recordings")

    recording = RecordingConfig(
        root=rec_root,
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
        realtime_device=_env_str(ana_raw.get("realtime_device"), default="auto"),
        detailed_device=_env_str(ana_raw.get("detailed_device"), default="auto"),
        cpu_threads=int(ana_raw.get("cpu_threads", 0)),
        remote_url=_env_str(ana_raw.get("remote_url", "")),
        remote_timeout=float(ana_raw.get("remote_timeout", 10.0)),
        remote_jpeg_quality=int(ana_raw.get("remote_jpeg_quality", 80)),
        detailed_model=str(ana_raw.get("detailed_model", "yolov8s.pt")),
        detailed_imgsz=int(ana_raw.get("detailed_imgsz", 1280)),
        detailed_conf=float(ana_raw.get("detailed_conf", 0.25)),
        detailed_interval=float(ana_raw.get("detailed_interval", 2.0)),
        tracker_config=str(ana_raw.get("tracker_config", "config/bytetrack_live.yaml")),
        pipe_width=int(ana_raw.get("pipe_width", 0)),
        source=str(ana_raw.get("source", "main")),
        dynamic_fps=(
            DynamicFpsConfig(enabled=bool(ana_raw.get("dynamic_fps")))
            if isinstance(ana_raw.get("dynamic_fps"), bool)
            else DynamicFpsConfig(
                enabled=bool(ana_raw.get("dynamic_fps", {}).get("enabled", True)),
                idle_fps=float(ana_raw.get("dynamic_fps", {}).get("idle_fps", 3.0)),
                boost_fps=float(ana_raw.get("dynamic_fps", {}).get("boost_fps", 15.0)),
                motion_threshold=float(ana_raw.get("dynamic_fps", {}).get("motion_threshold", 0.015)),
                boost_cooldown=float(ana_raw.get("dynamic_fps", {}).get("boost_cooldown", 8.0)),
                motion_gate=bool(ana_raw.get("dynamic_fps", {}).get("motion_gate", True)),
            )
            if isinstance(ana_raw.get("dynamic_fps"), dict)
            else DynamicFpsConfig()
        ),
        night_mode=(
            NightModeConfig(enabled=bool(ana_raw.get("night_mode")))
            if isinstance(ana_raw.get("night_mode"), bool)
            else NightModeConfig(
                enabled=bool(ana_raw.get("night_mode", {}).get("enabled", True)),
                confidence=float(ana_raw.get("night_mode", {}).get("confidence", 0.20)),
                tracker_config=str(ana_raw.get("night_mode", {}).get("tracker_config", "config/bytetrack_mild.yaml")),
                imgsz=(int(ana_raw.get("night_mode", {}).get("imgsz")) if ana_raw.get("night_mode", {}).get("imgsz") is not None else None),
                night_threshold=float(ana_raw.get("night_mode", {}).get("night_threshold", 6.0)),
                day_threshold=float(ana_raw.get("night_mode", {}).get("day_threshold", 15.0)),
                confirm_seconds=float(ana_raw.get("night_mode", {}).get("confirm_seconds", 6.0)),
                cooldown_seconds=float(ana_raw.get("night_mode", {}).get("cooldown_seconds", 30.0)),
            )
            if isinstance(ana_raw.get("night_mode"), dict)
            else NightModeConfig()
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
        name = _env_str(c.get("name"), default="Camera")
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
            record=bool(c.get("record", True)),
            confidence=float(c["confidence"]) if c.get("confidence") is not None else None,
            model=str(c["model"]) if c.get("model") else None,
            imgsz=int(c["imgsz"]) if c.get("imgsz") is not None else None,
            tracker_config=str(c["tracker_config"]) if c.get("tracker_config") else None,
            device=str(c["device"]) if c.get("device") else None,
            night_confidence=float(c["night_confidence"]) if c.get("night_confidence") is not None else None,
            night_tracker_config=str(c["night_tracker_config"]) if c.get("night_tracker_config") else None,
            night_imgsz=int(c["night_imgsz"]) if c.get("night_imgsz") is not None else None,
        ))

    return LiveConfig(
        recording=recording,
        analysis=analysis,
        faces=faces,
        notifications=notifications,
        cameras=cameras,
    )
