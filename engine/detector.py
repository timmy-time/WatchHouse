"""Video and thumbnail object detection and tracking using Ultralytics YOLOv8 and ByteTrack."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import os
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]


@dataclass
class TrackDetection:
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]
    frame_idx: int


@dataclass
class FrameDetections:
    frame_idx: int
    detections: List[TrackDetection] = field(default_factory=list)


def _parse_track_boxes(r, frame_idx: int) -> List[TrackDetection]:
    frame_dets: List[TrackDetection] = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            class_name = r.names[cls_id]
            conf = float(box.conf[0].item())
            # Track id may be None if tracker hasn't assigned an ID yet
            track_id = int(box.id[0].item()) if box.id is not None else -1

            xyxy = tuple(box.xyxy[0].tolist())
            xywh = tuple(box.xywh[0].tolist())

            frame_dets.append(
                TrackDetection(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    bbox_xywh=(xywh[0], xywh[1], xywh[2], xywh[3]),
                    frame_idx=frame_idx,
                )
            )
    return frame_dets


def configure_cpu_inference(cpu_threads: int = 0) -> None:
    """Bound CPU inference to `cpu_threads` torch threads (0 = leave torch's default).

    Process-wide by design: call once per process that runs CPU inference, before
    the first forward pass. Inter-op threads are pinned to 1 because a single-image
    forward pass has nothing to parallelise across ops — extra inter-op threads only
    add scheduling noise and CPU wake-ups.
    """
    if cpu_threads <= 0:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already initialised (e.g. a forward pass ran first); intra-op still applies.
        pass


class ClipDetector:
    """Wrapper around YOLOv8 model for batched inference and video tracking."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: Union[int, str] = 0,
        imgsz: int = 640,
        tracker_config: str = "bytetrack.yaml",
    ):
        """Wrapper around a YOLO model.

        device: int/str index is passed to Ultralytics (which overwrites
        CUDA_VISIBLE_DEVICES for any non-empty value). Pass "" to let the
        process-pinned CUDA_VISIBLE_DEVICES decide — required for multi-GPU
        setups where each process owns one GPU. Pass "cpu" for CPU inference:
        Ultralytics then hides the GPUs in this process, so any ffmpeg child
        needing GPU decode must be spawned with an explicit CUDA_VISIBLE_DEVICES
        (see CameraStream) rather than inheriting the mutated environment.
        """
        self.device = device
        self.model_path = model_path
        self.imgsz = imgsz
        self.tracker_config = tracker_config
        # Ultralytics model load with target CUDA device
        if YOLO is None:
            raise RuntimeError("ultralytics is required for ClipDetector")
        self.model = YOLO(model_path)

    def detect_frame(self, frame, conf_threshold: float = 0.35) -> List[Detection]:
        """Stateless single-frame detection (no tracker state)."""
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=conf_threshold,
            verbose=False,
            imgsz=self.imgsz,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            class_name = r.names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = tuple(box.xyxy[0].tolist())
            xywh = tuple(box.xywh[0].tolist())
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    bbox_xywh=(xywh[0], xywh[1], xywh[2], xywh[3]),
                )
            )
        return detections

    def detect_thumbnail(self, image_path: str, conf_threshold: float = 0.35) -> List[Detection]:
        """Perform fast single-frame object detection on thumbnail image."""
        if not os.path.exists(image_path):
            return []

        results = self.model.predict(
            source=image_path,
            device=self.device,
            conf=conf_threshold,
            verbose=False,
            imgsz=self.imgsz,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            class_name = r.names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = tuple(box.xyxy[0].tolist())
            xywh = tuple(box.xywh[0].tolist())
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    bbox_xywh=(xywh[0], xywh[1], xywh[2], xywh[3]),
                )
            )
        return detections
    def track_frame(self, frame, frame_idx: int, conf_threshold: float = 0.30) -> List[TrackDetection]:
        """Track objects through a single frame using YOLOv8 + ByteTrack."""
        results = self.model.track(
            source=frame,
            persist=True,
            conf=conf_threshold,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
        )
        if not results:
            return []
        return _parse_track_boxes(results[0], frame_idx)


    def track_video(
        self,
        video_path: str,
        vid_stride: int = 15,
        conf_threshold: float = 0.30,
    ) -> List[FrameDetections]:
        """Track objects through video using YOLOv8 + ByteTrack across subsampled frames."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Stream track generator across frames
        track_stream = self.model.track(
            source=video_path,
            vid_stride=vid_stride,
            device=self.device,
            persist=True,
            conf=conf_threshold,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False,
            imgsz=self.imgsz,
        )

        frames_result: List[FrameDetections] = []
        frame_counter = 0

        for r in track_stream:
            frame_dets = _parse_track_boxes(r, frame_counter)
            frames_result.append(
                FrameDetections(
                    frame_idx=frame_counter,
                    detections=frame_dets,
                )
            )
            frame_counter += 1

        return frames_result
