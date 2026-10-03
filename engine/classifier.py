"""Stationary vehicle suppression and event classification logic."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Set, Tuple

from engine.detector import FrameDetections, TrackDetection


class Verdict(str, Enum):
    KEEP = "KEEP"
    DISCARD = "DISCARD"
    ERROR = "ERROR"


HIGH_VALUE_CLASSES: Set[str] = {
    "person",
    "bicycle",
    "motorcycle",
    "dog",
    "cat",
    "bird",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

VEHICLE_CLASSES: Set[str] = {
    "car",
    "truck",
    "bus",
}


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union (IoU) for two xyxy bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


@dataclass
class TrackSummary:
    track_id: int
    class_name: str
    frame_count: int
    total_frames: int
    avg_confidence: float
    max_confidence: float
    min_iou_start: float
    max_displacement_px: float
    normalized_displacement: float
    frame_ratio: float
    is_stationary: bool
    is_high_value: bool
    is_moving_vehicle: bool
    classification_label: str
    first_bbox: Tuple[float, float, float, float]
    last_bbox: Tuple[float, float, float, float]


@dataclass
class EventDecision:
    verdict: Verdict
    primary_reason: str
    confidence: float
    high_value_detected: bool
    moving_vehicle_detected: bool
    stationary_vehicle_count: int
    active_track_summaries: List[TrackSummary] = field(default_factory=list)


def extract_track_features(
    track_id: int,
    detections: List[TrackDetection],
    total_video_frames: int,
) -> TrackSummary:
    """Extract kinematic and spatial stability features from a track trajectory."""
    sorted_dets = sorted(detections, key=lambda d: d.frame_idx)
    class_name = sorted_dets[0].class_name
    confidences = [d.confidence for d in sorted_dets]
    avg_conf = sum(confidences) / len(confidences)
    max_conf = max(confidences)

    bboxes = [d.bbox_xyxy for d in sorted_dets]
    first_bbox = bboxes[0]
    last_bbox = bboxes[-1]

    # Initial dimension for normalization
    w0 = max(1.0, first_bbox[2] - first_bbox[0])
    h0 = max(1.0, first_bbox[3] - first_bbox[1])
    scale_norm = math.sqrt(w0 * h0)

    # IoU relative to first bounding box
    ious_from_start = [compute_iou(first_bbox, b) for b in bboxes]
    min_iou_start = min(ious_from_start)

    # Centroids
    centroids = [
        ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        for b in bboxes
    ]

    # Max pairwise displacement
    max_disp_px = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            dist = math.hypot(dx, dy)
            if dist > max_disp_px:
                max_disp_px = dist

    norm_disp = max_disp_px / scale_norm
    frame_ratio = len(sorted_dets) / max(1, total_video_frames)

    is_high_value = class_name in HIGH_VALUE_CLASSES
    is_vehicle = class_name in VEHICLE_CLASSES

    # Stationary vehicle criteria:
    # 1. High overlap with initial position (min_iou_start >= 0.65)
    # 2. Low displacement relative to object size (norm_disp <= 0.25)
    is_stationary = False
    is_moving_vehicle = False
    classification_label = "other"

    if is_vehicle:
        if min_iou_start >= 0.65 and norm_disp <= 0.25:
            is_stationary = True
            classification_label = "stationary_vehicle"
        elif norm_disp > 0.40 and (min_iou_start < 0.60 or max_disp_px > 60.0) and len(sorted_dets) >= 3:
            is_moving_vehicle = True
            classification_label = "moving_vehicle"
        else:
            # Ambiguous/minor jitter: if low displacement, treat as stationary
            if norm_disp <= 0.30:
                is_stationary = True
                classification_label = "stationary_vehicle"
            else:
                is_moving_vehicle = True
                classification_label = "moving_vehicle"
    elif is_high_value:
        classification_label = f"high_value_{class_name}"

    return TrackSummary(
        track_id=track_id,
        class_name=class_name,
        frame_count=len(sorted_dets),
        total_frames=total_video_frames,
        avg_confidence=round(avg_conf, 3),
        max_confidence=round(max_conf, 3),
        min_iou_start=round(min_iou_start, 3),
        max_displacement_px=round(max_disp_px, 1),
        normalized_displacement=round(norm_disp, 3),
        frame_ratio=round(frame_ratio, 3),
        is_stationary=is_stationary,
        is_high_value=is_high_value,
        is_moving_vehicle=is_moving_vehicle,
        classification_label=classification_label,
        first_bbox=first_bbox,
        last_bbox=last_bbox,
    )


def classify_clip_events(
    frame_detections: List[FrameDetections],
) -> EventDecision:
    """Analyze all tracks in a clip and produce the final KEEP or DISCARD verdict."""
    total_frames = len(frame_detections)
    if total_frames == 0:
        return EventDecision(
            verdict=Verdict.DISCARD,
            primary_reason="no_frames_recorded",
            confidence=1.0,
            high_value_detected=False,
            moving_vehicle_detected=False,
            stationary_vehicle_count=0,
        )

    # Group detections by track_id
    tracks_map: Dict[int, List[TrackDetection]] = {}
    untracked_high_value: List[TrackDetection] = []

    for fd in frame_detections:
        for det in fd.detections:
            if det.track_id > 0:
                tracks_map.setdefault(det.track_id, []).append(det)
            else:
                if det.class_name in HIGH_VALUE_CLASSES:
                    untracked_high_value.append(det)

    # Synthesize TrackSummary for each track
    summaries: List[TrackSummary] = []
    for tid, dets in tracks_map.items():
        summary = extract_track_features(tid, dets, total_frames)
        summaries.append(summary)

    # 1. Check for High-Value Objects (person, dog, cat, bike, etc.)
    # Confirm if >= 2 frames or 1 frame with high confidence
    for s in summaries:
        if s.is_high_value:
            if s.frame_count >= 2 and s.avg_confidence >= 0.35:
                return EventDecision(
                    verdict=Verdict.KEEP,
                    primary_reason=f"{s.class_name}_detected",
                    confidence=s.max_confidence,
                    high_value_detected=True,
                    moving_vehicle_detected=False,
                    stationary_vehicle_count=sum(1 for x in summaries if x.is_stationary),
                    active_track_summaries=summaries,
                )
            elif s.frame_count == 1 and s.max_confidence >= 0.65:
                return EventDecision(
                    verdict=Verdict.KEEP,
                    primary_reason=f"{s.class_name}_detected",
                    confidence=s.max_confidence,
                    high_value_detected=True,
                    moving_vehicle_detected=False,
                    stationary_vehicle_count=sum(1 for x in summaries if x.is_stationary),
                    active_track_summaries=summaries,
                )

    # Also check untracked high value detections if ByteTrack missed assigning a track ID
    if untracked_high_value:
        high_conf_untracked = [d for d in untracked_high_value if d.confidence >= 0.55]
        if len(high_conf_untracked) >= 2 or any(d.confidence >= 0.70 for d in untracked_high_value):
            top_det = max(untracked_high_value, key=lambda d: d.confidence)
            return EventDecision(
                verdict=Verdict.KEEP,
                primary_reason=f"{top_det.class_name}_detected",
                confidence=round(top_det.confidence, 3),
                high_value_detected=True,
                moving_vehicle_detected=False,
                stationary_vehicle_count=sum(1 for x in summaries if x.is_stationary),
                active_track_summaries=summaries,
            )

    # 2. Check for Moving Vehicles
    moving_vehicles = [s for s in summaries if s.is_moving_vehicle]
    if moving_vehicles:
        top_moving = max(moving_vehicles, key=lambda s: s.normalized_displacement)
        return EventDecision(
            verdict=Verdict.KEEP,
            primary_reason="moving_vehicle",
            confidence=top_moving.avg_confidence,
            high_value_detected=False,
            moving_vehicle_detected=True,
            stationary_vehicle_count=sum(1 for x in summaries if x.is_stationary),
            active_track_summaries=summaries,
        )

    # 3. Check for Stationary Vehicles Only
    stationary_vehicles = [s for s in summaries if s.is_stationary]
    if stationary_vehicles and not moving_vehicles:
        return EventDecision(
            verdict=Verdict.DISCARD,
            primary_reason="stationary_vehicles_only",
            confidence=max(s.avg_confidence for s in stationary_vehicles),
            high_value_detected=False,
            moving_vehicle_detected=False,
            stationary_vehicle_count=len(stationary_vehicles),
            active_track_summaries=summaries,
        )

    # 4. If any vehicles existed but didn't meet moving criteria, classify as stationary/nuisance
    all_vehicles = [s for s in summaries if s.class_name in VEHICLE_CLASSES]
    if all_vehicles:
        return EventDecision(
            verdict=Verdict.DISCARD,
            primary_reason="stationary_vehicles_only",
            confidence=max(s.avg_confidence for s in all_vehicles),
            high_value_detected=False,
            moving_vehicle_detected=False,
            stationary_vehicle_count=len(all_vehicles),
            active_track_summaries=summaries,
        )

    # 5. Empty / No Objects Detected
    return EventDecision(
        verdict=Verdict.DISCARD,
        primary_reason="no_objects_detected",
        confidence=1.0,
        high_value_detected=False,
        moving_vehicle_detected=False,
        stationary_vehicle_count=0,
        active_track_summaries=summaries,
    )


LIVE_MIN_HIGH_VALUE_FRAMES = 3
LIVE_MIN_HIGH_VALUE_CONF = 0.40
LIVE_MIN_VEHICLE_FRAMES = 5


def track_qualifies_live(summary: TrackSummary) -> bool:
    if summary.is_high_value:
        # Inanimate stationary object filter (e.g. engine blocks, tire piles, garden ornaments):
        # If a person track is frozen static (IoU >= 0.80, displacement <= 0.08) with marginal confidence (< 0.60),
        # it is an inanimate false positive, not a living person.
        if summary.class_name == "person":
            if summary.frame_count >= 4 and summary.min_iou_start >= 0.80 and summary.normalized_displacement <= 0.08:
                if summary.avg_confidence < 0.60:
                    return False
        return summary.frame_count >= LIVE_MIN_HIGH_VALUE_FRAMES and summary.avg_confidence >= LIVE_MIN_HIGH_VALUE_CONF
    return summary.is_moving_vehicle and summary.frame_count >= LIVE_MIN_VEHICLE_FRAMES
