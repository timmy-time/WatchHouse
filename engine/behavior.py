"""Behavior classification for live CCTV analytics."""

from collections import deque
from dataclasses import dataclass, field
import math
import statistics
from typing import Deque, Dict, List, Optional, Set, Tuple

from engine.classifier import HIGH_VALUE_CLASSES, VEHICLE_CLASSES, extract_track_features
from engine.detector import TrackDetection

ANIMAL_CLASSES: Set[str] = HIGH_VALUE_CLASSES - {"person", "bicycle", "motorcycle"}


@dataclass
class Zone:
    name: str
    type: str  # "entry"
    polygon: List[Tuple[float, float]]


@dataclass
class TrackState:
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    first_center: Tuple[float, float]
    head: List[Tuple[float, TrackDetection]] = field(default_factory=list)
    window: Deque[Tuple[float, TrackDetection]] = field(default_factory=lambda: deque(maxlen=100))
    zone_seconds: Dict[str, float] = field(default_factory=dict)
    qualified: bool = False
    behaviors: Set[str] = field(default_factory=set)
    last_eval: float = 0.0
    notified_kinds: Set[str] = field(default_factory=set)


def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray casting point-in-polygon test on normalized coordinates."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def observe(
    state: TrackState,
    t: float,
    det: TrackDetection,
    zones: List[Zone],
    frame_w: int,
    frame_h: int,
    fps: int = 5,
) -> None:
    """Update track state with a new observation."""
    head_limit = fps * 5
    if len(state.head) < head_limit:
        state.head.append((t, det))
    state.window.append((t, det))

    previous_t = state.last_seen if state.last_seen > 0.0 else t
    dt = min(1.0, max(0.0, t - previous_t))

    x1, y1, x2, y2 = det.bbox_xyxy
    feet_x = ((x1 + x2) / 2.0) / frame_w
    feet_y = y2 / frame_h

    for zone in zones:
        if zone.type == "entry" and point_in_polygon(feet_x, feet_y, zone.polygon):
            state.zone_seconds[zone.name] = state.zone_seconds.get(zone.name, 0.0) + dt

    state.last_seen = t


def detect_behaviors(
    state: TrackState,
    zones: List[Zone],
    frame_w: int,
    frame_h: int,
    now: float,
    closed: bool = False,
) -> Set[str]:
    """Evaluate and accumulate behaviors for a track state."""
    new_behaviors: Set[str] = set()

    # Approaching (any class)
    for zone_name, secs in state.zone_seconds.items():
        if secs >= 1.0:
            new_behaviors.add("approaching")
            break

    # Person-specific behaviors
    if state.class_name == "person" and state.window:
        heights = [d.bbox_xyxy[3] - d.bbox_xyxy[1] for _, d in state.window]
        median_h = max(1.0, statistics.median(heights)) if heights else 1.0

        # Loitering: duration >= 45s and distance from first_center < 1.5 * median_h
        if (state.last_seen - state.first_seen) >= 45.0:
            cur_det = state.window[-1][1]
            cur_cx = (cur_det.bbox_xyxy[0] + cur_det.bbox_xyxy[2]) / 2.0
            cur_cy = (cur_det.bbox_xyxy[1] + cur_det.bbox_xyxy[3]) / 2.0
            dist = math.hypot(cur_cx - state.first_center[0], cur_cy - state.first_center[1])
            if dist < 1.5 * median_h:
                new_behaviors.add("loitering")

        # Running: window observations in the last 2.0s >= 5 and
        # (centroid path length / elapsed seconds / median bbox height) >= 1.6
        recent = [(t, d) for t, d in state.window if t >= now - 2.0]
        if len(recent) >= 5:
            elapsed = recent[-1][0] - recent[0][0]
            if elapsed > 0.0:
                centers = [
                    ((d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2.0, (d.bbox_xyxy[1] + d.bbox_xyxy[3]) / 2.0)
                    for _, d in recent
                ]
                path_len = sum(
                    math.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1])
                    for i in range(1, len(centers))
                )
                speed = (path_len / elapsed) / median_h
                if speed >= 1.6:
                    new_behaviors.add("running")

        # Passing by: closed only, not approaching, duration < 20s, |last_cx - first_cx| >= 0.3 * frame_w
        if closed:
            has_approaching = "approaching" in state.behaviors or "approaching" in new_behaviors
            duration = state.last_seen - state.first_seen
            if not has_approaching and duration < 20.0:
                last_det = state.window[-1][1]
                last_cx = (last_det.bbox_xyxy[0] + last_det.bbox_xyxy[2]) / 2.0
                if abs(last_cx - state.first_center[0]) >= 0.3 * frame_w:
                    new_behaviors.add("passing_by")

    # Vehicle-specific behaviors
    if state.class_name in VEHICLE_CLASSES and state.window:
        head_dets = [d for _, d in state.head]
        tail_dets = [d for t, d in state.window if t >= now - 5.0]

        if len(head_dets) >= 3 and len(tail_dets) >= 3:
            head_summary = extract_track_features(state.track_id, head_dets, len(head_dets))
            tail_summary = extract_track_features(state.track_id, tail_dets, len(tail_dets))

            if head_summary.is_moving_vehicle and tail_summary.is_stationary:
                new_behaviors.add("vehicle_arrived")
            if head_summary.is_stationary and tail_summary.is_moving_vehicle:
                new_behaviors.add("vehicle_departed")

        if closed:
            all_dets = [d for _, d in state.window]
            if len(all_dets) >= 3:
                full_summary = extract_track_features(state.track_id, all_dets, len(all_dets))
                arrived = "vehicle_arrived" in state.behaviors or "vehicle_arrived" in new_behaviors
                departed = "vehicle_departed" in state.behaviors or "vehicle_departed" in new_behaviors
                if full_summary.is_moving_vehicle and not arrived and not departed:
                    new_behaviors.add("vehicle_passing")

    # Animal present
    if state.class_name in ANIMAL_CLASSES and state.qualified:
        new_behaviors.add("animal_present")

    state.behaviors.update(new_behaviors)
    return set(state.behaviors)
