"""Scenery anchors and vehicle fingerprinting for persistent vehicle personalization."""

from dataclasses import dataclass
import json
import logging
import math
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from engine.live.db import EventStore

logger = logging.getLogger(__name__)


@dataclass
class VehicleSignature:
    aspect_ratio: float
    color_name: str
    hsv_bins: List[float]  # 8 hue bins + 4 saturation bins, L2-normalized


@dataclass
class VehicleSlot:
    id: int
    camera: str
    name: str
    box: Tuple[float, float, float, float]  # [x1, y1, x2, y2] normalized [0..1]
    sig: VehicleSignature
    is_friendly: bool = True


def _classify_color_name(hsv_crop: np.ndarray) -> str:
    """Classify the dominant perceived color of a vehicle body crop in HSV space."""
    if hsv_crop.size == 0:
        return "unknown"

    h = hsv_crop[:, :, 0]
    s = hsv_crop[:, :, 1]
    v = hsv_crop[:, :, 2]

    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))

    # Achromatic checks (brightness / saturation dominant)
    if mean_v < 50:
        return "black"
    if mean_v > 185 and mean_s < 45:
        return "white"
    if mean_s < 45:
        return "silver/gray"

    # Chromatic check: consider pixels with decent saturation & brightness
    chromatic_mask = (s > 40) & (v > 45)
    if not np.any(chromatic_mask):
        return "silver/gray"

    h_chromatic = h[chromatic_mask]
    hist, _ = np.histogram(h_chromatic, bins=18, range=(0, 180))
    dominant_bin = int(np.argmax(hist))
    dom_h = dominant_bin * 10

    # OpenCV Hue [0..180)
    if dom_h < 15 or dom_h >= 165:
        return "red"
    if 15 <= dom_h < 30:
        return "orange"
    if 30 <= dom_h < 40:
        return "yellow"
    if 40 <= dom_h < 85:
        return "green"
    if 85 <= dom_h < 135:
        return "blue"
    return "purple"


def extract_vehicle_signature(
    frame: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
) -> VehicleSignature:
    """Extract lightweight appearance signature (aspect ratio + color + HSV bins)."""
    fh, fw = frame.shape[:2]
    bx1, by1, bx2, by2 = bbox_xyxy

    w = max(1.0, bx2 - bx1)
    h = max(1.0, by2 - by1)
    aspect_ratio = float(round(w / h, 2))

    # Sample central 70% of the bounding box to avoid ground/background bleed
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    crop_w = max(4, int(w * 0.70))
    crop_h = max(4, int(h * 0.70))

    x1 = max(0, min(fw - 1, int(cx - crop_w / 2.0)))
    y1 = max(0, min(fh - 1, int(cy - crop_h / 2.0)))
    x2 = max(x1 + 1, min(fw, int(cx + crop_w / 2.0)))
    y2 = max(y1 + 1, min(fh, int(cy + crop_h / 2.0)))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return VehicleSignature(aspect_ratio, "unknown", [0.0] * 12)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_name = _classify_color_name(hsv)

    # 8-bin Hue histogram on pixels with S > 30 and V > 35
    mask = (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 35)
    if np.any(mask):
        h_vals = hsv[:, :, 0][mask]
        h_hist, _ = np.histogram(h_vals, bins=8, range=(0, 180))
    else:
        h_hist = np.zeros(8, dtype=np.float32)

    # 4-bin Saturation histogram
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=4, range=(0, 256))

    combined = np.concatenate([h_hist.astype(np.float32), s_hist.astype(np.float32)])
    norm = float(np.linalg.norm(combined))
    if norm > 1e-6:
        combined /= norm

    return VehicleSignature(
        aspect_ratio=aspect_ratio,
        color_name=color_name,
        hsv_bins=[float(round(v, 4)) for v in combined.tolist()],
    )


def compare_signatures(sig1: VehicleSignature, sig2: VehicleSignature) -> float:
    """Compare two vehicle signatures, returning similarity score in [0.0, 1.0]."""
    # 1. Aspect ratio similarity
    max_ar = max(sig1.aspect_ratio, sig2.aspect_ratio, 0.1)
    diff_ar = abs(sig1.aspect_ratio - sig2.aspect_ratio)
    ar_sim = max(0.0, 1.0 - (diff_ar / max_ar))

    # 2. Histogram cosine similarity
    v1 = np.array(sig1.hsv_bins, dtype=np.float32)
    v2 = np.array(sig2.hsv_bins, dtype=np.float32)
    hist_sim = 0.0
    if len(v1) == len(v2) and len(v1) > 0:
        dot = float(np.dot(v1, v2))
        hist_sim = max(0.0, min(1.0, dot))

    # 3. Dominant color name match
    color_bonus = 0.15 if (sig1.color_name != "unknown" and sig1.color_name == sig2.color_name) else 0.0

    score = (0.55 * hist_sim) + (0.30 * ar_sim) + color_bonus
    return min(1.0, max(0.0, score))


def compute_box_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two bounding boxes (xyxy in normalized or pixel coordinates)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0.0 else 0.0


class SceneryManager:
    """Manages persistent vehicle parking slots and suppresses stationary vehicle false alarms."""

    def __init__(self, store: EventStore):
        self.store = store
        self.lock = threading.RLock()
        self.slots_by_camera: Dict[str, List[VehicleSlot]] = {}
        self.last_reload = 0.0
        self.reload()

    def reload(self) -> None:
        """Reload all registered vehicle slots from the database."""
        with self.lock:
            all_slots = self.store.list_vehicle_slots()
            grouped: Dict[str, List[VehicleSlot]] = {}
            for r in all_slots:
                cam = r["camera"]
                try:
                    box_list = json.loads(r["slot_box"])
                    box = (float(box_list[0]), float(box_list[1]), float(box_list[2]), float(box_list[3]))
                    sig_dict = json.loads(r["appearance_sig"])
                    sig = VehicleSignature(
                        aspect_ratio=float(sig_dict.get("aspect_ratio", 1.5)),
                        color_name=str(r["color_name"]),
                        hsv_bins=list(sig_dict.get("hsv_bins", [])),
                    )
                    slot = VehicleSlot(
                        id=r["id"],
                        camera=cam,
                        name=r["name"],
                        box=box,
                        sig=sig,
                        is_friendly=bool(r["is_friendly"]),
                    )
                    grouped.setdefault(cam, []).append(slot)
                except Exception as exc:
                    logger.warning(f"Failed to parse vehicle slot {r.get('id')}: {exc}")

            self.last_reload = time.time()
            self.slots_by_camera = grouped

    def get_slots(self, camera: str) -> List[VehicleSlot]:
        if time.time() - self.last_reload > 10.0:
            self.reload()
        with self.lock:
            return list(self.slots_by_camera.get(camera, []))

    def match_slot(
        self,
        camera: str,
        det_box_norm: Tuple[float, float, float, float],
        frame: Optional[np.ndarray] = None,
        det_box_px: Optional[Tuple[float, float, float, float]] = None,
        min_iou: float = 0.40,
    ) -> Optional[Tuple[VehicleSlot, float]]:
        """Match a detected vehicle against registered slots for this camera.

        Returns (matched_slot, confidence_score) if a spatial match exists.
        """
        slots = self.get_slots(camera)
        if not slots:
            return None

        det_cx = (det_box_norm[0] + det_box_norm[2]) / 2.0
        det_cy = (det_box_norm[1] + det_box_norm[3]) / 2.0

        best_slot: Optional[VehicleSlot] = None
        best_score = 0.0

        for slot in slots:
            iou = compute_box_iou(det_box_norm, slot.box)

            # Check if centroid is inside slot box
            inside = (
                slot.box[0] <= det_cx <= slot.box[2]
                and slot.box[1] <= det_cy <= slot.box[3]
            )

            if iou >= min_iou or inside:
                score = iou
                if frame is not None and det_box_px is not None and slot.sig.hsv_bins:
                    current_sig = extract_vehicle_signature(frame, det_box_px)
                    sig_sim = compare_signatures(current_sig, slot.sig)
                    score = (0.5 * iou) + (0.5 * sig_sim)

                if score > best_score:
                    best_score = score
                    best_slot = slot

        if best_slot is not None and best_score >= 0.35:
            return best_slot, best_score

        return None
