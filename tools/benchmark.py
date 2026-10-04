#!/usr/bin/env python3
"""Detection & tracking benchmark against labelled objects in a camera clip.

Runs the *live* detection path (ClipDetector.track_frame → ByteTrack) over a
recorded clip for several model/resolution/threshold/ROI variants and reports
per-object detection rate, flicker gaps, track fragmentation and throughput.

Usage (inside the analysis-engine image, e.g. on the idle GPU):
  CUDA_VISIBLE_DEVICES=1 python3 tools/benchmark.py \
      --clip output/bench/foliage.mp4 \
      --gt output/bench/gt_foliage.json \
      --stride 2 --out output/bench/results.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.detector import ClipDetector, TrackDetection  # noqa: E402

# ---------------------------------------------------------------------------
# Variants: each is a full detection configuration to evaluate.
# `roi` crops [x1,y1,x2,y2] (normalized) and scales it before inference, then
# maps boxes back to full-frame coordinates.
# ---------------------------------------------------------------------------
VARIANTS: List[dict] = [
    {"name": "n@640 conf.30 (live baseline)", "model": "yolov8n.pt", "imgsz": 640, "conf": 0.30},
    {"name": "n@640 conf.20",                 "model": "yolov8n.pt", "imgsz": 640, "conf": 0.20},
    {"name": "n@960 conf.25",                 "model": "yolov8n.pt", "imgsz": 960, "conf": 0.25},
    {"name": "n@1280 conf.25",                "model": "yolov8n.pt", "imgsz": 1280, "conf": 0.25},
    {"name": "s@640 conf.25",                 "model": "yolov8s.pt", "imgsz": 640, "conf": 0.25},
    {"name": "s@960 conf.25",                 "model": "yolov8s.pt", "imgsz": 960, "conf": 0.25},
    {"name": "ROIx2 n@640 conf.25",           "model": "yolov8n.pt", "imgsz": 640, "conf": 0.25,
     "roi": [0.35, 0.0, 1.0, 0.45], "roi_scale": 2.0},
    {"name": "ROIx2 s@960 conf.25",           "model": "yolov8s.pt", "imgsz": 960, "conf": 0.25,
     "roi": [0.35, 0.0, 1.0, 0.45], "roi_scale": 2.0},
]


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_inside(box, region) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


class VariantRunner:
    """Runs one variant over the clip and collects per-object statistics."""

    def __init__(self, variant: dict, device: str = "", tracker: Optional[str] = None):
        self.variant = variant
        self.detector = ClipDetector(
            model_path=variant["model"],
            device=device,
            imgsz=variant["imgsz"],
            tracker_config=tracker or "bytetrack.yaml",
        )
        self.roi = variant.get("roi")
        self.roi_scale = float(variant.get("roi_scale", 1.0))

    def detect(self, frame: np.ndarray, frame_idx: int) -> List[TrackDetection]:
        """Track one frame; handles ROI crop + coordinate mapping."""
        h, w = frame.shape[:2]
        if self.roi:
            x1 = int(self.roi[0] * w); y1 = int(self.roi[1] * h)
            x2 = int(self.roi[2] * w); y2 = int(self.roi[3] * h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return []
            if self.roi_scale != 1.0:
                crop = cv2.resize(
                    crop, (int(crop.shape[1] * self.roi_scale), int(crop.shape[0] * self.roi_scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
            dets = self.detector.track_frame(crop, frame_idx, conf_threshold=self.variant["conf"])
            mapped: List[TrackDetection] = []
            for d in dets:
                bx1 = d.bbox_xyxy[0] / self.roi_scale + x1
                by1 = d.bbox_xyxy[1] / self.roi_scale + y1
                bx2 = d.bbox_xyxy[2] / self.roi_scale + x1
                by2 = d.bbox_xyxy[3] / self.roi_scale + y1
                mapped.append(TrackDetection(
                    track_id=d.track_id, class_name=d.class_name, confidence=d.confidence,
                    bbox_xyxy=(bx1, by1, bx2, by2),
                    bbox_xywh=((bx1 + bx2) / 2, (by1 + by2) / 2, bx2 - bx1, by2 - by1),
                    frame_idx=frame_idx,
                ))
            return mapped
        return self.detector.track_frame(frame, frame_idx, conf_threshold=self.variant["conf"])


def run_variant(
    variant: dict,
    clip: str,
    gt_objects: List[dict],
    stride: int,
    device: str = "",
    tracker: Optional[str] = None,
) -> dict:
    runner = VariantRunner(variant, device=device, tracker=tracker)

    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {clip}")

    stats = {
        o["id"]: {
            "hits_iou": 0, "hits_center": 0, "frames": 0, "ious": [], "class_hist": {},
            "ids": set(), "runs": [], "cur_run": 0, "centers": [],
            "prev_center": None, "max_jump": 0.0, "jumps": 0,
        }
        for o in gt_objects
    }

    frame_idx = 0
    sampled = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            h, w = frame.shape[:2]
            dets = runner.detect(frame, frame_idx)
            sampled += 1

            for obj in gt_objects:
                s = stats[obj["id"]]
                s["frames"] += 1
                gt_box = tuple(obj["box_norm"])
                best = None
                best_iou = 0.0
                for d in dets:
                    if d.class_name not in obj["accept_classes"]:
                        continue
                    dn = (d.bbox_xyxy[0] / w, d.bbox_xyxy[1] / h, d.bbox_xyxy[2] / w, d.bbox_xyxy[3] / h)
                    v = iou(dn, gt_box)
                    if v > best_iou:
                        best_iou = v
                        best = (d, dn)

                iou_hit = best_iou >= 0.15
                center_hit = best is not None and center_inside(best[1], gt_box)
                if iou_hit:
                    s["hits_iou"] += 1
                if center_hit:
                    s["hits_center"] += 1

                if iou_hit or center_hit:
                    s["cur_run"] += 1
                    if best is not None:
                        s["ious"].append(round(best_iou, 3))
                        s["class_hist"][best[0].class_name] = s["class_hist"].get(best[0].class_name, 0) + 1
                        s["ids"].add(best[0].track_id)
                        c = ((best[1][0] + best[1][2]) / 2.0, (best[1][1] + best[1][3]) / 2.0)
                        s["centers"].append(c)
                        if s["prev_center"] is not None:
                            jump = ((c[0] - s["prev_center"][0]) ** 2 + (c[1] - s["prev_center"][1]) ** 2) ** 0.5
                            s["max_jump"] = max(s["max_jump"], jump)
                            if jump > 0.15:
                                s["jumps"] += 1
                        s["prev_center"] = c
                else:
                    if s["cur_run"] > 0:
                        s["runs"].append(s["cur_run"])
                    s["cur_run"] = 0
        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0

    for obj in gt_objects:
        s = stats[obj["id"]]
        if s["cur_run"] > 0:
            s["runs"].append(s["cur_run"])
        total = max(1, s["frames"])
        centers = s["centers"]
        jitter = 0.0
        if len(centers) > 1:
            arr = np.array(centers)
            jitter = float(np.mean(np.std(arr, axis=0)))

        s["det_rate_iou"] = round(s["hits_iou"] / total, 3)
        s["det_rate_center"] = round(s["hits_center"] / total, 3)
        s["mean_iou"] = round(float(np.mean(s["ious"])), 3) if s["ious"] else 0.0
        s["unique_tracks"] = len(s["ids"])
        s["longest_run"] = max(s["runs"]) if s["runs"] else 0
        s["gap_count"] = total - s["hits_center"]
        s["center_jitter"] = round(jitter, 5)
        s["max_jump"] = round(s["max_jump"], 3)
        del s["ids"], s["centers"], s["ious"], s["prev_center"]

    return {
        "variant": variant["name"],
        "frames_sampled": sampled,
        "ms_per_frame": round(elapsed / max(1, sampled) * 1000, 1),
        "objects": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="", help='"" = respect CUDA_VISIBLE_DEVICES')
    ap.add_argument("--tracker", default=None)
    ap.add_argument("--model-dir", default="", help="prefix for model paths, e.g. /models/")
    ap.add_argument("--only", default="", help="substring filter on variant names")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    objects = gt["objects"]
    print(f"clip={args.clip} stride={args.stride} objects={[o['id'] for o in objects]}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print()

    results = []
    for variant in VARIANTS:
        if args.only and args.only.lower() not in variant["name"].lower():
            continue
        v = dict(variant)
        if args.model_dir:
            v["model"] = os.path.join(args.model_dir, v["model"])
        print(f"--- {v['name']} ---", flush=True)
        try:
            res = run_variant(v, args.clip, objects, args.stride, device=args.device, tracker=args.tracker)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue
        results.append(res)
        for oid, s in res["objects"].items():
            print(
                f"  {oid:16s} det_iou={s['det_rate_iou']:.2f} det_center={s['det_rate_center']:.2f} "
                f"meanIoU={s['mean_iou']:.2f} tracks={s['unique_tracks']:<3} longest_run={s['longest_run']:<4} "
                f"gaps={s['gap_count']:<4} jitter={s['center_jitter']:.4f} "
                f"max_jump={s['max_jump']:.2f} jumps={s['jumps']:<3} classes={s['class_hist']}"
            )
        print(f"  ms/frame={res['ms_per_frame']}  (capacity {1000.0 / max(0.1, res['ms_per_frame']):.0f} fps)")

    print()
    print("=== SUMMARY (higher det=better, fewer tracks/lower jitter=better) ===")
    header = f"{'variant':28s} " + " ".join(f"{o['id'][:12]:>12s}" for o in objects) + "   ms/f"
    print(header)
    for r in results:
        row = f"{r['variant'][:28]:28s} "
        for o in objects:
            s = r["objects"][o["id"]]
            row += f"{s['det_rate_center']:>12.2f}"
        row += f" {r['ms_per_frame']:>6.1f}"
        print(row)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"gt": gt, "results": results}, f, indent=2, default=str)
        print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
