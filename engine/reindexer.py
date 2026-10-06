"""Historical clip and event video face re-indexing engine."""

from datetime import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from engine.faces import FaceEngine, FaceGallery, FaceCandidate
from engine.live.db import EventStore
from engine.reporter import parse_clip_metadata

logger = logging.getLogger(__name__)


class FaceReindexer:
    """Scans historical event clips or archive videos to extract, embed, and index faces.

    Extracts bounding boxes for every person in a frame, runs YuNet face detection on the
    person crop, computes SFace embeddings, matches against known identities or clusters
    into unknown faces for human review, and updates event metadata.
    """

    def __init__(
        self,
        store: EventStore,
        output_dir: str,
        face_engine: Optional[FaceEngine] = None,
        gallery: Optional[FaceGallery] = None,
        detector: Optional[Any] = None,
        model_dir: str = "/opt/models",
    ):
        self.store = store
        self.output_dir = output_dir

        if face_engine is None:
            # Fallback to local model paths if /opt/models is not populated
            paths_to_check = [model_dir, "models", "/app/models"]
            valid_dir = next(
                (
                    d for d in paths_to_check
                    if os.path.exists(os.path.join(d, "face_detection_yunet_2023mar.onnx"))
                ),
                model_dir,
            )
            try:
                face_engine = FaceEngine(model_dir=valid_dir)
            except Exception as exc:
                logger.warning(f"FaceEngine could not be initialized: {exc}")
                face_engine = None

        self.face_engine = face_engine

        if gallery is None and self.face_engine is not None:
            gallery = FaceGallery(self.store)
        self.gallery = gallery

        if detector is None:
            try:
                from engine.detector import ClipDetector
                detector = ClipDetector(model_path="yolov8n.pt", imgsz=640)
            except Exception as exc:
                logger.warning(f"ClipDetector could not be initialized: {exc}")
                detector = None

        self.detector = detector

    def reindex_clip(
        self,
        clip_path: str,
        event_id: Optional[int] = None,
        camera: Optional[str] = None,
        timestamp: Optional[float] = None,
        stride: int = 5,
        min_quality: float = 0.40,
    ) -> Dict[str, Any]:
        """Scan a video clip, detect persons, extract faces, and register them in the gallery."""
        if not os.path.exists(clip_path):
            return {"error": f"Clip not found: {clip_path}", "faces_indexed": 0}

        if self.face_engine is None:
            return {"error": "Face engine not available", "faces_indexed": 0}

        meta = parse_clip_metadata(clip_path)
        camera_name = camera or meta.get("camera", "Unknown")
        clip_ts = timestamp or time.time()

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return {"error": f"Failed to open video: {clip_path}", "faces_indexed": 0}

        frame_idx = 0
        faces_indexed = 0
        matched_identities = set()
        new_clusters = set()

        crops_dir = os.path.join(self.output_dir, "faces/crops")
        context_dir = os.path.join(self.output_dir, "faces/context")
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(context_dir, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % stride == 0:
                    frame_h, frame_w = frame.shape[:2]

                    # 1. Detect objects if detector available
                    person_boxes: List[Tuple[float, float, float, float]] = []
                    if self.detector is not None:
                        dets = self.detector.detect_frame(frame, conf_threshold=0.25)
                        person_dets = [d for d in dets if d.class_name == "person"]
                        person_dets.sort(
                            key=lambda d: (d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]),
                            reverse=True,
                        )
                        person_boxes = [d.bbox_xyxy for d in person_dets[:6]]
                    else:
                        # Fallback: scan whole frame if person detector unavailable
                        person_boxes = [(0.0, 0.0, float(frame_w), float(frame_h))]

                    # 2. Extract faces from each person box
                    for box in person_boxes:
                        cands = self.face_engine.detect_in_person(frame, box)
                        if not cands:
                            continue

                        cands.sort(key=lambda c: c.quality, reverse=True)
                        cand = cands[0]
                        if cand.quality < min_quality:
                            continue

                        aligned, feat = self.face_engine.embed(frame, cand)
                        with self.gallery.lock:
                            matched_id, cluster_id, match_score = self.gallery.assign(feat)

                            # Save face crop image
                            face_uid = f"{int(clip_ts)}_{frame_idx}_{faces_indexed}"
                            crop_filename = f"{face_uid}.jpg"
                            ctx_filename = f"{face_uid}_ctx.jpg"

                            crop_abs = os.path.join(crops_dir, crop_filename)
                            ctx_abs = os.path.join(context_dir, ctx_filename)

                            cv2.imwrite(crop_abs, aligned)
                            # Save context crop around face
                            fx1, fy1, fx2, fy2 = [int(v) for v in cand.box]
                            pad = int(max(fx2 - fx1, fy2 - fy1) * 0.8)
                            cx1 = max(0, fx1 - pad)
                            cy1 = max(0, fy1 - pad)
                            cx2 = min(frame_w, fx2 + pad)
                            cy2 = min(frame_h, fy2 + pad)
                            cv2.imwrite(ctx_abs, frame[cy1:cy2, cx1:cx2])

                            crop_rel = f"faces/crops/{crop_filename}"
                            ctx_rel = f"faces/context/{ctx_filename}"
                            fw = cand.box[2] - cand.box[0]

                            face_id = self.store.insert_face(
                                event_id=event_id,
                                camera=camera_name,
                                track_id=None,
                                captured_at=clip_ts,
                                crop_path=crop_rel,
                                context_path=ctx_rel,
                                embedding=feat.tobytes(),
                                quality=cand.quality,
                                det_score=cand.score,
                                face_width=fw,
                                identity_id=matched_id,
                                cluster_id=cluster_id,
                                match_score=match_score,
                            )
                            faces_indexed += 1
                            if matched_id:
                                matched_identities.add(matched_id)
                            elif cluster_id:
                                new_clusters.add(cluster_id)
                frame_idx += 1
        finally:
            cap.release()

        return {
            "clip_path": clip_path,
            "event_id": event_id,
            "frames_scanned": frame_idx,
            "faces_indexed": faces_indexed,
            "identities_matched": list(matched_identities),
            "clusters_assigned": list(new_clusters),
        }

    def reindex_events_directory(
        self,
        events_root: Optional[str] = None,
        camera: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Scan event directory for mp4 files and index faces into the gallery."""
        root = events_root or os.path.join(self.output_dir, "events")
        if not os.path.exists(root):
            return {"error": f"Events directory not found: {root}", "processed": 0}

        processed = 0
        total_faces = 0
        results = []

        for dirpath, _, filenames in os.walk(root):
            for f in sorted(filenames):
                if not f.endswith(".mp4"):
                    continue

                full_path = os.path.join(dirpath, f)
                # Try to extract event_id from filename like 350.mp4
                base_name = os.path.splitext(f)[0]
                event_id = int(base_name) if base_name.isdigit() else None

                # Derive camera from path (e.g. output/events/Driveway/20261006/350.mp4)
                parts = dirpath.replace("\\", "/").split("/")
                cam_name = camera
                if not cam_name and len(parts) >= 2:
                    # Look for directory under events/
                    try:
                        ev_idx = parts.index("events")
                        if ev_idx + 1 < len(parts):
                            cam_name = parts[ev_idx + 1]
                    except ValueError:
                        pass

                res = self.reindex_clip(
                    full_path,
                    event_id=event_id,
                    camera=cam_name or "Unknown",
                )
                if not res.get("error"):
                    processed += 1
                    total_faces += res.get("faces_indexed", 0)
                    results.append(res)

                if processed >= limit:
                    break
            if processed >= limit:
                break

        return {
            "processed_clips": processed,
            "total_faces_indexed": total_faces,
            "details": results,
        }
