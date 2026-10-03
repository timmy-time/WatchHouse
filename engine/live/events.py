"""Live event lifecycle, qualification, behavior tracking, and face recognition."""

from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from engine.behavior import (
    ANIMAL_CLASSES,
    TrackState,
    Zone,
    detect_behaviors,
    observe,
)
from engine.classifier import (
    HIGH_VALUE_CLASSES,
    VEHICLE_CLASSES,
    extract_track_features,
    track_qualifies_live,
)
from engine.detector import TrackDetection
from engine.faces import FaceEngine, FaceGallery
from engine.live.config import CameraConfig, LiveConfig
from engine.live.db import EventStore
from engine.live.notify import Notifier

logger = logging.getLogger(__name__)

CLASS_PRIORITY = {
    "person": 3,
    "animal": 2,
    "vehicle": 1,
}


def _classify_category(class_name: str) -> str:
    if class_name == "person":
        return "person"
    if class_name in ANIMAL_CLASSES:
        return "animal"
    return "vehicle"


def _draw_annotation(
    frame: np.ndarray,
    dets: List[TrackDetection],
    zones: List[Zone],
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Draw bounding boxes and zone polygons onto a copy of the frame."""
    canvas = frame.copy()

    # Draw zones in yellow
    for zone in zones:
        if len(zone.polygon) >= 3:
            pts = np.array(
                [[int(pt[0] * frame_w), int(pt[1] * frame_h)] for pt in zone.polygon],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(
                canvas,
                zone.name,
                (pts[0][0], max(20, pts[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

    # Draw detections
    for d in dets:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        if d.class_name in HIGH_VALUE_CLASSES:
            color = (0, 255, 0)  # Green
        elif d.class_name in VEHICLE_CLASSES:
            color = (0, 0, 255)  # Red for vehicles
        else:
            color = (128, 128, 128)  # Gray

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} #{d.track_id} {d.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return canvas


class EventManager:
    """Stateful event lifecycle manager for a single camera stream."""

    def __init__(
        self,
        cam: CameraConfig,
        cfg: LiveConfig,
        store: EventStore,
        notifier: Notifier,
        output_dir: str,
        face_engine: Optional[FaceEngine] = None,
        gallery: Optional[FaceGallery] = None,
        on_closed: Optional[Callable[[int, float, float], None]] = None,
    ):
        self.cam = cam
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.output_dir = output_dir
        self.face_engine = face_engine
        self.gallery = gallery
        self.on_closed = on_closed

        self.tracks: Dict[int, TrackState] = {}
        self.event_tracks: Dict[int, TrackState] = {}

        self.open_event_id: Optional[int] = None
        self.open_event_started: float = 0.0
        self.open_event_primary_class: str = "vehicle"
        self.open_event_max_conf: float = 0.0
        self.open_event_behaviors: Set[str] = set()
        self.event_notified_kinds: Set[str] = set()

        self.last_preview_time: float = 0.0
        self.face_track_last_embed: Dict[int, float] = {}
        self.face_track_best_quality: Dict[int, float] = {}
        self.last_event_ended_at: float = 0.0

    def process(
        self,
        frame: Optional[np.ndarray],
        dets: List[TrackDetection],
        now: float,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Process detections from one frame, update tracks, manage events."""
        # 1. Update / create TrackStates for relevant classes
        relevant_dets = [
            d for d in dets
            if d.track_id > 0 and (d.class_name in HIGH_VALUE_CLASSES or d.class_name in VEHICLE_CLASSES)
        ]

        for d in relevant_dets:
            x1, y1, x2, y2 = d.bbox_xyxy
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            if d.track_id not in self.tracks:
                state = TrackState(
                    track_id=d.track_id,
                    class_name=d.class_name,
                    first_seen=now,
                    last_seen=now,
                    first_center=center,
                )
                self.tracks[d.track_id] = state
            else:
                state = self.tracks[d.track_id]

            observe(state, now, d, self.cam.zones, frame_w, frame_h, fps=self.cfg.analysis.fps)

        # 2. Track qualification
        for tid, state in list(self.tracks.items()):
            if not state.qualified:
                if state.class_name in HIGH_VALUE_CLASSES:
                    # Check every frame
                    window_dets = [d for _, d in state.window]
                    summary = extract_track_features(state.track_id, window_dets, len(window_dets))
                    if track_qualifies_live(summary):
                        state.qualified = True
                elif state.class_name in VEHICLE_CLASSES:
                    # Check every 1.0s
                    if now - state.last_eval >= 1.0:
                        state.last_eval = now
                        window_dets = [d for _, d in state.window]
                        summary = extract_track_features(state.track_id, window_dets, len(window_dets))
                        if track_qualifies_live(summary):
                            state.qualified = True

        qualified_active = [s for s in self.tracks.values() if s.qualified]

        # 3. Event open & primary class upgrade
        if self.open_event_id is None and qualified_active:
            earliest_started = max(min(s.first_seen for s in qualified_active), self.last_event_ended_at)
            # Determine initial primary class
            best_cat = "vehicle"
            best_score = 0
            for s in qualified_active:
                cat = _classify_category(s.class_name)
                score = CLASS_PRIORITY.get(cat, 0)
                if score > best_score:
                    best_score = score
                    best_cat = cat

            self.open_event_started = earliest_started
            self.open_event_primary_class = best_cat
            self.open_event_max_conf = max((d.confidence for d in relevant_dets), default=0.5)
            self.open_event_behaviors = set()
            self.event_notified_kinds = set()
            self.event_tracks = {}

            self.open_event_id = self.store.create_event(
                camera=self.cam.name,
                started_at=earliest_started,
                status="open",
                primary_class=best_cat,
                behaviors="[]",
                max_confidence=self.open_event_max_conf,
            )

            for s in qualified_active:
                self.event_tracks[s.track_id] = s

            # Thumbnail generation
            thumb_rel = None
            if frame is not None:
                date_str = datetime.utcfromtimestamp(now).strftime("%Y%m%d")
                thumb_rel = f"events/{self.cam.slug}/{date_str}/{self.open_event_id}.jpg"
                full_thumb = os.path.join(self.output_dir, thumb_rel)
                os.makedirs(os.path.dirname(full_thumb), exist_ok=True)
                ann = _draw_annotation(frame, relevant_dets, self.cam.zones, frame_w, frame_h)
                resized_thumb = cv2.resize(ann, (1280, 720))
                cv2.imwrite(full_thumb, resized_thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                self.store.update_event(self.open_event_id, thumb_path=thumb_rel)

            # 4. Notifications on open
            if best_cat == "person":
                self.event_notified_kinds.add("person")
                self.notifier.notify(
                    event_id=self.open_event_id,
                    camera=self.cam.name,
                    kind="person",
                    title=f"Person at {self.cam.name}",
                    body=f"Detected person at {self.cam.name}",
                    image_rel=thumb_rel,
                )
            elif best_cat == "animal":
                # Find animal class
                animal_cls = next((s.class_name for s in qualified_active if s.class_name in ANIMAL_CLASSES), "animal")
                if animal_cls in self.cfg.notifications.notify_classes:
                    self.event_notified_kinds.add("animal")
                    self.notifier.notify(
                        event_id=self.open_event_id,
                        camera=self.cam.name,
                        kind="animal",
                        title=f"{animal_cls.capitalize()} at {self.cam.name}",
                        body=f"Detected {animal_cls} at {self.cam.name}",
                        image_rel=thumb_rel,
                    )
            # Vehicles not notified on open

        elif self.open_event_id is not None:
            # Check for primary class upgrade
            for s in qualified_active:
                self.event_tracks[s.track_id] = s
                cat = _classify_category(s.class_name)
                if CLASS_PRIORITY.get(cat, 0) > CLASS_PRIORITY.get(self.open_event_primary_class, 0):
                    self.open_event_primary_class = cat
                    self.store.update_event(self.open_event_id, primary_class=cat)

            if relevant_dets:
                m_conf = max(d.confidence for d in relevant_dets)
                if m_conf > self.open_event_max_conf:
                    self.open_event_max_conf = m_conf
                    self.store.update_event(self.open_event_id, max_confidence=m_conf)

        # 5. Behavior evaluation every ~1.0s per active track
        if self.open_event_id is not None:
            updated_behaviors = False
            for s in list(self.tracks.values()):
                if now - s.last_eval >= 1.0 or not s.behaviors:
                    b_set = detect_behaviors(s, self.cam.zones, frame_w, frame_h, now, closed=False)
                    for b in b_set:
                        if b not in self.open_event_behaviors:
                            self.open_event_behaviors.add(b)
                            updated_behaviors = True

                            # Notifications per behavior
                            if b == "approaching" and "approaching" not in self.event_notified_kinds:
                                self.event_notified_kinds.add("approaching")
                                self.notifier.notify(
                                    event_id=self.open_event_id,
                                    camera=self.cam.name,
                                    kind="approaching",
                                    title=f"{s.class_name.capitalize()} approaching {self.cam.name}",
                                    body=f"{s.class_name.capitalize()} approaching {self.cam.name}",
                                )
                            elif b == "loitering" and "loitering" not in self.event_notified_kinds:
                                self.event_notified_kinds.add("loitering")
                                self.notifier.notify(
                                    event_id=self.open_event_id,
                                    camera=self.cam.name,
                                    kind="loitering",
                                    title=f"Loitering at {self.cam.name}",
                                    body=f"Person loitering at {self.cam.name}",
                                )
                            elif b == "running" and "running" not in self.event_notified_kinds:
                                self.event_notified_kinds.add("running")
                                self.notifier.notify(
                                    event_id=self.open_event_id,
                                    camera=self.cam.name,
                                    kind="running",
                                    title=f"Running at {self.cam.name}",
                                    body=f"Person running at {self.cam.name}",
                                )
                            elif b == "vehicle_arrived" and "vehicle" not in self.event_notified_kinds:
                                self.event_notified_kinds.add("vehicle")
                                self.notifier.notify(
                                    event_id=self.open_event_id,
                                    camera=self.cam.name,
                                    kind="vehicle",
                                    title=f"Vehicle arrived at {self.cam.name}",
                                    body=f"Vehicle arrived at {self.cam.name}",
                                )
                            elif b == "vehicle_departed" and "vehicle" not in self.event_notified_kinds:
                                self.event_notified_kinds.add("vehicle")
                                self.notifier.notify(
                                    event_id=self.open_event_id,
                                    camera=self.cam.name,
                                    kind="vehicle",
                                    title=f"Vehicle left {self.cam.name}",
                                    body=f"Vehicle left {self.cam.name}",
                                )

            if updated_behaviors:
                self.store.update_event(
                    self.open_event_id,
                    behaviors=json.dumps(sorted(self.open_event_behaviors)),
                )

        # 6. Face detection & recognition (only when face_engine is available and frame provided)
        if (
            self.face_engine is not None
            and self.gallery is not None
            and frame is not None
            and self.open_event_id is not None
        ):
            person_dets = [
                d for d in relevant_dets
                if d.class_name == "person"
                and d.track_id in self.tracks
                and self.tracks[d.track_id].qualified
            ]
            person_dets.sort(
                key=lambda d: (d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]),
                reverse=True,
            )
            for det in person_dets[:2]:
                tid = det.track_id
                if now - self.face_track_last_embed.get(tid, 0.0) < 1.0:
                    continue

                cands = self.face_engine.detect_in_person(frame, det.bbox_xyxy)
                if not cands:
                    continue

                cands.sort(key=lambda c: c.quality, reverse=True)
                cand = cands[0]
                best_q = self.face_track_best_quality.get(tid, 0.0)
                if cand.quality > 1.10 * best_q:
                    self.face_track_last_embed[tid] = now
                    self.face_track_best_quality[tid] = cand.quality
                    aligned, feat = self.face_engine.embed(frame, cand)

                    with self.gallery.lock:
                        matched_id, cluster_id, match_score = self.gallery.assign(feat)
                        date_str = datetime.utcfromtimestamp(now).strftime("%Y%m%d")
                        existing = [
                            f for f in self.store.faces_for_event(self.open_event_id)
                            if f["track_id"] == tid
                        ]

                        fw = cand.box[2] - cand.box[0]
                        if existing:
                            face_row = existing[0]
                            face_id = face_row["id"]
                            crop_rel = face_row["crop_path"]
                            ctx_rel = face_row["context_path"]
                            self.store.update_face(
                                face_id,
                                captured_at=now,
                                embedding=feat.tobytes(),
                                quality=cand.quality,
                                det_score=cand.score,
                                face_width=fw,
                                identity_id=matched_id,
                                cluster_id=cluster_id,
                                match_score=match_score,
                            )
                        else:
                            face_id = self.store.insert_face(
                                event_id=self.open_event_id,
                                camera=self.cam.name,
                                track_id=tid,
                                captured_at=now,
                                crop_path="",
                                context_path="",
                                embedding=feat.tobytes(),
                                quality=cand.quality,
                                det_score=cand.score,
                                face_width=fw,
                                identity_id=matched_id,
                                cluster_id=cluster_id,
                                match_score=match_score,
                            )
                            crop_rel = f"faces/{date_str}/{face_id}.jpg"
                            ctx_rel = f"faces/{date_str}/{face_id}_ctx.jpg"
                            self.store.update_face(face_id, crop_path=crop_rel, context_path=ctx_rel)

                        # Write crops to disk
                        full_crop = os.path.join(self.output_dir, crop_rel)
                        os.makedirs(os.path.dirname(full_crop), exist_ok=True)
                        cv2.imwrite(full_crop, aligned)

                        # Context crop: face box expanded 2x, max side 256
                        fcx = (cand.box[0] + cand.box[2]) / 2.0
                        fcy = (cand.box[1] + cand.box[3]) / 2.0
                        fcw = (cand.box[2] - cand.box[0]) * 2.0
                        fch = (cand.box[3] - cand.box[1]) * 2.0
                        ctx_x1 = max(0, int(fcx - fcw / 2.0))
                        ctx_y1 = max(0, int(fcy - fch / 2.0))
                        ctx_x2 = min(frame_w, int(fcx + fcw / 2.0))
                        ctx_y2 = min(frame_h, int(fcy + fch / 2.0))
                        ctx_img = frame[ctx_y1:ctx_y2, ctx_x1:ctx_x2]

                        if ctx_img.size > 0:
                            ms = max(ctx_img.shape[0], ctx_img.shape[1])
                            if ms > 256:
                                s = 256.0 / ms
                                ctx_img = cv2.resize(
                                    ctx_img,
                                    (max(1, int(ctx_img.shape[1] * s)), max(1, int(ctx_img.shape[0] * s))),
                                )
                            full_ctx = os.path.join(self.output_dir, ctx_rel)
                            os.makedirs(os.path.dirname(full_ctx), exist_ok=True)
                            cv2.imwrite(full_ctx, ctx_img)

                        self.gallery.add_assigned(matched_id, cluster_id, feat)

                        # Face notification
                        if matched_id is not None:
                            kind = f"identity:{matched_id}"
                            if kind not in self.event_notified_kinds:
                                self.event_notified_kinds.add(kind)
                                id_name = "Known person"
                                for ident in self.store.list_identities():
                                    if ident["id"] == matched_id:
                                        id_name = ident["name"]
                                        break
                                self.notifier.notify(
                                    event_id=self.open_event_id,
                                    camera=self.cam.name,
                                    kind=kind,
                                    title=f"{id_name} at {self.cam.name}",
                                    body=f"Identified {id_name} at {self.cam.name}",
                                    image_rel=ctx_rel,
                                )
                        elif cand.quality >= 0.50:
                            kind = "unknown_face"
                            if kind not in self.event_notified_kinds:
                                self.event_notified_kinds.add(kind)
                                self.notifier.notify(
                                    event_id=self.open_event_id,
                                    camera=self.cam.name,
                                    kind=kind,
                                    title=f"Unknown person at {self.cam.name}",
                                    body=f"Detected unknown face at {self.cam.name}",
                                    image_rel=ctx_rel,
                                )

        # 7. Track expiry: not seen for 6.0s
        for tid, s in list(self.tracks.items()):
            if now - s.last_seen >= 6.0:
                final_b = detect_behaviors(s, self.cam.zones, frame_w, frame_h, now, closed=True)
                if self.open_event_id is not None and final_b:
                    self.open_event_behaviors.update(final_b)
                    self.store.update_event(
                        self.open_event_id,
                        behaviors=json.dumps(sorted(self.open_event_behaviors)),
                    )
                self.tracks.pop(tid, None)

        # 8. Close event
        if self.open_event_id is not None:
            force_close = (now - self.open_event_started) >= self.cfg.analysis.max_event_seconds
            all_inactive = all(
                (now - s.last_seen) >= self.cfg.analysis.post_roll_seconds
                for s in self.event_tracks.values()
            )

            if force_close or all_inactive:
                ended_at = max((s.last_seen for s in self.event_tracks.values()), default=now)
                summaries = [
                    asdict(extract_track_features(t.track_id, [d for _, d in t.window], len(t.window)))
                    for t in self.event_tracks.values()
                    if t.window
                ]
                closed_event_id = self.open_event_id
                start_ts = self.open_event_started
                self.store.update_event(
                    closed_event_id,
                    ended_at=ended_at,
                    status="closed",
                    behaviors=json.dumps(sorted(self.open_event_behaviors)),
                    track_summaries=json.dumps(summaries),
                )

                if self.on_closed:
                    try:
                        self.on_closed(closed_event_id, start_ts, ended_at)
                    except Exception as exc:
                        logger.error(f"{self.cam.name}: on_closed callback error: {exc}")

                self.last_event_ended_at = ended_at
                if force_close:
                    for t in self.tracks.values():
                        t.first_seen = ended_at
                # Reset open event state
                self.open_event_id = None
                self.open_event_started = 0.0
                self.open_event_primary_class = "vehicle"
                self.open_event_max_conf = 0.0
                self.open_event_behaviors = set()
                self.event_notified_kinds = set()
                self.event_tracks = {}

        # 9. Live preview generation (<= 2 fps)
        if frame is not None and (now - self.last_preview_time) >= 0.5:
            self.last_preview_time = now
            try:
                prev_rel = f"live/preview/{self.cam.slug}.jpg"
                full_prev = os.path.join(self.output_dir, prev_rel)
                os.makedirs(os.path.dirname(full_prev), exist_ok=True)
                ann = _draw_annotation(frame, relevant_dets, self.cam.zones, frame_w, frame_h)
                resized_prev = cv2.resize(ann, (960, 540))
                tmp_prev = full_prev + ".tmp.jpg"
                cv2.imwrite(tmp_prev, resized_prev, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                os.replace(tmp_prev, full_prev)
            except Exception as exc:
                logger.warning(f"{self.cam.name}: Failed to write preview image: {exc}")
