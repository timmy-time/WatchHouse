"""Face detection, recognition, and gallery clustering using OpenCV YuNet and SFace."""

from dataclasses import dataclass
import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from engine.live.db import EventStore


@dataclass
class FaceCandidate:
    row: np.ndarray  # 15-float YuNet row in full-frame coordinates
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    frontal: float
    sharpness: float
    quality: float


YUNET_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_NAME = "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def _ensure_face_models(model_dir: str = "/opt/models") -> Tuple[str, str]:
    """Locate YuNet and SFace models or auto-download them into models/ on first run."""
    candidates = [model_dir, "models", "/app/models", "/opt/models", os.path.expanduser("~/.cache/watchhouse/models")]
    for c in candidates:
        yp = os.path.join(c, YUNET_NAME)
        sp = os.path.join(c, SFACE_NAME)
        if os.path.exists(yp) and os.path.exists(sp):
            return yp, sp

    target_dir = "models" if (os.path.exists("models") or not os.path.exists("/opt/models")) else "/opt/models"
    os.makedirs(target_dir, exist_ok=True)
    yp = os.path.join(target_dir, YUNET_NAME)
    sp = os.path.join(target_dir, SFACE_NAME)

    import urllib.request
    if not os.path.exists(yp):
        urllib.request.urlretrieve(YUNET_URL, yp)
    if not os.path.exists(sp):
        urllib.request.urlretrieve(SFACE_URL, sp)
    return yp, sp

class FaceEngine:
    """Thread-local face detector (YuNet) and recognizer (SFace)."""

    def __init__(
        self,
        model_dir: str = "/opt/models",
        min_face_px: int = 40,
        min_det_score: float = 0.80,
    ):
        self.model_dir = model_dir
        self.min_face_px = min_face_px
        self.min_det_score = min_det_score
        yunet_path, sface_path = _ensure_face_models(model_dir)
        self.detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3, 50)
        self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    def detect_in_person(
        self,
        frame: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
    ) -> List[FaceCandidate]:
        """Detect face candidates within a person bounding box."""
        frame_h, frame_w = frame.shape[:2]
        bx1, by1, bx2, by2 = bbox_xyxy
        bw = bx2 - bx1
        bh = by2 - by1

        # Crop: x expanded by w/4 each side, top raised by h/10, height limited to 60%
        crop_x1 = max(0, int(bx1 - bw * 0.25))
        crop_x2 = min(frame_w, int(bx2 + bw * 0.25))
        crop_y1 = max(0, int(by1 - bh * 0.10))
        crop_y2 = min(frame_h, int(by1 + bh * 0.60))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return []

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 10 or crop_w < 10:
            return []

        s = min(320.0 / max(crop_h, crop_w), 2.0)
        resized_w = max(1, int(crop_w * s))
        resized_h = max(1, int(crop_h * s))
        resized = cv2.resize(crop, (resized_w, resized_h))

        self.detector.setInputSize((resized_w, resized_h))
        _, faces = self.detector.detect(resized)

        if faces is None or len(faces) == 0:
            return []

        candidates: List[FaceCandidate] = []
        for face in faces:
            # Map coordinates back to full frame
            mapped_row = face.astype(np.float32).copy()
            mapped_row[0] = face[0] / s + crop_x1
            mapped_row[1] = face[1] / s + crop_y1
            mapped_row[2] = face[2] / s
            mapped_row[3] = face[3] / s

            # Landmarks: right eye, left eye, nose, right mouth, left mouth
            for k in range(4, 14, 2):
                mapped_row[k] = face[k] / s + crop_x1
                mapped_row[k + 1] = face[k + 1] / s + crop_y1

            fw = float(mapped_row[2])
            fh = float(mapped_row[3])
            score = float(face[14])

            # Frontality metric
            re_x = float(mapped_row[4])
            le_x = float(mapped_row[6])
            nose_x = float(mapped_row[8])
            eye_dist = max(1e-3, abs(re_x - le_x))
            r = (nose_x - min(re_x, le_x)) / eye_dist
            frontal = 1.0 - min(1.0, abs(r - 0.5) * 2.0)

            # Sharpness metric
            fx1 = max(0, min(frame_w - 1, int(mapped_row[0])))
            fy1 = max(0, min(frame_h - 1, int(mapped_row[1])))
            fx2 = max(0, min(frame_w, int(mapped_row[0] + fw)))
            fy2 = max(0, min(frame_h, int(mapped_row[1] + fh)))

            if fx2 > fx1 and fy2 > fy1:
                patch = cv2.cvtColor(frame[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)
                sharpness_raw = float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size > 0 else 0.0
            else:
                sharpness_raw = 0.0

            sharpness = min(1.0, sharpness_raw / 100.0)
            quality = score * min(1.0, fw / 80.0) * frontal * sharpness

            # Quality gates
            if (
                fw >= self.min_face_px
                and score >= self.min_det_score
                and frontal >= 0.3
                and sharpness_raw >= 30.0
            ):
                candidates.append(
                    FaceCandidate(
                        row=mapped_row,
                        box=(float(mapped_row[0]), float(mapped_row[1]), float(mapped_row[0] + fw), float(mapped_row[1] + fh)),
                        score=score,
                        frontal=frontal,
                        sharpness=sharpness,
                        quality=quality,
                    )
                )

        return candidates

    def embed(self, frame: np.ndarray, cand: FaceCandidate) -> Tuple[np.ndarray, np.ndarray]:
        """Align crop and extract 128-d L2-normalized feature vector."""
        aligned = self.recognizer.alignCrop(frame, cand.row)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if norm > 1e-6:
            feat = feat / norm
        return aligned, feat

    def embed_image(
        self,
        image: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, FaceCandidate]]:
        """Detect and embed the largest face in an uploaded image."""
        img_h, img_w = image.shape[:2]
        if max(img_h, img_w) > 1280:
            scale = 1280.0 / max(img_h, img_w)
            resized = cv2.resize(image, (int(img_w * scale), int(img_h * scale)))
        else:
            scale = 1.0
            resized = image

        self.detector.setInputSize((resized.shape[1], resized.shape[0]))
        _, faces = self.detector.detect(resized)
        if faces is None or len(faces) == 0:
            return None

        # Filter score >= 0.6 and take largest
        valid = [f for f in faces if float(f[14]) >= 0.60]
        if not valid:
            return None

        valid.sort(key=lambda f: float(f[2]) * float(f[3]), reverse=True)
        best_face = valid[0]

        mapped_row = best_face.astype(np.float32).copy()
        if scale != 1.0:
            mapped_row[0:4] /= scale
            mapped_row[4:14] /= scale

        fw = float(mapped_row[2])
        fh = float(mapped_row[3])
        score = float(best_face[14])

        re_x = float(mapped_row[4])
        le_x = float(mapped_row[6])
        nose_x = float(mapped_row[8])
        eye_dist = max(1e-3, abs(re_x - le_x))
        r = (nose_x - min(re_x, le_x)) / eye_dist
        frontal = 1.0 - min(1.0, abs(r - 0.5) * 2.0)

        cand = FaceCandidate(
            row=mapped_row,
            box=(float(mapped_row[0]), float(mapped_row[1]), float(mapped_row[0] + fw), float(mapped_row[1] + fh)),
            score=score,
            frontal=frontal,
            sharpness=1.0,
            quality=score,
        )

        aligned, feat = self.embed(image, cand)
        return aligned, feat, cand


class FaceGallery:
    """Thread-safe gallery managing face clustering and identity recognition."""

    def __init__(
        self,
        store: EventStore,
        match_threshold: float = 0.40,
        cluster_threshold: float = 0.45,
    ):
        self.store = store
        self.match_threshold = match_threshold
        self.cluster_threshold = cluster_threshold
        self.lock = threading.RLock()
        self.last_refresh = 0.0
        self.labeled_ids: List[int] = []
        self.labeled_embs: Optional[np.ndarray] = None
        self.unknown_cids: List[int] = []
        self.unknown_embs: Optional[np.ndarray] = None
        self.refresh()

    def refresh(self) -> None:
        """Reload labeled and unknown embeddings from database."""
        with self.lock:
            labeled = self.store.labeled_faces()
            unknown = self.store.unknown_faces()

            if labeled:
                self.labeled_ids = [iid for _, iid, _ in labeled]
                self.labeled_embs = np.stack([emb for _, _, emb in labeled], axis=0)
            else:
                self.labeled_ids = []
                self.labeled_embs = None

            if unknown:
                self.unknown_cids = [cid for _, cid, _ in unknown]
                self.unknown_embs = np.stack([emb for _, _, emb in unknown], axis=0)
            else:
                self.unknown_cids = []
                self.unknown_embs = None

            self.last_refresh = time.time()

    def assign(
        self,
        feat: np.ndarray,
    ) -> Tuple[Optional[int], Optional[int], float]:
        """Assign feature vector to an identity, existing cluster, or new cluster.

        Returns (identity_id, cluster_id, match_score).
        Caller should hold gallery.lock across assign and store insertion.
        """
        if time.time() - self.last_refresh > 30.0:
            self.refresh()

        norm = float(np.linalg.norm(feat))
        if norm > 1e-6:
            feat = feat / norm

        # 1. Check labeled identities
        best_id: Optional[int] = None
        best_id_score: float = 0.0
        if self.labeled_embs is not None and len(self.labeled_ids) > 0:
            sims = np.dot(self.labeled_embs, feat)
            # Find best per identity
            for idx, iid in enumerate(self.labeled_ids):
                s = float(sims[idx])
                if s > best_id_score:
                    best_id_score = s
                    best_id = iid

            if best_id is not None and best_id_score >= self.match_threshold:
                return best_id, None, best_id_score

        # 2. Check unknown clusters
        best_cid: Optional[int] = None
        best_cid_score: float = 0.0
        if self.unknown_embs is not None and len(self.unknown_cids) > 0:
            u_sims = np.dot(self.unknown_embs, feat)
            for idx, cid in enumerate(self.unknown_cids):
                s = float(u_sims[idx])
                if s > best_cid_score:
                    best_cid_score = s
                    best_cid = cid

            if best_cid is not None and best_cid_score >= self.cluster_threshold:
                return None, best_cid, best_cid_score

        # 3. New cluster
        new_cid = self.store.next_cluster_id()
        best_score = max(0.0, best_id_score, best_cid_score)
        return None, new_cid, best_score

    def add_assigned(
        self,
        identity_id: Optional[int],
        cluster_id: Optional[int],
        feat: np.ndarray,
    ) -> None:
        """Immediately update in-memory matrices after insertion."""
        with self.lock:
            norm = float(np.linalg.norm(feat))
            if norm > 1e-6:
                feat = feat / norm
            feat_row = feat.reshape(1, -1)

            if identity_id is not None:
                self.labeled_ids.append(identity_id)
                if self.labeled_embs is None:
                    self.labeled_embs = feat_row
                else:
                    self.labeled_embs = np.vstack([self.labeled_embs, feat_row])
            elif cluster_id is not None:
                self.unknown_cids.append(cluster_id)
                if self.unknown_embs is None:
                    self.unknown_embs = feat_row
                else:
                    self.unknown_embs = np.vstack([self.unknown_embs, feat_row])
