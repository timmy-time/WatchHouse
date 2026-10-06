"""Unit tests for historical clip and event video face re-indexing."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from engine.faces import FaceCandidate
from engine.live.db import EventStore
from engine.reindexer import FaceReindexer


class TestFaceReindexer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "events.db")
        self.store = EventStore(self.db_path)
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_video(self, path: str, num_frames: int = 15):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 15.0, (320, 240))
        for _ in range(num_frames):
            frame = np.full((240, 320, 3), 128, dtype=np.uint8)
            # Draw synthetic person silhouette
            cv2.rectangle(frame, (100, 40), (220, 220), (200, 150, 100), -1)
            # Draw synthetic face
            cv2.circle(frame, (160, 80), 25, (230, 200, 180), -1)
            writer.write(frame)
        writer.release()

    def test_reindex_missing_clip_returns_error(self):
        reindexer = FaceReindexer(store=self.store, output_dir=self.output_dir)
        res = reindexer.reindex_clip("/nonexistent/path.mp4")
        self.assertIn("error", res)
        self.assertEqual(res["faces_indexed"], 0)

    def test_reindex_clip_extracts_and_inserts_faces(self):
        clip_path = os.path.join(self.test_dir, "event_100.mp4")
        self._create_synthetic_video(clip_path, num_frames=12)

        # Mock detector detecting 2 persons
        class MockDetection:
            def __init__(self, bbox):
                self.class_name = "person"
                self.bbox_xyxy = bbox

        mock_detector = MagicMock()
        mock_detector.detect_frame.return_value = [
            MockDetection((80, 20, 240, 220)),
            MockDetection((10, 10, 70, 150)),
        ]

        # Mock face engine
        mock_face_engine = MagicMock()
        mock_cand = FaceCandidate(
            row=np.zeros(15),
            box=(135.0, 55.0, 185.0, 105.0),
            score=0.92,
            frontal=0.95,
            sharpness=1.0,
            quality=0.88,
        )
        mock_face_engine.detect_in_person.return_value = [mock_cand]
        mock_feat = np.ones(128, dtype=np.float32)
        mock_aligned = np.full((112, 112, 3), 150, dtype=np.uint8)
        mock_face_engine.embed.return_value = (mock_aligned, mock_feat)

        # Mock gallery
        mock_gallery = MagicMock()
        mock_gallery.assign.return_value = (None, 1, 0.0)

        reindexer = FaceReindexer(
            store=self.store,
            output_dir=self.output_dir,
            face_engine=mock_face_engine,
            gallery=mock_gallery,
            detector=mock_detector,
        )

        res = reindexer.reindex_clip(
            clip_path,
            event_id=100,
            camera="Camera A",
            stride=5,
        )

        self.assertEqual(res["event_id"], 100)
        self.assertGreater(res["faces_indexed"], 0)
        self.assertIn(1, res["clusters_assigned"])

        # Verify faces inserted into database
        faces = self.store.faces_for_event(100)
        self.assertGreater(len(faces), 0)
        self.assertEqual(faces[0]["camera"], "Camera A")
        self.assertEqual(faces[0]["cluster_id"], 1)

        # Verify crop files exist on disk
        crop_abs = os.path.join(self.output_dir, faces[0]["crop_path"])
        self.assertTrue(os.path.exists(crop_abs))

    def test_reindex_events_directory_walks_and_processes(self):
        events_dir = os.path.join(self.test_dir, "events/Camera B/20261006")
        clip1 = os.path.join(events_dir, "201.mp4")
        clip2 = os.path.join(events_dir, "202.mp4")
        self._create_synthetic_video(clip1, num_frames=6)
        self._create_synthetic_video(clip2, num_frames=6)

        mock_face_engine = MagicMock()
        mock_face_engine.detect_in_person.return_value = []
        reindexer = FaceReindexer(
            store=self.store,
            output_dir=self.output_dir,
            face_engine=mock_face_engine,
        )

        summary = reindexer.reindex_events_directory(
            events_root=os.path.join(self.test_dir, "events"),
            limit=10,
        )
        self.assertEqual(summary["processed_clips"], 2)


if __name__ == "__main__":
    unittest.main()
