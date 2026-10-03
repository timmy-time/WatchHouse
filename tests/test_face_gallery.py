"""Unit tests for FaceGallery clustering and identity matching."""

import os
import shutil
import tempfile
import threading
import unittest

import numpy as np

from engine.faces import FaceGallery
from engine.live.db import EventStore


class TestFaceGallery(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "events.db")
        self.store = EventStore(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_unit_vector(self, index: int, dim: int = 128) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        v[index] = 1.0
        return v

    def test_identity_match_above_threshold(self):
        # Insert labeled identity Alice with unit vector 0
        alice_id = self.store.create_identity("Alice")
        v0 = self._make_unit_vector(0)
        self.store.insert_face(
            event_id=1,
            camera="camera_c",
            track_id=1,
            captured_at=100.0,
            crop_path="f1.jpg",
            context_path="ctx1.jpg",
            embedding=v0.tobytes(),
            quality=0.9,
            det_score=0.95,
            face_width=50.0,
            identity_id=alice_id,
        )

        gallery = FaceGallery(self.store, match_threshold=0.40, cluster_threshold=0.45)

        # Query with vector very similar to Alice (cosine ~0.8)
        # v_query = 0.8 * v0 + 0.6 * v1 (norm = 1.0, dot product with v0 = 0.8)
        v1 = self._make_unit_vector(1)
        query = 0.8 * v0 + 0.6 * v1
        with gallery.lock:
            matched_id, cluster_id, score = gallery.assign(query)

        self.assertEqual(matched_id, alice_id)
        self.assertIsNone(cluster_id)
        self.assertAlmostEqual(score, 0.8, places=4)

    def test_unknown_cluster_match_above_threshold(self):
        # Insert unknown face in cluster 10 with unit vector 2
        v2 = self._make_unit_vector(2)
        self.store.insert_face(
            event_id=1,
            camera="camera_c",
            track_id=1,
            captured_at=100.0,
            crop_path="f2.jpg",
            context_path="ctx2.jpg",
            embedding=v2.tobytes(),
            quality=0.9,
            det_score=0.95,
            face_width=50.0,
            identity_id=None,
            cluster_id=10,
        )

        gallery = FaceGallery(self.store, match_threshold=0.40, cluster_threshold=0.45)

        # Query with vector close to cluster 10 (cosine = 0.5 > 0.45)
        v3 = self._make_unit_vector(3)
        query = 0.5 * v2 + float(np.sqrt(0.75)) * v3
        with gallery.lock:
            matched_id, cluster_id, score = gallery.assign(query)

        self.assertIsNone(matched_id)
        self.assertEqual(cluster_id, 10)
        self.assertAlmostEqual(score, 0.5, places=4)

    def test_orthogonal_vector_creates_new_cluster(self):
        # Insert unknown face in cluster 1 with unit vector 0
        v0 = self._make_unit_vector(0)
        self.store.insert_face(
            event_id=1,
            camera="camera_c",
            track_id=1,
            captured_at=100.0,
            crop_path="f0.jpg",
            context_path="ctx0.jpg",
            embedding=v0.tobytes(),
            quality=0.9,
            det_score=0.95,
            face_width=50.0,
            identity_id=None,
            cluster_id=1,
        )

        gallery = FaceGallery(self.store, match_threshold=0.40, cluster_threshold=0.45)

        # Query with completely orthogonal vector (unit vector 5, dot = 0.0)
        v5 = self._make_unit_vector(5)
        with gallery.lock:
            matched_id, cluster_id, score = gallery.assign(v5)

        self.assertIsNone(matched_id)
        # Next cluster id should be max(1) + 1 = 2
        self.assertEqual(cluster_id, 2)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_assign_under_lock_with_stale_gallery_does_not_deadlock(self):
        # Identity added after gallery construction (e.g. via web photo upload),
        # then gallery becomes >30s stale: assign() must refresh without
        # self-deadlocking on the lock the caller already holds.
        gallery = FaceGallery(self.store, match_threshold=0.40, cluster_threshold=0.45)
        bob_id = self.store.create_identity("Bob")
        v0 = self._make_unit_vector(0)
        self.store.insert_face(
            event_id=None, camera="upload", track_id=None, captured_at=1.0,
            crop_path="a.jpg", context_path="a_ctx.jpg", embedding=v0.tobytes(),
            quality=1.0, det_score=0.99, face_width=100.0, identity_id=bob_id,
        )
        gallery.last_refresh = 0.0  # force stale

        result = {}

        def run():
            with gallery.lock:
                result["r"] = gallery.assign(v0)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "assign() deadlocked on stale gallery refresh")
        self.assertEqual(result["r"][0], bob_id)


if __name__ == "__main__":
    unittest.main()
