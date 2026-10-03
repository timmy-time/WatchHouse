"""Unit tests for Web API endpoints and operations."""

import json
import os
import shutil
import tempfile
import unittest

from fastapi.testclient import TestClient

from engine.web.app import create_app


class TestWebApi(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "output")
        self.clips_dir = os.path.join(self.test_dir, "clips")
        self.config_path = os.path.join(self.test_dir, "live.yaml")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)

        # Write dummy live.yaml
        with open(self.config_path, "w") as f:
            f.write("recording: {root: /tmp}\n")

        # Create dummy analysis_results.json
        self.results_path = os.path.join(self.output_dir, "analysis_results.json")
        sample_clip = os.path.join(self.clips_dir, "20260103/sample.mp4")
        os.makedirs(os.path.dirname(sample_clip), exist_ok=True)
        with open(sample_clip, "wb") as f:
            f.write(b"mp4")

        archive_data = {
            "total_analyzed": 2,
            "kept_clips": 1,
            "discarded_clips": 1,
            "reduction_percentage": 50.0,
            "results": [
                {
                    "clip_path": sample_clip,
                    "thumbnail_path": sample_clip.replace(".mp4", ".png"),
                    "camera": "Camera C",
                    "verdict": "KEEP",
                    "reason": "person_detected",
                    "confidence": 0.9,
                },
                {
                    "clip_path": "/other/clip2.mp4",
                    "thumbnail_path": None,
                    "camera": "Camera A",
                    "verdict": "DISCARD",
                    "reason": "stationary_vehicles_only",
                    "confidence": 0.8,
                },
            ],
        }
        with open(self.results_path, "w") as f:
            json.dump(archive_data, f)

        self.app = create_app(self.output_dir, self.clips_dir, self.config_path)
        self.client = TestClient(self.app)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_health(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"ok": True})

    def test_archive_summary_and_filter(self):
        # Summary
        res = self.client.get("/api/archive/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_analyzed"], 2)
        self.assertEqual(data["reduction_percentage"], 50.0)
        self.assertNotIn("results", data)

        # Filter verdict=KEEP
        res = self.client.get("/api/archive?verdict=KEEP")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(len(data["items"]), 1)
        item = data["items"][0]
        self.assertEqual(item["verdict"], "KEEP")
        self.assertEqual(item["clip_url"], "/media/clips/20260103/sample.mp4")

        # Filter verdict=DISCARD
        res = self.client.get("/api/archive?verdict=DISCARD")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["clip_url"], None)

    def test_identity_crud_and_duplicate(self):
        # Create identity
        res = self.client.post("/api/identities", json={"name": "Alice"})
        self.assertEqual(res.status_code, 201)
        alice = res.json()
        self.assertEqual(alice["name"], "Alice")
        alice_id = alice["id"]

        # Duplicate name -> 409
        res_dup = self.client.post("/api/identities", json={"name": "Alice"})
        self.assertEqual(res_dup.status_code, 409)

        # Rename
        res_rename = self.client.patch(f"/api/identities/{alice_id}", json={"name": "Alice Smith"})
        self.assertEqual(res_rename.status_code, 200)
        self.assertEqual(res_rename.json()["name"], "Alice Smith")

        # List
        res_list = self.client.get("/api/identities")
        self.assertEqual(res_list.status_code, 200)
        self.assertTrue(any(i["name"] == "Alice Smith" for i in res_list.json()))

        # Delete
        res_del = self.client.delete(f"/api/identities/{alice_id}")
        self.assertEqual(res_del.status_code, 200)

        # Delete non-existent -> 404
        res_del404 = self.client.delete(f"/api/identities/{alice_id}")
        self.assertEqual(res_del404.status_code, 404)

    def test_cluster_assign_by_name(self):
        # Insert a face in cluster 42 into the DB
        store = self.app.state.store if hasattr(self.app.state, "store") else None
        # Access store directly from app
        from engine.live.db import EventStore
        store = EventStore(os.path.join(self.output_dir, "live/events.db"))
        fid = store.insert_face(
            event_id=1,
            camera="Camera C",
            track_id=1,
            captured_at=100.0,
            crop_path="",
            context_path="",
            embedding=b"\x00" * 512,
            quality=0.8,
            det_score=0.9,
            face_width=50.0,
            identity_id=None,
            cluster_id=42,
        )

        # Verify cluster appears in /api/faces/clusters
        res = self.client.get("/api/faces/clusters")
        self.assertEqual(res.status_code, 200)
        clusters = res.json()
        self.assertTrue(any(c["cluster_id"] == 42 for c in clusters))

        # Assign cluster 42 by name "Bob"
        res_assign = self.client.post(
            "/api/faces/clusters/42/assign",
            json={"name": "Bob"},
        )
        self.assertEqual(res_assign.status_code, 200)

        # Cluster 42 should no longer exist in unknown clusters
        res_after = self.client.get("/api/faces/clusters")
        self.assertFalse(any(c["cluster_id"] == 42 for c in res_after.json()))

        # "Bob" identity should now exist with face_count >= 1
        res_ids = self.client.get("/api/identities")
        bob_id = next((i for i in res_ids.json() if i["name"] == "Bob"), None)
        self.assertIsNotNone(bob_id)
        self.assertEqual(bob_id["face_count"], 1)

    def test_events_filter_by_behavior(self):
        from engine.live.db import EventStore
        store = EventStore(os.path.join(self.output_dir, "live/events.db"))

        eid1 = store.create_event(
            camera="Camera C",
            started_at=10.0,
            status="closed",
            primary_class="person",
            behaviors=json.dumps(["approaching", "loitering"]),
        )
        eid2 = store.create_event(
            camera="Camera A",
            started_at=20.0,
            status="closed",
            primary_class="vehicle",
            behaviors=json.dumps(["vehicle_arrived"]),
        )

        # Filter by behavior=approaching
        res = self.client.get("/api/events?behavior=approaching")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["id"], eid1)

        # Filter by behavior=vehicle_arrived
        res2 = self.client.get("/api/events?behavior=vehicle_arrived")
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["total"], 1)
        self.assertEqual(res2.json()["items"][0]["id"], eid2)

    def test_scenery_slots_api(self):
        # Create slot
        res = self.client.post(
            "/api/scenery/slots",
            json={
                "camera": "Camera A",
                "name": "Parked Parked Car 1",
                "slot_box": [0.15, 0.35, 0.55, 0.75],
                "is_friendly": True,
                "color_name": "red",
            },
        )
        self.assertEqual(res.status_code, 201)
        slot = res.json()
        self.assertEqual(slot["name"], "Parked Parked Car 1")
        slot_id = slot["id"]

        # List slots
        res_list = self.client.get("/api/scenery/slots?camera=Camera A")
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual(len(res_list.json()), 1)
        self.assertEqual(res_list.json()[0]["color_name"], "red")

        # Update slot
        res_patch = self.client.patch(
            f"/api/scenery/slots/{slot_id}",
            json={"name": "Parked Car 1"},
        )
        self.assertEqual(res_patch.status_code, 200)
        self.assertEqual(res_patch.json()["name"], "Parked Car 1")

        # Delete slot
        res_del = self.client.delete(f"/api/scenery/slots/{slot_id}")
        self.assertEqual(res_del.status_code, 200)

        # Verify deleted
        res_del404 = self.client.delete(f"/api/scenery/slots/{slot_id}")
        self.assertEqual(res_del404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
