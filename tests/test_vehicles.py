"""Unit tests for cross-camera vehicle identity (vehicles, linking, sightings)."""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest

from engine.live.db import EventStore


class TestVehicleIdentity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db = os.path.join(self.test_dir, "events.db")
        self.store = EventStore(self.db)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _slot(self, camera="Camera B", name="Parked Car 1"):
        sid = self.store.create_vehicle_slot(
            camera=camera, name=name, slot_box=json.dumps([0.1, 0.2, 0.3, 0.4]),
            color_name="silver/gray", appearance_sig=json.dumps({"aspect_ratio": 1.5, "hsv_bins": []}),
            is_friendly=1,
        )
        return sid

    def test_slot_creation_creates_vehicle_link(self):
        sid = self._slot()
        vid = self.store.get_or_create_vehicle("Parked Car 1")
        self.assertTrue(self.store.link_slot_vehicle(sid, vid))

        slot = self.store.get_vehicle_slot(sid)
        self.assertEqual(slot["vehicle_id"], vid)

        veh = next(v for v in self.store.list_vehicles() if v["id"] == vid)
        self.assertEqual(veh["cameras"], ["Camera B"])
        self.assertEqual(len(veh["slots"]), 1)

    def test_get_or_create_is_idempotent(self):
        a = self.store.get_or_create_vehicle("Same Car")
        b = self.store.get_or_create_vehicle("Same Car")
        self.assertEqual(a, b)
        self.assertEqual(len(self.store.list_vehicles()), 1)

    def test_two_cameras_link_to_one_vehicle(self):
        v = self.store.get_or_create_vehicle("Parked Car 2")
        s1 = self._slot(camera="Camera B", name="Parked Car 2")
        s2 = self._slot(camera="Camera A", name="Parked Car 2")
        self.store.link_slot_vehicle(s1, v)
        self.store.link_slot_vehicle(s2, v)

        veh = next(x for x in self.store.list_vehicles() if x["id"] == v)
        self.assertEqual(veh["cameras"], ["Camera A", "Camera B"])
        self.assertEqual(len(veh["slots"]), 2)

    def test_rename_vehicle_syncs_slot_labels(self):
        v = self.store.get_or_create_vehicle("Old Name")
        sid = self._slot(name="Old Name")
        self.store.link_slot_vehicle(sid, v)

        self.store.rename_vehicle(v, "New Name")
        self.assertEqual(self.store.get_vehicle_slot(sid)["name"], "New Name")
        self.assertEqual(self.store.get_vehicle(v)["name"], "New Name")

    def test_link_slot_vehicle_rejects_missing_vehicle(self):
        sid = self._slot()
        self.assertFalse(self.store.link_slot_vehicle(sid, 9999))

    def test_unlink_keeps_slot(self):
        v = self.store.get_or_create_vehicle("Temp")
        sid = self._slot()
        self.store.link_slot_vehicle(sid, v)
        self.store.link_slot_vehicle(sid, None)
        self.assertIsNone(self.store.get_vehicle_slot(sid)["vehicle_id"])
        self.assertIsNotNone(self.store.get_vehicle_slot(sid))  # slot survives

    def test_delete_vehicle_unlinks_slots_and_drops_sightings(self):
        v = self.store.get_or_create_vehicle("Doomed")
        sid = self._slot()
        self.store.link_slot_vehicle(sid, v)
        self.store.record_sighting(v, "Camera B")

        self.assertTrue(self.store.delete_vehicle(v))
        self.assertIsNone(self.store.get_vehicle_slot(sid)["vehicle_id"])
        self.assertEqual(self.store.list_vehicles(), [])
        self.assertFalse(self.store.delete_vehicle(v))  # already gone

    def test_sighting_upsert_counts_hits(self):
        v = self.store.get_or_create_vehicle("Seen Car")
        self.store.record_sighting(v, "Camera B")
        self.store.record_sighting(v, "Camera B")
        self.store.record_sighting(v, "Camera A")

        veh = next(x for x in self.store.list_vehicles() if x["id"] == v)
        by_cam = {s["camera"]: s["hits"] for s in veh["sightings"]}
        self.assertEqual(by_cam, {"Camera B": 2, "Camera A": 1})
        self.assertEqual(veh["cameras"], ["Camera A", "Camera B"])

    def test_migration_links_legacy_slots_by_name(self):
        """A DB created before the vehicles table gets linked during migration."""
        legacy = os.path.join(self.test_dir, "legacy.db")
        conn = sqlite3.connect(legacy)
        conn.executescript(
            """
            CREATE TABLE vehicle_slots(
                id INTEGER PRIMARY KEY, camera TEXT NOT NULL, name TEXT NOT NULL,
                slot_box TEXT NOT NULL, color_name TEXT NOT NULL, appearance_sig TEXT NOT NULL,
                is_friendly INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO vehicle_slots (camera, name, slot_box, color_name, appearance_sig, created_at, updated_at)"
            " VALUES ('Camera A', 'Parked Car 1', '[0,0,1,1]', 'blue', '{}', 1, 1)"
        )
        conn.commit()
        conn.close()

        store = EventStore(legacy)
        vehicles = store.list_vehicles()
        self.assertEqual([v["name"] for v in vehicles], ["Parked Car 1"])
        slot = store.list_vehicle_slots(camera="Camera A")[0]
        self.assertEqual(slot["vehicle_id"], vehicles[0]["id"])


if __name__ == "__main__":
    unittest.main()
