"""Unit tests for SceneryManager and vehicle appearance fingerprinting."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from engine.live.db import EventStore
from engine.scenery import (
    SceneryManager,
    VehicleSignature,
    compare_signatures,
    compute_box_iou,
    extract_vehicle_signature,
)


class TestScenery(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "events.db")
        self.store = EventStore(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_box_iou(self):
        b1 = (0.1, 0.1, 0.5, 0.5)
        b2 = (0.1, 0.1, 0.5, 0.5)
        self.assertAlmostEqual(compute_box_iou(b1, b2), 1.0)

        b3 = (0.6, 0.6, 0.9, 0.9)
        self.assertEqual(compute_box_iou(b1, b3), 0.0)

        b4 = (0.3, 0.1, 0.7, 0.5)  # 50% horizontal overlap
        iou = compute_box_iou(b1, b4)
        self.assertGreater(iou, 0.25)
        self.assertLess(iou, 0.40)

    def test_extract_and_compare_signatures(self):
        # Create a synthetic 100x200 BGR red vehicle image (aspect ratio ~2.0)
        # Red in BGR: B=20, G=20, R=200
        red_img = np.full((100, 200, 3), (20, 20, 200), dtype=np.uint8)
        bbox = (10.0, 10.0, 190.0, 90.0)

        sig_red1 = extract_vehicle_signature(red_img, bbox)
        self.assertEqual(sig_red1.color_name, "red")
        self.assertGreater(len(sig_red1.hsv_bins), 0)

        # Create another red vehicle with slightly different aspect ratio
        red_img2 = np.full((100, 180, 3), (25, 25, 190), dtype=np.uint8)
        sig_red2 = extract_vehicle_signature(red_img2, (10.0, 10.0, 170.0, 90.0))
        self.assertEqual(sig_red2.color_name, "red")

        # Create a silver/gray vehicle (aspect ratio ~1.5)
        # Gray in BGR: B=160, G=160, R=160
        silver_img = np.full((100, 150, 3), (160, 160, 160), dtype=np.uint8)
        sig_silver = extract_vehicle_signature(silver_img, (10.0, 10.0, 140.0, 90.0))
        self.assertIn(sig_silver.color_name, ("silver/gray", "white"))

        # Compare red1 vs red2 -> high similarity
        sim_red = compare_signatures(sig_red1, sig_red2)
        # Compare red1 vs silver -> much lower similarity
        sim_diff = compare_signatures(sig_red1, sig_silver)

        self.assertGreater(sim_red, 0.75)
        self.assertLess(sim_diff, 0.60)
        self.assertGreater(sim_red, sim_diff + 0.20)

    def test_scenery_manager_matching(self):
        # Register a slot for "Camera A": Parked Car 1 in [0.2, 0.3, 0.6, 0.7]
        slot_box = [0.20, 0.30, 0.60, 0.70]
        sig_data = {"aspect_ratio": 1.8, "hsv_bins": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]}
        slot_id = self.store.create_vehicle_slot(
            camera="Camera A",
            name="Parked Car 1",
            slot_box=json.dumps(slot_box),
            color_name="red",
            appearance_sig=json.dumps(sig_data),
            is_friendly=1,
        )

        manager = SceneryManager(self.store)
        slots = manager.get_slots("Camera A")
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0].name, "Parked Car 1")

        # Match detection overlapping slot [0.22, 0.32, 0.58, 0.68]
        det_box = (0.22, 0.32, 0.58, 0.68)
        matched = manager.match_slot("Camera A", det_box)
        self.assertIsNotNone(matched)
        slot, conf = matched
        self.assertEqual(slot.name, "Parked Car 1")
        self.assertGreater(conf, 0.50)

        # Non-matching detection far away [0.8, 0.8, 0.95, 0.95]
        det_far = (0.80, 0.80, 0.95, 0.95)
        matched_far = manager.match_slot("Camera A", det_far)
        self.assertIsNone(matched_far)

        # Other camera has no slots
        matched_other = manager.match_slot("Camera C", det_box)
        self.assertIsNone(matched_other)

    def test_animal_slot_occupancy(self):
        slot_box = [0.10, 0.10, 0.40, 0.40]
        self.store.create_vehicle_slot(
            camera="Camera D",
            name="Buster",
            slot_box=json.dumps(slot_box),
            color_name="golden",
            appearance_sig="{}",
            is_friendly=True,
        )
        manager = SceneryManager(self.store)
        slot = manager.get_slots("Camera D")[0]

        class DummyDetection:
            def __init__(self, class_name, bbox):
                self.class_name = class_name
                self.bbox_xyxy = bbox

        dog_det = DummyDetection("dog", (150, 150, 350, 350))
        self.assertTrue(manager.is_slot_occupied(slot, [dog_det], 1000, 1000))

        bird_det = DummyDetection("bird", (800, 800, 900, 900))
        self.assertFalse(manager.is_slot_occupied(slot, [bird_det], 1000, 1000))


if __name__ == "__main__":
    unittest.main()
