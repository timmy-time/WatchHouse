"""Unit tests for stationary vehicle filter and event classification logic."""

import unittest
from engine.classifier import (
    Verdict,
    classify_clip_events,
    compute_iou,
    extract_track_features,
)
from engine.detector import FrameDetections, TrackDetection


class TestClassifier(unittest.TestCase):

    def test_compute_iou(self):
        # Identical boxes
        b1 = (100.0, 100.0, 200.0, 200.0)
        self.assertAlmostEqual(compute_iou(b1, b1), 1.0)

        # Disjoint boxes
        b2 = (300.0, 300.0, 400.0, 400.0)
        self.assertEqual(compute_iou(b1, b2), 0.0)

        # Partial overlap (50% horizontal, 100% vertical)
        b3 = (150.0, 100.0, 250.0, 200.0)
        # Inter: 50 * 100 = 5000; Union: 10000 + 10000 - 5000 = 15000 -> 1/3
        self.assertAlmostEqual(compute_iou(b1, b3), 1.0 / 3.0)

    def test_stationary_vehicle_discard(self):
        # Vehicle staying in almost the same spot across 20 frames with slight jitter
        dets = []
        for i in range(20):
            jitter = (i % 3) * 1.5
            d = TrackDetection(
                track_id=1,
                class_name="car",
                confidence=0.85,
                bbox_xyxy=(500.0 + jitter, 400.0 + jitter, 700.0 + jitter, 600.0 + jitter),
                bbox_xywh=(600.0 + jitter, 500.0 + jitter, 200.0, 200.0),
                frame_idx=i,
            )
            dets.append(FrameDetections(frame_idx=i, detections=[d]))

        decision = classify_clip_events(dets)
        self.assertEqual(decision.verdict, Verdict.DISCARD)
        self.assertEqual(decision.primary_reason, "stationary_vehicles_only")
        self.assertEqual(decision.stationary_vehicle_count, 1)

    def test_moving_vehicle_keep(self):
        # Vehicle driving across scene (x moving from 100 to 800)
        dets = []
        for i in range(15):
            x1 = 100.0 + i * 50.0
            d = TrackDetection(
                track_id=2,
                class_name="truck",
                confidence=0.80,
                bbox_xyxy=(x1, 300.0, x1 + 200.0, 450.0),
                bbox_xywh=(x1 + 100.0, 375.0, 200.0, 150.0),
                frame_idx=i,
            )
            dets.append(FrameDetections(frame_idx=i, detections=[d]))

        decision = classify_clip_events(dets)
        self.assertEqual(decision.verdict, Verdict.KEEP)
        self.assertEqual(decision.primary_reason, "moving_vehicle")
        self.assertTrue(decision.moving_vehicle_detected)

    def test_person_detected_keep(self):
        # Parked car present PLUS a person walking by
        dets = []
        for i in range(10):
            frame_list = [
                # Parked car
                TrackDetection(
                    track_id=1,
                    class_name="car",
                    confidence=0.90,
                    bbox_xyxy=(200.0, 200.0, 400.0, 400.0),
                    bbox_xywh=(300.0, 300.0, 200.0, 200.0),
                    frame_idx=i,
                ),
            ]
            if i >= 3:
                # Person enters
                frame_list.append(
                    TrackDetection(
                        track_id=2,
                        class_name="person",
                        confidence=0.75,
                        bbox_xyxy=(50.0 + i * 10, 150.0, 90.0 + i * 10, 250.0),
                        bbox_xywh=(70.0 + i * 10, 200.0, 40.0, 100.0),
                        frame_idx=i,
                    )
                )
            dets.append(FrameDetections(frame_idx=i, detections=frame_list))

        decision = classify_clip_events(dets)
        self.assertEqual(decision.verdict, Verdict.KEEP)
        self.assertEqual(decision.primary_reason, "person_detected")
        self.assertTrue(decision.high_value_detected)

    def test_no_objects_discard(self):
        dets = [FrameDetections(frame_idx=i, detections=[]) for i in range(10)]
        decision = classify_clip_events(dets)
        self.assertEqual(decision.verdict, Verdict.DISCARD)
        self.assertEqual(decision.primary_reason, "no_objects_detected")


if __name__ == "__main__":
    unittest.main()
