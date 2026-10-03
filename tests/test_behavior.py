"""Unit tests for behavior classification."""

import unittest
from collections import deque

from engine.behavior import (
    Zone,
    TrackState,
    observe,
    detect_behaviors,
    point_in_polygon,
)
from engine.detector import TrackDetection


class TestBehaviorClassification(unittest.TestCase):
    def setUp(self):
        self.frame_w = 1920
        self.frame_h = 1080
        # Porch zone: [0.2, 0.6] to [0.6, 1.0]
        self.porch_zone = Zone(
            name="porch",
            type="entry",
            polygon=[(0.20, 0.60), (0.60, 0.60), (0.60, 1.0), (0.20, 1.0)],
        )
        self.zones = [self.porch_zone]

    def _make_det(
        self,
        track_id: int,
        class_name: str,
        frame_idx: int,
        bbox_xyxy: tuple,
        conf: float = 0.8,
    ) -> TrackDetection:
        x1, y1, x2, y2 = bbox_xyxy
        w = x2 - x1
        h = y2 - y1
        return TrackDetection(
            track_id=track_id,
            class_name=class_name,
            confidence=conf,
            bbox_xyxy=bbox_xyxy,
            bbox_xywh=(x1 + w / 2, y1 + h / 2, w, h),
            frame_idx=frame_idx,
        )

    def test_point_in_polygon(self):
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertTrue(point_in_polygon(0.5, 0.5, poly))
        self.assertFalse(point_in_polygon(1.5, 0.5, poly))
        self.assertFalse(point_in_polygon(-0.1, 0.5, poly))

    def test_approaching_behavior(self):
        # Feet inside porch zone: x in [0.2, 0.6], y in [0.6, 1.0]
        # Frame: 1920x1080. Porch x: [384, 1152], y: [648, 1080]
        # Box: [500, 700, 600, 900] -> feet at (550, 900) -> norm (0.286, 0.833) -> inside!
        feet_in_box = (500.0, 700.0, 600.0, 900.0)
        state = TrackState(
            track_id=1,
            class_name="person",
            first_seen=0.0,
            last_seen=0.0,
            first_center=(550.0, 800.0),
        )

        # 6 frames at 5 fps = 1.2 seconds in zone
        for i in range(7):
            t = i * 0.2
            det = self._make_det(1, "person", i, feet_in_box)
            observe(state, t, det, self.zones, self.frame_w, self.frame_h)

        behaviors = detect_behaviors(state, self.zones, self.frame_w, self.frame_h, now=1.2)
        self.assertIn("approaching", behaviors)

        # Feet outside porch zone
        feet_out_box = (50.0, 100.0, 150.0, 300.0)
        state_out = TrackState(
            track_id=2,
            class_name="person",
            first_seen=0.0,
            last_seen=0.0,
            first_center=(100.0, 200.0),
        )
        for i in range(7):
            t = i * 0.2
            det = self._make_det(2, "person", i, feet_out_box)
            observe(state_out, t, det, self.zones, self.frame_w, self.frame_h)

        behaviors_out = detect_behaviors(state_out, self.zones, self.frame_w, self.frame_h, now=1.2)
        self.assertNotIn("approaching", behaviors_out)

    def test_loitering_behavior(self):
        # Person stays in place for 50s vs 30s
        box = (500.0, 500.0, 600.0, 700.0)  # height = 200
        center = (550.0, 600.0)

        # 50s static person
        state_50 = TrackState(
            track_id=1,
            class_name="person",
            first_seen=0.0,
            last_seen=0.0,
            first_center=center,
        )
        # Sample at 1 fps for duration test
        for i in range(51):
            t = float(i)
            # slight jitter of 5 px
            b = (box[0] + (i % 2), box[1], box[2] + (i % 2), box[3])
            det = self._make_det(1, "person", i, b)
            observe(state_50, t, det, [], self.frame_w, self.frame_h)

        behaviors_50 = detect_behaviors(state_50, [], self.frame_w, self.frame_h, now=50.0)
        self.assertIn("loitering", behaviors_50)

        # 30s static person -> duration < 45s -> not loitering
        state_30 = TrackState(
            track_id=2,
            class_name="person",
            first_seen=0.0,
            last_seen=0.0,
            first_center=center,
        )
        for i in range(31):
            t = float(i)
            det = self._make_det(2, "person", i, box)
            observe(state_30, t, det, [], self.frame_w, self.frame_h)

        behaviors_30 = detect_behaviors(state_30, [], self.frame_w, self.frame_h, now=30.0)
        self.assertNotIn("loitering", behaviors_30)

    def test_running_behavior(self):
        # Fast person: speed >= 1.6 bbox heights / sec
        # Box height = 100 px. In 2.0 s (10 frames at 5 fps), person moves 2.0 * 2.0 * 100 = 400 px
        state = TrackState(
            track_id=1,
            class_name="person",
            first_seen=0.0,
            last_seen=0.0,
            first_center=(200.0, 500.0),
        )
        for i in range(10):
            t = i * 0.2
            # Move by 40 px each frame = 200 px/sec = 2.0 heights/sec
            x1 = 200.0 + i * 40.0
            y1 = 500.0
            x2 = x1 + 50.0
            y2 = y1 + 100.0
            det = self._make_det(1, "person", i, (x1, y1, x2, y2))
            observe(state, t, det, [], self.frame_w, self.frame_h)

        behaviors = detect_behaviors(state, [], self.frame_w, self.frame_h, now=1.8)
        self.assertIn("running", behaviors)

    def test_passing_by_behavior(self):
        # Closed 8s left->right street walk: duration < 20s, not approaching, |last_cx - first_cx| >= 0.3 * frame_w
        # 0.3 * 1920 = 576 px
        state = TrackState(
            track_id=1,
            class_name="person",
            first_seen=0.0,
            last_seen=0.0,
            first_center=(200.0, 300.0),
        )
        # 8s duration at 5 fps = 41 frames
        total_frames = 41
        for i in range(total_frames):
            t = i * 0.2
            # Walk from x=200 to x=900 (diff 700 > 576)
            x1 = 200.0 + i * (700.0 / (total_frames - 1))
            y1 = 250.0
            x2 = x1 + 60.0
            y2 = y1 + 150.0
            det = self._make_det(1, "person", i, (x1, y1, x2, y2))
            observe(state, t, det, [], self.frame_w, self.frame_h)

        # Unclosed -> not passing_by
        b_open = detect_behaviors(state, [], self.frame_w, self.frame_h, now=8.0, closed=False)
        self.assertNotIn("passing_by", b_open)

        # Closed -> passing_by
        b_closed = detect_behaviors(state, [], self.frame_w, self.frame_h, now=8.0, closed=True)
        self.assertIn("passing_by", b_closed)

    def test_vehicle_arrived_and_departed(self):
        # Car moving then still -> vehicle_arrived
        # Moving in head: displacement significant
        state_arrived = TrackState(
            track_id=10,
            class_name="car",
            first_seen=0.0,
            last_seen=0.0,
            first_center=(100.0, 500.0),
        )
        # First 15 frames: moving (x moves from 100 to 500)
        for i in range(15):
            t = i * 0.2
            x = 100.0 + i * 25.0
            det = self._make_det(10, "car", i, (x, 500.0, x + 200.0, 650.0))
            observe(state_arrived, t, det, [], self.frame_w, self.frame_h)

        # Next 30 frames (6s): stationary at x=500
        for i in range(15, 45):
            t = i * 0.2
            det = self._make_det(10, "car", i, (500.0, 500.0, 700.0, 650.0))
            observe(state_arrived, t, det, [], self.frame_w, self.frame_h)

        now = 45 * 0.2
        behaviors_arr = detect_behaviors(state_arrived, [], self.frame_w, self.frame_h, now=now)
        self.assertIn("vehicle_arrived", behaviors_arr)

        # Car still then moving -> vehicle_departed
        state_departed = TrackState(
            track_id=11,
            class_name="car",
            first_seen=0.0,
            last_seen=0.0,
            first_center=(500.0, 500.0),
        )
        # First 25 frames (5s): stationary at x=500
        for i in range(25):
            t = i * 0.2
            det = self._make_det(11, "car", i, (500.0, 500.0, 700.0, 650.0))
            observe(state_departed, t, det, [], self.frame_w, self.frame_h)

        # Next 25 frames (5s): moving (x moves from 500 to 900)
        for i in range(25, 50):
            t = i * 0.2
            x = 500.0 + (i - 25) * 16.0
            det = self._make_det(11, "car", i, (x, 500.0, x + 200.0, 650.0))
            observe(state_departed, t, det, [], self.frame_w, self.frame_h)

        now = 50 * 0.2
        behaviors_dep = detect_behaviors(state_departed, [], self.frame_w, self.frame_h, now=now)
        self.assertIn("vehicle_departed", behaviors_dep)


if __name__ == "__main__":
    unittest.main()
