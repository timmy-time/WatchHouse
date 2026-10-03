"""Unit tests for EventManager lifecycle, qualification, and closing."""

import os
import shutil
import tempfile
import unittest

from engine.detector import TrackDetection
from engine.live.config import CameraConfig, LiveConfig
from engine.live.db import EventStore
from engine.live.events import EventManager


class FakeNotifier:
    def __init__(self):
        self.calls = []

    def notify(self, event_id, camera, kind, title, body, image_rel=None, extra=None):
        self.calls.append({
            "event_id": event_id,
            "camera": camera,
            "kind": kind,
            "title": title,
            "body": body,
            "image_rel": image_rel,
            "extra": extra,
        })


class TestLiveEvents(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "events.db")
        self.store = EventStore(self.db_path)
        self.notifier = FakeNotifier()
        self.cfg = LiveConfig()
        self.cfg.analysis.post_roll_seconds = 10
        self.cfg.analysis.max_event_seconds = 300
        self.cam = CameraConfig(name="Camera C", url="", gpu=0, zones=[], slug="Camera_C")
        self.closed_calls = []

        def _on_closed(event_id, start_ts, end_ts):
            self.closed_calls.append((event_id, start_ts, end_ts))

        self.on_closed = _on_closed
        self.manager = EventManager(
            cam=self.cam,
            cfg=self.cfg,
            store=self.store,
            notifier=self.notifier,
            output_dir=self.test_dir,
            face_engine=None,
            gallery=None,
            on_closed=self.on_closed,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_det(self, track_id: int, class_name: str, frame_idx: int, bbox: tuple, conf: float = 0.6):
        x1, y1, x2, y2 = bbox
        return TrackDetection(
            track_id=track_id,
            class_name=class_name,
            confidence=conf,
            bbox_xyxy=bbox,
            bbox_xywh=((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
            frame_idx=frame_idx,
        )

    def test_parked_car_produces_zero_events(self):
        # Parked car: completely stationary box for 60 seconds (300 frames at 5 fps)
        # Bbox doesn't move -> min_iou = 1.0, normalized_displacement = 0.0 -> stationary vehicle
        car_box = (100.0, 100.0, 300.0, 250.0)
        for i in range(300):
            now = i * 0.2
            det = self._make_det(track_id=1, class_name="car", frame_idx=i, bbox=car_box, conf=0.85)
            self.manager.process(frame=None, dets=[det], now=now, frame_w=1920, frame_h=1080)

        total, items = self.store.list_events()
        self.assertEqual(total, 0)
        self.assertEqual(len(self.notifier.calls), 0)
        self.assertIsNone(self.manager.open_event_id)

    def test_person_track_opens_and_closes_event(self):
        # 1. 2 frames of person conf 0.6 -> not qualified yet (needs >= 3 frames)
        box = (200.0, 200.0, 300.0, 450.0)
        for i in range(2):
            now = i * 0.2
            det = self._make_det(track_id=2, class_name="person", frame_idx=i, bbox=box, conf=0.6)
            self.manager.process(frame=None, dets=[det], now=now, frame_w=1920, frame_h=1080)

        self.assertIsNone(self.manager.open_event_id)
        self.assertEqual(len(self.notifier.calls), 0)

        # 2. 3rd frame -> qualifies -> event opens!
        now = 2 * 0.2
        det = self._make_det(track_id=2, class_name="person", frame_idx=2, bbox=box, conf=0.6)
        self.manager.process(frame=None, dets=[det], now=now, frame_w=1920, frame_h=1080)

        self.assertIsNotNone(self.manager.open_event_id)
        self.assertEqual(len(self.notifier.calls), 1)
        self.assertEqual(self.notifier.calls[0]["kind"], "person")
        self.assertEqual(self.notifier.calls[0]["title"], "Person at Camera C")

        event_id = self.manager.open_event_id

        # 3. Person stays visible until t=5.0s
        for i in range(3, 26):
            now = i * 0.2
            det = self._make_det(track_id=2, class_name="person", frame_idx=i, bbox=box, conf=0.6)
            self.manager.process(frame=None, dets=[det], now=now, frame_w=1920, frame_h=1080)

        # Person still visible, still only 1 person notification
        self.assertEqual(len(self.notifier.calls), 1)
        last_seen = 25 * 0.2  # 5.0s

        # 4. Person disappears. Run loop without detections for 9s (t = 5.0 to 14.0)
        # post_roll_seconds = 10 -> still open at 9s
        for i in range(26, 71):
            now = i * 0.2
            self.manager.process(frame=None, dets=[], now=now, frame_w=1920, frame_h=1080)

        self.assertIsNotNone(self.manager.open_event_id)
        self.assertEqual(len(self.closed_calls), 0)

        # 5. At t = 15.2s (elapsed > 10s post-roll), event closes!
        now = 15.2
        self.manager.process(frame=None, dets=[], now=now, frame_w=1920, frame_h=1080)

        self.assertIsNone(self.manager.open_event_id)
        self.assertEqual(len(self.closed_calls), 1)
        closed_id, start_ts, end_ts = self.closed_calls[0]
        self.assertEqual(closed_id, event_id)
        self.assertAlmostEqual(end_ts, last_seen, places=2)

        # Verify DB row
        ev = self.store.get_event(event_id)
        self.assertEqual(ev["status"], "closed")
        self.assertEqual(ev["primary_class"], "person")

    def test_force_close_at_max_event_seconds(self):
        # Configure max_event_seconds = 20s
        self.cfg.analysis.max_event_seconds = 20
        box = (200.0, 200.0, 300.0, 450.0)

        # Feed person detections from t=0 to t=25s (continuously present)
        for i in range(126):
            now = i * 0.2
            det = self._make_det(track_id=3, class_name="person", frame_idx=i, bbox=box, conf=0.7)
            self.manager.process(frame=None, dets=[det], now=now, frame_w=1920, frame_h=1080)

        # Event should have been force closed at ~20s, and a new one opened
        self.assertGreaterEqual(len(self.closed_calls), 1)
        first_closed = self.closed_calls[0]
        self.assertAlmostEqual(first_closed[2] - first_closed[1], 20.0, delta=1.0)
        # Manager should have a second event open for the continuous track
        self.assertIsNotNone(self.manager.open_event_id)
        self.assertNotEqual(self.manager.open_event_id, first_closed[0])


if __name__ == "__main__":
    unittest.main()
