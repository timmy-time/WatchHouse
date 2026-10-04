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

    def test_anchored_vehicle_suppressed_and_departed(self):
        import json
        from engine.scenery import SceneryManager

        # Register a slot for Camera C: [0.1, 0.1, 0.3, 0.3]
        slot_box = [0.10, 0.10, 0.30, 0.30]
        self.store.create_vehicle_slot(
            camera="Camera C",
            name="Parked Car 1",
            slot_box=json.dumps(slot_box),
            color_name="red",
            appearance_sig=json.dumps({"aspect_ratio": 1.5, "hsv_bins": []}),
            is_friendly=1,
        )
        scenery = SceneryManager(self.store)
        self.manager.scenery = scenery

        # 1. Car sitting in the slot in pixels (frame 1920x1080)
        # Slot in pixels: [192, 108, 576, 324]
        car_box = (200.0, 115.0, 560.0, 315.0)

        # Run 50 frames with jitter (simulating wind/shadows)
        for i in range(50):
            now = i * 0.2
            jitter_x = (i % 3) * 5.0
            b = (car_box[0] + jitter_x, car_box[1], car_box[2] + jitter_x, car_box[3])
            det = self._make_det(track_id=10, class_name="car", frame_idx=i, bbox=b, conf=0.85)
            self.manager.process(frame=None, dets=[det], now=now, frame_w=1920, frame_h=1080)

        # Vehicle is anchored -> must NOT open any event
        self.assertIsNone(self.manager.open_event_id)
        self.assertEqual(len(self.notifier.calls), 0)

        # 2. Car drives out of slot: moved far away to [1200, 500, 1560, 700]
        for i in range(50, 70):
            now = i * 0.2
            # Moving away
            step = (i - 50) * 40.0
            move_b = (car_box[0] + step, car_box[1] + step, car_box[2] + step, car_box[3] + step)
            det = self._make_det(track_id=10, class_name="car", frame_idx=i, bbox=move_b, conf=0.85)
            self.manager.process(frame=None, dets=[det], now=now, frame_w=1920, frame_h=1080)

        # Moving vehicle event opens with departed behavior
        self.assertIsNotNone(self.manager.open_event_id)
        self.assertIn("vehicle_departed", self.manager.open_event_behaviors)

    def test_ignore_mask_suppresses_detection(self):
        from engine.behavior import Zone
        # Add an ignore zone covering [0.65, 0.55] to [1.0, 1.0] (engine pile location)
        ignore_zone = Zone(
            name="engine_pile",
            type="ignore",
            polygon=[(0.65, 0.55), (1.0, 0.55), (1.0, 1.0), (0.65, 1.0)],
        )
        self.cam.zones.append(ignore_zone)

        # Detection center inside ignore mask: [1300, 630, 1880, 1070]
        engine_box = (1300.0, 630.0, 1880.0, 1070.0)
        for i in range(10):
            now = i * 0.2
            det = self._make_det(track_id=16, class_name="person", frame_idx=i, bbox=engine_box, conf=0.75)
            self.manager.process(frame=None, dets=[det], now=now, frame_w=1920, frame_h=1080)

        # Should be filtered out by ignore zone, zero events opened
        self.assertIsNone(self.manager.open_event_id)
        self.assertEqual(len(self.manager.tracks), 0)

    def test_live_only_camera_finalizes_without_clip(self):
        """record=False cameras close as 'finalized' with no clip and no on_closed."""
        cam = CameraConfig(
            name="Camera D", url="rtsp://dvr/back", gpu=0, zones=[],
            slug="Camera_D", record=False,
        )
        closed_calls = []
        manager = EventManager(
            cam=cam,
            cfg=self.cfg,
            store=self.store,
            notifier=self.notifier,
            output_dir=self.test_dir,
            face_engine=None,
            gallery=None,
            on_closed=lambda eid, s, e: closed_calls.append(eid),
        )

        box = (200.0, 200.0, 300.0, 450.0)
        for i in range(4):
            now = i * 0.2
            det = self._make_det(track_id=7, class_name="person", frame_idx=i, bbox=box, conf=0.7)
            manager.process(frame=None, dets=[det], now=now, frame_w=1920, frame_h=1080)

        event_id = manager.open_event_id
        self.assertIsNotNone(event_id)

        # Person gone: after post_roll the event closes
        for i in range(4, 90):
            manager.process(frame=None, dets=[], now=i * 0.2, frame_w=1920, frame_h=1080)

        ev = self.store.get_event(event_id)
        self.assertEqual(ev["status"], "finalized")     # complete, not awaiting a clip
        self.assertIsNone(ev["clip_path"])              # nothing written
        self.assertEqual(closed_calls, [])              # no clip assembly requested


if __name__ == "__main__":
    unittest.main()
