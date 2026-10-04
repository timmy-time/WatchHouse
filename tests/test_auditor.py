"""Unit tests for the background high-resolution EventAuditor."""

import os
import shutil
import tempfile
import time
import unittest

import cv2
import numpy as np

from engine.live.auditor import EventAuditor, _extract_clip_keyframes
from engine.live.db import EventStore


class TestEventAuditor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "events.db")
        self.store = EventStore(self.db_path)
        self.video_path = os.path.join(self.test_dir, "clip.mp4")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_test_video(self, frames: int = 30, size=(64, 48)) -> None:
        writer = cv2.VideoWriter(
            self.video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            size,
        )
        for i in range(frames):
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            img[:, :] = (i * 5) % 255
            writer.write(img)
        writer.release()

    def test_extract_clip_keyframes_returns_frames(self):
        self._make_test_video(frames=30)
        frames = _extract_clip_keyframes(self.video_path, num_frames=4)
        self.assertEqual(len(frames), 4)
        for f in frames:
            self.assertEqual(f.shape, (48, 64, 3))

    def test_extract_clip_keyframes_missing_file(self):
        self.assertEqual(_extract_clip_keyframes(os.path.join(self.test_dir, "nope.mp4")), [])

    def test_audit_event_without_clip_is_noop(self):
        # Event that has no clip_path -> auditor must silently skip, not crash
        eid = self.store.create_event(
            camera="Camera A",
            started_at=time.time(),
            status="closed",
            primary_class="vehicle",
        )
        auditor = EventAuditor(
            store=self.store,
            output_dir=self.test_dir,
            model_name="yolov8n.pt",
            device=0,
        )
        auditor.start()
        auditor.audit_event(eid)
        auditor.audit_event(None)  # sentinel-safe: stop path
        time.sleep(0.5)
        auditor.stop()

        # Event must remain untouched (no behaviors added)
        ev = self.store.get_event(eid)
        self.assertEqual(ev["behaviors"], "[]")

    def test_finalizer_callback_contract(self):
        from engine.live.clipper import Finalizer
        from engine.live.config import CameraConfig, LiveConfig

        cam = CameraConfig(name="Camera A", url="", gpu=0, zones=[], slug="Camera A")
        cfg = LiveConfig()
        recorded = []

        finalizer = Finalizer(
            cam, cfg, self.store, self.test_dir,
            on_finalized=lambda eid: recorded.append(eid),
        )
        # Callback is optional-safe and only invoked on successful finalize;
        # verify the wiring attribute contract here.
        self.assertTrue(callable(finalizer.on_finalized))
        finalizer.on_finalized(42)
        self.assertEqual(recorded, [42])

    def test_idle_detector_release(self):
        auditor = EventAuditor(
            store=self.store,
            output_dir=self.test_dir,
            idle_unload_seconds=5.0,
        )
        sentinel = object()
        now = time.time()

        # Recent work -> keep model resident
        auditor.detector = sentinel
        auditor.last_work_time = now - 1.0
        auditor._release_detector_if_idle(now)
        self.assertIs(auditor.detector, sentinel)

        # Idle past threshold -> release model context
        auditor.last_work_time = now - 10.0
        auditor._release_detector_if_idle(now)
        self.assertIsNone(auditor.detector)

        # Subsequent call with no detector must not raise
        auditor._release_detector_if_idle(now + 1.0)
        self.assertIsNone(auditor.detector)


if __name__ == "__main__":
    unittest.main()
