"""Unit tests for the GPU-1 DetailedVerifier (core logic + parent-side application)."""

import json
import os
import shutil
import tempfile
import time
import unittest

import cv2
import numpy as np

from engine.detector import Detection
from engine.live.auditor import (
    DetailedVerifier,
    VerifierCore,
    VerifierResult,
    _extract_clip_keyframes,
    evaluate_detections,
)
from engine.live.db import EventStore


class FakeNotifier:
    def __init__(self):
        self.calls = []

    def notify(self, event_id, camera, kind, title, body, image_rel=None, extra=None):
        self.calls.append({"event_id": event_id, "camera": camera, "kind": kind, "title": title})


class FakeDetector:
    """Returns canned detections regardless of input frame."""

    def __init__(self, dets):
        self._dets = dets

    def detect_frame(self, frame, conf_threshold=0.25):
        return list(self._dets)


def _det(cls, conf):
    return Detection(class_name=cls, confidence=conf, bbox_xyxy=(0, 0, 10, 10), bbox_xywh=(5, 5, 10, 10))


class TestEvaluateDetections(unittest.TestCase):
    def test_missed_is_found_minus_seen_intersect_notify(self):
        dets = [_det("person", 0.7), _det("car", 0.9), _det("cat", 0.5)]
        found, missed = evaluate_detections(dets, seen_classes={"car"}, notify_classes={"person", "dog"})
        self.assertEqual(set(found.keys()), {"person", "car", "cat"})
        # car was seen; cat is not a notify class
        self.assertEqual(set(missed.keys()), {"person"})

    def test_keeps_highest_confidence_per_class(self):
        dets = [_det("person", 0.51), _det("person", 0.83)]
        found, _ = evaluate_detections(dets, seen_classes=set(), notify_classes={"person"})
        self.assertAlmostEqual(found["person"], 0.83)


class TestVerifierCore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.test_dir, "clip.mp4")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _core(self, dets):
        core = VerifierCore(
            model_name="yolov8s.pt",
            imgsz=1280,
            conf_threshold=0.25,
            notify_classes={"person", "dog", "cat"},
        )
        core.detector = FakeDetector(dets)
        return core

    def test_verify_frame_returns_only_missed_notify_classes(self):
        core = self._core([_det("person", 0.72), _det("car", 0.9)])
        missed = core.verify_frame(np.zeros((16, 16, 3), dtype=np.uint8), seen_classes={"car"})
        self.assertEqual(set(missed.keys()), {"person"})
        self.assertAlmostEqual(missed["person"], 0.72)

    def test_verify_frame_no_miss_when_realtime_saw_it(self):
        core = self._core([_det("person", 0.8)])
        missed = core.verify_frame(np.zeros((16, 16, 3), dtype=np.uint8), seen_classes={"person"})
        self.assertEqual(missed, {})

    def test_audit_clip_sweeps_keyframes(self):
        writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for i in range(30):
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            img[:, :] = (i * 5) % 255
            writer.write(img)
        writer.release()

        core = self._core([_det("dog", 0.66)])
        found = core.audit_clip(self.video_path)
        self.assertEqual(set(found.keys()), {"dog"})
        self.assertAlmostEqual(found["dog"], 0.66)

    def test_extract_keyframes_shapes(self):
        writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for i in range(30):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        frames = _extract_clip_keyframes(self.video_path, num_frames=4)
        self.assertEqual(len(frames), 4)
        for f in frames:
            self.assertEqual(f.shape, (48, 64, 3))

    def test_idle_detector_release(self):
        core = self._core([])
        sentinel = object()
        core.detector = sentinel
        core.idle_unload_seconds = 5.0
        now = time.time()

        core.last_work_time = now - 1.0
        self.assertFalse(core.release_detector_if_idle(now))
        self.assertIs(core.detector, sentinel)

        core.last_work_time = now - 10.0
        self.assertTrue(core.release_detector_if_idle(now))
        self.assertIsNone(core.detector)

        # No detector -> no-op
        self.assertFalse(core.release_detector_if_idle(now + 1.0))


class TestDetailedVerifierParent(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "events.db")
        self.store = EventStore(self.db_path)
        self.notifier = FakeNotifier()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _verifier(self, **kw):
        return DetailedVerifier(
            store=self.store,
            output_dir=self.test_dir,
            model_name="yolov8s.pt",
            device=1,
            imgsz=1280,
            notify_classes={"person", "dog", "cat"},
            notifier=self.notifier,
            **kw,
        )

    def _event(self, camera="Camera B", behaviors=("approaching",)):
        return self.store.create_event(
            camera=camera, started_at=time.time(), status="open",
            primary_class="vehicle", behaviors=json.dumps(list(behaviors)),
        )

    def test_verified_miss_tags_event_and_alerts(self):
        v = self._verifier()
        eid = self._event()
        v.apply_result(VerifierResult(
            kind="verified", camera="Camera B", event_id=eid,
            found={"person": 0.71}, missed={"person": 0.71},
        ))
        self.assertEqual(len(self.notifier.calls), 1)
        self.assertEqual(self.notifier.calls[0]["kind"], "detailed_person")
        self.assertEqual(self.notifier.calls[0]["event_id"], eid)

        behaviors = set(json.loads(self.store.get_event(eid)["behaviors"]))
        self.assertIn("verified_person", behaviors)
        self.assertIn("approaching", behaviors)  # existing tags preserved

    def test_verified_without_miss_does_not_alert(self):
        v = self._verifier()
        v.apply_result(VerifierResult(kind="verified", camera="Camera A", event_id=None,
                                      found={"person": 0.8}, missed={}))
        self.assertEqual(len(self.notifier.calls), 0)

    def test_audited_tags_event_without_alert(self):
        v = self._verifier()
        eid = self._event()
        v.apply_result(VerifierResult(kind="audited", event_id=eid, found={"dog": 0.6}))
        behaviors = set(json.loads(self.store.get_event(eid)["behaviors"]))
        self.assertIn("audited_dog", behaviors)
        self.assertEqual(len(self.notifier.calls), 0)

    def test_apply_result_is_idempotent(self):
        v = self._verifier()
        eid = self._event()
        for _ in range(2):
            v.apply_result(VerifierResult(kind="audited", event_id=eid, found={"dog": 0.6}))
        behaviors = json.loads(self.store.get_event(eid)["behaviors"])
        self.assertEqual(behaviors.count("audited_dog"), 1)

    def test_submit_sample_drops_when_queue_full(self):
        v = self._verifier(queue_size=1)
        v._in_q.put_nowait("blocker")
        ok = v.submit_sample("Camera A", None, np.zeros((8, 8, 3), dtype=np.uint8), set())
        self.assertFalse(ok)

    def test_audit_event_ignores_missing_clip(self):
        v = self._verifier()
        eid = self._event()
        v.audit_event(eid)  # no clip_path -> must not raise nor queue
        self.assertEqual(v._in_q.qsize(), 0)

    def test_status_shape(self):
        v = self._verifier()
        st = v.status()
        for key in ("device", "model", "imgsz", "child_alive", "samples_verified", "misses_caught"):
            self.assertIn(key, st)
        self.assertEqual(st["device"], 1)
        self.assertFalse(st["child_alive"])  # not started


if __name__ == "__main__":
    unittest.main()
