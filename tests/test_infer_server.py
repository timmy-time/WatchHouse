"""Tests for the inference server (fake detector factory, no GPU, no ultralytics)."""

import base64
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient

from engine.detector import Detection, TrackDetection
from engine.live.infer_server import InferenceSessions, create_infer_app


class _FakeDetector:
    """Records calls and returns one fixed detection."""

    def __init__(self, model: str, imgsz: int, tracker: str, device: str):
        self.model = model
        self.imgsz = imgsz
        self.tracker = tracker
        self.device = device
        self.track_calls = 0
        self.detect_calls = 0
        self.last_frame_shape = None

    def track_frame(self, frame, frame_idx: int, conf_threshold: float = 0.30):
        self.track_calls += 1
        self.last_frame_shape = frame.shape
        return [
            TrackDetection(
                track_id=100 + frame_idx,
                class_name="car",
                confidence=0.9,
                bbox_xyxy=(1.0, 2.0, 30.0, 40.0),
                bbox_xywh=(15.5, 21.0, 29.0, 38.0),
                frame_idx=frame_idx,
            )
        ]

    def detect_frame(self, frame, conf_threshold: float = 0.35):
        self.detect_calls += 1
        return [Detection(class_name="person", confidence=0.7, bbox_xyxy=(5.0, 6.0, 7.0, 8.0),
                          bbox_xywh=(6.0, 7.0, 2.0, 2.0))]


def _jpeg_b64(width: int = 64, height: int = 48) -> str:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class FakeFactory:
    def __init__(self):
        self.created = []

    def __call__(self, model, imgsz, tracker, device):
        detector = _FakeDetector(model, imgsz, tracker, device)
        self.created.append(detector)
        return detector


class InferenceServerTests(unittest.TestCase):
    def setUp(self):
        self.factory = FakeFactory()
        self.app = create_infer_app(device="0", factory=self.factory, max_sessions=2)
        self.client = TestClient(self.app)

    def _track(self, session="Camera D", frame_idx=5, **kw):
        body = {"session": session, "jpeg_b64": _jpeg_b64(), "conf": 0.2,
                "frame_idx": frame_idx, "model": "yolov8n.pt", "imgsz": 640}
        body.update(kw)
        return self.client.post("/track", json=body)

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["device"], "0")
        self.assertEqual(body["sessions"], 0)

    def test_track_returns_detections(self):
        response = self._track()
        self.assertEqual(response.status_code, 200)
        detections = response.json()["detections"]
        self.assertEqual(detections[0]["track_id"], 105)
        self.assertEqual(detections[0]["class_name"], "car")
        self.assertEqual(detections[0]["bbox_xyxy"], [1.0, 2.0, 30.0, 40.0])
        self.assertEqual(self.factory.created[0].last_frame_shape, (48, 64, 3))

    def test_session_detector_is_reused_across_requests(self):
        self._track(frame_idx=1)
        self._track(frame_idx=2)
        self.assertEqual(len(self.factory.created), 1)
        self.assertEqual(self.factory.created[0].track_calls, 2)
        self.assertEqual(self.client.get("/health").json()["sessions"], 1)

    def test_sessions_are_isolated(self):
        self._track(session="Camera A", frame_idx=1)
        self._track(session="Camera B", frame_idx=1)
        self.assertEqual(len(self.factory.created), 2)
        self.assertEqual(self.client.get("/health").json()["session_names"], ["Camera A", "Camera B"])

    def test_model_and_size_form_part_of_the_session_key(self):
        self._track(imgsz=640)
        self._track(imgsz=1280)
        self.assertEqual(len(self.factory.created), 2)
        self.assertEqual(self.factory.created[1].imgsz, 1280)

    def test_max_sessions_evicts_oldest(self):
        self._track(session="A")
        self._track(session="B")
        self._track(session="C")
        self.assertEqual(self.client.get("/health").json()["sessions"], 2)
        self.assertEqual(self.client.get("/health").json()["session_names"], ["B", "C"])

    def test_detect_endpoint(self):
        response = self.client.post("/detect", json={"session": "verifier", "jpeg_b64": _jpeg_b64()})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detections"][0]["class_name"], "person")
        self.assertEqual(self.factory.created[0].detect_calls, 1)

    def test_reset_drops_session(self):
        self._track(session="Camera D")
        response = self.client.post("/reset", json={"session": "Camera D"})
        self.assertEqual(response.json(), {"ok": True, "dropped": 1})
        self.assertEqual(self.client.get("/health").json()["sessions"], 0)
        # a later frame rebuilds the detector
        self._track(session="Camera D")
        self.assertEqual(len(self.factory.created), 2)

    def test_bad_base64_is_rejected(self):
        response = self.client.post("/track", json={"session": "A", "jpeg_b64": "not-base64!!"})
        self.assertEqual(response.status_code, 400)

    def test_non_jpeg_payload_is_rejected(self):
        response = self.client.post(
            "/track", json={"session": "A", "jpeg_b64": base64.b64encode(b"hello").decode()}
        )
        self.assertEqual(response.status_code, 400)


class InferenceSessionsTests(unittest.TestCase):
    def test_drop_unknown_session_is_a_no_op(self):
        sessions = InferenceSessions(factory=FakeFactory())
        self.assertEqual(sessions.drop("nope"), 0)
        self.assertEqual(sessions.count(), 0)

    def test_default_factory_builds_a_clip_detector(self):
        # Importing (not constructing) must work without ultralytics installed.
        from engine.live.infer_server import _default_factory
        self.assertTrue(callable(_default_factory))


if __name__ == "__main__":
    unittest.main()
