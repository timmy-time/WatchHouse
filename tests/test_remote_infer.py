"""Tests for the remote inference client (stub HTTP server, no GPU, no ultralytics)."""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from engine.live.config import resolve_inference_device
from engine.live.remote_infer import RemoteDetector


class _StubHandler(BaseHTTPRequestHandler):
    requests = []
    response = {"detections": []}
    status = 200
    delay = 0.0

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append({"path": self.path, "body": body})
        if self.delay:
            time.sleep(self.delay)
        payload = json.dumps(type(self).response).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence test noise
        pass


class RemoteDetectorTests(unittest.TestCase):
    def setUp(self):
        _StubHandler.requests = []
        _StubHandler.response = {"detections": []}
        _StubHandler.status = 200
        _StubHandler.delay = 0.0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.frame = np.zeros((396, 704, 3), dtype=np.uint8)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _detector(self, **kw):
        return RemoteDetector(url=self.url, session="Camera D", model_path="yolov8n.pt",
                              imgsz=640, tracker_config="bytetrack.yaml", **kw)

    def test_track_frame_round_trips_detections(self):
        _StubHandler.response = {
            "detections": [
                {"track_id": 7, "class_name": "car", "confidence": 0.83,
                 "bbox_xyxy": [10.0, 20.0, 110.0, 220.0], "frame_idx": 42},
                {"track_id": 8, "class_name": "person", "confidence": 0.51,
                 "bbox_xyxy": [0.0, 0.0, 50.0, 200.0], "frame_idx": 42},
            ]
        }
        dets = self._detector().track_frame(self.frame, frame_idx=42, conf_threshold=0.2)

        self.assertEqual([d.track_id for d in dets], [7, 8])
        self.assertEqual([d.class_name for d in dets], ["car", "person"])
        self.assertAlmostEqual(dets[0].confidence, 0.83)
        self.assertEqual(dets[0].bbox_xyxy, (10.0, 20.0, 110.0, 220.0))
        # xywh is derived locally from xyxy
        self.assertEqual(dets[0].bbox_xywh, (60.0, 120.0, 100.0, 200.0))
        self.assertEqual(dets[0].frame_idx, 42)

        sent = _StubHandler.requests[-1]
        self.assertEqual(sent["path"], "/track")
        self.assertEqual(sent["body"]["session"], "Camera D")
        self.assertEqual(sent["body"]["frame_idx"], 42)
        self.assertEqual(sent["body"]["conf"], 0.2)
        self.assertEqual(sent["body"]["model"], "yolov8n.pt")
        self.assertEqual(sent["body"]["imgsz"], 640)
        self.assertGreater(len(sent["body"]["jpeg_b64"]), 100)

    def test_frame_is_sent_as_a_decodable_jpeg(self):
        import base64
        import cv2

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[10:60, 20:120] = 255
        self._detector(jpeg_quality=70).detect_frame(frame, conf_threshold=0.3)

        raw = base64.b64decode(_StubHandler.requests[-1]["body"]["jpeg_b64"])
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, (100, 200, 3))

    def test_detect_frame_parses_stateless_detections(self):
        _StubHandler.response = {
            "detections": [{"class_name": "dog", "confidence": 0.66, "bbox_xyxy": [1, 2, 3, 4]}]
        }
        dets = self._detector().detect_frame(self.frame)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].class_name, "dog")
        self.assertEqual(_StubHandler.requests[-1]["path"], "/detect")

    def test_http_error_returns_empty_and_counts(self):
        _StubHandler.status = 500
        detector = self._detector()
        self.assertEqual(detector.track_frame(self.frame, frame_idx=1), [])
        stats = detector.stats()
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["calls"], 0)
        self.assertIn("500", stats["last_error"])

    def test_timeout_returns_empty_without_raising(self):
        _StubHandler.delay = 0.4
        detector = self._detector(timeout=0.15)
        started = time.perf_counter()
        self.assertEqual(detector.track_frame(self.frame, frame_idx=1), [])
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(detector.stats()["errors"], 1)

    def test_successful_calls_track_latency(self):
        detector = self._detector()
        detector.track_frame(self.frame, frame_idx=1)
        stats = detector.stats()
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["errors"], 0)
        self.assertGreaterEqual(stats["last_ms"], 0.0)
        self.assertEqual(stats["backend"], "remote")

    def test_reset_posts_session(self):
        detector = self._detector()
        self.assertTrue(detector.reset())
        self.assertEqual(_StubHandler.requests[-1]["path"], "/reset")
        self.assertEqual(_StubHandler.requests[-1]["body"], {"session": "Camera D"})

    def test_reset_failure_is_not_fatal(self):
        self.server.shutdown()
        detector = self._detector(timeout=0.2)
        self.assertFalse(detector.reset())

    def test_malformed_detections_are_skipped(self):
        _StubHandler.response = {
            "detections": [
                {"track_id": 1, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [1, 2, 3]},
                {"track_id": 2, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [1, 2, 3, 4]},
            ]
        }
        dets = self._detector().track_frame(self.frame, frame_idx=1)
        self.assertEqual([d.track_id for d in dets], [2])

    def test_requires_url(self):
        with self.assertRaises(ValueError):
            RemoteDetector(url="", session="x")


class WorkerBackendSelectionTests(unittest.TestCase):
    """The worker must pick the backend from the resolved device."""

    def _worker(self, device: str, remote_url: str = "http://10.0.0.5:8099"):
        from engine.live.config import AnalysisConfig, CameraConfig, LiveConfig
        from engine.live.worker import CameraWorker

        worker = CameraWorker.__new__(CameraWorker)
        worker.cam = CameraConfig(name="Camera D", url="rtsp://dvr/back", slug="Camera_D")
        worker.cfg = LiveConfig(analysis=AnalysisConfig(realtime_device=device, remote_url=remote_url))
        worker.model_path = "yolov8n.pt"
        worker.imgsz = 640
        worker.tracker_config = "bytetrack.yaml"
        worker.inference_device = resolve_inference_device(None, device)
        return worker

    def test_remote_device_builds_remote_detector(self):
        detector = self._worker("remote")._build_detector()
        self.assertIsInstance(detector, RemoteDetector)
        self.assertEqual(detector.url, "http://10.0.0.5:8099")
        self.assertEqual(detector.session, "Camera_D")
        self.assertEqual(detector.model_path, "yolov8n.pt")
        detector.close()

    def test_remote_device_needs_a_url(self):
        # Without a url the worker falls back to local GPU inference (no RemoteDetector).
        with self.assertRaises(ValueError):
            self._worker("remote", remote_url="")._build_detector()

    def test_local_device_builds_clip_detector(self):
        # Ultralytics is absent outside the container, so construction raises — which
        # proves the local branch was taken rather than the remote one.
        with self.assertRaises(RuntimeError):
            self._worker("gpu")._build_detector()

    def test_remote_config_requires_remote_device(self):
        self.assertEqual(resolve_inference_device("remote"), "remote")
        self.assertEqual(resolve_inference_device(None, "remote"), "remote")
        self.assertEqual(resolve_inference_device("gpu", "remote"), "gpu")
        self.assertEqual(self._worker("remote").inference_device, "remote")


class RemoteConfigParsingTests(unittest.TestCase):
    def test_remote_keys_parse(self):
        import os
        import tempfile

        from engine.live.config import load_live_config

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "live.yaml")
            with open(path, "w") as fh:
                fh.write(
                    """
analysis:
  realtime_device: remote
  detailed_device: remote
  remote_url: "http://192.0.2.20:8099"
  remote_timeout: 4.5
  remote_jpeg_quality: 65
cameras:
  - name: Camera D
    url: "rtsp://dvr/back"
"""
                )
            cfg = load_live_config(path)
        self.assertEqual(cfg.analysis.realtime_device, "remote")
        self.assertEqual(cfg.analysis.detailed_device, "remote")
        self.assertEqual(cfg.analysis.remote_url, "http://192.0.2.20:8099")
        self.assertEqual(cfg.analysis.remote_timeout, 4.5)
        self.assertEqual(cfg.analysis.remote_jpeg_quality, 65)

    def test_remote_defaults_are_inert(self):
        from engine.live.config import AnalysisConfig

        cfg = AnalysisConfig()
        self.assertEqual(cfg.remote_url, "")
        self.assertEqual(resolve_inference_device(cfg.realtime_device), "gpu")


if __name__ == "__main__":
    unittest.main()
