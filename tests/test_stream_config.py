"""Unit tests for the dual-input RTSP stream command and live config parsing."""

import os
import shutil
import tempfile
import unittest

from engine.live.config import CameraConfig, RecordingConfig, load_live_config, resolve_inference_device
from engine.live.stream import CameraStream

MAIN = "rtsp://dvr:554/main"
SUB = "rtsp://dvr:554/sub"


def _cam(**kw):
    base = dict(name="Camera A", url=MAIN, gpu=0, zones=[], slug="Camera A")
    base.update(kw)
    return CameraConfig(**base)


class TestStreamCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stream(self, cam):
        rec = RecordingConfig(root=self.tmp, segment_seconds=10, ring_minutes=15)
        return CameraStream(cam, rec, fps=15)

    def test_single_input_without_substream(self):
        cmd = self._stream(_cam()).build_cmd(hwaccel=True)
        self.assertEqual(cmd.count("-i"), 1)
        self.assertIn(MAIN, cmd)
        # recording + pipe both map from input 0
        maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]
        self.assertEqual(maps, ["0:v:0", "0:a:0?", "0:v:0"])
        self.assertIn("fps=15", cmd)

    def test_dual_input_maps_sub_for_analysis_and_main_for_recording(self):
        cmd = self._stream(_cam(sub_url=SUB)).build_cmd(hwaccel=True)
        self.assertEqual(cmd.count("-i"), 2)
        self.assertIn(SUB, cmd)
        self.assertIn(MAIN, cmd)
        # First input is the analysis source, second is the recording source
        self.assertLess(cmd.index(SUB), cmd.index(MAIN))
        maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]
        self.assertEqual(maps, ["1:v:0", "1:a:0?", "0:v:0"])
        # Recording is stream copy; pipe keeps the fps filter
        self.assertIn("copy", cmd)
        self.assertIn("fps=15", cmd)

    def test_substream_failure_switches_back_to_main(self):
        stream = self._stream(_cam(sub_url=SUB))
        self.assertIn(SUB, stream.build_cmd(hwaccel=False))
        stream.sub_failed = True
        cmd = stream.build_cmd(hwaccel=False)
        self.assertEqual(cmd.count("-i"), 1)
        self.assertNotIn(SUB, cmd)

    def test_status_reports_source(self):
        stream = self._stream(_cam(sub_url=SUB))
        self.assertEqual(stream.status()["source"], "sub")
        stream.sub_failed = True
        self.assertEqual(stream.status()["source"], "main")

    def test_cpu_decode_omits_hwaccel(self):
        cmd = self._stream(_cam()).build_cmd(hwaccel=False)
        self.assertNotIn("-hwaccel", cmd)
        cmd_hw = self._stream(_cam()).build_cmd(hwaccel=True)
        self.assertIn("-hwaccel", cmd_hw)

    def test_live_only_camera_writes_nothing(self):
        cmd = self._stream(_cam(record=False)).build_cmd(hwaccel=True)
        # No segment muxer, no file output, single input, raw pipe only
        self.assertNotIn("segment", cmd)
        self.assertNotIn("-strftime", cmd)
        self.assertFalse(any(".mp4" in arg for arg in cmd))
        self.assertEqual(cmd.count("-i"), 1)
        self.assertEqual(cmd[-3:], ["-f", "rawvideo", "pipe:1"])
        # No recordings directory should be created for live-only cameras
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "Camera A")))

    def test_live_only_camera_prefers_substream(self):
        cmd = self._stream(_cam(record=False, sub_url=SUB)).build_cmd(hwaccel=False)
        self.assertIn(SUB, cmd)
        self.assertNotIn(MAIN, cmd)

    def test_recording_camera_still_creates_dir(self):
        self._stream(_cam()).build_cmd(hwaccel=False)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "Camera A")))


class TestLiveConfigParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, text: str) -> str:
        path = os.path.join(self.tmp, "live.yaml")
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_dual_gpu_and_substream_fields(self):
        path = self._write(
            """
analysis:
  fps: 15
  realtime_gpu: 0
  detailed_gpu: 1
  detailed_model: yolov8s.pt
  detailed_imgsz: 1280
  tracker_config: config/bytetrack_live.yaml
cameras:
  - name: Camera A
    url: "rtsp://dvr/main"
    sub_url: "rtsp://dvr/sub"
    gpu: 0
"""
        )
        cfg = load_live_config(path)
        self.assertEqual(cfg.analysis.fps, 15)
        self.assertEqual(cfg.analysis.realtime_gpu, 0)
        self.assertEqual(cfg.analysis.detailed_gpu, 1)
        self.assertEqual(cfg.analysis.detailed_model, "yolov8s.pt")
        self.assertEqual(cfg.analysis.detailed_imgsz, 1280)
        self.assertEqual(cfg.analysis.dynamic_fps.idle_fps, 3.0)
        self.assertEqual(cfg.analysis.dynamic_fps.boost_fps, 15.0)
        self.assertEqual(cfg.cameras[0].sub_url, "rtsp://dvr/sub")

    def test_missing_sub_url_defaults_empty(self):
        path = self._write(
            """
cameras:
  - name: Camera A
    url: "rtsp://dvr/main"
"""
        )
        cfg = load_live_config(path)
        self.assertEqual(cfg.cameras[0].sub_url, "")
        self.assertEqual(cfg.analysis.realtime_gpu, 0)
        self.assertEqual(cfg.analysis.detailed_gpu, 1)

    def test_per_camera_overrides_parsing(self):
        path = self._write(
            """
cameras:
  - name: Camera A
    url: "rtsp://dvr/main"
  - name: Camera D
    url: "rtsp://dvr/back"
    confidence: 0.20
    tracker_config: config/bytetrack_flicker.yaml
"""
        )
        cfg = load_live_config(path)
        by_name = {c.name: c for c in cfg.cameras}
        self.assertIsNone(by_name["Camera A"].confidence)         # falls back to analysis.*
        self.assertEqual(by_name["Camera D"].confidence, 0.20)
        self.assertEqual(by_name["Camera D"].tracker_config, "config/bytetrack_flicker.yaml")

    def test_record_flag_parsing(self):
        path = self._write(
            """
cameras:
  - name: Camera A
    url: "rtsp://dvr/main"
  - name: Camera D
    url: "rtsp://dvr/back"
    record: false
"""
        )
        cfg = load_live_config(path)
        by_name = {c.name: c for c in cfg.cameras}
        self.assertTrue(by_name["Camera A"].record)      # default is recording
        self.assertFalse(by_name["Camera D"].record)    # live-only


class TestInferenceDeviceResolution(unittest.TestCase):
    def test_auto_and_gpu_keep_gpu_behaviour(self):
        for spec in ("auto", "gpu", "cuda", "", None):
            self.assertEqual(resolve_inference_device(spec), "gpu", spec)

    def test_cpu_is_selected(self):
        self.assertEqual(resolve_inference_device("cpu"), "cpu")
        self.assertEqual(resolve_inference_device(" CPU "), "cpu")

    def test_unknown_falls_back_to_gpu(self):
        self.assertEqual(resolve_inference_device("tpu"), "gpu")

    def test_default_used_when_camera_has_no_override(self):
        self.assertEqual(resolve_inference_device(None, "cpu"), "cpu")
        self.assertEqual(resolve_inference_device("", "cpu"), "cpu")
        self.assertEqual(resolve_inference_device("gpu", "cpu"), "gpu")

    def test_config_parses_device_knobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "live.yaml")
            with open(path, "w") as fh:
                fh.write(
                    """
analysis:
  realtime_device: cpu
  detailed_device: gpu
  cpu_threads: 3
cameras:
  - name: Camera A
    url: "rtsp://dvr/main"
  - name: Camera D
    url: "rtsp://dvr/back"
    device: gpu
"""
                )
            cfg = load_live_config(path)
            self.assertEqual(cfg.analysis.realtime_device, "cpu")
            self.assertEqual(cfg.analysis.detailed_device, "gpu")
            self.assertEqual(cfg.analysis.cpu_threads, 3)
            by_name = {c.name: c for c in cfg.cameras}
            self.assertIsNone(by_name["Camera A"].device)
            self.assertEqual(by_name["Camera D"].device, "gpu")
            # Camera A inherits the global CPU setting, Camera D overrides it back to GPU
            self.assertEqual(resolve_inference_device(by_name["Camera A"].device, cfg.analysis.realtime_device), "cpu")
            self.assertEqual(resolve_inference_device(by_name["Camera D"].device, cfg.analysis.realtime_device), "gpu")

    def test_device_defaults_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "live.yaml")
            with open(path, "w") as fh:
                fh.write("cameras:\n  - name: Camera A\n    url: \"rtsp://dvr/main\"\n")
            cfg = load_live_config(path)
            self.assertEqual(cfg.analysis.realtime_device, "auto")
            self.assertEqual(cfg.analysis.detailed_device, "auto")
            self.assertEqual(cfg.analysis.cpu_threads, 0)
            self.assertIsNone(cfg.cameras[0].device)


class TestFfmpegDecodeDevice(unittest.TestCase):
    """ffmpeg must keep decoding on the GPU even after Ultralytics hides it from this process."""

    def _stream(self, cam):
        rec = RecordingConfig(root="/tmp", segment_seconds=10, ring_minutes=15)
        return CameraStream(cam, rec, fps=15)

    def test_explicit_decode_gpu_pins_child_env_and_index_zero(self):
        stream = self._stream(_cam(decode_gpu=1))
        self.assertEqual(stream._hwaccel_flags(), ["-hwaccel", "cuda", "-hwaccel_device", "0"])
        env = stream._decode_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_child_env_survives_torch_hiding_the_gpus(self):
        # Ultralytics sets CUDA_VISIBLE_DEVICES=-1 in the process when inference runs on CPU.
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            env = self._stream(_cam(decode_gpu=0))._decode_env()
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

    def test_without_decode_gpu_legacy_flags_and_inherited_env(self):
        stream = self._stream(_cam(gpu=1))
        self.assertEqual(stream._hwaccel_flags(), ["-hwaccel", "cuda", "-hwaccel_device", "1"])
        self.assertIsNone(stream._decode_env())

    def test_build_cmd_uses_decode_flags(self):
        cmd = self._stream(_cam(decode_gpu=1, record=False)).build_cmd(hwaccel=True)
        self.assertEqual(cmd[cmd.index("-hwaccel_device") + 1], "0")
        self.assertEqual(cmd.count("-hwaccel"), 1)


if __name__ == "__main__":
    unittest.main()
