"""Unit tests for automatic day/night lighting mode detection and tracking profile adaptation."""

import unittest
import numpy as np

from engine.live.lighting import (
    NightModeConfig,
    LightingModeTracker,
    measure_saturation,
)


class TestLightingModeTracker(unittest.TestCase):
    def test_measure_saturation_grayscale_vs_color(self):
        # Pure monochrome grayscale frame (R=G=B)
        bw_frame = np.full((180, 320, 3), 128, dtype=np.uint8)
        sat_bw = measure_saturation(bw_frame)
        self.assertEqual(sat_bw, 0.0)

        # Full saturated color frame
        color_frame = np.full((180, 320, 3), [255, 0, 0], dtype=np.uint8)  # pure blue
        sat_color = measure_saturation(color_frame)
        self.assertGreater(sat_color, 150.0)

        # Empty or None frame
        self.assertEqual(measure_saturation(None), 0.0)
        self.assertEqual(measure_saturation(np.zeros((0, 0, 3), dtype=np.uint8)), 0.0)

    def test_startup_initialization_at_night(self):
        cfg = NightModeConfig(night_threshold=6.0, day_threshold=15.0)
        tracker = LightingModeTracker(cfg)

        # Boot in dark IR mode (sat=1.0) -> should initialize directly into night mode
        frame = np.full((90, 160, 3), 100, dtype=np.uint8)
        changed, mode, sat = tracker.update(frame, now=100.0)
        self.assertTrue(changed)
        self.assertEqual(mode, "night")
        self.assertEqual(tracker.current_mode, "night")

    def test_startup_initialization_in_daylight(self):
        cfg = NightModeConfig(night_threshold=6.0, day_threshold=15.0)
        tracker = LightingModeTracker(cfg)

        # Boot in daylight (sat=100.0) -> should initialize in day mode
        frame = np.full((90, 160, 3), [200, 100, 50], dtype=np.uint8)
        changed, mode, sat = tracker.update(frame, now=100.0)
        self.assertFalse(changed)
        self.assertEqual(mode, "day")
        self.assertEqual(tracker.current_mode, "day")

    def test_hysteresis_and_debounce_prevent_sporadic_flapping(self):
        cfg = NightModeConfig(
            night_threshold=6.0,
            day_threshold=15.0,
            confirm_seconds=4.0,
            cooldown_seconds=15.0,
        )
        tracker = LightingModeTracker(cfg)

        # Start in day mode
        color_frame = np.full((90, 160, 3), [200, 100, 50], dtype=np.uint8)
        tracker.update(color_frame, now=1.0)
        self.assertEqual(tracker.current_mode, "day")

        # Brief shadow/headlight glitch for 2 seconds (less than 4s confirm window)
        ir_frame = np.full((90, 160, 3), 100, dtype=np.uint8)
        for t in [2.0, 3.0]:
            changed, mode, _ = tracker.update(ir_frame, now=t)
            self.assertFalse(changed)
            self.assertEqual(mode, "day")

        # Color returns at t=4s -> glitch cancelled
        tracker.update(color_frame, now=4.0)
        self.assertEqual(tracker.current_mode, "day")

        # Sustained night transition starting at t=10s
        for t in [10.0, 11.0, 12.0, 13.0]:
            changed, mode, _ = tracker.update(ir_frame, now=t)
            self.assertFalse(changed)  # Still in confirmation window
            self.assertEqual(mode, "day")

        # At t=14.5s (4.5s of continuous night >= 4.0s confirmation window), mode switches cleanly
        changed, mode, _ = tracker.update(ir_frame, now=14.5)
        self.assertTrue(changed)
        self.assertEqual(mode, "night")
        self.assertEqual(tracker.current_mode, "night")

        # Cooldown lock: brief light burst at t=16.0s (only 1.5s after switch, cooldown is 15s)
        # Cannot switch back during cooldown
        for t in [16.0, 17.0, 18.0, 20.0]:
            changed, mode, _ = tracker.update(color_frame, now=t)
            self.assertFalse(changed)
            self.assertEqual(mode, "night")

        # Once cooldown has elapsed (now >= 14.5 + 15 = 29.5s) and daylight persists for confirm window:
        for t in [30.0, 31.0, 32.0, 33.0]:
            changed, mode, _ = tracker.update(color_frame, now=t)
            self.assertFalse(changed)
            self.assertEqual(mode, "night")

        # At t=34.5s (4.5s continuous daylight), switches back to day cleanly
        changed, mode, _ = tracker.update(color_frame, now=34.5)
        self.assertTrue(changed)
        self.assertEqual(mode, "day")

    def test_deadband_preserves_current_mode(self):
        cfg = NightModeConfig(night_threshold=6.0, day_threshold=15.0)
        tracker = LightingModeTracker(cfg)

        # Set initial day mode
        color_frame = np.full((90, 160, 3), [200, 100, 50], dtype=np.uint8)
        tracker.update(color_frame, now=1.0)

        # Twilight frame whose saturation sits inside the deadband (e.g. 10.0)
        # In deadband, no candidate is formed
        deadband_frame = np.full((90, 160, 3), [110, 115, 100], dtype=np.uint8)
        for t in range(2, 20):
            changed, mode, _ = tracker.update(deadband_frame, now=float(t))
            self.assertFalse(changed)
            self.assertEqual(mode, "day")


class TestWorkerLightingIntegration(unittest.TestCase):
    def test_worker_switches_profiles_and_updates_detector(self):
        from engine.live.config import CameraConfig, LiveConfig, AnalysisConfig
        from engine.live.worker import CameraWorker

        cfg = LiveConfig(
            analysis=AnalysisConfig(
                confidence=0.30,
                imgsz=640,
                tracker_config="config/bytetrack_live.yaml",
                night_mode=NightModeConfig(
                    enabled=True,
                    confidence=0.18,
                    tracker_config="config/bytetrack_mild.yaml",
                    imgsz=960,
                ),
            )
        )
        cam = CameraConfig(name="Driveway", url="", slug="Driveway")

        worker = CameraWorker.__new__(CameraWorker)
        worker.cam = cam
        worker.cfg = cfg
        worker.conf_threshold = cfg.analysis.confidence
        worker.tracker_config = cfg.analysis.tracker_config
        worker.imgsz = cfg.analysis.imgsz

        worker.day_conf = worker.conf_threshold
        worker.day_tracker = worker.tracker_config
        worker.day_imgsz = worker.imgsz

        night_cfg = cfg.analysis.night_mode
        worker.night_conf = night_cfg.confidence
        worker.night_tracker = night_cfg.tracker_config
        worker.night_imgsz = night_cfg.imgsz

        class DummyDetector:
            def __init__(self):
                self.tracker_config = "config/bytetrack_live.yaml"
                self.imgsz = 640

        worker.detector = DummyDetector()
        worker.lighting_mode = "day"

        # Switch to night
        worker._apply_lighting_mode("night")
        self.assertEqual(worker.lighting_mode, "night")
        self.assertEqual(worker.conf_threshold, 0.18)
        self.assertEqual(worker.tracker_config, "config/bytetrack_mild.yaml")
        self.assertEqual(worker.imgsz, 960)
        self.assertEqual(worker.detector.tracker_config, "config/bytetrack_mild.yaml")
        self.assertEqual(worker.detector.imgsz, 960)

        # Switch back to day
        worker._apply_lighting_mode("day")
        self.assertEqual(worker.lighting_mode, "day")
        self.assertEqual(worker.conf_threshold, 0.30)
        self.assertEqual(worker.tracker_config, "config/bytetrack_live.yaml")
        self.assertEqual(worker.imgsz, 640)
        self.assertEqual(worker.detector.tracker_config, "config/bytetrack_live.yaml")
        self.assertEqual(worker.detector.imgsz, 640)

if __name__ == "__main__":
    unittest.main()
