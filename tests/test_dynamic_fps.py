"""Unit tests for DynamicFpsController and motion-triggered boost."""

import unittest
import numpy as np

from engine.live.dynamic_fps import DynamicFpsConfig, DynamicFpsController


class TestDynamicFps(unittest.TestCase):
    def setUp(self):
        self.cfg = DynamicFpsConfig(
            enabled=True,
            idle_fps=2.0,       # 1 frame every 0.5s
            boost_fps=5.0,      # 1 frame every 0.2s
            motion_threshold=0.015,
            boost_cooldown=5.0, # 5s cooldown for test
        )
        self.controller = DynamicFpsController(self.cfg)

    def test_idle_throttling(self):
        # Create a static black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # First frame at t=0.0 -> should infer
        infer0, mode0, target_fps0 = self.controller.should_infer(0.0, frame, event_active=False)
        self.assertTrue(infer0)
        self.assertEqual(mode0, "idle")
        self.assertEqual(target_fps0, 2.0)

        # Next frame at t=0.1s -> within 0.5s interval -> should skip
        infer1, mode1, target_fps1 = self.controller.should_infer(0.1, frame, event_active=False)
        self.assertFalse(infer1)
        self.assertEqual(mode1, "idle")

        # Next frame at t=0.49s -> still within 0.5s interval -> should skip
        infer2, _, _ = self.controller.should_infer(0.49, frame, event_active=False)
        self.assertFalse(infer2)

        # Next frame at t=0.51s -> >0.5s elapsed -> should infer
        infer3, mode3, _ = self.controller.should_infer(0.51, frame, event_active=False)
        self.assertTrue(infer3)
        self.assertEqual(mode3, "idle")

    def test_instant_boost_on_event_active(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Start in idle at t=0.0
        self.controller.should_infer(0.0, frame, event_active=False)
        self.assertEqual(self.controller.current_mode, "idle")

        # At t=0.2s: event becomes active (e.g. person detected) -> instant boost!
        infer_event, mode_event, target_fps = self.controller.should_infer(0.2, frame, event_active=True)
        self.assertTrue(infer_event)
        self.assertEqual(mode_event, "boost")
        self.assertEqual(target_fps, 5.0)

        # Next frame at t=0.4s (0.2s interval at 5 FPS) -> should infer again!
        infer_next, mode_next, _ = self.controller.should_infer(0.4, frame, event_active=False)
        self.assertTrue(infer_next)
        self.assertEqual(mode_next, "boost")

    def test_boost_cooldown_hysteresis(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Trigger boost at t=1.0s via active track
        self.controller.should_infer(1.0, frame, event_active=False, active_tracks=1)
        self.assertEqual(self.controller.current_mode, "boost")

        # At t=3.0s (2s after activity): event ends, but within 5s cooldown -> stays in boost
        infer3, mode3, target_fps3 = self.controller.should_infer(3.0, frame, event_active=False, active_tracks=0)
        self.assertTrue(infer3)
        self.assertEqual(mode3, "boost")
        self.assertEqual(target_fps3, 5.0)

        # At t=6.5s (>5.0s after t=1.0s activity): cooldown expires -> returns to idle
        infer6, mode6, target_fps6 = self.controller.should_infer(6.5, frame, event_active=False, active_tracks=0)
        self.assertTrue(infer6)
        self.assertEqual(mode6, "idle")
        self.assertEqual(target_fps6, 2.0)

    def test_motion_detection_trigger(self):
        # Frame 1: black image
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        self.controller.should_infer(0.0, frame1)

        # Frame 2: significant white block (large motion > 1.5% pixels)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[100:300, 100:300] = 255  # 200x200 white rectangle

        infer, mode, target_fps = self.controller.should_infer(0.2, frame2)
        self.assertTrue(infer)
        self.assertEqual(mode, "boost")
        self.assertEqual(target_fps, 5.0)

    def test_pir_motion_gating_sleep_mode(self):
        cfg = DynamicFpsConfig(
            enabled=True,
            idle_fps=1.0,
            boost_fps=15.0,
            motion_threshold=0.015,
            boost_cooldown=3.0,
            motion_gate=True,
        )
        controller = DynamicFpsController(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Stationary scene with 1 tracked entity (no pixel motion)
        controller.should_infer(0.0, frame, active_tracks=1)
        # After cooldown (e.g. at t=4.0s > 3.0s), camera drops to idle / sleep!
        _, mode, target_fps = controller.should_infer(4.0, frame, active_tracks=1)
        self.assertEqual(mode, "idle")
        self.assertEqual(target_fps, 1.0)

        # Now actual motion occurs (white block moved into frame)
        moving_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        moving_frame[100:300, 100:300] = 255
        infer, mode_wake, target_fps_wake = controller.should_infer(4.2, moving_frame, active_tracks=1)
        self.assertTrue(infer)
        self.assertEqual(mode_wake, "boost")
        self.assertEqual(target_fps_wake, 15.0)


if __name__ == "__main__":
    unittest.main()
