"""Unit tests for worker-side helpers (box velocity for client-side prediction)."""

import unittest
from collections import deque

from engine.detector import TrackDetection
from engine.live.worker import _track_velocity


class FakeTrack:
    def __init__(self, window):
        self.window = deque(window)


def _det(cx: float, cy: float, frame_idx: int = 0, w: float = 100.0, h: float = 100.0) -> TrackDetection:
    return TrackDetection(
        track_id=1,
        class_name="car",
        confidence=0.8,
        bbox_xyxy=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        bbox_xywh=(cx, cy, w, h),
        frame_idx=frame_idx,
    )


class TestTrackVelocity(unittest.TestCase):
    def test_stationary_track_has_zero_velocity(self):
        track = FakeTrack([(t * 0.2, _det(500.0, 500.0)) for t in range(10)])
        vx, vy = _track_velocity(track, 1000, 1000)
        self.assertAlmostEqual(vx, 0.0, places=3)
        self.assertAlmostEqual(vy, 0.0, places=3)

    def test_moving_track_velocity_matches_motion(self):
        # 100 px per second to the right on a 1000 px wide frame => 0.1 units/s
        track = FakeTrack([(t * 0.2, _det(100.0 + t * 20.0, 500.0)) for t in range(10)])
        vx, vy = _track_velocity(track, 1000, 1000)
        self.assertAlmostEqual(vx, 0.1, places=2)
        self.assertAlmostEqual(vy, 0.0, places=3)

    def test_single_observation_is_zero(self):
        track = FakeTrack([(0.0, _det(100.0, 100.0))])
        self.assertEqual(_track_velocity(track, 1000, 1000), (0.0, 0.0))

    def test_velocity_is_clamped(self):
        # Teleport (track id swap) must not produce absurd predictions
        track = FakeTrack([(0.0, _det(50.0, 50.0)), (0.25, _det(950.0, 950.0))])
        vx, vy = _track_velocity(track, 1000, 1000)
        self.assertLessEqual(abs(vx), 3.0)
        self.assertLessEqual(abs(vy), 3.0)


class TestTrackerResolution(unittest.TestCase):
    def test_per_camera_tracker_overrides_global(self):
        from engine.live.config import CameraConfig, LiveConfig
        from engine.live.worker import _resolve_tracker_config

        cfg = LiveConfig()
        cfg.analysis.tracker_config = "config/bytetrack_live.yaml"
        cam_override = CameraConfig(name="Camera D", url="x", tracker_config="config/bytetrack_flicker.yaml")
        cam_plain = CameraConfig(name="Camera A", url="x")

        self.assertEqual(_resolve_tracker_config(cfg, cam_override), "config/bytetrack_flicker.yaml")
        self.assertEqual(_resolve_tracker_config(cfg, cam_plain), "config/bytetrack_live.yaml")
        # Missing files fall back to the Ultralytics default
        cfg.analysis.tracker_config = "config/does_not_exist.yaml"
        self.assertEqual(_resolve_tracker_config(cfg, CameraConfig(name="X", url="y")), "bytetrack.yaml")


if __name__ == "__main__":
    unittest.main()
