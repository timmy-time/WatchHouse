"""Dynamic FPS analysis controller with motion-triggered boost and cooldown hysteresis."""

from dataclasses import dataclass
import time
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DynamicFpsConfig:
    enabled: bool = True
    idle_fps: float = 2.0
    boost_fps: float = 5.0
    motion_threshold: float = 0.015  # Fraction of pixels changed (1.5%)
    boost_cooldown: float = 12.0     # Stay in boost for 12s after last motion/event


class DynamicFpsController:
    """Controls analysis frame rate per camera based on scene activity and events."""

    def __init__(self, cfg: Optional[DynamicFpsConfig] = None):
        self.cfg = cfg or DynamicFpsConfig()
        self.last_infer_time: float = -999999.0
        self.last_boost_time: float = -999999.0
        self.prev_gray_small: Optional[np.ndarray] = None
        self.current_mode: str = "idle"
        self.last_motion_score: float = 0.0

    def check_motion(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Fast motion estimation using downscaled 160x90 frame differencing (~0.2ms)."""
        if frame is None or frame.size == 0:
            return False, 0.0

        # Downscale to 160x90
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray_small is None:
            self.prev_gray_small = gray_blurred
            return False, 0.0

        # Compute absolute difference
        diff = cv2.absdiff(self.prev_gray_small, gray_blurred)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        non_zero = int(np.count_nonzero(thresh))
        total_pixels = 160 * 90
        motion_score = non_zero / total_pixels

        self.prev_gray_small = gray_blurred
        self.last_motion_score = motion_score
        has_motion = motion_score >= self.cfg.motion_threshold
        return has_motion, motion_score

    def should_infer(
        self,
        now: float,
        frame: Optional[np.ndarray],
        event_active: bool = False,
        active_tracks: int = 0,
    ) -> Tuple[bool, str, float]:
        """Decide whether to execute YOLO inference on this frame.

        Returns (should_run_yolo: bool, mode: "idle" | "boost", target_fps: float).
        """
        if not self.cfg.enabled:
            self.current_mode = "boost"
            return True, "boost", self.cfg.boost_fps

        has_motion = False
        if frame is not None:
            has_motion, _ = self.check_motion(frame)

        # Trigger boost on motion, active open event, or active tracked objects
        if event_active or active_tracks > 0 or has_motion:
            self.last_boost_time = now

        in_boost = (now - self.last_boost_time) < self.cfg.boost_cooldown
        mode = "boost" if in_boost else "idle"
        target_fps = self.cfg.boost_fps if in_boost else self.cfg.idle_fps
        self.current_mode = mode

        interval = 1.0 / max(0.5, target_fps)
        elapsed = now - self.last_infer_time

        # Run inference if sufficient time has elapsed since previous inference
        if elapsed >= interval or self.last_infer_time < -1000.0:
            self.last_infer_time = now
            return True, mode, target_fps

        return False, mode, target_fps

    def force_boost(self, now: float) -> None:
        """Force the controller into boost mode."""
        self.last_boost_time = now
        self.current_mode = "boost"
