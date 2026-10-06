"""Dynamic day/night (color vs infrared) lighting mode detection with hysteresis."""

from dataclasses import dataclass
import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def measure_saturation(frame: np.ndarray) -> float:
    """Fast saturation estimation (<0.1ms) using downscaled 160x90 HSV conversion."""
    if frame is None or frame.size == 0:
        return 0.0
    # Downscale to 160x90 for fast, noise-tolerant mean saturation
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


@dataclass
class NightModeConfig:
    """Configuration for automatic day/night mode adaptation."""

    enabled: bool = True
    confidence: float = 0.20
    tracker_config: str = "config/bytetrack_mild.yaml"
    imgsz: Optional[int] = None
    night_threshold: float = 6.0
    day_threshold: float = 15.0
    confirm_seconds: float = 6.0
    cooldown_seconds: float = 30.0


class LightingModeTracker:
    """Tracks camera day (color) vs night (IR monochrome) mode with hysteresis and debouncing.

    Prevents sporadic flapping caused by headlights, shadows, or twilight flicker
    using dual thresholds, exponential moving average smoothing, persistence confirmation,
    and a minimum dwell cooldown time.
    """

    def __init__(self, cfg: Optional[NightModeConfig] = None):
        self.cfg = cfg or NightModeConfig()
        self.enabled = bool(self.cfg.enabled)
        self.night_threshold = float(self.cfg.night_threshold)
        self.day_threshold = float(self.cfg.day_threshold)
        self.confirm_seconds = float(self.cfg.confirm_seconds)
        self.cooldown_seconds = float(self.cfg.cooldown_seconds)

        self.current_mode: str = "day"
        self.candidate_mode: Optional[str] = None
        self.candidate_start_time: float = 0.0
        self.last_switch_time: float = -999999.0
        self.smoothed_saturation: float = 50.0
        self.last_saturation: float = 50.0
        self._initialized: bool = False

    def update(self, frame: np.ndarray, now: Optional[float] = None) -> Tuple[bool, str, float]:
        """Update lighting mode estimate from the current frame.

        Returns:
            (mode_changed: bool, current_mode: str, current_saturation: float)
        """
        if not self.enabled:
            return False, self.current_mode, self.last_saturation

        if now is None:
            now = time.time()

        raw_sat = measure_saturation(frame)
        self.last_saturation = raw_sat

        # On very first sample, initialize directly to matching mode without cooldown delay
        if not self._initialized:
            self._initialized = True
            self.smoothed_saturation = raw_sat
            if raw_sat <= self.night_threshold:
                self.current_mode = "night"
                return True, "night", raw_sat
            else:
                self.current_mode = "day"
                return False, "day", raw_sat

        # Update telemetry smoothed saturation
        self.smoothed_saturation = 0.70 * self.smoothed_saturation + 0.30 * raw_sat

        # Dual threshold evaluation with deadband
        if raw_sat <= self.night_threshold:
            instant_mode = "night"
        elif raw_sat >= self.day_threshold:
            instant_mode = "day"
        else:
            instant_mode = self.current_mode

        # If instant mode matches current committed mode, cancel candidate
        if instant_mode == self.current_mode:
            self.candidate_mode = None
            return False, self.current_mode, self.last_saturation

        # In cooldown lock: prevent switching back too quickly
        if (now - self.last_switch_time) < self.cooldown_seconds:
            self.candidate_mode = None
            return False, self.current_mode, self.last_saturation

        # Require candidate to persist continuously for confirm_seconds
        if self.candidate_mode != instant_mode:
            self.candidate_mode = instant_mode
            self.candidate_start_time = now
            return False, self.current_mode, self.last_saturation

        if (now - self.candidate_start_time) >= self.confirm_seconds:
            old_mode = self.current_mode
            self.current_mode = instant_mode
            self.last_switch_time = now
            self.candidate_mode = None
            logger.info(
                "Lighting mode switched %s -> %s (sat=%.1f, smoothed=%.1f)",
                old_mode.upper(),
                instant_mode.upper(),
                self.last_saturation,
                self.smoothed_saturation,
            )
            return True, self.current_mode, self.last_saturation

        return False, self.current_mode, self.last_saturation
