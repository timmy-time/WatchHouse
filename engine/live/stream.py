"""RTSP stream capture and video recording with NVDEC / CPU decode fallback."""

import logging
import os
import subprocess
import select
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from engine.live.config import CameraConfig, RecordingConfig

logger = logging.getLogger(__name__)

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "/usr/bin/ffprobe")


def _read_exact(stream, n: int) -> Optional[bytes]:
    """Read exactly n bytes from a binary stream, handling chunked pipe transfers."""
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        nbytes = stream.readinto(view[pos:])
        if not nbytes:
            return None
        pos += nbytes
    return bytes(buf)


class CameraStream:
    """Manages an RTSP ffmpeg pipeline capturing rolling MP4 segments and raw frames."""

    def __init__(
        self,
        cam: CameraConfig,
        rec_cfg: RecordingConfig,
        fps: int = 5,
        pipe_width: int = 0,
        source: str = "main",
    ):
        self.cam = cam
        self.rec_cfg = rec_cfg
        self.fps = fps
        self.pipe_width = max(0, int(pipe_width))
        self.source = (source or "main").strip().lower()

        self.width = 1920
        self.height = 1080
        # Analysis pipe frame size (== source size unless the pipe is downscaled)
        self.pipe_w = self.width
        self.pipe_h = self.height
        self.use_hwaccel = True
        self.nvdec_fail_count = 0
        self.sub_failed = False
        self.sub_fail_count = 0

        self.proc: Optional[subprocess.Popen] = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.latest_frame: Optional[Tuple[float, np.ndarray]] = None

        self.restarts = 0
        self.last_frame_time = 0.0
        self.fps_in = 0.0
        self.last_error: Optional[str] = None

        self.supervisor_thread = threading.Thread(target=self._supervisor_loop, daemon=True)

    def _analysis_url(self) -> str:
        """URL feeding the realtime analysis pipe.

        "sub" prefers the DVR substream while it is healthy; "main" always uses the
        main stream, whose framing defines the dashboard's normalized coordinates.
        """
        if self.source == "sub" and self.cam.sub_url and not self.sub_failed:
            return self.cam.sub_url
        return self.cam.url

    def _analysis_filter(self) -> str:
        """Filters applied to the pipe: optional downscale, then the analysis frame rate."""
        parts = []
        if self.pipe_width and self.width > self.pipe_width:
            parts.append(f"scale={self.pipe_width}:-2")
        parts.append(f"fps={self.fps}")
        return ",".join(parts)

    def _resolve_pipe_size(self) -> None:
        """Track the size ffmpeg will actually emit, so the reader stays in sync."""
        if self.pipe_width and self.width > self.pipe_width:
            out_w = self.pipe_width
            out_h = int(round(self.height * out_w / self.width / 2.0)) * 2
            self.pipe_w, self.pipe_h = out_w, out_h
        else:
            self.pipe_w, self.pipe_h = self.width, self.height

    def _hwaccel_flags(self) -> List[str]:
        """ffmpeg hardware-decode flags for this camera.

        With an explicit decode GPU the child is spawned with its own single-entry
        CUDA_VISIBLE_DEVICES mask, so the visible index is always 0.
        """
        index = 0 if self.cam.decode_gpu is not None else int(self.cam.gpu or 0)
        return ["-hwaccel", "cuda", "-hwaccel_device", str(index)]

    def _decode_env(self) -> Optional[dict]:
        """Child environment pinning ffmpeg to the decode GPU.

        Ultralytics rewrites CUDA_VISIBLE_DEVICES to -1 in this process when a
        camera runs CPU inference; giving ffmpeg its own mask keeps GPU decode
        (and the recording copy path) working regardless.
        """
        if self.cam.decode_gpu is None:
            return None
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.cam.decode_gpu)
        return env

    def probe(self) -> Tuple[int, int]:
        """Probe the analysis stream resolution using ffprobe."""
        cmd = [
            FFPROBE_BIN,
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            "10000000",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            self._analysis_url(),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and res.stdout.strip():
                parts = res.stdout.strip().split("\n")[0].split(",")
                w, h = int(parts[0]), int(parts[1])
                self.width = w
                self.height = h
                self._resolve_pipe_size()
                return w, h
        except Exception as exc:
            logger.warning(f"{self.cam.name}: ffprobe failed ({exc}), falling back to 1920x1080")
        return self.width, self.height

    def build_cmd(self, hwaccel: bool) -> List[str]:
        """Build the ffmpeg command: record main stream, pipe analysis frames.

        Modes:
          record=True  -> up to two RTSP inputs: analysis source (substream when
                          configured) -> rawvideo pipe, main stream -> MP4 segments.
          record=False -> live-only: a single input feeds the rawvideo pipe and
                          nothing is written to disk (no segments, no clips).
        """
        rec_dir = os.path.join(self.rec_cfg.root, self.cam.slug)

        pipe_args = [
            "-map", "0:v:0",
            "-vf", self._analysis_filter(),
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]

        cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "warning", "-nostdin"]

        # --- Live-only camera: inference pipe, zero disk output ---
        if not self.cam.record:
            cmd += ["-rtsp_transport", "tcp", "-stimeout", "10000000"]
            if hwaccel:
                cmd += self._hwaccel_flags()
            cmd += ["-i", self._analysis_url()]
            return cmd + pipe_args

        # --- Recording camera ---
        os.makedirs(rec_dir, exist_ok=True)
        segment_pattern = os.path.join(rec_dir, "%Y%m%d-%H%M%S.mp4")
        segment_args = [
            "-f", "segment",
            "-segment_time", str(self.rec_cfg.segment_seconds),
            "-segment_format", "mp4",
            "-reset_timestamps", "1",
            "-strftime", "1",
            segment_pattern,
        ]

        analysis_url = self._analysis_url()
        if analysis_url != self.cam.url:
            # --- Twin inputs: analysis decodes the secondary stream, recording copies the main ---
            if hwaccel:
                cmd += self._hwaccel_flags()
            cmd += [
                "-rtsp_transport", "tcp", "-stimeout", "10000000",
                "-i", analysis_url,
                "-rtsp_transport", "tcp", "-stimeout", "10000000",
                "-i", self.cam.url,
                # Output 1: MP4 segments, copied from main (no decode cost)
                "-map", "1:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "32k",
            ] + segment_args + pipe_args
            return cmd

        # --- Single input: same stream for recording and analysis ---
        cmd += ["-rtsp_transport", "tcp", "-stimeout", "10000000"]
        if hwaccel:
            cmd += self._hwaccel_flags()
        cmd += [
            "-i", self.cam.url,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "32k",
        ] + segment_args + pipe_args
        return cmd

    def start(self) -> None:
        """Start the supervisor thread."""
        self.probe()
        self.supervisor_thread.start()

    def _supervisor_loop(self) -> None:
        backoff_steps = [2, 4, 8, 16, 30]
        backoff_idx = 0

        while not self.stop_event.is_set():
            cmd = self.build_cmd(hwaccel=self.use_hwaccel)
            start_time = time.time()
            logger.info(
                f"{self.cam.name}: Starting ffmpeg (hwaccel={self.use_hwaccel}, source={self.source}, "
                f"{self.width}x{self.height} -> {self.pipe_w}x{self.pipe_h} @ {self.fps}fps)"
            )

            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10 * 1024 * 1024,
                    env=self._decode_env() if self.use_hwaccel else None,
                )
            except Exception as exc:
                self.last_error = str(exc)
                logger.error(f"{self.cam.name}: Failed to spawn ffmpeg: {exc}")
                time.sleep(5)
                continue
            self.last_frame_time = start_time

            frame_size = self.pipe_w * self.pipe_h * 3
            fps_count = 0
            fps_start = time.time()

            # Read frames until process dies or timeout
            while not self.stop_event.is_set():
                if self.proc.poll() is not None:
                    break

                # Wait for data with timeout to detect stream stalls
                rlist, _, _ = select.select([self.proc.stdout], [], [], 1.0)
                if not rlist:
                    if time.time() - self.last_frame_time > 15.0:
                        logger.warning(f"{self.cam.name}: No frames received for 15s; restarting")
                        break
                    continue

                # Read raw frame bytes
                raw = _read_exact(self.proc.stdout, frame_size)
                if not raw:
                    break

                now = time.time()
                self.last_frame_time = now
                fps_count += 1
                if now - fps_start >= 5.0:
                    self.fps_in = round(fps_count / (now - fps_start), 1)
                    fps_count = 0
                    fps_start = now

                arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.pipe_h, self.pipe_w, 3))
                with self.condition:
                    self.latest_frame = (now, arr)
                    self.condition.notify_all()

            # Process exited or broke
            elapsed = time.time() - start_time
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=3.0)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass

            if self.stop_event.is_set():
                break

            self.restarts += 1

            # Check NVDEC fallback: 3 consecutive exits within 10s of start
            if self.use_hwaccel and elapsed < 10.0:
                self.nvdec_fail_count += 1
                if self.nvdec_fail_count >= 3:
                    self.use_hwaccel = False
                    logger.warning(f"{self.cam.name}: NVDEC failed 3x, falling back to CPU decode")

            # Substream fallback: 2 rapid failures while using the substream -> main-only
            if self.cam.sub_url and not self.sub_failed and elapsed < 10.0:
                self.sub_fail_count += 1
                if self.sub_fail_count >= 2:
                    self.sub_failed = True
                    logger.warning(
                        f"{self.cam.name}: substream failed {self.sub_fail_count}x, "
                        f"falling back to main stream for analysis"
                    )
                    self.probe()

            if elapsed >= 60.0:
                # Reset backoff and failure counters after 60s of healthy running
                backoff_idx = 0
                self.nvdec_fail_count = 0
            if self.sub_failed and self.sub_fail_count > 0 and elapsed >= 60.0:
                was_failed = self.sub_failed
                self.sub_failed = False
                self.sub_fail_count = 0
                if was_failed:
                    logger.info(f"{self.cam.name}: retrying substream after healthy main-only run")

            delay = backoff_steps[min(backoff_idx, len(backoff_steps) - 1)]
            backoff_idx = min(backoff_idx + 1, len(backoff_steps) - 1)
            logger.info(f"{self.cam.name}: Restarting stream in {delay}s...")

            # Sleep with stop_event check
            self.stop_event.wait(delay)

    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[float, np.ndarray]]:
        """Fetch the latest available frame; blocks up to timeout seconds."""
        with self.condition:
            if self.latest_frame is None:
                self.condition.wait(timeout=timeout)
            frame = self.latest_frame
            self.latest_frame = None
            return frame

    def status(self) -> dict:
        """Return stream health and decode status."""
        connected = (
            self.proc is not None
            and self.proc.poll() is None
            and (time.time() - self.last_frame_time < 15.0 if self.last_frame_time > 0 else False)
        )
        return {
            "connected": connected,
            "decode": "nvdec" if self.use_hwaccel else "cpu",
            "source": "sub" if (self.source == "sub" and self.cam.sub_url and not self.sub_failed) else "main",
            "pipe": f"{self.pipe_w}x{self.pipe_h}",
            "fps_in": self.fps_in,
            "restarts": self.restarts,
            "error": self.last_error,
        }

    def stop(self) -> None:
        """Terminate ffmpeg process and stop supervisor thread."""
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

        if self.supervisor_thread.is_alive():
            self.supervisor_thread.join(timeout=3.0)
