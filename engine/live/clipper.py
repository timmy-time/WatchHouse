"""Rolling MP4 segment management, clip assembly, and ring pruning."""

from datetime import datetime, timezone
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from typing import List, Optional, Set, Tuple

from engine.live.config import CameraConfig, LiveConfig
from engine.live.db import EventStore

logger = logging.getLogger(__name__)

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "/usr/bin/ffprobe")
SEGMENT_RE = re.compile(r"^(\d{8}-\d{6})\.mp4$")


def list_segments(directory: str) -> List[Tuple[float, str]]:
    """Scan directory for %Y%m%d-%H%M%S.mp4 segments, return sorted (utc_epoch, full_path)."""
    if not os.path.exists(directory):
        return []

    segments: List[Tuple[float, str]] = []
    for fname in os.listdir(directory):
        match = SEGMENT_RE.match(fname)
        if not match:
            continue
        ts_str = match.group(1)
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
            epoch = dt.timestamp()
            full_path = os.path.join(directory, fname)
            segments.append((epoch, full_path))
        except ValueError:
            continue

    segments.sort(key=lambda s: s[0])
    return segments


def select_segments(segments: List[Tuple[float, str]], start: float, end: float) -> List[str]:
    """Select segments spanning [start, end), excluding the newest unfinalized segment."""
    if len(segments) < 2:
        return []

    selected: List[str] = []
    # Segment i spans [s_i, s_{i+1})
    for i in range(len(segments) - 1):
        s_curr, path_curr = segments[i]
        s_next, _ = segments[i + 1]

        if s_curr < end and s_next > start:
            selected.append(path_curr)

    return selected


def is_valid_mp4(path: str) -> bool:
    """Check if an MP4 segment is readable and not truncated (e.g. missing moov atom)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if os.path.getsize(path) > 1024:
        try:
            res = subprocess.run(
                [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode != 0:
                logger.warning(f"Skipping corrupt or incomplete MP4 segment {path}: {res.stderr.strip()}")
                return False
        except Exception:
            pass
    return True


def assemble_clip(paths: List[str], out_path: str) -> None:
    """Concatenate MP4 segments using ffmpeg concat demuxer with stream copy."""
    valid_paths = [p for p in paths if is_valid_mp4(p)]
    if not valid_paths:
        raise ValueError("Cannot assemble clip: no valid MP4 segments found")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        list_file = f.name
        for p in valid_paths:
            abs_p = os.path.abspath(p)
            escaped = abs_p.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        out_path,
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed (code {res.returncode}): {res.stderr}")
    finally:
        if os.path.exists(list_file):
            try:
                os.remove(list_file)
            except OSError:
                pass


def prune_segments(
    segments: List[Tuple[float, str]],
    now: float,
    ring_minutes: int,
    protected_start: Optional[float] = None,
) -> int:
    """Delete segments older than ring_minutes, preserving those >= protected_start."""
    if len(segments) < 2:
        return 0

    cutoff = now - (ring_minutes * 60.0)
    if protected_start is not None:
        effective_cutoff = min(cutoff, protected_start)
    else:
        effective_cutoff = cutoff

    deleted = 0
    # Segments[:-1] can be pruned; newest segment is always preserved
    for i in range(len(segments) - 1):
        s_curr, path_curr = segments[i]
        s_next, _ = segments[i + 1]

        # Segment ends at s_next; if s_next < effective_cutoff, it's expired
        if s_next < effective_cutoff:
            try:
                if os.path.exists(path_curr):
                    os.remove(path_curr)
                    deleted += 1
            except OSError as exc:
                logger.warning(f"Failed to delete pruned segment {path_curr}: {exc}")

    return deleted


class Finalizer:
    """Asynchronous clip assembler and ring buffer pruner for a camera."""

    def __init__(
        self,
        cam: CameraConfig,
        cfg: LiveConfig,
        store: EventStore,
        output_dir: str,
    ):
        self.cam = cam
        self.cfg = cfg
        self.store = store
        self.output_dir = output_dir

        self.queue: queue.Queue = queue.Queue()
        self.pending_items: List[Tuple[int, float, float]] = []
        self.pending_lock = threading.Lock()

        self.stop_event = threading.Event()
        self.rec_dir = os.path.join(cfg.recording.root, cam.slug)
        self.last_prune = 0.0

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def enqueue(self, event_id: int, start: float, end: float) -> None:
        with self.pending_lock:
            self.pending_items.append((event_id, start, end))
        self.queue.put((event_id, start, end))

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                self._check_prune()
                continue

            if item is None:
                self.queue.task_done()
                break

            event_id, start_ts, end_ts = item
            try:
                self._process_item(event_id, start_ts, end_ts)
            except Exception as exc:
                logger.error(f"{self.cam.name}: Finalizer error for event {event_id}: {exc}")
                self.store.update_event(event_id, status="error", error=str(exc))
            finally:
                with self.pending_lock:
                    if (event_id, start_ts, end_ts) in self.pending_items:
                        self.pending_items.remove((event_id, start_ts, end_ts))
                self.queue.task_done()

            self._check_prune()

    def _process_item(self, event_id: int, start_ts: float, end_ts: float) -> None:
        post_roll = self.cfg.analysis.post_roll_seconds
        pre_roll = self.cfg.analysis.pre_roll_seconds
        target_time = end_ts + post_roll

        # Wait until a segment starting >= target_time exists (poll 2s, max 60s)
        deadline = time.time() + 60.0
        while time.time() < deadline and not self.stop_event.is_set():
            segs = list_segments(self.rec_dir)
            if any(s >= target_time for s, _ in segs):
                break
            self.stop_event.wait(2.0)

        segs = list_segments(self.rec_dir)
        clip_start = start_ts - pre_roll
        clip_end = end_ts + post_roll
        selected = select_segments(segs, clip_start, clip_end)

        if not selected:
            logger.warning(f"{self.cam.name}: No segments selectable for event {event_id}")
            self.store.update_event(
                event_id,
                status="error",
                error="No recording segments found for event window",
            )
            return

        date_str = datetime.utcfromtimestamp(start_ts).strftime("%Y%m%d")
        clip_rel = f"events/{self.cam.slug}/{date_str}/{event_id}.mp4"
        clip_full = os.path.join(self.output_dir, clip_rel)

        try:
            assemble_clip(selected, clip_full)
            self.store.update_event(event_id, status="finalized", clip_path=clip_rel)
            logger.info(f"{self.cam.name}: Event {event_id} finalized with {len(selected)} segments -> {clip_rel}")
        except Exception as exc:
            logger.error(f"{self.cam.name}: Assembly failed for event {event_id}: {exc}")
            self.store.update_event(event_id, status="error", error=str(exc))

    def _check_prune(self) -> None:
        now = time.time()
        if now - self.last_prune < 30.0:
            return
        self.last_prune = now

        pre_roll = self.cfg.analysis.pre_roll_seconds
        with self.pending_lock:
            if self.pending_items:
                protected = min(item[1] for item in self.pending_items) - pre_roll
            else:
                protected = None

        segs = list_segments(self.rec_dir)
        prune_segments(segs, now, self.cfg.recording.ring_minutes, protected_start=protected)

    def stop(self) -> None:
        self.stop_event.set()
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
