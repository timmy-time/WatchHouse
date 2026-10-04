"""Multi-GPU multiprocessing execution pipeline for surveillance clip analysis."""

from dataclasses import asdict
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2

from engine.classifier import (
    HIGH_VALUE_CLASSES,
    Verdict,
    classify_clip_events,
)
from engine.detector import ClipDetector
from engine.reporter import parse_clip_metadata
from engine.storage import batch_store_keeps


def _worker_loop(
    gpu_id: int,
    model_path: str,
    vid_stride: int,
    conf_threshold: float,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
):
    """Worker process targeting a dedicated GPU device."""
    # Pin the GPU via env and pass device="" so ultralytics' select_device()
    # never rewrites CUDA_VISIBLE_DEVICES (it overwrites it for any index).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        detector = ClipDetector(model_path=model_path, device="")
    except Exception:
        # No CUDA available in this worker -> fall back to CPU
        detector = ClipDetector(model_path=model_path, device="cpu")

    while True:
        task = task_queue.get()
        if task is None:
            break

        clip_path, thumb_path = task
        meta = parse_clip_metadata(clip_path)

        try:
            start_t = time.time()

            # 1. Thumbnail hint (fast check if thumbnail exists)
            thumb_hint: Optional[str] = None
            thumb_high_value = False
            thumb_conf = 0.0
            if thumb_path and os.path.exists(thumb_path):
                thumb_dets = detector.detect_thumbnail(thumb_path, conf_threshold=conf_threshold)
                for td in thumb_dets:
                    if td.class_name in HIGH_VALUE_CLASSES and td.confidence >= 0.50:
                        thumb_high_value = True
                        thumb_hint = f"thumb_{td.class_name}"
                        thumb_conf = td.confidence
                        break

            # 2. Video tracking across subsampled frames
            frame_detections = detector.track_video(
                video_path=clip_path,
                vid_stride=vid_stride,
                conf_threshold=conf_threshold,
            )

            # 3. Decision Engine
            decision = classify_clip_events(frame_detections)

            # If thumbnail strongly saw a person/animal but video tracking missed ID, keep
            verdict = decision.verdict
            primary_reason = decision.primary_reason
            confidence = decision.confidence
            if verdict == Verdict.DISCARD and thumb_high_value:
                verdict = Verdict.KEEP
                primary_reason = f"{thumb_hint}_detected"
                confidence = thumb_conf

            # Serialize track summaries
            tracks_info = [asdict(s) for s in decision.active_track_summaries]

            proc_time = round(time.time() - start_t, 3)

            result_record = {
                "clip_path": clip_path,
                "thumbnail_path": thumb_path,
                "clip_rel": os.path.basename(clip_path),
                "thumbnail_rel": os.path.basename(thumb_path) if thumb_path else "",
                "filename": meta["filename"],
                "camera": meta["camera"],
                "timestamp": meta["timestamp"],
                "verdict": verdict.value,
                "primary_reason": primary_reason,
                "confidence": round(confidence, 3),
                "high_value_detected": decision.high_value_detected or thumb_high_value,
                "moving_vehicle_detected": decision.moving_vehicle_detected,
                "stationary_vehicle_count": decision.stationary_vehicle_count,
                "duration_frames": len(frame_detections),
                "processing_time_sec": proc_time,
                "detected_tracks": tracks_info,
            }

        except Exception as exc:
            result_record = {
                "clip_path": clip_path,
                "thumbnail_path": thumb_path,
                "clip_rel": os.path.basename(clip_path),
                "thumbnail_rel": os.path.basename(thumb_path) if thumb_path else "",
                "filename": meta["filename"],
                "camera": meta["camera"],
                "timestamp": meta["timestamp"],
                "verdict": Verdict.ERROR.value,
                "primary_reason": f"error: {str(exc)}",
                "confidence": 0.0,
                "high_value_detected": False,
                "moving_vehicle_detected": False,
                "stationary_vehicle_count": 0,
                "duration_frames": 0,
                "processing_time_sec": 0.0,
                "detected_tracks": [],
            }

        result_queue.put(result_record)


class AnalysisPipeline:
    """Multi-process, multi-GPU surveillance event analysis pipeline."""

    def __init__(
        self,
        gpus: List[int] = [0, 1],
        model_path: str = "yolov8n.pt",
        vid_stride: int = 15,
        conf_threshold: float = 0.30,
    ):
        self.gpus = gpus
        self.model_path = model_path
        self.vid_stride = vid_stride
        self.conf_threshold = conf_threshold

    def discover_clips(self, input_dir: str) -> List[Tuple[str, Optional[str]]]:
        """Find all .mp4 video files and pair them with companion thumbnails if present."""
        pairs: List[Tuple[str, Optional[str]]] = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".mp4"):
                    clip_path = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]
                    thumb_candidate = os.path.join(root, base_name + ".png")
                    thumb_path = thumb_candidate if os.path.exists(thumb_candidate) else None
                    pairs.append((clip_path, thumb_path))

        pairs.sort(key=lambda x: x[0])
        return pairs

    def process_batch(
        self,
        clip_pairs: List[Tuple[str, Optional[str]]],
    ) -> List[Dict[str, Any]]:
        """Distribute clips across GPU worker processes and collect results."""
        total = len(clip_pairs)
        if total == 0:
            return []

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        # Enqueue all tasks
        for pair in clip_pairs:
            task_queue.put(pair)

        # Worker count matches GPU list (or 1 if no GPUs specified)
        worker_gpus = self.gpus if self.gpus else [0]
        workers: List[mp.Process] = []

        for gpu_id in worker_gpus:
            task_queue.put(None)  # Poison pill per worker
            p = ctx.Process(
                target=_worker_loop,
                args=(
                    gpu_id,
                    self.model_path,
                    self.vid_stride,
                    self.conf_threshold,
                    task_queue,
                    result_queue,
                ),
            )
            p.start()
            workers.append(p)

        results: List[Dict[str, Any]] = []
        start_time = time.time()
        kept_count = 0
        discard_count = 0

        for idx in range(1, total + 1):
            record = result_queue.get()
            results.append(record)
            if record["verdict"] == Verdict.KEEP.value:
                kept_count += 1
            else:
                discard_count += 1

            elapsed = max(0.001, time.time() - start_time)
            rate = idx / elapsed
            print(
                f"[{idx:3d}/{total:3d}] ({elapsed:5.1f}s | {rate:4.1f} clips/s) "
                f"KEEP: {kept_count:2d} | DISCARD: {discard_count:3d} | "
                f"{record['camera']:10s} -> {record['verdict']:7s} ({record['primary_reason']})",
                flush=True,
            )

        for p in workers:
            p.join()

        return results

    def run_analysis(
        self,
        input_dir: str,
        output_dir: str,
        cache: Optional[Any] = None,
        store_keep: bool = True,
        clip_pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Execute analysis with optional caching and automated KEEP storage.

        - Skips clips already analyzed if present in cache (matching size and mtime).
        - Runs multi-GPU tracking on newly discovered/modified clips.
        - Updates cache file.
        - Automatically saves KEEP clips to output/clips/<foldername>/<filename>.
        """
        if clip_pairs is None:
            clip_pairs = self.discover_clips(input_dir)

        total_discovered = len(clip_pairs)
        cached_records: List[Dict[str, Any]] = []
        uncached_pairs: List[Tuple[str, Optional[str]]] = clip_pairs
        cfg_fingerprint = {
            "model": self.model_path,
            "vid_stride": self.vid_stride,
            "conf": self.conf_threshold,
        }

        if cache is not None:
            uncached_pairs, cached_records = cache.filter_uncached(
                clip_pairs, input_dir, config=cfg_fingerprint
            )
        cached_count = len(cached_records)
        to_process_count = len(uncached_pairs)

        print(
            f"Discovered {total_discovered} clips: "
            f"{cached_count} cached (skipping GPU analysis), "
            f"{to_process_count} new to analyze."
        )

        new_results: List[Dict[str, Any]] = []
        if to_process_count > 0:
            new_results = self.process_batch(uncached_pairs)
            if cache is not None:
                for rec in new_results:
                    cache.put(rec["clip_path"], input_dir, rec, config=cfg_fingerprint)
                cache.save()

        all_results = cached_records + new_results
        all_results.sort(
            key=lambda r: (r.get("camera", ""), r.get("timestamp", ""), r.get("filename", ""))
        )

        stored_count = 0
        if store_keep:
            stored_count = batch_store_keeps(all_results, input_dir, output_dir)
            if stored_count > 0:
                print(f"Stored {stored_count} KEEP clips in {os.path.join(output_dir, 'clips')}")

        stats = {
            "total_discovered": total_discovered,
            "cached_count": cached_count,
            "new_processed_count": to_process_count,
            "stored_keep_count": stored_count,
        }
        return all_results, stats
