#!/usr/bin/env python3
"""Surveillance Video Event Analysis Engine CLI.

Supports scanning surveillance archives, filtering out stationary vehicles,
exporting high-value events, and continuously watching incoming directories.
"""

import argparse
import os
import sys
import time
from typing import List

from engine.pipeline import AnalysisPipeline
from engine.reporter import AnalysisReporter
from engine.cache import AnalysisCache


def parse_gpu_ids(gpu_str: str) -> List[int]:
    """Parse comma-separated GPU string (e.g. '0,1') into list of integers."""
    if not gpu_str:
        return []
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip().isdigit()]


def cmd_scan(args: argparse.Namespace) -> int:
    """Run batch scan on directory of clips."""
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    gpus = parse_gpu_ids(args.gpus)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    print(f"=== Video Event Analysis Engine ===")
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"GPUs:             {gpus or 'CPU'}")
    print(f"Model:            {args.model}")
    print(f"Video Stride:     {args.vid_stride}")
    print(f"Confidence:       {args.confidence}")
    print(f"===================================")

    cache = None
    if not getattr(args, "no_cache", False):
        cache_file = getattr(args, "cache_file", None) or os.path.join(output_dir, "analysis_cache.json")
        cache = AnalysisCache(cache_file)

    store_keep = not getattr(args, "no_store_keep", False)

    pipeline = AnalysisPipeline(
        gpus=gpus,
        model_path=args.model,
        vid_stride=args.vid_stride,
        conf_threshold=args.confidence,
    )

    start_time = time.time()
    results, stats = pipeline.run_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        cache=cache,
        store_keep=store_keep,
    )
    total_time = time.time() - start_time

    if not results:
        print("No .mp4 files found.")
        return 0

    reporter = AnalysisReporter(output_dir)
    json_path = reporter.write_json_report(results, total_time)
    html_path = reporter.write_html_gallery(results, total_time)
    kept_count = sum(1 for r in results if r["verdict"] == "KEEP")
    discard_count = sum(1 for r in results if r["verdict"] == "DISCARD")
    reduction = round((discard_count / max(1, len(results))) * 100.0, 1)

    print("\n=== Analysis Complete ===")
    print(f"Total Clips in Library: {stats['total_discovered']}")
    print(f"Skipped (Cached):       {stats['cached_count']}")
    print(f"Newly Analyzed:         {stats['new_processed_count']}")
    print(f"Kept Events:            {kept_count}")
    print(f"Discarded (Filtered):   {discard_count} ({reduction}% noise reduction)")
    if store_keep and stats['stored_keep_count'] > 0:
        print(f"KEEP Clips Stored at:   {os.path.join(output_dir, 'clips')}/<foldername>/")
    print(f"Total Runtime:          {round(total_time, 2)}s")
    print(f"JSON Report:            {json_path}")
    print(f"HTML Gallery:           {html_path}")
    print("=========================")

    if getattr(args, "export_dir", None):
        action = getattr(args, "action", "copy")
        exp_count = reporter.export_kept_clips(results, args.export_dir, action=action)
        print(f"Exported {exp_count} kept clips to {args.export_dir} ({action})")

    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Continuously poll an incoming directory for new surveillance clips."""
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    gpus = parse_gpu_ids(args.gpus)
    interval = args.interval

    print(f"=== Surveillance Watch Daemon ===")
    print(f"Watching:         {input_dir} (poll interval: {interval}s)")
    print(f"Output Directory: {output_dir}")
    print(f"GPUs:             {gpus or 'CPU'}")
    print("Press Ctrl+C to stop.")

    cache = None
    if not getattr(args, "no_cache", False):
        cache_file = getattr(args, "cache_file", None) or os.path.join(output_dir, "analysis_cache.json")
        cache = AnalysisCache(cache_file)

    store_keep = not getattr(args, "no_store_keep", False)

    pipeline = AnalysisPipeline(
        gpus=gpus,
        model_path=args.model,
        vid_stride=args.vid_stride,
        conf_threshold=args.confidence,
    )
    reporter = AnalysisReporter(output_dir)
    all_results: List[dict] = []
    start_time = time.time()

    try:
        while True:
            batch_results, stats = pipeline.run_analysis(
                input_dir=input_dir,
                output_dir=output_dir,
                cache=cache,
                store_keep=store_keep,
            )
            all_results = batch_results

            if stats["new_processed_count"] > 0:
                total_time = time.time() - start_time
                reporter.write_json_report(all_results, total_time)
                reporter.write_html_gallery(all_results, total_time)

                if getattr(args, "export_dir", None):
                    action = getattr(args, "action", "copy")
                    reporter.export_kept_clips(all_results, args.export_dir, action=action)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping surveillance watch daemon.")
        return 0

def cmd_live(args: argparse.Namespace) -> int:
    """Run live RTSP surveillance analysis engine."""
    from engine.live.worker import run_live
    output_dir = os.path.abspath(args.output)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            output_dir = os.path.abspath("output")
            os.makedirs(output_dir, exist_ok=True)
    return run_live(config_path=args.config, output_dir=output_dir)


def cmd_serve(args: argparse.Namespace) -> int:
    """Run web dashboard and API server."""
    import uvicorn
    from engine.web.app import create_app
    output_dir = os.path.abspath(args.output)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            output_dir = os.path.abspath("output")
            os.makedirs(output_dir, exist_ok=True)
    app = create_app(output_dir=output_dir, clips_dir=args.clips, config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_infer_server(args: argparse.Namespace) -> int:
    """Run the standalone inference server (for offloading camera inference)."""
    import uvicorn
    from engine.live.infer_server import create_infer_app
    app = create_infer_app(device=args.device, max_sessions=args.max_sessions)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0

def cmd_reindex_faces(args: argparse.Namespace) -> int:
    """Run face re-indexing on historical event clips."""
    from engine.live.db import EventStore
    from engine.reindexer import FaceReindexer
    output_dir = os.path.abspath(args.output)
    db_path = os.path.join(output_dir, "live/events.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(output_dir, "events.db")
    store = EventStore(db_path)
    reindexer = FaceReindexer(store=store, output_dir=output_dir)
    if args.clip:
        res = reindexer.reindex_clip(os.path.abspath(args.clip), event_id=args.event_id, camera=args.camera)
        print(f"Re-indexed clip: {res.get('faces_indexed', 0)} faces found.")
        return 0
    res = reindexer.reindex_events_directory(events_root=args.events_dir, camera=args.camera, limit=args.limit)
    print(f"Re-indexed {res.get('processed_clips', 0)} clips: {res.get('total_faces_indexed', 0)} faces indexed.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Surveillance Video Event Analysis Engine - Filter parked cars & nuisance motion"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Scan command
    scan_p = subparsers.add_parser("scan", help="Batch analyze surveillance clips in a directory")
    scan_p.add_argument("--input", "-i", default="/data/clips", help="Input directory containing .mp4 clips")
    scan_p.add_argument("--output", "-o", default="/data/output", help="Output directory for reports and gallery")
    scan_p.add_argument("--gpus", default="0,1", help="Comma-separated GPU indices (e.g. '0,1' or '0')")
    scan_p.add_argument("--vid-stride", type=int, default=15, help="Video frame subsampling stride (default: 15)")
    scan_p.add_argument("--confidence", type=float, default=0.30, help="Detection confidence threshold (default: 0.30)")
    scan_p.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name (default: yolov8n.pt)")
    scan_p.set_defaults(func=cmd_scan)
    scan_p.add_argument("--no-cache", action="store_true", help="Disable cache and re-analyze all clips")
    scan_p.add_argument("--no-store-keep", action="store_true", help="Disable auto-storing KEEP clips to output/clips/<foldername>/")
    scan_p.add_argument("--cache-file", default=None, help="Custom cache file path (default: <output>/analysis_cache.json)")

    # Filter / Export command
    filter_p = subparsers.add_parser("filter", help="Scan and export/copy only valuable event clips")
    filter_p.add_argument("--input", "-i", default="/data/clips", help="Input directory containing .mp4 clips")
    filter_p.add_argument("--output", "-o", default="/data/output", help="Output directory for reports and gallery")
    filter_p.add_argument("--export-dir", "-e", required=True, help="Destination directory to copy/move kept clips")
    filter_p.add_argument("--action", choices=["copy", "move"], default="copy", help="File action (copy or move)")
    filter_p.add_argument("--gpus", default="0,1", help="Comma-separated GPU indices")
    filter_p.add_argument("--vid-stride", type=int, default=15, help="Video frame subsampling stride")
    filter_p.add_argument("--confidence", type=float, default=0.30, help="Detection confidence threshold")
    filter_p.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name")
    filter_p.set_defaults(func=cmd_scan)
    filter_p.add_argument("--no-cache", action="store_true", help="Disable cache and re-analyze all clips")
    filter_p.add_argument("--no-store-keep", action="store_true", help="Disable auto-storing KEEP clips to output/clips/<foldername>/")
    filter_p.add_argument("--cache-file", default=None, help="Custom cache file path")

    # Watch command
    watch_p = subparsers.add_parser("watch", help="Continuously watch directory and analyze incoming clips")
    watch_p.add_argument("--input", "-i", default="/data/clips", help="Input directory to watch")
    watch_p.add_argument("--output", "-o", default="/data/output", help="Output directory for reports and gallery")
    watch_p.add_argument("--interval", type=float, default=5.0, help="Poll interval in seconds (default: 5.0)")
    watch_p.add_argument("--export-dir", "-e", default=None, help="Optional destination to export kept clips")
    watch_p.add_argument("--action", choices=["copy", "move"], default="copy", help="File action (copy or move)")
    watch_p.add_argument("--gpus", default="0,1", help="Comma-separated GPU indices")
    watch_p.add_argument("--vid-stride", type=int, default=15, help="Video frame subsampling stride")
    watch_p.add_argument("--confidence", type=float, default=0.30, help="Detection confidence threshold")
    watch_p.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name")
    watch_p.set_defaults(func=cmd_watch)
    watch_p.add_argument("--no-cache", action="store_true", help="Disable cache and re-analyze all clips")
    watch_p.add_argument("--no-store-keep", action="store_true", help="Disable auto-storing KEEP clips to output/clips/<foldername>/")
    watch_p.add_argument("--cache-file", default=None, help="Custom cache file path")


    # Live command
    live_p = subparsers.add_parser("live", help="Run live RTSP surveillance analysis engine")
    live_p.add_argument("--config", "-c", default="/app/config/live.yaml", help="Path to live.yaml configuration")
    live_p.add_argument("--output", "-o", default="/data/output", help="Output directory for events and preview")
    live_p.set_defaults(func=cmd_live)

    # Serve command
    serve_p = subparsers.add_parser("serve", help="Run web dashboard and API server")
    serve_p.add_argument("--output", "-o", default="/data/output", help="Output directory containing events and results")
    serve_p.add_argument("--clips", default="/data/clips", help="Clips directory for historical video playback")
    serve_p.add_argument("--config", "-c", default="/app/config/live.yaml", help="Path to live.yaml configuration")
    serve_p.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    serve_p.set_defaults(func=cmd_serve)

    # Inference server command (offload camera inference to another machine)
    infer_p = subparsers.add_parser(
        "infer-server", help="Serve YOLO inference over HTTP so another host can offload camera inference"
    )
    infer_p.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    infer_p.add_argument("--port", type=int, default=8099, help="Port to bind (default: 8099)")
    infer_p.add_argument("--device", default="0", help="Torch device for inference ('0', 'cuda:0' or 'cpu')")
    infer_p.add_argument("--max-sessions", type=int, default=8, help="Max cached camera sessions (default: 8)")
    infer_p.set_defaults(func=cmd_infer_server)

    # Face re-indexing command
    reindex_p = subparsers.add_parser(
        "reindex-faces", help="Scan event clips to detect, embed, and index faces into the gallery"
    )
    reindex_p.add_argument("--output", "-o", default="output", help="Output directory containing events and events.db")
    reindex_p.add_argument("--events-dir", default=None, help="Directory containing event .mp4 clips")
    reindex_p.add_argument("--clip", default=None, help="Specific .mp4 clip to reindex")
    reindex_p.add_argument("--event-id", type=int, default=None, help="Specific event ID to associate")
    reindex_p.add_argument("--limit", type=int, default=50, help="Max clips to process (default: 50)")
    reindex_p.add_argument("--camera", default=None, help="Filter by camera name")
    reindex_p.set_defaults(func=cmd_reindex_faces)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
