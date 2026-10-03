# Surveillance Video Event Analysis Engine

Automated, dual-GPU accelerated surveillance video analysis engine designed to evaluate security camera clips and thumbnails, filter out stationary parked vehicles and nuisance environmental triggers (spiderwebs, insect glare, wind foliage, headlight sweeps), and retain genuine events (pedestrians, moving vehicles, and animals).

---

## Performance Summary

Tested on **444 clips** (~30 seconds each, 1080p) across three cameras (`Camera A`, `Camera B`, `Camera C`) utilizing **2x GPU GPUs**:

| Metric | Result |
| :--- | :--- |
| **Total Clips Analyzed** | 444 clips |
| **Total Processing Time** | ~304 seconds (~5 minutes) |
| **Throughput** | 1.46 clips / second (~36x faster than real-time) |
| **False Alarms Filtered** | 420 clips discarded |
| **Storage / Noise Reduction** | **94.6% noise reduction** |
| **Valuable Events Kept** | 24 confirmed events (all true people & moving vehicles) |
| **Subsequent Cached Scans** | **0.02 seconds** (100% skipped, 0 GPU compute) |
---

## System Architecture

1. **Dual-GPU Pipeline (`engine/pipeline.py`)**:
   - Uses `multiprocessing.get_context("spawn")` to partition workloads evenly across `GPU 0` and `GPU 1`.
   - Video frame subsampling with `vid_stride=15` (~1.6 fps on 25fps video), inspecting ~29 frames per 30-second clip for rapid inference.
2. **Object Detection & ByteTrack Tracking (`engine/detector.py`)**:
   - YOLOv8n object detection coupled with ByteTrack multi-object tracking.
   - Dual-stage analysis: fast companion thumbnail triage (`.png`) + multi-frame video trajectory tracking (`.mp4`).
3. **Stationary Vehicle Suppression Engine (`engine/classifier.py`)**:
   - For every tracked vehicle (`car`, `truck`, `bus`), calculates:
     - Minimum start-to-current bounding box overlap: $\min_t \text{IoU}(B_0, B_t) \ge 0.65$
     - Normalized centroid displacement: $\frac{\max \|\mathbf{c}_i - \mathbf{c}_j\|}{\sqrt{w \times h}} \le 0.25$
   - Vehicles meeting these criteria are flagged as `STATIONARY_VEHICLE` and discarded, preventing parked cars from causing false saves.
   - Moving vehicles ($\Delta d > 60\text{px}$, displacement $> 0.40$, or $\text{IoU} < 0.60$) and high-value objects (`person`, `dog`, `cat`, etc.) are classified as `KEEP`.
4. **Analysis Cache (`engine/cache.py`)**:
   - Stores clip verdicts and kinematic metrics in `output/analysis_cache.json` keyed by relative path.
   - Validates file size and modification time (`mtime`) on disk.
   - When scanning, already-analyzed clips are skipped in microseconds without GPU inference, ensuring all GPU compute is directed purely to newly uploaded folders and clips.
5. **Automatic KEEP Storage (`engine/storage.py`)**:
   - All clips with verdict `KEEP` are automatically organized and copied into `output/clips/<foldername>/<clip.mp4>`, mirroring the original folder structure (e.g. `output/clips/20260103/` and `output/clips/20260524/`).
   - Companion `.png` thumbnails are preserved alongside the video clips.
6. **Rich Reporting & Visual Gallery (`engine/reporter.py`)**:
   - `output/analysis_results.json`: Machine-readable breakdown with per-object kinematic tracking data and confidence metrics.
   - `output/gallery.html`: Self-contained interactive HTML review dashboard with statistical cards, verdict filters ("Keep Only", "Discarded", camera selector), and an inline video modal player.

---

## Quickstart (Docker Compose)

### 1. Run Batch Scan Across Dual GPU GPUs
```bash
docker compose run --rm analysis-engine
```
This mounts `./clips` as input and writes output to `./output/`.

### 2. View Results
Open `output/gallery.html` in any web browser to view the interactive gallery, inspect reasons, and play kept vs discarded videos.

---

## CLI Usage

### `scan`
Scan an entire directory of surveillance clips and generate JSON + HTML reports:
```bash
python3 main.py scan \
  --input /path/to/clips \
  --output /path/to/output \
  --gpus 0,1 \
  --vid-stride 15 \
  --confidence 0.30
```

### `filter`
Scan and automatically copy (or move) only `KEEP` clips into a clean folder structure organized by camera:
```bash
python3 main.py filter \
  --input /path/to/clips \
  --output /path/to/output \
  --export-dir /path/to/saved_events \
  --action copy \
  --gpus 0,1
```

### `watch`
Run as a background daemon continuously watching an incoming directory for new clips:
```bash
python3 main.py watch \
  --input /path/to/incoming_clips \
  --output /path/to/output \
  --export-dir /path/to/saved_events \
  --interval 5.0 \
  --gpus 0,1
```

---

## Unit & Ground-Truth Tests

Run the test suite inside the container:
```bash
# Unit tests
docker compose run --rm --entrypoint python3 analysis-engine tests/test_classifier.py

# Ground-truth validation (5 verified test cases)
docker compose run --rm --entrypoint python3 analysis-engine tests/test_ground_truth.py
```
