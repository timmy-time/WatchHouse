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

## Offloading Live Camera Inference (Windows / WSL2)

The live engine can send its per-frame YOLO work to another machine, so the camera
server only decodes, tracks events, records and serves the dashboard. Frames go out
as JPEG and tracked boxes come back, one tracker session per camera.

**1. On the Windows / WSL2 box** (GTX 1650 = sm_75, works with current CUDA wheels):

```bash
git clone <this repo> && cd watchhouse
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # pulls torch + ultralytics
python main.py infer-server --host 0.0.0.0 --port 8099 --device 0
curl http://localhost:8099/health        # {"ok":true,"device":"0","sessions":0,...}
```

Model weights download on first use (`yolov8n.pt`), or copy the `.pt` files over.
On Windows itself, add an inbound firewall rule for TCP 8099. From WSL2 the LAN also
needs a port proxy to reach the NAT'd WSL VM:

```powershell
netsh interface portproxy add v4tov4 listenport=8099 listenaddress=0.0.0.0 `
  connectport=8099 connectaddress=$(wsl hostname -I).Trim()
netsh advfirewall firewall add rule name="CCTV infer" dir=in action=allow protocol=TCP localport=8099
```

**2. On the camera server**, point the analysis at it (`config/live.yaml`):

```yaml
analysis:
  realtime_device: remote                    # or per camera: device: remote
  remote_url: "http://<windows-lan-ip>:8099"
  remote_timeout: 10.0
  remote_jpeg_quality: 80
  detailed_device: remote                    # optional: move the detailed verifier too
```

**3. Verify**: startup logs say `inference=remote`, and `output/live/status.json`
carries `inference_stats` per camera (calls, errors, avg round-trip ms).

Notes: ~35-55 KB per frame at 704×396 q80 (~0.5 MB/s per camera at 15 fps). If the
remote host is unreachable or slower than the frame rate, frames keep decoding and
the camera simply analyses fewer frames; the client returns no detections, logs one
warning per 20 s and counts the failures in the status file.

---

## Unit & Ground-Truth Tests

Run the test suite inside the container:
```bash
# Unit tests
docker compose run --rm --entrypoint python3 analysis-engine tests/test_classifier.py

# Ground-truth validation (5 verified test cases)
docker compose run --rm --entrypoint python3 analysis-engine tests/test_ground_truth.py
```
