# WatchHouse

AI-powered video intelligence that watches your live camera feeds and saved footage, keeping what matters.

WatchHouse is a self-hosted, privacy-first video intelligence engine for IP cameras. It analyses both live RTSP streams and archived clip directories on hardware you already own: nothing leaves your network, and there is no cloud service, account, or telemetry. Its job is to separate genuine events — people, animals, moving vehicles — from the constant noise that fills up a security archive, so the clips worth keeping are the only ones you see.

The core problem it solves is parked cars. A camera pointed at a parking area records a vehicle sitting still for hours, and every spiderweb, headlight sweep, wind gust and IR glare starts another "event". WatchHouse tracks objects across frames and scores their kinematics, so a car that never moves is discarded while a car that pulls away is kept.

---

## Why WatchHouse

- **Stationary-vehicle suppression by tracked kinematics, not single-frame detection.** Vehicles are tracked with ByteTrack across a subsampled frame sequence; a vehicle that maintains bounding-box overlap and stays within a normalized centroid-displacement budget is classified `STATIONARY_VEHICLE` and discarded, regardless of how many frames it appears in.
- **Asymmetric keep rules for high-value classes.** People and animals trigger `KEEP` on detection alone, while displacement thresholds apply only to vehicles. A person standing still is still an event.
- **An analysis cache that makes rescans effectively free.** Verdicts and kinematic metrics are stored in `output/analysis_cache.json` keyed by path relative to the input root and validated against file size and mtime, so re-scanning a growing archive only touches newly arrived footage.
- **Scales across one or more GPUs.** Worker processes are partitioned across every GPU you provide, with a CPU fallback when CUDA is unavailable, so throughput grows with the hardware you have rather than assuming a fixed configuration.
- **Two pipelines, one engine.** The same detector and classifier drive a batch pipeline over recorded clips and a live RTSP pipeline that analyses, records, and raises events as they happen.
- **A web dashboard, not just a report file.** Live view with MJPEG preview, an archive browsable by camera, event class and date, plus scenery/vehicle-slot registration so static objects can be absorbed instead of re-alerting.
- **Optional remote inference offload.** Move per-frame detection to a machine with a supported GPU — including a Windows/WSL2 host — while the camera server keeps decoding, recording and serving the dashboard.

---

## Architecture

### Batch pipeline (`engine/`)

Processes an existing directory of recorded `.mp4` clips (plus optional companion `.png` thumbnails, which can be used as a fast triage path) and produces JSON, HTML and an organised `KEEP` tree.

```
discover clips → cache filter → worker processes (1 per GPU) → classify → store KEEP → report
```

| Module | Responsibility |
| :--- | :--- |
| `engine/pipeline.py` | Spawns worker processes with `multiprocessing.get_context("spawn")`, partitions clips across the requested GPUs, falls back to CPU per worker when CUDA is unavailable. |
| `engine/detector.py` | YOLOv8 object detection coupled with ByteTrack multi-object tracking. Frames are subsampled (`vid_stride`, default 15 ≈ 1.6 fps on 25 fps video). |
| `engine/classifier.py` | Stationary-vehicle suppression and verdict assignment. Computes start-to-current IoU and normalized centroid displacement per track and labels vehicles `STATIONARY_VEHICLE` or `KEEP`. |
| `engine/cache.py` | `output/analysis_cache.json` keyed by path relative to the input root, validated by file size and mtime. |
| `engine/storage.py` | Mirrors `KEEP` clips into `output/clips/<folder>/<clip>.mp4`, preserving folder structure and companion thumbnails. |
| `engine/reporter.py` | Writes `output/analysis_results.json` and a self-contained `output/gallery.html` with filters and an inline video modal. |

### Live pipeline (`engine/live/`)

Consumes RTSP streams continuously, records segments, opens and closes events, and publishes notifications and a dashboard feed.

| Module | Responsibility |
| :--- | :--- |
| `engine/live/stream.py` | RTSP decode via FFmpeg with hardware acceleration, and stream selection (main vs substream). |
| `engine/live/worker.py` | Per-camera workers: frame loop, dynamic frame rate, scheduling of realtime vs detailed inference. |
| `engine/live/events.py` | Event lifecycle (open/close/cooldown), zones, notification dispatch. |
| `engine/live/infer_server.py` | FastAPI inference server exporting `/track`, `/detect`, `/reset`, `/health` for remote offload. |
| `engine/live/remote_infer.py` | Client that JPEG-encodes frames to a remote host and parses tracked boxes back. |
| `engine/web/` | Dashboard and API: live view, MJPEG, archive by camera/event class/date, scenery and vehicle-slot registration. |

### Scaling

Clips (batch) or cameras (live) are distributed across the GPUs you pass in. With `--gpus 0,1` two worker processes each own one device; with `--gpus 0` a single worker does all the work. Passing no GPUs, or running where CUDA is unavailable, falls back to CPU inference at substantially reduced throughput.

---

## Performance

Measured on a handful of 1080p camera feeds; your numbers will vary with clip length, resolution, scene activity and hardware.

| Metric | Result |
| :--- | :--- |
| **Throughput** | Well above real-time — roughly one clip per second per GPU on 30-second 1080p clips |
| **Noise reduction** | The large majority of clips discarded in a parked-vehicle-dominated scene |
| **Kept events** | Confirmed people and moving vehicles |
| **Subsequent cached scans** | Near-instant: every unchanged clip skipped with zero GPU compute |

---

## Requirements

- **Docker** with the **NVIDIA Container Toolkit** for GPU acceleration.
- **An NVIDIA GPU with a CUDA-capable driver.** Any supported compute capability works; pre-Turing cards can be pinned to an older CUDA/PyTorch base image if current wheels drop them.
- **CPU-only is supported**: omit the GPU reservation in `docker-compose.yml`. Inference falls back to CPU automatically, with significantly lower throughput — expect a few frames per second across a handful of cameras rather than many.
- `docker compose` v2.

---

## Quickstart (Docker Compose)

### 1. Configure

Copy the example environment file and fill in your streams:

```bash
cp .env.example .env
```

### 2. Batch scan an archive

Drop recorded clips into `./clips`, then run the batch profile:

```bash
docker compose run --rm analysis-engine
```

This mounts `./clips` read-only as input and writes results to `./output/`. The `analysis-engine` service runs `scan --input /data/clips --output /data/output --gpus 0,1`; edit `command:` in `docker-compose.yml` to change the arguments.

### 3. View results

- Open `output/gallery.html` in a browser for the self-contained static gallery.
- Or run the web dashboard:

```bash
docker compose up -d web
```

The dashboard listens on host port **18080** (container 8080). Set `DASHBOARD_USER` and `DASHBOARD_PASSWORD` in `.env` to enable HTTP Basic Auth; leave both blank to run unauthenticated on a trusted network.

---

## CLI Reference

All commands live in `main.py` at the repository root.

### `scan` — batch analysis

Analyse every `.mp4` in a directory and write JSON + HTML reports.

```bash
python3 main.py scan \
  --input /path/to/clips \
  --output /path/to/output \
  --gpus 0,1 \
  --vid-stride 15 \
  --confidence 0.30
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input`, `-i` | `/data/clips` | Directory containing `.mp4` clips. |
| `--output`, `-o` | `/data/output` | Destination for reports and gallery. |
| `--gpus` | `0,1` | Comma-separated GPU indices, e.g. `0` or `0,1`. |
| `--vid-stride` | `15` | Frame subsampling stride. |
| `--confidence` | `0.30` | Detection confidence threshold. |
| `--model` | `yolov8n.pt` | YOLO model path or name. |
| `--no-cache` | off | Disable the cache and re-analyse everything. |
| `--no-store-keep` | off | Skip copying `KEEP` clips to `output/clips/`. |
| `--cache-file` | `<output>/analysis_cache.json` | Custom cache path. |

### `filter` — scan and export kept clips

Runs the same analysis and additionally copies or moves `KEEP` clips into a clean directory.

```bash
python3 main.py filter \
  --input /path/to/clips \
  --output /path/to/output \
  --export-dir /path/to/saved_events \
  --action copy \
  --gpus 0,1
```

Accepts all `scan` flags plus `--export-dir`/`-e` (required) and `--action` (`copy` or `move`).

### `watch` — continuous directory watcher

Polls an incoming directory and analyses new clips as they land.

```bash
python3 main.py watch \
  --input /path/to/incoming_clips \
  --output /path/to/output \
  --export-dir /path/to/saved_events \
  --interval 5.0 \
  --gpus 0,1
```

Accepts all `scan` flags plus `--interval` (seconds, default `5.0`) and optional `--export-dir`/`--action`.

### `live` — live RTSP pipeline

Decode, analyse, record and raise events for the cameras in a YAML config.

```bash
python3 main.py live --config /app/config/live.yaml --output /data/output
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--config`, `-c` | `/app/config/live.yaml` | Path to the live configuration. |
| `--output`, `-o` | `/data/output` | Output directory for events and preview assets. |

### `serve` — web dashboard and API

Serve the dashboard, archive API, live MJPEG view and scenery registration UI.

```bash
python3 main.py serve \
  --output /data/output \
  --clips /data/clips \
  --config /app/config/live.yaml \
  --host 0.0.0.0 \
  --port 8080
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--output`, `-o` | `/data/output` | Directory containing events and results. |
| `--clips` | `/data/clips` | Clips directory for historical playback. |
| `--config`, `-c` | `/app/config/live.yaml` | Live configuration (camera and zone metadata). |
| `--host` | `0.0.0.0` | Bind interface. |
| `--port` | `8080` | Bind port. |

In Docker Compose the `web` service runs this command and publishes it on host port 18080.

### `infer-server` — remote inference endpoint

Serve YOLO inference over HTTP so another machine can offload its per-frame detection.

```bash
python3 main.py infer-server --host 0.0.0.0 --port 8099 --device 0
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--host` | `0.0.0.0` | Bind interface. |
| `--port` | `8099` | Bind port. |
| `--device` | `0` | Torch device: `0`, `cuda:0` or `cpu`. |
| `--max-sessions` | `8` | Maximum cached camera sessions. |

---

## Live Cameras and Dashboard

### Configuration

`config/live.yaml` holds the recording settings, the analysis block, notifications, and one entry per camera:

```yaml
analysis:
  fps: 15
  model: yolov8n.pt
  confidence: 0.30
  imgsz: 640
  # Analysis pipe width. This drives per-frame CPU work (rawvideo read, motion diff,
  # preview encode) and detector input. 0 keeps the native source resolution.
  pipe_width: 0
  # Which stream feeds the pipe: "main" or "sub".
  source: main
  # "auto"/"gpu" runs YOLO on the configured GPU, "cpu" on this machine, "remote"
  # on another host via remote_url. Cameras can override with their own `device:`.
  realtime_device: auto
  detailed_device: auto
  remote_url: ""
  remote_timeout: 10.0
  remote_jpeg_quality: 80

notifications:
  cooldown_seconds: 60
  dashboard_url: "${DASHBOARD_URL}"
  notify_classes: [person, dog, cat, bear, horse, cow, sheep]
  ntfy: {url: "${NTFY_URL}", topic: "${NTFY_TOPIC}", token: "${NTFY_TOKEN}"}
  webhook: {url: "${WEBHOOK_URL}"}

cameras:
  - name: Drive
    url: "rtsp://user:pass@<camera-host>:554/stream1"       # main stream, used for recording
    sub_url: "rtsp://user:pass@<camera-host>:554/stream2"   # optional substream for analysis
    gpu: 0
    record: true
    confidence: 0.30
    device: auto                 # optional per-camera override: cpu | gpu | remote | auto
    zones:
      - {name: property, type: entry, polygon: [[0.12, 0.30], [1.0, 0.30], [1.0, 1.0], [0.05, 1.0]]}
      - {name: ignore_me, type: ignore, polygon: [[0.65, 0.55], [1.0, 0.55], [1.0, 1.0], [0.65, 1.0]]}
```

Per-camera options:

| Key | Meaning |
| :--- | :--- |
| `name` | Display name; also the camera column in reports and the archive. |
| `url` | Main RTSP stream. Used for recording. |
| `sub_url` | Optional lower-resolution substream. |
| `gpu` | Preferred GPU index for this camera's inference. |
| `device` | Optional per-camera override of `realtime_device`. |
| `record` | `false` to run live inference without writing segments or event clips. |
| `confidence` | Per-camera detection confidence. A lower value helps with partially occluded objects. |
| `tracker_config` | Optional path to a ByteTrack profile for this camera. |
| `zones` | Named polygons with `type: entry` (alerts) or `type: ignore` (suppressed). |

Environment variables (`CAM_*_URL`, `CAM_*_SUB_URL`, `NTFY_*`, `WEBHOOK_URL`, `DASHBOARD_URL`, `DASHBOARD_USER`, `DASHBOARD_PASSWORD`) are interpolated by docker compose from `.env`. Because the compose file lists each variable explicitly under `environment:`, a variable present in `.env` but missing from that list never reaches the container — add it in both places.

### Main stream vs substream

Record from the **main** stream and analyse from the **substream** when available.

- The main stream is what ends up in your clips, so keep its resolution and quality.
- The substream is the recommended analysis path: far less CPU per frame (raw decode, motion diff and preview encode all scale with frame size) with the same detection quality at a smaller detector input.
- Caveat: substreams frequently have a different aspect ratio and field of view from the main stream. Zones and scenery slots are drawn in the framing of whichever stream feeds the pipe (`analysis.source`), so switching a camera between `main` and `sub` means redrawing its zones.
- Many cameras expose substreams on a separate RTSP path; check your camera's documentation. If there is no substream, leave `sub_url` blank and analyse the main stream, optionally lowering `analysis.pipe_width` instead.

### Dashboard

The `web` service serves:

- **Live view** — per-camera MJPEG preview with boxes, zones and registered vehicle slots overlaid.
- **Archive** — kept and discarded clips, filterable by camera, event class and date range.
- **Scenery and vehicle slots** — register static regions and parking slots so persistent objects are absorbed rather than re-alerted.

Put the dashboard behind Basic Auth by setting `DASHBOARD_USER` and `DASHBOARD_PASSWORD`. If both are blank the middleware is not installed and the dashboard is open to anyone who can reach the port.

### Notifications

Events are dispatched to:

- **ntfy** (`NTFY_URL`, `NTFY_TOPIC`, optional `NTFY_TOKEN`) — push to your phone or desktop.
- **Generic webhook** (`WEBHOOK_URL`) — any HTTP endpoint.
- **`DASHBOARD_URL`** — used to build links back into the archive.

`notifications.cooldown_seconds` rate-limits repeats, and `notify_classes` restricts which object classes trigger an alert.

---

## Remote Inference Offload

The live pipeline can send its per-frame YOLO work to another machine, so the camera server only decodes, tracks events, records and serves the dashboard. Frames travel as JPEG and tracked boxes come back, with one tracker session per camera on the remote side.

This is useful when the camera server has no GPU (or a weak one) but another machine on the network does — including a Windows host running WSL2.

### 1. On the machine with the GPU

```bash
git clone <this repo> && cd WatchHouse
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python main.py infer-server --host 0.0.0.0 --port 8099 --device 0
curl http://localhost:8099/health
```

Model weights download on first use (`yolov8n.pt`), or copy existing `.pt` files across. On Windows, add an inbound firewall rule for TCP 8099. If the GPU host runs inside WSL2, the LAN also needs a port proxy to reach the NAT'd WSL VM:

```powershell
netsh interface portproxy add v4tov4 listenport=8099 listenaddress=0.0.0.0 `
  connectport=8099 connectaddress=$(wsl hostname -I).Trim()
netsh advfirewall firewall add rule name="WatchHouse infer" dir=in action=allow protocol=TCP localport=8099
```

### 2. On the camera server

Point the analysis at the remote host in `config/live.yaml`:

```yaml
analysis:
  realtime_device: remote                    # or per camera: device: remote
  remote_url: "http://<remote-host>:8099"
  remote_timeout: 10.0
  remote_jpeg_quality: 80
  detailed_device: remote                    # optional: move the detailed verifier too
```

Both the realtime cameras and the detailed verifier can be offloaded.

### 3. Verify

Startup logs report the inference backend, and `output/live/status.json` carries per-camera `inference_stats` (calls, errors, average round-trip milliseconds).

**Notes.** Each frame is roughly 35–55 KB at 704×396 with `remote_jpeg_quality: 80` — on the order of 0.5 MB/s per camera at 15 fps, so a fast LAN is fine but a congested Wi-Fi link is not. If the remote host is unreachable or slower than the frame rate, frames keep decoding and the camera simply analyses fewer of them: the client returns no detections, logs one warning every 20 seconds, and counts the failures in the status file. Nothing else stalls.

---

## Testing

The test suite lives in `tests/` and runs inside the analysis container:

```bash
docker compose run --rm --entrypoint python3 analysis-engine tests/test_classifier.py
```

Replace the filename with any other `tests/test_*.py` module. Coverage includes the classifier's kinematic thresholds, the analysis cache, KEEP storage, the live event lifecycle, the remote inference client and server, the web API and archive, and the dynamic frame-rate controller.

`tests/test_ground_truth.py` is an optional, footage-dependent validation: it checks the classifier against a small set of hand-labelled clips from a specific deployment. It only passes when you supply your own labelled footage in the layout it expects — a fresh clone should treat the remaining unit tests as the meaningful suite.

---

## License

WatchHouse is released under the **AGPL-3.0**. See [LICENSE](LICENSE).
