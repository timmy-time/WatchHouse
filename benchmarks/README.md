# Benchmarks

Ground-truth evaluation of the detection/tracking stack against labelled objects
in real camera feeds.

## Camera D — cars behind foliage (the hard case)

Ground truth (`foliage_gt.json`): two parked vehicles on the far side of the road
seen through/behind the foliage fence and street trees:

| id | label | box (normalized) | note |
|----|-------|------------------|------|
| truck_parked | truck | 0.69, 0.069, 0.941, 0.378 | prime mover parked far side of road |
| hyundai_parked | car | 0.452, 0.12, 0.521, 0.23 | sedan behind tree branches |

### Capture a clip

```bash
docker compose run --rm --entrypoint bash live -c '
  /usr/bin/ffmpeg -hide_banner -loglevel error -rtsp_transport tcp -stimeout 10000000 \
    -i "$CAM_CAMD_URL" -t 120 -map 0:v:0 -c copy -y /data/output/bench/foliage.mp4'
```
(Audio must be dropped: the DVR streams pcm_alaw, which MP4 cannot hold.)

### Run the sweep

```bash
docker compose run --rm \
  -v "$PWD/tools:/app/tools:ro" -v "$PWD/models:/models:ro" -v "$PWD/config:/bench-config:ro" \
  -e CUDA_VISIBLE_DEVICES=1 --entrypoint python3 analysis-engine \
  tools/benchmark.py --clip /data/output/bench/foliage.mp4 \
                     --gt /data/output/bench/gt_foliage.json \
                     --stride 2 --model-dir /models/ \
                     --out /data/output/bench/results.json
```
Run it on the *detailed* GPU (CUDA_VISIBLE_DEVICES=1) so live realtime inference is
not disturbed. `--tracker /bench-config/bytetrack_mild.yaml` swaps the tracker profile (the canonical
profiles live in `config/`, mounted here as `/bench-config`);
`--only <substring>` filters variants.

### Tracker comparison (yolov8n@640 conf .20, 120 s clip)

| tracker profile | truck det | truck gaps | truck longest run | hyundai det | hyundai gaps | hyundai IDs |
|---|---|---|---|---|---|---|
| default (buffer 30, match .80) | 1.00 | 2 | 571 | 0.92 | 69 | 5 |
| flicker (buffer 90, match .95) | 1.00 | 0 | **900** | 0.97 | 28 | 14 |
| **mild (buffer 90, match .80) — deployed** | 1.00 | 0 | **900** | 0.97 | 33 | **11** |

The aggressive .95 association wins on paper but steals IDs on a busy road
(tracks starting on the parked car then flying 10-13 object-widths away, seen in
the live event log) — the mild profile keeps the same continuity with churn 11 vs 14.

### Detector sweep (120 s clip, stride 2 ≈ 12.5 fps, 2026-10-04)

| config | hyundai det | missing frames | truck continuity | ms/frame |
|---|---|---|---|---|
| LIVE BASELINE · yolov8n@640 conf .30 | 0.83 | 151 | 571-frame run | 29.7 |
| yolov8n@640 conf .20 | 0.92 | 69 | 571-frame run | 26.6 |
| **conf .20 + flicker tracker (deployed)** | **0.97** | **28** | **900 = whole clip, 0 gaps** | 29.5 |
| yolov8s@960 + flicker tracker | 0.98 | 16 | 655-frame run | 37.7 |

Findings:
- **Detection was not the main failure — track fragmentation was.** A parked car
  produced 5–14 ByteTrack IDs per 2 minutes; fragmented tracks whose boxes jump
  are classified as `vehicle_arrived/departed/passing`, which spammed events.
- Raising resolution/model buys a little (0.97→0.98) for +27% inference cost.
- ROI-crop inference did **not** beat simply using a better model/resolution and
  cannot see the rest of the scene — rejected.
- Deployed fix: per-camera `confidence: 0.20` + flicker-tolerant ByteTrack
  (`config/bytetrack_flicker.yaml`: high_thresh .15 / low .05 / buffer 90 /
  match_thresh .95). Scoped to the Camera D only — looser association is risky in
  heavy-traffic scenes.

### Live outcome (deployed config)

- Parked cars are now **anchored to their friendly slots** in the live feed
  (`anchored=Parked Truck ...`, `anchored=Parked Hyundai`), including detections
  at conf 0.23 that the old 0.30 threshold discarded.
- **Slot-occupancy absorption** (`engine/live/events.py`): while a friendly slot
  shows continuous vehicle presence, any detection inside it is attributed to the
  parked vehicle and cannot qualify for an event — so box jitter from foliage no
  longer fires `vehicle_arrived/departed/passing`.
- Before: 39% of Camera D events came from parked-car churn with impossible
  kinematics (displacement 10-13x object width, `min_iou_start` 0.00).
  After: 1 of 9, and the rest are genuine passing traffic (displacement 1-5x,
  outside the slots).

Re-run the sweep after any detector/tracker change and compare against these tables.
