"""Reporting module: generates structured JSON outputs and an interactive HTML review gallery."""

from dataclasses import asdict
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional
from engine.classifier import EventDecision, Verdict


_HEX_HASH_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def parse_clip_metadata(clip_path: str) -> Dict[str, str]:
    """Extract camera name and timestamp from standard clip filenames.

    Positional convention (underscore-joined, any DVR brand)::

        <channel:8hex>_<n>_<DVR-id>_<Camera Name>_<hash:32hex>_<YYYYMMDDHHMMSS>.<ext>

    The camera is the field immediately before the trailing 32-hex digest; a
    shorter 4-field form (<n>_<channel>_<Camera Name>_<timestamp>) is also
    accepted. The DVR id itself is never interpreted, so any recorder works.
    """
    filename = os.path.basename(clip_path)
    parts = filename.split("_")
    camera = "Unknown"
    timestamp = ""
    # A 32-hex field is a content hash, not a camera: the camera name is the
    # field directly before it. This is DVR-agnostic and reproduces the
    # canonical 6-field layout without hardcoding a positional index.
    hash_idx = next(
        (i for i in range(len(parts) - 1, 0, -1) if _HEX_HASH_RE.match(parts[i])),
        None,
    )
    if hash_idx is not None:
        camera = parts[hash_idx - 1]
        timestamp = parts[-1].replace(".mp4", "").replace(".png", "")
    elif len(parts) >= 6:
        camera = parts[3]
        timestamp = parts[-1].replace(".mp4", "").replace(".png", "")
    elif len(parts) >= 4:
        camera = parts[2]
        timestamp = parts[-1].replace(".mp4", "").replace(".png", "")

    return {
        "filename": filename,
        "camera": camera,
        "timestamp": timestamp,
    }


class AnalysisReporter:
    """Produces JSON summary and HTML gallery for event analysis results."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_json_report(
        self,
        records: List[Dict[str, Any]],
        runtime_seconds: float,
    ) -> str:
        """Write detailed results to analysis_results.json."""
        total = len(records)
        kept = sum(1 for r in records if r["verdict"] == Verdict.KEEP.value)
        discarded = sum(1 for r in records if r["verdict"] == Verdict.DISCARD.value)
        errors = sum(1 for r in records if r["verdict"] == Verdict.ERROR.value)
        reduction_pct = round(((discarded) / max(1, total)) * 100.0, 1)

        summary = {
            "total_clips": total,
            "kept_count": kept,
            "discarded_count": discarded,
            "error_count": errors,
            "reduction_percentage": reduction_pct,
            "runtime_seconds": round(runtime_seconds, 2),
            "results": records,
        }

        json_path = os.path.join(self.output_dir, "analysis_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return json_path

    def write_html_gallery(
        self,
        records: List[Dict[str, Any]],
        runtime_seconds: float,
    ) -> str:
        """Generate a self-contained, responsive HTML gallery dashboard."""
        total = len(records)
        kept = sum(1 for r in records if r["verdict"] == Verdict.KEEP.value)
        discarded = sum(1 for r in records if r["verdict"] == Verdict.DISCARD.value)
        reduction_pct = round(((discarded) / max(1, total)) * 100.0, 1)

        # Unique cameras
        cameras = sorted(list({r.get("camera", "Unknown") for r in records}))

        # Ensure relative paths from output_dir to media files
        clean_records = []
        for r in records:
            item = dict(r)
            clip_p = r.get("clip_path")
            thumb_p = r.get("thumbnail_path")
            if clip_p and os.path.isabs(clip_p):
                try:
                    item["clip_rel"] = os.path.relpath(clip_p, start=self.output_dir)
                except Exception:
                    item["clip_rel"] = os.path.basename(clip_p)
            if thumb_p and os.path.isabs(thumb_p):
                try:
                    item["thumbnail_rel"] = os.path.relpath(thumb_p, start=self.output_dir)
                except Exception:
                    item["thumbnail_rel"] = os.path.basename(thumb_p)
            clean_records.append(item)

        # Generate JSON string for JS data
        records_json = json.dumps(clean_records)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WatchHouse Event Analysis Gallery</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent-keep: #10b981;
      --accent-discard: #64748b;
      --accent-error: #ef4444;
      --border: #334155;
      --card-hover: #273549;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{ font-size: 1.5rem; font-weight: 700; }}
    .stats-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }}
    .stat-val {{
      font-size: 2rem;
      font-weight: 800;
    }}
    .stat-lbl {{
      font-size: 0.85rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .keep-val {{ color: var(--accent-keep); }}
    .discard-val {{ color: var(--accent-discard); }}
    .reduct-val {{ color: #38bdf8; }}

    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 24px;
      background: var(--card-bg);
      padding: 12px 16px;
      border-radius: 8px;
      align-items: center;
    }}
    .filter-btn {{
      background: #334155;
      color: var(--text);
      border: none;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 500;
      transition: all 0.2s;
    }}
    .filter-btn:hover {{ background: #475569; }}
    .filter-btn.active {{
      background: #3b82f6;
      color: white;
    }}
    select {{
      background: #334155;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
      outline: none;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s, border-color 0.15s;
    }}
    .card:hover {{
      transform: translateY(-2px);
      border-color: #475569;
    }}
    .card.is-keep {{ border-top: 4px solid var(--accent-keep); }}
    .card.is-discard {{ border-top: 4px solid var(--accent-discard); }}
    .card.is-error {{ border-top: 4px solid var(--accent-error); }}

    .thumb-wrap {{
      position: relative;
      width: 100%;
      height: 180px;
      background: #000;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .thumb-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}
    .play-overlay {{
      position: absolute;
      width: 44px;
      height: 44px;
      background: rgba(0,0,0,0.65);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.2s;
    }}
    .thumb-wrap:hover .play-overlay {{
      transform: scale(1.15);
      background: rgba(0,0,0,0.85);
    }}
    .play-icon {{
      width: 0;
      height: 0;
      border-top: 9px solid transparent;
      border-bottom: 9px solid transparent;
      border-left: 15px solid white;
      margin-left: 4px;
    }}
    .card-body {{
      padding: 14px;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
    }}
    .badge-keep {{ background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid #059669; }}
    .badge-discard {{ background: rgba(100,116,139,0.2); color: #cbd5e1; border: 1px solid #475569; }}
    .badge-error {{ background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid #dc2626; }}
    .camera-tag {{ font-size: 0.85rem; font-weight: 600; color: #38bdf8; }}
    .reason-text {{ font-size: 0.85rem; color: var(--text); }}
    .meta-sub {{ font-size: 0.75rem; color: var(--muted); }}

    /* Modal */
    .modal {{
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.85);
      z-index: 1000;
      align-items: center;
      justify-content: center;
    }}
    .modal.active {{ display: flex; }}
    .modal-content {{
      background: var(--card-bg);
      border-radius: 8px;
      max-width: 900px;
      width: 90%;
      padding: 20px;
      position: relative;
    }}
    .close-btn {{
      position: absolute;
      top: 10px; right: 14px;
      color: var(--muted);
      font-size: 1.8rem;
      cursor: pointer;
      line-height: 1;
    }}
    .close-btn:hover {{ color: white; }}
    video {{
      width: 100%;
      max-height: 550px;
      border-radius: 6px;
      margin-top: 10px;
      outline: none;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>WatchHouse Event Analysis</h1>
      <div class="meta-sub">Accelerated Detection Pipeline &bull; Stride 15 &bull; Runtime: {round(runtime_seconds, 1)}s</div>
    </div>
  </header>

  <div class="stats-bar">
    <div class="stat-card">
      <div class="stat-val">{total}</div>
      <div class="stat-lbl">Total Clips</div>
    </div>
    <div class="stat-card">
      <div class="stat-val keep-val">{kept}</div>
      <div class="stat-lbl">Kept Events (Valuable)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val discard-val">{discarded}</div>
      <div class="stat-lbl">Discarded (False Triggers)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val reduct-val">{reduction_pct}%</div>
      <div class="stat-lbl">Storage / Noise Reduction</div>
    </div>
  </div>

  <div class="controls">
    <span style="font-size: 0.9rem; color: var(--muted); margin-right: 4px;">Filter:</span>
    <button class="filter-btn active" onclick="setVerdictFilter('ALL', this)">All ({total})</button>
    <button class="filter-btn" onclick="setVerdictFilter('KEEP', this)">Keep Only ({kept})</button>
    <button class="filter-btn" onclick="setVerdictFilter('DISCARD', this)">Discarded ({discarded})</button>

    <div style="margin-left: auto; display: flex; gap: 8px; align-items: center;">
      <span style="font-size: 0.9rem; color: var(--muted);">Camera:</span>
      <select id="cameraFilter" onchange="applyFilters()">
        <option value="ALL">All Cameras</option>
        {"".join(f'<option value="{c}">{c}</option>' for c in cameras)}
      </select>
    </div>
  </div>

  <div class="grid" id="clipsGrid"></div>

  <!-- Video Modal -->
  <div class="modal" id="videoModal" onclick="closeModal(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
      <span class="close-btn" onclick="closeModal()">&times;</span>
      <h3 id="modalTitle" style="font-size: 1.1rem; margin-bottom: 6px;"></h3>
      <div id="modalSubtitle" class="meta-sub"></div>
      <video id="modalVideo" controls></video>
    </div>
  </div>

  <script>
    const DATA = {records_json};
    let currentVerdict = 'ALL';

    function setVerdictFilter(v, btn) {{
      currentVerdict = v;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyFilters();
    }}

    function applyFilters() {{
      const cam = document.getElementById('cameraFilter').value;
      const filtered = DATA.filter(item => {{
        const matchVerdict = (currentVerdict === 'ALL') || (item.verdict === currentVerdict);
        const matchCam = (cam === 'ALL') || (item.camera === cam);
        return matchVerdict && matchCam;
      }});
      renderGrid(filtered);
    }}

    function renderGrid(items) {{
      const grid = document.getElementById('clipsGrid');
      if (items.length === 0) {{
        grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--muted);">No matching clips found.</div>';
        return;
      }}
      grid.innerHTML = items.map(item => {{
        const isKeep = item.verdict === 'KEEP';
        const cardClass = isKeep ? 'is-keep' : (item.verdict === 'DISCARD' ? 'is-discard' : 'is-error');
        const badgeClass = isKeep ? 'badge-keep' : (item.verdict === 'DISCARD' ? 'badge-discard' : 'badge-error');
        const thumbSrc = item.thumbnail_rel || item.thumbnail_path || '';
        const videoSrc = item.clip_rel || item.clip_path || '';

        return `
          <div class="card ${{cardClass}}">
            <div class="thumb-wrap" onclick="playClip('${{videoSrc}}', '${{item.filename}}', '${{item.primary_reason}}')">
              ${{thumbSrc ? `<img src="${{thumbSrc}}" alt="thumbnail" loading="lazy">` : `<div style="color:var(--muted)">No Preview</div>`}}
              <div class="play-overlay"><div class="play-icon"></div></div>
            </div>
            <div class="card-body">
              <div class="card-header">
                <span class="badge ${{badgeClass}}">${{item.verdict}}</span>
                <span class="camera-tag">${{item.camera}}</span>
              </div>
              <div class="reason-text"><strong>Reason:</strong> ${{item.primary_reason.replace(/_/g, ' ')}}</div>
              <div class="meta-sub" title="${{item.filename}}">${{item.timestamp || item.filename}}</div>
            </div>
          </div>
        `;
      }}).join('');
    }}

    function playClip(src, title, reason) {{
      const modal = document.getElementById('videoModal');
      const video = document.getElementById('modalVideo');
      document.getElementById('modalTitle').textContent = title;
      document.getElementById('modalSubtitle').textContent = 'Event reason: ' + reason.replace(/_/g, ' ');
      video.src = src;
      modal.classList.add('active');
      video.play().catch(e => console.log('Autoplay blocked', e));
    }}

    function closeModal() {{
      const modal = document.getElementById('videoModal');
      const video = document.getElementById('modalVideo');
      video.pause();
      video.src = '';
      modal.classList.remove('active');
    }}

    // Initial render
    applyFilters();
  </script>
</body>
</html>
"""
        html_path = os.path.join(self.output_dir, "gallery.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path

    def export_kept_clips(
        self,
        records: List[Dict[str, Any]],
        export_dir: str,
        action: str = "copy",
    ) -> int:
        """Copy or move kept clips to an organized export directory structure."""
        os.makedirs(export_dir, exist_ok=True)
        count = 0
        for r in records:
            if r["verdict"] == Verdict.KEEP.value:
                src_path = r["clip_path"]
                if not os.path.exists(src_path):
                    continue

                cam = r.get("camera", "Unknown")
                dest_dir = os.path.join(export_dir, "keep", cam)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(src_path))

                if action == "move":
                    shutil.move(src_path, dest_path)
                else:
                    shutil.copy2(src_path, dest_path)

                # Also copy thumbnail if available
                thumb_src = r.get("thumbnail_path")
                if thumb_src and os.path.exists(thumb_src):
                    dest_thumb = os.path.join(dest_dir, os.path.basename(thumb_src))
                    if action == "move":
                        shutil.move(thumb_src, dest_thumb)
                    else:
                        shutil.copy2(thumb_src, dest_thumb)

                count += 1

        return count
