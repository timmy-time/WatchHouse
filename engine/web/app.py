"""FastAPI web application serving CCTV live streams, events, archive, and face recognition."""

import asyncio
import base64
from datetime import datetime
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

from engine.scenery import SceneryManager, extract_vehicle_signature
from engine.faces import FaceEngine
from engine.web.archive import (
    build_facets,
    build_index,
    describe,
    filter_indices,
    parse_class_filter,
    parse_date_bound,
    sort_indices,
)
from engine.live.config import load_live_config
from engine.live.db import EventStore

logger = logging.getLogger(__name__)


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self.username = username
        self.password = password

    async def dispatch(self, request: Request, call_next):
        # Allow pre-flight OPTIONS requests without auth
        if request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return Response(
                status_code=401,
                content="Unauthorized",
                headers={"WWW-Authenticate": 'Basic realm="cctv"'},
            )

        try:
            b64_creds = auth_header[6:].strip()
            decoded = base64.b64decode(b64_creds).decode("utf-8")
            user, pwd = decoded.split(":", 1)
        except Exception:
            return Response(
                status_code=401,
                content="Unauthorized",
                headers={"WWW-Authenticate": 'Basic realm="cctv"'},
            )

        user_ok = secrets.compare_digest(user, self.username)
        pwd_ok = secrets.compare_digest(pwd, self.password)
        if not (user_ok and pwd_ok):
            return Response(
                status_code=401,
                content="Unauthorized",
                headers={"WWW-Authenticate": 'Basic realm="cctv"'},
            )

        return await call_next(request)


# Request schemas
class IdentityCreate(BaseModel):
    name: str


class ClusterAssign(BaseModel):
    identity_id: Optional[int] = None
    name: Optional[str] = None


class FaceAssign(BaseModel):
    identity_id: Optional[int] = None



class SlotCreate(BaseModel):
    camera: str
    name: str
    slot_box: List[float]  # [x1, y1, x2, y2] normalized
    is_friendly: bool = True
    color_name: Optional[str] = None
    vehicle_id: Optional[int] = None       # link to an existing global vehicle
    vehicle_name: Optional[str] = None     # or create/reuse by name


class SlotUpdate(BaseModel):
    name: Optional[str] = None
    is_friendly: Optional[bool] = None
    vehicle_id: Optional[int] = None  # set to link; explicitly null to unlink


class VehicleCreate(BaseModel):
    name: str
    color_name: Optional[str] = ""

def create_app(output_dir: str, clips_dir: str, config_path: str) -> FastAPI:
    """Create and configure FastAPI application for CCTV dashboard."""
    output_dir = os.path.abspath(output_dir)
    clips_dir = os.path.abspath(clips_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    app = FastAPI(title="CCTV Analytics Dashboard")

    # Optional Basic Auth
    auth_user = os.environ.get("DASHBOARD_USER", "").strip()
    auth_pass = os.environ.get("DASHBOARD_PASSWORD", "").strip()
    if auth_user and auth_pass:
        app.add_middleware(BasicAuthMiddleware, username=auth_user, password=auth_pass)

    db_path = os.path.join(output_dir, "live/events.db")
    store = EventStore(db_path)
    scenery = SceneryManager(store)

    # Lazily initialize FaceEngine for photo uploads
    face_engine_lock = threading.Lock()
    face_engine: Optional[FaceEngine] = None

    def get_face_engine() -> FaceEngine:
        nonlocal face_engine
        with face_engine_lock:
            if face_engine is None:
                model_dir = os.environ.get("FACE_MODEL_DIR", "/opt/models")
                face_engine = FaceEngine(model_dir=model_dir, min_face_px=20, min_det_score=0.60)
            return face_engine

    # Cached analysis results
    archive_cache_lock = threading.Lock()
    archive_cache_mtime = 0.0
    archive_cache_data: Optional[Dict[str, Any]] = None
    archive_cache_index: List[Dict[str, Any]] = []

    def get_archive() -> tuple:
        """Return (analysis results, derived query index), reloading both when the file changes."""
        nonlocal archive_cache_mtime, archive_cache_data, archive_cache_index
        res_file = os.path.join(output_dir, "analysis_results.json")
        if not os.path.exists(res_file):
            raise HTTPException(status_code=404, detail="no analysis results")

        with archive_cache_lock:
            cur_mtime = os.path.getmtime(res_file)
            if archive_cache_data is None or cur_mtime != archive_cache_mtime:
                with open(res_file, "r", encoding="utf-8") as f:
                    archive_cache_data = json.load(f)
                archive_cache_index = build_index(archive_cache_data.get("results", []))
                archive_cache_mtime = cur_mtime
            return archive_cache_data, archive_cache_index

    def get_archive_data() -> Dict[str, Any]:
        return get_archive()[0]

    # Static mounts
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)

    app.mount("/media/clips", StaticFiles(directory=clips_dir), name="media_clips")
    app.mount("/media/output", StaticFiles(directory=output_dir), name="media_output")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"status": "ok", "message": "CCTV Web API active"}

    # --- Health ---

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    # --- Archive API ---

    @app.get("/api/archive/summary")
    async def archive_summary():
        data, index = get_archive()
        summary = {k: v for k, v in data.items() if k != "results"}
        summary.update(build_facets(data.get("results", []), index))
        return summary

    @app.get("/api/archive")
    async def archive_list(
        verdict: Optional[str] = None,
        camera: Optional[str] = None,
        reason: Optional[str] = None,
        class_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort: str = "date_desc",
        offset: int = 0,
        limit: int = 50,
    ):
        data, index = get_archive()
        results = data.get("results", [])
        limit = min(200, max(1, limit))

        indices = filter_indices(
            results,
            index,
            verdict=verdict,
            camera=camera,
            reason=reason,
            classes=parse_class_filter(class_name),
            date_from=parse_date_bound(date_from, is_end=False),
            date_to=parse_date_bound(date_to, is_end=True),
        )
        indices = sort_indices(results, index, indices, sort)
        total = len(indices)

        items = []
        for idx in indices[offset : offset + limit]:
            r = results[idx]
            item = {k: v for k, v in r.items() if k != "detected_tracks"}
            item["idx"] = idx
            item.update(describe(index, idx))

            # Compute URLs
            clip_path = r.get("clip_path")
            thumb_path = r.get("thumbnail_path")

            clip_url = None
            if clip_path:
                abs_clip = os.path.abspath(clip_path)
                if abs_clip.startswith(clips_dir):
                    rel = os.path.relpath(abs_clip, clips_dir)
                    clip_url = f"/media/clips/{rel}"
            item["clip_url"] = clip_url

            thumb_url = None
            if thumb_path:
                abs_thumb = os.path.abspath(thumb_path)
                if abs_thumb.startswith(clips_dir):
                    thumb_url = f"/media/clips/{os.path.relpath(abs_thumb, clips_dir)}"
                elif abs_thumb.startswith(output_dir):
                    thumb_url = f"/media/output/{os.path.relpath(abs_thumb, output_dir)}"
            item["thumb_url"] = thumb_url

            items.append(item)

        return {"total": total, "items": items}

    @app.get("/api/archive/{idx}")
    async def archive_item(idx: int):
        data, index = get_archive()
        results = data.get("results", [])
        if idx < 0 or idx >= len(results):
            raise HTTPException(status_code=404, detail="Archive item not found")
        item = dict(results[idx])
        item["idx"] = idx
        item.update(describe(index, idx))

        clip_path = item.get("clip_path")
        if clip_path:
            abs_clip = os.path.abspath(clip_path)
            if abs_clip.startswith(clips_dir):
                item["clip_url"] = f"/media/clips/{os.path.relpath(abs_clip, clips_dir)}"
            else:
                item["clip_url"] = None

        thumb_path = item.get("thumbnail_path")
        if thumb_path:
            abs_thumb = os.path.abspath(thumb_path)
            if abs_thumb.startswith(clips_dir):
                item["thumb_url"] = f"/media/clips/{os.path.relpath(abs_thumb, clips_dir)}"
            elif abs_thumb.startswith(output_dir):
                item["thumb_url"] = f"/media/output/{os.path.relpath(abs_thumb, output_dir)}"
            else:
                item["thumb_url"] = None

        return item

    # --- Live Status and Previews ---

    @app.get("/api/live/status")
    async def live_status():
        status_file = os.path.join(output_dir, "live/status.json")
        if os.path.exists(status_file):
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"cameras": {}}

    @app.get("/api/live/{slug}/snapshot.jpg")
    async def live_snapshot(slug: str):
        path = os.path.join(output_dir, f"live/preview/{slug}.jpg")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Snapshot not available")
        return FileResponse(path, media_type="image/jpeg")

    # Cached live config for zone geometry (reloads when the YAML changes)
    live_cfg_cache: Dict[str, Any] = {"mtime": 0.0, "cfg": None}

    def get_live_cfg():
        try:
            mtime = os.path.getmtime(config_path)
            if live_cfg_cache["cfg"] is None or mtime != live_cfg_cache["mtime"]:
                live_cfg_cache["cfg"] = load_live_config(config_path)
                live_cfg_cache["mtime"] = mtime
            return live_cfg_cache["cfg"]
        except Exception:
            return None

    @app.get("/api/live/{slug}/detections")
    async def live_camera_detections(slug: str):
        """Boxes + velocity for one camera (fast path), enriched with zones and slots."""
        cam_name = slug
        detections = []
        updated_at = None
        source = None
        open_event_id = None
        mode = None

        # Fast path: per-camera detections file written by the worker at ~4 Hz
        det_file = os.path.join(output_dir, f"live/detections/{slug}.json")
        if os.path.exists(det_file):
            try:
                with open(det_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Ignore files left over from a previous run
                if data.get("updated_at") and (time.time() - float(data["updated_at"])) < 5.0:
                    cam_name = data.get("camera", slug)
                    detections = data.get("detections", [])
                    updated_at = data.get("updated_at")
                    source = data.get("source")
            except Exception:
                pass

        # Status adds event/mode + is the fallback box source
        status_file = os.path.join(output_dir, "live/status.json")
        if os.path.exists(status_file):
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    status = json.load(f)
                for name, info in status.get("cameras", {}).items():
                    if info.get("slug") == slug:
                        cam_name = name
                        open_event_id = info.get("open_event_id")
                        mode = info.get("mode", "idle")
                        if not detections:
                            detections = info.get("detections", [])
                            updated_at = status.get("updated_at")
                        break
            except Exception:
                pass

        zones = []
        cfg = get_live_cfg()
        if cfg is not None:
            for c in cfg.cameras:
                if c.name == cam_name:
                    zones = [
                        {
                            "name": z.name,
                            "type": z.type,
                            "polygon": [[float(x), float(y)] for (x, y) in z.polygon],
                        }
                        for z in c.zones
                    ]
                    break

        slots = []
        for s in store.list_vehicle_slots(camera=cam_name):
            try:
                slots.append({
                    "name": s["name"],
                    "slot_box": json.loads(s["slot_box"]),
                    "is_friendly": bool(s["is_friendly"]),
                    "vehicle_id": s.get("vehicle_id"),
                })
            except Exception:
                continue

        return {
            "camera": cam_name,
            "slug": slug,
            "detections": detections,
            "zones": zones,
            "slots": slots,
            "updated_at": updated_at,
            "source": source,
            "open_event_id": open_event_id,
            "mode": mode or "idle",
        }

    @app.get("/api/live/{slug}/stream.mjpg")
    async def live_stream_mjpg(slug: str):
        preview_path = os.path.join(output_dir, f"live/preview/{slug}.jpg")

        async def stream_generator():
            last_mtime = 0.0
            while True:
                try:
                    if os.path.exists(preview_path):
                        mtime = os.path.getmtime(preview_path)
                        if mtime != last_mtime:
                            last_mtime = mtime
                            with open(preview_path, "rb") as f:
                                data = f.read()
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                            )
                    await asyncio.sleep(0.12)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(0.4)

        return StreamingResponse(
            stream_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # --- Events API ---

    @app.get("/api/events")
    async def events_list(
        camera: Optional[str] = None,
        behavior: Optional[str] = None,
        identity_id: Optional[int] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        offset: int = 0,
        limit: int = 50,
    ):
        limit = min(200, max(1, limit))
        total, raw_items = store.list_events(
            camera=camera,
            behavior=behavior,
            identity_id=identity_id,
            since=since,
            until=until,
            offset=offset,
            limit=limit,
        )

        items = []
        for r in raw_items:
            item = dict(r)
            try:
                item["behaviors"] = json.loads(item["behaviors"])
            except Exception:
                item["behaviors"] = []
            try:
                item["track_summaries"] = json.loads(item["track_summaries"])
            except Exception:
                item["track_summaries"] = []

            item["clip_url"] = f"/media/output/{item['clip_path']}" if item.get("clip_path") else None
            item["thumb_url"] = f"/media/output/{item['thumb_path']}" if item.get("thumb_path") else None

            faces = store.faces_for_event(item["id"])
            item["identities"] = sorted(
                list({f["identity_name"] for f in faces if f.get("identity_name")})
            )
            items.append(item)

        return {"total": total, "items": items}

    @app.get("/api/events/{event_id}")
    async def event_detail(event_id: int):
        ev = store.get_event(event_id)
        if not ev:
            raise HTTPException(status_code=404, detail="Event not found")

        item = dict(ev)
        try:
            item["behaviors"] = json.loads(item["behaviors"])
        except Exception:
            item["behaviors"] = []
        try:
            item["track_summaries"] = json.loads(item["track_summaries"])
        except Exception:
            item["track_summaries"] = []

        item["clip_url"] = f"/media/output/{item['clip_path']}" if item.get("clip_path") else None
        item["thumb_url"] = f"/media/output/{item['thumb_path']}" if item.get("thumb_path") else None

        raw_faces = store.faces_for_event(event_id)
        faces = []
        for f in raw_faces:
            faces.append({
                "id": f["id"],
                "crop_url": f"/media/output/{f['crop_path']}" if f.get("crop_path") else None,
                "context_url": f"/media/output/{f['context_path']}" if f.get("context_path") else None,
                "identity_name": f.get("identity_name"),
                "cluster_id": f.get("cluster_id"),
                "quality": f.get("quality"),
                "face_width": f.get("face_width"),
            })
        item["faces"] = faces
        return item

    # --- Real-time SSE Stream ---

    @app.get("/api/stream")
    async def sse_stream():
        async def event_generator():
            event_cursor = time.time() - 5.0
            initial_notifs = await asyncio.to_thread(store.notifications_since, 0, "dashboard")
            notif_cursor = max((n["id"] for n in initial_notifs), default=0)
            last_ping = time.time()

            while True:
                try:
                    # 1. Events updated since cursor
                    updated = await asyncio.to_thread(store.events_updated_since, event_cursor)
                    if updated:
                        for row in updated:
                            event_cursor = max(event_cursor, row["updated_at"])
                            ev_dict = dict(row)
                            try:
                                ev_dict["behaviors"] = json.loads(ev_dict["behaviors"])
                            except Exception:
                                pass
                            ev_dict["clip_url"] = (
                                f"/media/output/{ev_dict['clip_path']}" if ev_dict.get("clip_path") else None
                            )
                            ev_dict["thumb_url"] = (
                                f"/media/output/{ev_dict['thumb_path']}" if ev_dict.get("thumb_path") else None
                            )
                            yield f"event: event_update\ndata: {json.dumps(ev_dict)}\n\n"

                    # 2. Dashboard notifications
                    notifs = await asyncio.to_thread(store.notifications_since, notif_cursor, "dashboard")
                    if notifs:
                        for n in notifs:
                            notif_cursor = max(notif_cursor, n["id"])
                            yield f"event: notification\ndata: {json.dumps(dict(n))}\n\n"

                    # 3. Ping keepalive
                    now = time.time()
                    if now - last_ping >= 15.0:
                        last_ping = now
                        yield ": ping\n\n"

                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.debug(f"SSE error: {exc}")
                    await asyncio.sleep(1.0)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # --- Identities & Faces API ---

    @app.get("/api/identities")
    async def identities_list():
        identities = store.list_identities()
        result = []
        with store.lock:
            for ident in identities:
                cur = store.conn.execute(
                    "SELECT context_path FROM faces WHERE identity_id = ? AND context_path != '' LIMIT 6",
                    (ident["id"],),
                )
                sample_urls = [f"/media/output/{r['context_path']}" for r in cur.fetchall()]
                result.append({
                    "id": ident["id"],
                    "name": ident["name"],
                    "face_count": ident["face_count"],
                    "sample_urls": sample_urls,
                })
        return result

    @app.post("/api/identities", status_code=201)
    async def identity_create(payload: IdentityCreate):
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Name cannot be empty")
        try:
            iid = store.create_identity(name)
            return {"id": iid, "name": name}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="identity exists")

    @app.patch("/api/identities/{identity_id}")
    async def identity_rename(identity_id: int, payload: IdentityCreate):
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Name cannot be empty")
        # Check existence
        existing = [i for i in store.list_identities() if i["id"] == identity_id]
        if not existing:
            raise HTTPException(status_code=404, detail="identity not found")
        try:
            store.rename_identity(identity_id, name)
            return {"id": identity_id, "name": name}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="identity exists")

    @app.delete("/api/identities/{identity_id}")
    async def identity_delete(identity_id: int):
        try:
            store.delete_identity(identity_id)
            return {"ok": True}
        except LookupError:
            raise HTTPException(status_code=404, detail="identity not found")

    @app.get("/api/faces/clusters")
    async def faces_clusters():
        clusters = store.list_clusters()
        result = []
        with store.lock:
            for c in clusters:
                cid = c["cluster_id"]
                cur = store.conn.execute(
                    "SELECT context_path FROM faces WHERE cluster_id = ? AND context_path != '' LIMIT 6",
                    (cid,),
                )
                sample_urls = [f"/media/output/{r['context_path']}" for r in cur.fetchall()]
                result.append({
                    "cluster_id": cid,
                    "count": c["count"],
                    "last_seen": c["last_seen"],
                    "sample_urls": sample_urls,
                })
        return result

    @app.post("/api/faces/clusters/{cluster_id}/assign")
    async def cluster_assign(cluster_id: int, payload: ClusterAssign):
        target_id = payload.identity_id
        if not target_id and not payload.name:
            raise HTTPException(status_code=422, detail="identity_id or name required")

        if payload.name:
            name = payload.name.strip()
            existing = [i for i in store.list_identities() if i["name"].lower() == name.lower()]
            if existing:
                target_id = existing[0]["id"]
            else:
                target_id = store.create_identity(name)

        store.assign_cluster(cluster_id, target_id)
        return {"ok": True, "identity_id": target_id}

    @app.post("/api/faces/{face_id}/assign")
    async def face_assign(face_id: int, payload: FaceAssign):
        store.assign_face(face_id, payload.identity_id)
        return {"ok": True}

    @app.delete("/api/faces/{face_id}")
    async def face_delete(face_id: int):
        row = store.delete_face(face_id)
        if not row:
            raise HTTPException(status_code=404, detail="Face not found")

        # Delete crop files
        for key in ("crop_path", "context_path"):
            rel = row.get(key)
            if rel:
                full_p = os.path.join(output_dir, rel)
                if os.path.exists(full_p):
                    try:
                        os.remove(full_p)
                    except OSError:
                        pass
        return {"ok": True}

    class ReindexRequest(BaseModel):
        event_id: Optional[int] = None
        camera: Optional[str] = None
        limit: int = 50

    @app.post("/api/faces/reindex")
    async def faces_reindex(payload: Optional[ReindexRequest] = None):
        from engine.reindexer import FaceReindexer
        req = payload or ReindexRequest()
        reindexer = FaceReindexer(store=store, output_dir=output_dir, face_engine=face_engine, gallery=gallery)
        if req.event_id:
            event_row = store.get_event(req.event_id)
            if not event_row or not event_row.get("clip_path"):
                raise HTTPException(status_code=404, detail="Event or clip not found")
            full_clip = os.path.join(output_dir, event_row["clip_path"])
            return reindexer.reindex_clip(full_clip, event_id=req.event_id, camera=event_row.get("camera", "Unknown"))
        return reindexer.reindex_events_directory(camera=req.camera, limit=req.limit)

    @app.post("/api/identities/{identity_id}/photos", status_code=201)
    async def identity_upload_photo(identity_id: int, file: UploadFile = File(...)):
        # Verify identity exists
        existing = [i for i in store.list_identities() if i["id"] == identity_id]
        if not existing:
            raise HTTPException(status_code=404, detail="identity not found")

        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=422, detail="invalid image file")

        engine = get_face_engine()
        with face_engine_lock:
            res = engine.embed_image(img)

        if res is None:
            raise HTTPException(status_code=422, detail="no face detected")

        aligned, feat, cand = res
        now = time.time()
        date_str = datetime.utcfromtimestamp(now).strftime("%Y%m%d")

        fw = cand.box[2] - cand.box[0]
        face_id = store.insert_face(
            event_id=None,
            camera="upload",
            track_id=None,
            captured_at=now,
            crop_path="",
            context_path="",
            embedding=feat.tobytes(),
            quality=cand.quality,
            det_score=cand.score,
            face_width=fw,
            identity_id=identity_id,
        )

        crop_rel = f"faces/{date_str}/{face_id}.jpg"
        ctx_rel = f"faces/{date_str}/{face_id}_ctx.jpg"
        store.update_face(face_id, crop_path=crop_rel, context_path=ctx_rel)

        full_crop = os.path.join(output_dir, crop_rel)
        full_ctx = os.path.join(output_dir, ctx_rel)
        os.makedirs(os.path.dirname(full_crop), exist_ok=True)

        cv2.imwrite(full_crop, aligned)
        cv2.imwrite(full_ctx, aligned)

        return {
            "id": face_id,
            "crop_url": f"/media/output/{crop_rel}",
            "context_url": f"/media/output/{ctx_rel}",
        }


    # --- Scenery & Vehicle Slots API ---

    @app.get("/api/scenery/slots")
    async def scenery_slots_list(camera: Optional[str] = None):
        raw = store.list_vehicle_slots(camera=camera)
        res = []
        for r in raw:
            item = dict(r)
            try:
                item["slot_box"] = json.loads(item["slot_box"])
            except Exception:
                item["slot_box"] = []
            try:
                item["appearance_sig"] = json.loads(item["appearance_sig"])
            except Exception:
                item["appearance_sig"] = {}
            res.append(item)
        return res

    @app.post("/api/scenery/slots", status_code=201)
    async def scenery_slot_create(payload: SlotCreate):
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Slot name cannot be empty")
        if len(payload.slot_box) != 4:
            raise HTTPException(status_code=422, detail="slot_box must be [x1, y1, x2, y2]")

        from engine.live.config import slugify
        slug = slugify(payload.camera)
        preview_p = os.path.join(output_dir, f"live/preview/{slug}.jpg")

        color_name = payload.color_name or "unknown"
        sig_data = {"aspect_ratio": 1.5, "hsv_bins": []}

        if os.path.exists(preview_p):
            frame = cv2.imread(preview_p)
            if frame is not None:
                fh, fw = frame.shape[:2]
                box_px = (
                    payload.slot_box[0] * fw,
                    payload.slot_box[1] * fh,
                    payload.slot_box[2] * fw,
                    payload.slot_box[3] * fh,
                )
                sig = extract_vehicle_signature(frame, box_px)
                color_name = sig.color_name
                sig_data = {
                    "aspect_ratio": sig.aspect_ratio,
                    "hsv_bins": sig.hsv_bins,
                }

        # Cross-camera identity: link to an existing vehicle (by id or exact name)
        # or create one — creating a slot never duplicates a known vehicle name.
        vehicle_name = (payload.vehicle_name or name).strip()
        if payload.vehicle_id is not None:
            if store.get_vehicle(payload.vehicle_id) is None:
                raise HTTPException(status_code=404, detail="vehicle not found")
            vehicle_id = payload.vehicle_id
        else:
            vehicle_id = store.get_or_create_vehicle(vehicle_name, color_name)

        slot_id = store.create_vehicle_slot(
            camera=payload.camera,
            name=vehicle_name,
            slot_box=json.dumps(payload.slot_box),
            color_name=color_name,
            appearance_sig=json.dumps(sig_data),
            is_friendly=1 if payload.is_friendly else 0,
        )
        store.link_slot_vehicle(slot_id, vehicle_id)
        scenery.reload()
        return {
            "id": slot_id,
            "camera": payload.camera,
            "name": vehicle_name,
            "vehicle_id": vehicle_id,
            "slot_box": payload.slot_box,
            "color_name": color_name,
            "appearance_sig": sig_data,
            "is_friendly": payload.is_friendly,
        }

    @app.patch("/api/scenery/slots/{slot_id}")
    async def scenery_slot_update(slot_id: int, payload: SlotUpdate):
        slot = store.get_vehicle_slot(slot_id)
        if not slot:
            raise HTTPException(status_code=404, detail="Slot not found")

        fields = {}
        if payload.name is not None and "vehicle_id" not in payload.model_fields_set:
            fields["name"] = payload.name.strip()
        if payload.is_friendly is not None:
            fields["is_friendly"] = 1 if payload.is_friendly else 0
        if fields:
            store.update_vehicle_slot(slot_id, **fields)

        # Explicit vehicle_id (int = link, null = unlink) for cross-camera merging
        if "vehicle_id" in payload.model_fields_set:
            if payload.vehicle_id is None:
                store.link_slot_vehicle(slot_id, None)
            else:
                if not store.link_slot_vehicle(slot_id, payload.vehicle_id):
                    raise HTTPException(status_code=404, detail="vehicle not found")

        scenery.reload()
        updated = store.get_vehicle_slot(slot_id)
        res = dict(updated)
        try:
            res["slot_box"] = json.loads(res["slot_box"])
            res["appearance_sig"] = json.loads(res["appearance_sig"])
        except Exception:
            pass
        return res

    # --- Global vehicles (cross-camera identity) ---

    @app.get("/api/vehicles")
    async def vehicles_list():
        return store.list_vehicles()

    @app.post("/api/vehicles", status_code=201)
    async def vehicle_create(payload: VehicleCreate):
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Name cannot be empty")
        try:
            vid = store.create_vehicle(name, payload.color_name or "")
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="vehicle exists")
        return {"id": vid, "name": name}

    @app.patch("/api/vehicles/{vehicle_id}")
    async def vehicle_rename(vehicle_id: int, payload: VehicleCreate):
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Name cannot be empty")
        if store.get_vehicle(vehicle_id) is None:
            raise HTTPException(status_code=404, detail="vehicle not found")
        try:
            store.rename_vehicle(vehicle_id, name)
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="vehicle exists")
        scenery.reload()
        return {"id": vehicle_id, "name": name}

    @app.delete("/api/vehicles/{vehicle_id}")
    async def vehicle_delete(vehicle_id: int):
        if not store.delete_vehicle(vehicle_id):
            raise HTTPException(status_code=404, detail="vehicle not found")
        scenery.reload()
        return {"ok": True}

    @app.delete("/api/scenery/slots/{slot_id}")
    async def scenery_slot_delete(slot_id: int):
        ok = store.delete_vehicle_slot(slot_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Slot not found")
        scenery.reload()
        return {"ok": True}
    return app
