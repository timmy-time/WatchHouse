"""Query helpers for the archive view: DVR clock, event classes, filtering, sorting, facets.

The date/time of a clip comes from the DVR's own clock, which is embedded in the
filename (``..._<YYYYMMDDHHMMSS>.mp4``) and is the exact value the camera burns
into the on-screen display inside the video. Analysis records carry it as
``timestamp``, so sorting by date/time needs no OCR.
"""

import ast
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

SORT_KEYS = (
    "date_desc",
    "date_asc",
    "camera",
    "camera_desc",
    "class",
    "class_desc",
    "confidence",
    "confidence_desc",
    "duration",
    "duration_desc",
)
DEFAULT_SORT = "date_desc"

VEHICLE_CLASSES = frozenset(
    {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "tractor"}
)


def parse_dvr_timestamp(raw: Any) -> Optional[datetime]:
    """Parse the DVR clock (YYYYMMDDHHMMSS) into a naive datetime."""
    if raw is None:
        return None
    text = str(raw).strip()
    if len(text) < 14 or not text[:14].isdigit():
        return None
    try:
        return datetime.strptime(text[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def parse_tracks(raw: Any) -> List[Dict[str, Any]]:
    """Return the analysed tracks. Records store them as a Python repr, but accept real lists."""
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, dict)]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError):
        logger.debug("unparsable detected_tracks: %.80s", raw)
        return []
    if not isinstance(value, list):
        return []
    return [t for t in value if isinstance(t, dict)]


def track_class_counts(raw: Any) -> Dict[str, int]:
    """Per-class track counts for one record."""
    counts: Dict[str, int] = {}
    for track in parse_tracks(raw):
        name = track.get("class_name")
        if isinstance(name, str) and name.strip():
            key = name.strip().lower()
            counts[key] = counts.get(key, 0) + 1
    return counts


def build_index(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Derive the per-record query keys, positionally aligned with ``results``."""
    index: List[Dict[str, Any]] = []
    for record in results:
        dt = parse_dvr_timestamp(record.get("timestamp"))
        counts = track_class_counts(record.get("detected_tracks"))
        index.append(
            {
                "dt": dt,
                "datetime": dt.isoformat() if dt else None,
                "date": dt.strftime("%Y-%m-%d") if dt else None,
                "time": dt.strftime("%H:%M:%S") if dt else None,
                "class_counts": counts,
                "classes": sorted(counts),
            }
        )
    return index


def describe(index: Sequence[Dict[str, Any]], idx: int) -> Dict[str, Any]:
    """The derived fields added to every archive item."""
    meta = index[idx]
    return {
        "datetime": meta["datetime"],
        "date": meta["date"],
        "time": meta["time"],
        "classes": meta["classes"],
        "class_counts": meta["class_counts"],
    }


def parse_date_bound(raw: Optional[str], is_end: bool) -> Optional[datetime]:
    """Accept ``YYYY-MM-DD``, ``YYYY-MM-DD HH:MM`` or ``YYYY-MM-DD HH:MM:SS`` (``T`` separator too).

    A bare date covers the whole day: start of day for ``is_end=False``, end of day otherwise.
    """
    if raw is None:
        return None
    text = str(raw).strip().replace("T", " ")
    if not text:
        return None
    formats = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d" and is_end:
            return parsed.replace(hour=23, minute=59, second=59)
        return parsed
    logger.warning("ignoring unparsable date bound: %r", raw)
    return None


def parse_class_filter(raw: Optional[Any]) -> List[str]:
    """Normalise ``car,person`` / ``["car"]`` into a lowercase class list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts: Iterable[Any] = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        parts = raw
    else:
        return []
    return [str(p).strip().lower() for p in parts if str(p).strip()]


def filter_indices(
    results: Sequence[Dict[str, Any]],
    index: Sequence[Dict[str, Any]],
    *,
    verdict: Optional[str] = None,
    camera: Optional[str] = None,
    reason: Optional[str] = None,
    classes: Optional[Iterable[str]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[int]:
    """Indices of the records matching every supplied filter."""
    wanted = {c.lower() for c in (classes or [])}
    reason_needle = reason.strip().lower() if reason and reason.strip() else None
    keep: List[int] = []

    for idx, record in enumerate(results):
        if verdict and record.get("verdict") != verdict:
            continue
        if camera and record.get("camera") != camera:
            continue
        if reason_needle:
            haystack = " ".join(
                str(record.get(key) or "") for key in ("primary_reason", "reason")
            ).lower()
            if reason_needle not in haystack:
                continue

        meta = index[idx]
        if wanted and not wanted.intersection(meta["classes"]):
            continue

        if date_from is not None or date_to is not None:
            dt = meta["dt"]
            if dt is None:
                continue
            if date_from is not None and dt < date_from:
                continue
            if date_to is not None and dt > date_to:
                continue

        keep.append(idx)
    return keep


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sort_indices(
    results: Sequence[Dict[str, Any]],
    index: Sequence[Dict[str, Any]],
    indices: Iterable[int],
    sort_key: Optional[str] = None,
) -> List[int]:
    """Sort filtered indices. Unknown datetimes always sink to the end."""
    key = sort_key if sort_key in SORT_KEYS else DEFAULT_SORT
    out = list(indices)

    def timestamp(i: int) -> Optional[float]:
        dt = index[i]["dt"]
        return dt.timestamp() if dt is not None else None

    def confidence(i: int) -> float:
        return _to_float(results[i].get("confidence"))

    def duration(i: int) -> float:
        return _to_float(results[i].get("duration_frames"))

    # Stable base order: newest first, then original position.
    out.sort(key=lambda i: (timestamp(i) is None, -(timestamp(i) or 0.0), i))

    if key == "date_desc":
        return out
    if key == "date_asc":
        out.sort(key=lambda i: (timestamp(i) is None, timestamp(i) or 0.0))
    elif key in ("camera", "camera_desc"):
        out.sort(
            key=lambda i: str(results[i].get("camera") or "").lower(),
            reverse=key == "camera_desc",
        )
    elif key in ("class", "class_desc"):
        out.sort(
            key=lambda i: index[i]["classes"][0] if index[i]["classes"] else "\uffff",
            reverse=key == "class_desc",
        )
    elif key in ("confidence", "confidence_desc"):
        out.sort(key=confidence, reverse=key == "confidence_desc")
    elif key in ("duration", "duration_desc"):
        out.sort(key=duration, reverse=key == "duration_desc")
    return out


def build_facets(
    results: Sequence[Dict[str, Any]], index: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Camera/class counts and the covered date range across every record."""
    cameras: Dict[str, int] = {}
    classes: Dict[str, int] = {}
    stamps: List[datetime] = []

    for idx, record in enumerate(results):
        cam = str(record.get("camera") or "unknown")
        cameras[cam] = cameras.get(cam, 0) + 1
        for name in index[idx]["classes"]:
            classes[name] = classes.get(name, 0) + 1
        dt = index[idx]["dt"]
        if dt is not None:
            stamps.append(dt)

    def ranked(counts: Dict[str, int]) -> Dict[str, int]:
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "cameras": ranked(cameras),
        "classes": ranked(classes),
        "date_min": min(stamps).isoformat() if stamps else None,
        "date_max": max(stamps).isoformat() if stamps else None,
    }
