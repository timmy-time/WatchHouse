import os
import shutil
from typing import Any, Dict, List, Optional


def get_relative_folder(clip_path: str, input_dir: str, fallback: str = "default") -> str:
    """
    Extracts the relative folder of clip_path with respect to input_dir.
    If the relative directory is empty or '.', falls back to fallback.
    """
    try:
        rel_dir = os.path.dirname(os.path.relpath(clip_path, start=input_dir))
    except (ValueError, Exception):
        rel_dir = ""

    if not rel_dir or rel_dir == ".":
        return fallback or "default"
    return rel_dir


def store_keep_clip(
    record: Dict[str, Any],
    input_dir: str,
    output_dir: str,
    action: str = "copy",
) -> Dict[str, Optional[str]]:
    """
    Stores a KEEP clip and its optional thumbnail into output_dir/clips/<rel_folder>/...
    action can be 'copy' or 'move'.
    Updates record in-place with 'stored_clip_path' and 'stored_thumb_path'.
    """
    clip_path = record["clip_path"]
    thumb_path = record.get("thumbnail_path")

    # Resolve clip_path if path was recorded from a different container mount
    if not os.path.exists(clip_path):
        rel_hint = record.get("clip_rel") or os.path.basename(clip_path)
        candidate = os.path.join(input_dir, rel_hint)
        if os.path.exists(candidate):
            clip_path = candidate
        else:
            for root, _, files in os.walk(input_dir):
                if os.path.basename(clip_path) in files:
                    clip_path = os.path.join(root, os.path.basename(clip_path))
                    break

    if thumb_path and not os.path.exists(thumb_path):
        rel_thumb = record.get("thumbnail_rel") or os.path.basename(thumb_path)
        candidate_thumb = os.path.join(input_dir, rel_thumb)
        if os.path.exists(candidate_thumb):
            thumb_path = candidate_thumb
        else:
            for root, _, files in os.walk(input_dir):
                if os.path.basename(thumb_path) in files:
                    thumb_path = os.path.join(root, os.path.basename(thumb_path))
                    break
    camera_fallback = record.get("camera") or record.get("camera_name") or "default"
    rel_folder = get_relative_folder(clip_path, input_dir, fallback=camera_fallback)

    dest_dir = os.path.join(output_dir, "clips", rel_folder)
    os.makedirs(dest_dir, exist_ok=True)

    dest_clip = os.path.join(dest_dir, os.path.basename(clip_path))

    if action == "move":
        shutil.move(clip_path, dest_clip)
    else:  # copy
        if os.path.exists(clip_path) and (
            not os.path.exists(dest_clip) or os.path.getsize(dest_clip) != os.path.getsize(clip_path)
        ):
            shutil.copy2(clip_path, dest_clip)
    dest_thumb: Optional[str] = None
    if thumb_path and os.path.exists(thumb_path):
        dest_thumb = os.path.join(dest_dir, os.path.basename(thumb_path))
        if action == "move":
            shutil.move(thumb_path, dest_thumb)
        else:
            if not os.path.exists(dest_thumb) or os.path.getsize(dest_thumb) != os.path.getsize(thumb_path):
                shutil.copy2(thumb_path, dest_thumb)

    record["stored_clip_path"] = dest_clip
    record["stored_thumb_path"] = dest_thumb

    return {"stored_clip_path": dest_clip, "stored_thumb_path": dest_thumb}


def batch_store_keeps(
    records: List[Dict[str, Any]],
    input_dir: str,
    output_dir: str,
    action: str = "copy",
) -> int:
    """
    Iterates through records. If record['verdict'] == 'KEEP': calls store_keep_clip.
    Returns the count of stored KEEP clips.
    """
    count = 0
    for r in records:
        if r.get("verdict") == "KEEP":
            store_keep_clip(r, input_dir=input_dir, output_dir=output_dir, action=action)
            count += 1
    return count


class KeepStorage:
    """Convenience wrapper class for keep clip storage operations."""

    def __init__(self, input_dir: str, output_dir: str, action: str = "copy"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.action = action

    def get_relative_folder(self, clip_path: str, fallback: str = "default") -> str:
        return get_relative_folder(clip_path, self.input_dir, fallback=fallback)

    def store_clip(self, record: Dict[str, Any], action: Optional[str] = None) -> Dict[str, Optional[str]]:
        return store_keep_clip(
            record,
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            action=action or self.action,
        )

    def batch_store(self, records: List[Dict[str, Any]], action: Optional[str] = None) -> int:
        return batch_store_keeps(
            records,
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            action=action or self.action,
        )
