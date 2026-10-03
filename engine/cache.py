import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AnalysisCache:
    """Persistent analysis cache storing clip verdicts and metadata keyed by relative path.

    Validates file size and modification time to invalidate stale entries.
    """

    def __init__(self, cache_file_path: str):
        self.cache_file_path = os.path.abspath(cache_file_path)
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.entries = data
                else:
                    logger.warning("Cache file %s does not contain a dictionary, resetting", self.cache_file_path)
                    self.entries = {}
            except Exception as e:
                logger.warning("Failed to load cache from %s: %s", self.cache_file_path, e)
                self.entries = {}
        else:
            self.entries = {}

    def get(
        self,
        clip_path: str,
        input_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis record if file matches stored size, mtime, and config fingerprint."""
        try:
            rel_key = os.path.relpath(clip_path, start=input_dir)
        except ValueError:
            rel_key = os.path.abspath(clip_path)

        entry = self.entries.get(rel_key)
        if not entry or not isinstance(entry, dict):
            return None

        if not os.path.exists(clip_path):
            return None

        try:
            current_size = os.path.getsize(clip_path)
            current_mtime = os.path.getmtime(clip_path)
        except OSError:
            return None

        cached_size = entry.get("file_size")
        cached_mtime = entry.get("mtime")

        if cached_size != current_size:
            return None

        # Allow slight float precision variance if any, but mtime equality or close match
        if cached_mtime is None or abs(cached_mtime - current_mtime) > 1e-4:
            return None

        if config is not None:
            cached_config = entry.get("config")
            if cached_config != config:
                return None

        return entry.get("record")

    def put(
        self,
        clip_path: str,
        input_dir: str,
        record: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store analysis record with current file metadata and config fingerprint."""
        try:
            rel_key = os.path.relpath(clip_path, start=input_dir)
        except ValueError:
            rel_key = os.path.abspath(clip_path)

        try:
            size = os.path.getsize(clip_path)
            mtime = os.path.getmtime(clip_path)
        except OSError:
            size = 0
            mtime = 0.0

        self.entries[rel_key] = {
            "file_size": size,
            "mtime": mtime,
            "config": config,
            "record": record,
        }

    def save(self) -> None:
        """Atomically save cache entries to disk."""
        cache_dir = os.path.dirname(self.cache_file_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_dir,
                delete=False,
                prefix="cache_",
                suffix=".tmp",
            ) as f:
                temp_file = f.name
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_file, self.cache_file_path)
        except Exception as e:
            logger.error("Failed to save cache to %s: %s", self.cache_file_path, e)
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            raise

    def filter_uncached(
        self,
        clip_pairs: List[Tuple[str, Optional[str]]],
        input_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Tuple[str, Optional[str]]], List[Dict[str, Any]]]:
        """Partition clip pairs into uncached pairs needing analysis and cached records."""
        uncached_pairs: List[Tuple[str, Optional[str]]] = []
        cached_records: List[Dict[str, Any]] = []

        for pair in clip_pairs:
            clip_path = pair[0]
            record = self.get(clip_path, input_dir, config=config)
            if record is not None:
                cached_records.append(record)
            else:
                uncached_pairs.append(pair)

        return uncached_pairs, cached_records
