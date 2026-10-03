import os
import shutil
import tempfile
import time
import unittest

from engine.cache import AnalysisCache


class TestAnalysisCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "input")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache_file = os.path.join(self.output_dir, "analysis_cache.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_clip(self, subpath: str, content: bytes = b"dummy video content") -> str:
        full_path = os.path.join(self.input_dir, subpath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(content)
        return full_path

    def test_put_and_get_hit(self):
        clip = self._create_clip("sub/clip1.mp4", b"data123")
        cache = AnalysisCache(self.cache_file)

        record = {"verdict": "KEEP", "reason": "person_detected", "confidence": 0.88}
        cache.put(clip, self.input_dir, record)

        hit = cache.get(clip, self.input_dir)
        self.assertEqual(hit, record)

        # Save and reload from disk
        cache.save()
        self.assertTrue(os.path.exists(self.cache_file))

        loaded_cache = AnalysisCache(self.cache_file)
        hit_loaded = loaded_cache.get(clip, self.input_dir)
        self.assertEqual(hit_loaded, record)

    def test_miss_on_size_change(self):
        clip = self._create_clip("clip2.mp4", b"initial content")
        cache = AnalysisCache(self.cache_file)

        record = {"verdict": "DISCARD", "reason": "stationary_vehicles_only"}
        cache.put(clip, self.input_dir, record)
        self.assertEqual(cache.get(clip, self.input_dir), record)

        # Modify file size
        with open(clip, "wb") as f:
            f.write(b"modified and longer content")

        self.assertIsNone(cache.get(clip, self.input_dir))

    def test_miss_on_mtime_change(self):
        clip = self._create_clip("clip3.mp4", b"same size 123")
        cache = AnalysisCache(self.cache_file)

        record = {"verdict": "KEEP", "reason": "moving_vehicle"}
        cache.put(clip, self.input_dir, record)
        self.assertEqual(cache.get(clip, self.input_dir), record)

        # Change mtime without changing size
        stat = os.stat(clip)
        new_mtime = stat.st_mtime + 10.0
        os.utime(clip, (new_mtime, new_mtime))

        self.assertIsNone(cache.get(clip, self.input_dir))

    def test_miss_on_missing_file(self):
        clip = self._create_clip("clip_del.mp4", b"content")
        cache = AnalysisCache(self.cache_file)
        record = {"verdict": "KEEP"}
        cache.put(clip, self.input_dir, record)
        self.assertEqual(cache.get(clip, self.input_dir), record)

        os.remove(clip)
        self.assertIsNone(cache.get(clip, self.input_dir))

    def test_miss_on_config_mismatch(self):
        clip = self._create_clip("clip_cfg.mp4", b"cfg data")
        cache = AnalysisCache(self.cache_file)
        record = {"verdict": "KEEP"}
        cfg1 = {"model": "yolov8n.pt", "vid_stride": 15, "conf": 0.3}
        cfg2 = {"model": "yolov8s.pt", "vid_stride": 15, "conf": 0.3}

        cache.put(clip, self.input_dir, record, config=cfg1)
        self.assertEqual(cache.get(clip, self.input_dir, config=cfg1), record)
        self.assertIsNone(cache.get(clip, self.input_dir, config=cfg2))

    def test_legacy_null_config_misses_when_config_given(self):
        clip = self._create_clip("clip_legacy.mp4", b"legacy data")
        cache = AnalysisCache(self.cache_file)
        record = {"verdict": "KEEP"}
        cache.put(clip, self.input_dir, record, config=None)
        self.assertIsNone(cache.get(clip, self.input_dir, config={"model": "yolov8n.pt", "vid_stride": 15, "conf": 0.3}))

    def test_filter_uncached(self):
        clip1 = self._create_clip("20260103/c1.mp4", b"c1")
        thumb1 = self._create_clip("20260103/c1.png", b"t1")
        clip2 = self._create_clip("20260103/c2.mp4", b"c2")
        thumb2 = None
        clip3 = self._create_clip("20260103/c3.mp4", b"c3")
        thumb3 = self._create_clip("20260103/c3.png", b"t3")

        cache = AnalysisCache(self.cache_file)
        rec1 = {"clip_path": clip1, "verdict": "KEEP"}
        rec3 = {"clip_path": clip3, "verdict": "DISCARD"}

        cache.put(clip1, self.input_dir, rec1)
        cache.put(clip3, self.input_dir, rec3)

        pairs = [
            (clip1, thumb1),
            (clip2, thumb2),
            (clip3, thumb3),
        ]

        uncached_pairs, cached_records = cache.filter_uncached(pairs, self.input_dir)

        self.assertEqual(len(uncached_pairs), 1)
        self.assertEqual(uncached_pairs[0], (clip2, thumb2))

        self.assertEqual(len(cached_records), 2)
        self.assertIn(rec1, cached_records)
        self.assertIn(rec3, cached_records)


if __name__ == "__main__":
    unittest.main()
