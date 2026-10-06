"""Integration test verifying incremental caching and auto-storage of KEEP clips."""

import os
import shutil
import tempfile
import unittest

from engine.cache import AnalysisCache
from engine.pipeline import AnalysisPipeline


class TestIncrementalCache(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_cache_pipeline_")
        self.input_dir = os.path.join(self.test_dir, "input_clips")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_incremental_workflow(self):
        # 1. Setup sample clips in folderA
        folder_a = os.path.join(self.input_dir, "20260103")
        os.makedirs(folder_a, exist_ok=True)

        base_dir = "/data/clips" if os.path.exists("/data/clips") else "clips"
        clip1_src = os.path.join(
            base_dir,
            "20260103",
            "00000000_1_DVR000000_Camera A_00000000000000000000000000000001_20260103170429.mp4",
        )
        thumb1_src = clip1_src.replace(".mp4", ".png")

        clip1_dest = os.path.join(folder_a, os.path.basename(clip1_src))
        thumb1_dest = os.path.join(folder_a, os.path.basename(thumb1_src))
        shutil.copy2(clip1_src, clip1_dest)
        if os.path.exists(thumb1_src):
            shutil.copy2(thumb1_src, thumb1_dest)

        cache_path = os.path.join(self.output_dir, "analysis_cache.json")
        cache = AnalysisCache(cache_path)
        pipeline = AnalysisPipeline(gpus=[0], vid_stride=15, conf_threshold=0.30)

        # First run: should analyze clip1
        results1, stats1 = pipeline.run_analysis(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            cache=cache,
            store_keep=True,
        )
        self.assertEqual(stats1["total_discovered"], 1)
        self.assertEqual(stats1["cached_count"], 0)
        self.assertEqual(stats1["new_processed_count"], 1)
        self.assertEqual(results1[0]["verdict"], "KEEP")

        # Verify KEEP clip was automatically stored in output/clips/20260103/
        stored_clip = os.path.join(self.output_dir, "clips", "20260103", os.path.basename(clip1_src))
        self.assertTrue(os.path.exists(stored_clip), f"Expected {stored_clip} to exist")

        # Second run: should be 100% CACHED (0 clips to analyze)
        cache2 = AnalysisCache(cache_path)
        results2, stats2 = pipeline.run_analysis(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            cache=cache2,
            store_keep=True,
        )
        self.assertEqual(stats2["total_discovered"], 1)
        self.assertEqual(stats2["cached_count"], 1)
        self.assertEqual(stats2["new_processed_count"], 0)
        self.assertEqual(results2[0]["verdict"], "KEEP")

        # Third run: add a new folder (folderB) with another clip
        folder_b = os.path.join(self.input_dir, "20260524")
        os.makedirs(folder_b, exist_ok=True)
        clip2_src = os.path.join(
            base_dir,
            "20260103",
            "00000000_1_DVR000000_Camera C_00000000000000000000000000000003_20260103141555.mp4",
        )
        clip2_dest = os.path.join(folder_b, os.path.basename(clip2_src))
        shutil.copy2(clip2_src, clip2_dest)

        results3, stats3 = pipeline.run_analysis(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            cache=cache2,
            store_keep=True,
        )
        self.assertEqual(stats3["total_discovered"], 2)
        # clip1 from 20260103 must be skipped (cached)
        self.assertEqual(stats3["cached_count"], 1)
        # clip2 from 20260524 must be the ONLY clip analyzed
        self.assertEqual(stats3["new_processed_count"], 1)


if __name__ == "__main__":
    unittest.main()
