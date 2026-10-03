"""Unit tests for segment listing, segment selection, and ring buffer pruning."""

from datetime import datetime, timezone
import os
import shutil
import tempfile
import unittest

from engine.live.clipper import list_segments, select_segments, prune_segments


class TestClipper(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_segment(self, name: str) -> str:
        path = os.path.join(self.test_dir, name)
        with open(path, "wb") as f:
            f.write(b"dummy mp4 data")
        return path

    def test_list_and_select_segments(self):
        # Create 6 segments from 12:00:00 to 12:00:50 (10s segments)
        names = [
            "20261003-120000.mp4",
            "20261003-120010.mp4",
            "20261003-120020.mp4",
            "20261003-120030.mp4",
            "20261003-120040.mp4",
            "20261003-120050.mp4",  # Newest (being written)
        ]
        for name in names:
            self._create_segment(name)

        segs = list_segments(self.test_dir)
        self.assertEqual(len(segs), 6)

        # Base epoch for 2026-10-03 12:00:00 UTC
        base_epoch = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc).timestamp()

        # Query range [12:00:15, 12:00:35]
        # Should select:
        # segment 1 (12:00:10 - 12:00:20): spans [10, 20), overlaps 15 -> YES
        # segment 2 (12:00:20 - 12:00:30): spans [20, 30), overlaps [15, 35) -> YES
        # segment 3 (12:00:30 - 12:00:40): spans [30, 40), overlaps 35 -> YES
        # segment 4 (12:00:40 - 12:00:50): spans [40, 50), does not overlap -> NO
        # segment 5 (12:00:50): newest, excluded -> NO
        start = base_epoch + 15
        end = base_epoch + 35
        selected = select_segments(segs, start, end)

        expected_names = [
            "20261003-120010.mp4",
            "20261003-120020.mp4",
            "20261003-120030.mp4",
        ]
        self.assertEqual([os.path.basename(p) for p in selected], expected_names)

        # Test newest exclusion: query asking for [12:00:45, 12:00:55]
        # Should select segment 4 (12:00:40 - 12:00:50), but NOT segment 5 (newest)
        selected_late = select_segments(segs, base_epoch + 45, base_epoch + 55)
        self.assertEqual([os.path.basename(p) for p in selected_late], ["20261003-120040.mp4"])

    def test_prune_preserves_protected_segments(self):
        # Create segments spanning 20 minutes (12:00 to 12:20, 1 min apart)
        base_epoch = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        paths = []
        for minute in range(21):
            name = f"20261003-12{minute:02d}00.mp4"
            paths.append(self._create_segment(name))

        segs = list_segments(self.test_dir)
        now = base_epoch + (20 * 60)  # At 12:20:00
        ring_minutes = 15  # Retention cutoff: 12:05:00

        # Protect segments starting at 12:03:00 (earlier than 15 min cutoff)
        protected_start = base_epoch + (3 * 60)
        deleted = prune_segments(segs, now, ring_minutes, protected_start=protected_start)

        # Segments before 12:03:00 (12:00, 12:01, 12:02) end at 12:01, 12:02, 12:03 <= cutoff
        # So 12:00 and 12:01 should be deleted (end < 12:03), 12:02 ends at 12:03
        remaining_files = os.listdir(self.test_dir)
        self.assertNotIn("20261003-120000.mp4", remaining_files)
        self.assertNotIn("20261003-120100.mp4", remaining_files)
        # 12:03 and later must be preserved
        self.assertIn("20261003-120300.mp4", remaining_files)
        self.assertIn("20261003-121500.mp4", remaining_files)
        self.assertIn("20261003-122000.mp4", remaining_files)


if __name__ == "__main__":
    unittest.main()
