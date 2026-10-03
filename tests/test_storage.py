import os
import shutil
import tempfile
import unittest

from engine.storage import KeepStorage, batch_store_keeps, get_relative_folder, store_keep_clip


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.temp_dir.name, "input")
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_relative_folder(self):
        # Nested subfolder
        path1 = os.path.join(self.input_dir, "20260103", "clip1.mp4")
        self.assertEqual(get_relative_folder(path1, self.input_dir), "20260103")

        # Deeper nested subfolder
        path2 = os.path.join(self.input_dir, "sub", "cam1", "clip2.mp4")
        self.assertEqual(get_relative_folder(path2, self.input_dir), os.path.join("sub", "cam1"))

        # Direct root file -> fallback
        path3 = os.path.join(self.input_dir, "clip3.mp4")
        self.assertEqual(get_relative_folder(path3, self.input_dir, fallback="Camera A"), "Camera A")
        self.assertEqual(get_relative_folder(path3, self.input_dir), "default")

    def test_store_keep_clip_copy(self):
        subfolder = os.path.join(self.input_dir, "20260103")
        os.makedirs(subfolder, exist_ok=True)

        clip_path = os.path.join(subfolder, "clip1.mp4")
        thumb_path = os.path.join(subfolder, "clip1.png")

        with open(clip_path, "wb") as f:
            f.write(b"video_content_bytes")
        with open(thumb_path, "wb") as f:
            f.write(b"thumb_content_bytes")

        record = {
            "clip_path": clip_path,
            "thumbnail_path": thumb_path,
            "camera": "Camera A",
            "verdict": "KEEP",
        }

        res = store_keep_clip(record, self.input_dir, self.output_dir, action="copy")

        expected_clip = os.path.join(self.output_dir, "clips", "20260103", "clip1.mp4")
        expected_thumb = os.path.join(self.output_dir, "clips", "20260103", "clip1.png")

        self.assertEqual(res["stored_clip_path"], expected_clip)
        self.assertEqual(res["stored_thumb_path"], expected_thumb)
        self.assertEqual(record["stored_clip_path"], expected_clip)
        self.assertEqual(record["stored_thumb_path"], expected_thumb)

        self.assertTrue(os.path.exists(expected_clip))
        self.assertTrue(os.path.exists(expected_thumb))
        # Original files still exist in copy mode
        self.assertTrue(os.path.exists(clip_path))
        self.assertTrue(os.path.exists(thumb_path))

    def test_store_keep_clip_move(self):
        subfolder = os.path.join(self.input_dir, "20260103")
        os.makedirs(subfolder, exist_ok=True)

        clip_path = os.path.join(subfolder, "clip2.mp4")
        thumb_path = os.path.join(subfolder, "clip2.png")

        with open(clip_path, "wb") as f:
            f.write(b"video_content_bytes_move")
        with open(thumb_path, "wb") as f:
            f.write(b"thumb_content_bytes_move")

        record = {
            "clip_path": clip_path,
            "thumbnail_path": thumb_path,
            "camera": "Camera C",
            "verdict": "KEEP",
        }

        res = store_keep_clip(record, self.input_dir, self.output_dir, action="move")

        expected_clip = os.path.join(self.output_dir, "clips", "20260103", "clip2.mp4")
        expected_thumb = os.path.join(self.output_dir, "clips", "20260103", "clip2.png")

        self.assertEqual(res["stored_clip_path"], expected_clip)
        self.assertEqual(res["stored_thumb_path"], expected_thumb)
        self.assertTrue(os.path.exists(expected_clip))
        self.assertTrue(os.path.exists(expected_thumb))

        # Original files moved
        self.assertFalse(os.path.exists(clip_path))
        self.assertFalse(os.path.exists(thumb_path))

    def test_batch_store_keeps(self):
        subfolder = os.path.join(self.input_dir, "20260524")
        os.makedirs(subfolder, exist_ok=True)

        clip1 = os.path.join(subfolder, "keep1.mp4")
        clip2 = os.path.join(subfolder, "discard1.mp4")
        clip3 = os.path.join(subfolder, "keep2.mp4")

        for c in (clip1, clip2, clip3):
            with open(c, "wb") as f:
                f.write(b"data")

        records = [
            {"clip_path": clip1, "verdict": "KEEP", "camera": "Cam1"},
            {"clip_path": clip2, "verdict": "DISCARD", "camera": "Cam1"},
            {"clip_path": clip3, "verdict": "KEEP", "camera": "Cam1"},
        ]

        stored_count = batch_store_keeps(records, self.input_dir, self.output_dir, action="copy")
        self.assertEqual(stored_count, 2)

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "clips", "20260524", "keep1.mp4")))
        self.assertFalse(os.path.exists(os.path.join(self.output_dir, "clips", "20260524", "discard1.mp4")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "clips", "20260524", "keep2.mp4")))

    def test_keep_storage_class(self):
        subfolder = os.path.join(self.input_dir, "20260103")
        os.makedirs(subfolder, exist_ok=True)
        clip_path = os.path.join(subfolder, "test_class.mp4")
        with open(clip_path, "wb") as f:
            f.write(b"data")

        storage = KeepStorage(self.input_dir, self.output_dir, action="copy")
        record = {"clip_path": clip_path, "verdict": "KEEP"}
        res = storage.store_clip(record)
        self.assertTrue(os.path.exists(res["stored_clip_path"]))


if __name__ == "__main__":
    unittest.main()
