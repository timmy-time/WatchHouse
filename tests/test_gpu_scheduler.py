"""Unit tests for GpuScheduler camera allocation and metric polling."""

import unittest
from unittest.mock import patch

from engine.live.scheduler import GpuDevice, GpuScheduler


class TestGpuScheduler(unittest.TestCase):
    def test_mocked_dual_gpu_balancing(self):
        scheduler = GpuScheduler()
        # Mock dual GPUs
        scheduler.has_cuda = True
        scheduler.devices = {
            0: GpuDevice(index=0, name="GPU (Card 0)", total_memory_mb=8192),
            1: GpuDevice(index=1, name="GPU (Card 1)", total_memory_mb=8192),
        }

        # Camera 1 requested GPU 0 explicitly
        gpu_cam1 = scheduler.assign_camera("Camera A", requested_gpu=0)
        self.assertEqual(gpu_cam1, 0)
        self.assertIn("Camera A", scheduler.devices[0].assigned_cameras)

        # Camera 2 auto-assign (requested_gpu=None) -> should pick GPU 1 (0 cameras)
        gpu_cam2 = scheduler.assign_camera("Camera B", requested_gpu=None)
        self.assertEqual(gpu_cam2, 1)
        self.assertIn("Camera B", scheduler.devices[1].assigned_cameras)

        # Camera 3 auto-assign -> both have 1 camera, picks GPU 0 or 1
        gpu_cam3 = scheduler.assign_camera("Camera C", requested_gpu=None)
        self.assertIn(gpu_cam3, (0, 1))

        # Check metrics format
        metrics = scheduler.poll_metrics()
        self.assertIn("0", metrics)
        self.assertIn("1", metrics)
        self.assertEqual(metrics["0"]["name"], "GPU (Card 0)")
        self.assertIn("Camera A", metrics["0"]["cameras"])

    def test_cpu_mode_fallback(self):
        scheduler = GpuScheduler()
        scheduler.has_cuda = False
        scheduler.devices = {}

        gpu_idx = scheduler.assign_camera("Camera A", requested_gpu=None)
        self.assertEqual(gpu_idx, 0)

        metrics = scheduler.poll_metrics()
        self.assertIn("cpu", metrics)


if __name__ == "__main__":
    unittest.main()
