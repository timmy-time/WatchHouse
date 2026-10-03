"""Dual-GPU scheduler and hardware resource monitor for multi-camera live analytics."""

from dataclasses import dataclass, field
import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GpuDevice:
    index: int
    name: str
    total_memory_mb: int = 8192
    used_memory_mb: int = 0
    utilization_pct: int = 0
    assigned_cameras: List[str] = field(default_factory=list)


class GpuScheduler:
    """Discovers, balances, and monitors CUDA GPUs across camera workers."""

    def __init__(self):
        self.lock = threading.RLock()
        self.devices: Dict[int, GpuDevice] = {}
        self.has_cuda: bool = False
        self._discover_devices()

    def _discover_devices(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                self.has_cuda = True
                count = torch.cuda.device_count()
                for i in range(count):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    total_mb = int(props.total_memory / (1024 * 1024))
                    self.devices[i] = GpuDevice(
                        index=i,
                        name=name,
                        total_memory_mb=total_mb,
                    )
                logger.info(f"GpuScheduler discovered {count} CUDA device(s)")
                return
        except Exception as exc:
            logger.debug(f"PyTorch CUDA probe error: {exc}")

        # Fallback to nvidia-smi if torch failed or not loaded yet
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                res = subprocess.run(
                    [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    lines = res.stdout.strip().split("\n")
                    for line in lines:
                        if not line.strip():
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        idx = int(parts[0])
                        name = parts[1]
                        total_mb = int(float(parts[2]))
                        self.devices[idx] = GpuDevice(index=idx, name=name, total_memory_mb=total_mb)
                    if self.devices:
                        self.has_cuda = True
                        logger.info(f"GpuScheduler discovered {len(self.devices)} GPU(s) via nvidia-smi")
                        return
            except Exception as exc:
                logger.debug(f"nvidia-smi probe error: {exc}")

        logger.info("GpuScheduler running in CPU-only mode (no CUDA devices found)")

    def assign_camera(self, camera_name: str, requested_gpu: Optional[int] = None) -> int:
        """Assign camera to requested GPU, or auto-balance across available GPUs."""
        with self.lock:
            if not self.has_cuda or not self.devices:
                return 0

            # If user explicitly requested a valid GPU, honor it
            if requested_gpu is not None and requested_gpu in self.devices:
                chosen_idx = requested_gpu
            else:
                # Auto-balance: pick GPU with fewest assigned cameras
                chosen_idx = min(
                    self.devices.keys(),
                    key=lambda idx: len(self.devices[idx].assigned_cameras),
                )

            dev = self.devices[chosen_idx]
            if camera_name not in dev.assigned_cameras:
                dev.assigned_cameras.append(camera_name)

            logger.info(
                f"GpuScheduler: Assigned camera '{camera_name}' -> GPU {chosen_idx} "
                f"({dev.name}, {len(dev.assigned_cameras)} cameras assigned)"
            )
            return chosen_idx

    def poll_metrics(self) -> Dict[str, Any]:
        """Poll real-time GPU utilization and memory metrics."""
        with self.lock:
            if not self.has_cuda or not self.devices:
                return {
                    "cpu": {
                        "name": "CPU Host",
                        "assigned_cameras": [
                            cam for d in self.devices.values() for cam in d.assigned_cameras
                        ],
                    }
                }

            smi = shutil.which("nvidia-smi")
            if smi:
                try:
                    res = subprocess.run(
                        [
                            smi,
                            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if res.returncode == 0:
                        for line in res.stdout.strip().split("\n"):
                            if not line.strip():
                                continue
                            parts = [p.strip() for p in line.split(",")]
                            idx = int(parts[0])
                            if idx in self.devices:
                                self.devices[idx].utilization_pct = int(float(parts[1]))
                                self.devices[idx].used_memory_mb = int(float(parts[2]))
                                self.devices[idx].total_memory_mb = int(float(parts[3]))
                except Exception:
                    pass

            return {
                str(idx): {
                    "name": dev.name,
                    "memory_used_mb": dev.used_memory_mb,
                    "memory_total_mb": dev.total_memory_mb,
                    "utilization_pct": dev.utilization_pct,
                    "cameras": list(dev.assigned_cameras),
                }
                for idx, dev in sorted(self.devices.items())
            }
