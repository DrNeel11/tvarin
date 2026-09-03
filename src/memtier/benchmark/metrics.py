"""System/process metric sampling for benchmark runs.

GPU utilization is sampled from a background thread because it can only be
read as an instantaneous snapshot (no cumulative counter), unlike VRAM peak
which torch tracks for us.
"""

from __future__ import annotations

import threading
import time

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


class GpuUtilSampler:
    """Background-thread sampler for GPU/VRAM utilization percentage."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        self.device_index = device_index
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._mem_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        if _NVML_OK:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                self._handle = None

    def _run(self):
        while not self._stop.is_set():
            if self._handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self._samples.append(util.gpu)
                    self._mem_samples.append(util.memory)
                except Exception:
                    pass
            time.sleep(self.interval_s)

    def __enter__(self):
        self._samples = []
        self._mem_samples = []
        self._stop.clear()
        if self._handle is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    @property
    def avg_gpu_util_pct(self) -> float | None:
        return sum(self._samples) / len(self._samples) if self._samples else None

    @property
    def max_gpu_util_pct(self) -> float | None:
        return max(self._samples) if self._samples else None


def process_rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def system_ram_used_bytes() -> int:
    vm = psutil.virtual_memory()
    return vm.total - vm.available


def cpu_percent(interval: float | None = None) -> float:
    return psutil.cpu_percent(interval=interval)


def nvml_available() -> bool:
    return _NVML_OK
