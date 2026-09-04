from __future__ import annotations

from collections import deque
from typing import Deque, Dict

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class PerformanceMonitor:
    def __init__(self, window: int = 60) -> None:
        self._ms: Deque[float] = deque(maxlen=window)
        self._proc = psutil.Process() if psutil else None

    def record(self, total_ms: float) -> None:
        self._ms.append(total_ms)

    def snapshot(self) -> Dict[str, float]:
        fps = 0.0
        lat = 0.0
        if self._ms:
            lat = sum(self._ms) / len(self._ms)
            fps = 1000.0 / max(lat, 1e-6)
        cpu = self._proc.cpu_percent(interval=None) if self._proc else -1.0
        mem = (self._proc.memory_info().rss / (1024 * 1024)) if self._proc else -1.0
        return {
            "fps": fps,
            "latency_ms": lat,
            "cpu_percent": cpu,
            "rss_mb": mem,
            "gpu_percent": -1.0,  # filled on Jetson via tegrastats later
        }
