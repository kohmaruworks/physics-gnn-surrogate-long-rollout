"""
Inference timing with CUDA events (GPU) or ``perf_counter`` (CPU fallback).

Warm-up iterations reduce allocator / kernel-cache bias for ROI-grade timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch


@dataclass
class InferenceProfiler:
    """
    Measure mean wall time of ``forward_fn()`` per call.

    On CUDA, uses ``torch.cuda.Event`` pairs with synchronization; on CPU uses
    ``time.perf_counter``.
    """

    device: torch.device
    warmup_iters: int = 5
    benchmark_iters: int = 30

    def __post_init__(self) -> None:
        self._use_cuda = self.device.type == "cuda" and torch.cuda.is_available()

    def measure_mean_seconds(self, forward_fn: Callable[[], Tensor]) -> float:
        """Return mean elapsed seconds over ``benchmark_iters`` after ``warmup_iters`` warm-ups."""
        for _ in range(self.warmup_iters):
            if self._use_cuda:
                torch.cuda.synchronize(self.device)
            _ = forward_fn()
            if self._use_cuda:
                torch.cuda.synchronize(self.device)

        if self._use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            times_ms: List[float] = []
            for _ in range(self.benchmark_iters):
                torch.cuda.synchronize(self.device)
                starter.record()
                _ = forward_fn()
                ender.record()
                torch.cuda.synchronize(self.device)
                times_ms.append(float(starter.elapsed_time(ender)))
            ms_mean = sum(times_ms) / float(len(times_ms))
            return ms_mean / 1000.0

        import time

        secs: List[float] = []
        for _ in range(self.benchmark_iters):
            t0 = time.perf_counter()
            _ = forward_fn()
            secs.append(time.perf_counter() - t0)
        return sum(secs) / float(len(secs))
