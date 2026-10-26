"""Measure elapsed time and traced Python memory for one operation."""

import time
import tracemalloc
from dataclasses import dataclass
from typing import Optional


_BYTES_PER_MEGABYTE = 1024.0 * 1024.0


@dataclass(frozen=True)
class PerformanceMeasurement:
    """Measurements from an operation; memory is None if tracing was already active."""

    execution_time_ms: float
    peak_traced_memory_mb: Optional[float]
    tracing_was_already_active: bool


class PerformanceMonitor:
    """Measure wall-clock duration and peak traced allocations around a call.

    Memory measurements cover Python allocations tracked by ``tracemalloc`` during
    a tracing session started by this monitor. They do not represent total process
    memory. An existing tracing session is left untouched; its historical peak
    cannot be isolated safely, so peak memory is reported as unavailable.
    """

    def __init__(self) -> None:
        self._started_at: Optional[float] = None
        self._owns_tracing = False
        self._tracing_was_already_active = False
        self._measurement: Optional[PerformanceMeasurement] = None

    @property
    def is_active(self) -> bool:
        return self._started_at is not None

    @property
    def measurement(self) -> Optional[PerformanceMeasurement]:
        """The most recent completed measurement, including after a context block."""
        return self._measurement

    def start(self) -> None:
        """Start timing and begin tracing only if no other code is tracing."""
        if self.is_active:
            raise RuntimeError("This performance monitor is already active.")

        tracing_was_already_active = tracemalloc.is_tracing()
        started_at = time.perf_counter()
        if not tracing_was_already_active:
            tracemalloc.start()
        self._tracing_was_already_active = tracing_was_already_active
        self._owns_tracing = not tracing_was_already_active
        self._started_at = started_at
        self._measurement = None

    def stop(self) -> PerformanceMeasurement:
        """Stop the measurement and release only tracing started by this monitor."""
        if self._started_at is None:
            raise RuntimeError("This performance monitor has not been started.")

        start_time = self._started_at
        owns_tracing = self._owns_tracing
        already_active = self._tracing_was_already_active
        try:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            peak_memory_mb: Optional[float] = None
            if owns_tracing:
                _, peak_bytes = tracemalloc.get_traced_memory()
                peak_memory_mb = peak_bytes / _BYTES_PER_MEGABYTE
            measurement = PerformanceMeasurement(
                execution_time_ms=elapsed_ms,
                peak_traced_memory_mb=peak_memory_mb,
                tracing_was_already_active=already_active,
            )
            self._measurement = measurement
            return measurement
        finally:
            if owns_tracing and tracemalloc.is_tracing():
                tracemalloc.stop()
            self._started_at = None
            self._owns_tracing = False
            self._tracing_was_already_active = False

    def __enter__(self) -> "PerformanceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop()
        return False
