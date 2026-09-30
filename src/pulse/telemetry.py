"""Performance monitoring for pulse's problems and the ``pulse`` CLI.

The interface is a superset of fenicsx-beat's ``beat.telemetry`` (``track_time``,
``record_ksp``, ``advance_step``, ``display_summary``, ``save_summary``), adding
``record_snes`` (Newton iterations) and ``count`` (named events), so a coupled driver can hand
the same :class:`PerformanceMonitor` to both solvers. Timings are wall-clock seconds on each
rank; summaries report rank 0's.
"""

import abc
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

logger = logging.getLogger(__name__)


class BaseMonitor(abc.ABC):
    @abc.abstractmethod
    @contextmanager
    def track_time(self, name: str):
        yield

    @abc.abstractmethod
    def record_ksp(self, ksp: PETSc.KSP) -> None: ...

    @abc.abstractmethod
    def record_snes(self, snes: PETSc.SNES) -> None: ...

    @abc.abstractmethod
    def count(self, name: str, n: int = 1) -> None: ...

    @abc.abstractmethod
    def advance_step(self, t0: float, t1: float) -> None: ...


class NullMonitor(BaseMonitor):
    """Does nothing; the default everywhere."""

    @contextmanager
    def track_time(self, name: str):
        yield

    def record_ksp(self, ksp: PETSc.KSP) -> None:
        pass

    def record_snes(self, snes: PETSc.SNES) -> None:
        pass

    def count(self, name: str, n: int = 1) -> None:
        pass

    def advance_step(self, t0: float, t1: float) -> None:
        pass


class PerformanceMonitor(BaseMonitor):
    """Accumulates timings, Newton/KSP statistics and event counts; logs a line every
    ``log_frequency`` steps and can display/save a summary."""

    def __init__(self, log_frequency: int = 1, comm: MPI.Intracomm = MPI.COMM_WORLD):
        self.log_frequency = log_frequency
        self.comm = comm
        self.step_counter = 0
        self.timings: dict[str, float] = {}
        self.counters: dict[str, int] = {}
        self.newton_total_iterations = 0
        self.newton_max_iterations = 0
        self.newton_last_iterations = 0
        self.newton_failures = 0
        self.ksp_total_iterations = 0
        self.ksp_max_iterations = 0
        self.ksp_last_iterations = 0
        self.ksp_last_converged_reason = 0

    @contextmanager
    def track_time(self, name: str):
        tic = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - tic)

    def count(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def _add_linear(self, iterations: int) -> None:
        self.ksp_last_iterations = iterations
        self.ksp_total_iterations += iterations
        self.ksp_max_iterations = max(self.ksp_max_iterations, iterations)

    def record_ksp(self, ksp: PETSc.KSP) -> None:
        try:
            self._add_linear(int(ksp.getIterationNumber()))
            self.ksp_last_converged_reason = int(ksp.getConvergedReason())
        except PETSc.Error:
            pass

    def record_snes(self, snes: PETSc.SNES) -> None:
        """Newton iterations and the linear iterations accumulated over them."""
        try:
            iterations = int(snes.getIterationNumber())
            self.newton_last_iterations = iterations
            self.newton_total_iterations += iterations
            self.newton_max_iterations = max(self.newton_max_iterations, iterations)
            if int(snes.getConvergedReason()) <= 0:
                self.newton_failures += 1
            self._add_linear(int(snes.getLinearSolveIterations()))
            self.ksp_last_converged_reason = int(snes.getKSP().getConvergedReason())
        except PETSc.Error:
            pass

    def advance_step(self, t0: float, t1: float) -> None:
        self.step_counter += 1
        if self.log_frequency <= 0 or self.step_counter % self.log_frequency != 0:
            return
        timing_text = ", ".join(f"{name}={value:.6f}s" for name, value in self.timings.items())
        counter_text = "".join(f", {name}={value}" for name, value in self.counters.items())
        logger.info(
            f"Mechanics step timing step={self.step_counter}, t=({t0:.6g}, {t1:.6g}), "
            f"newton_iterations={self.newton_last_iterations}, "
            f"ksp_iterations={self.ksp_last_iterations}{counter_text}, {timing_text}",
        )

    def summary(self) -> dict:
        return {
            "total_steps": self.step_counter,
            "newton": {
                "total_iterations": self.newton_total_iterations,
                "max_iterations": self.newton_max_iterations,
                "failures": self.newton_failures,
            },
            "ksp": {
                "total_iterations": self.ksp_total_iterations,
                "max_iterations": self.ksp_max_iterations,
            },
            "counters": dict(self.counters),
            "timings": dict(self.timings),
        }

    def display_summary(self) -> None:
        """Log a formatted summary (rank 0 only)."""
        if self.comm.rank != 0:
            return
        lines = ["", "=" * 50, f"{'PERFORMANCE SUMMARY':^50}", "=" * 50]
        lines.append(f"Total steps:              {self.step_counter}")
        lines.append(f"Newton total iterations:  {self.newton_total_iterations}")
        lines.append(f"Newton max iterations:    {self.newton_max_iterations}")
        lines.append(f"Newton failures:          {self.newton_failures}")
        lines.append(f"KSP total iterations:     {self.ksp_total_iterations}")
        for name, value in sorted(self.counters.items()):
            lines.append(f"{name + ':':<26}{value}")
        lines += ["-" * 50, f"{'Metric':<35} | {'Time (s)':>10}", "-" * 50]
        for name, duration in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{name:<35} | {duration:>10.4f}")
        lines.append("=" * 50)
        logger.info("\n".join(lines))

    def save_summary(self, filepath: str | Path) -> None:
        """Write :meth:`summary` as JSON (rank 0 only)."""
        if self.comm.rank != 0:
            return
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(self.summary(), indent=4))
        logger.info(f"Performance summary saved to {filepath}")
