"""Couplings between a mechanics problem and the 0D models that drive its cavities.

A *coupling* owns the solve of one step: it decides what the cavities are constrained by, sets
up the problem for the step, solves it, and keeps whatever 0D state it carries. Three styles
exist, one per demo in `demo/time_dependent/`:

- `CycleCoupling`: the five-phase Alya-style cycle of `pulse.cycle`, a Windkessel per cavity.
- `SplitCoupling`: any 0D model whose right-hand side can be called with numbers, stepped by
  forward Euler between one mechanics solve and the next.
- `MonolithicCoupling`: a 0D model written in UFL whose states are unknowns of the same Newton
  system as the displacement.

`NoCoupling` is the plain mechanics step. A `StepHook` is the other half of the contract: state
that is not a 0D model but must advance, commit and roll back with every step, such as an
injected crossbridge model. Neither protocol knows about the CLI; `pulse.cli` only builds these
from a config, and simcardemsx can build them itself.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import dolfinx

from . import cycle
from .problem import Cavity, CavityControl

logger = logging.getLogger(__name__)

__all__ = [
    "Coupling",
    "CycleCoupling",
    "NoCoupling",
    "Phase",
    "StepHook",
]


@runtime_checkable
class Coupling(Protocol):
    """What drives one mechanics step and the 0D state that goes with it.

    The order of calls is: `cavities` and `problem_kwargs` while the problem is built,
    `attach` once it exists, then `initialize` on a fresh run *or* `load_state_dict` on a
    restart (never both: a restart must not solve), then `advance` once per (sub)step.
    """

    def cavities(self, mesh: dolfinx.mesh.Mesh) -> list[Cavity]:
        """The problem's cavities. Called once, while the problem is built."""
        ...

    def problem_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments for the problem; a ``"parameters"`` entry is merged into
        the problem's parameters rather than passed on."""
        ...

    def attach(self, problem: Any) -> None:
        """Wire the coupling to the built problem. Must not change any state."""
        ...

    def initialize(self, t0: float) -> None:
        """Set the initial 0D state at `t0` (fresh runs only); may solve."""
        ...

    def advance(self, t: float, dt: float) -> bool:
        """Solve the step from `t` to `t + dt`. On ``False`` the problem's state Functions and
        the coupling's state are as they were before the call, except for (a) per-step inputs
        the coupling sets from its own state before every solve (cavity controls, volume or
        input Constants), which may hold the failed attempt's values, and (b) solver hints such
        as `CycleController`'s pending preconditioner refresh."""
        ...

    def record(self) -> dict[str, float]:
        """Values at the current time, one column each in ``loads.csv``. Fixed keys."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Everything the coupling carries between steps, JSON-able."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore what `state_dict` returned."""
        ...


@runtime_checkable
class StepHook(Protocol):
    """State that advances with every mechanics step, e.g. an injected crossbridge model.

    `before_solve` runs before each solve attempt, `after_solve` only after a converged one.
    When an attempt fails, the hook is rolled back to its state before `before_solve` (through
    `state_dict`/`load_state_dict` and the values of `restart_functions`), so a halved retry
    starts from the same state. Restart names must not start with ``mechanics_``.
    """

    def before_solve(self, t: float, dt: float) -> None: ...

    def after_solve(self, t: float, dt: float) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]: ...


@dataclass(frozen=True, slots=True)
class Phase:
    """A time-derived 0D input: ``t mod period``, e.g. the ``beat_phase`` of Regazzoni's
    circuit once its `timing` component is dropped."""

    period: float

    def __call__(self, t: float) -> float:
        return t % self.period


class NoCoupling:
    """The plain mechanics step: solve, and put the state back if Newton fails."""

    def __init__(self) -> None:
        self.problem: Any = None

    def cavities(self, mesh: dolfinx.mesh.Mesh) -> list[Cavity]:
        return []

    def problem_kwargs(self) -> dict[str, Any]:
        return {}

    def attach(self, problem: Any) -> None:
        self.problem = problem

    def initialize(self, t0: float) -> None:
        pass

    def advance(self, t: float, dt: float) -> bool:
        ok = self.problem.solve()
        if not ok:
            self.problem.reset_states()
        return ok

    def record(self) -> dict[str, float]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        pass


class CycleCoupling:
    """The five-phase cycle of `pulse.cycle`, one `CavityControl` per cavity.

    `advance(t, dt)` is `CycleController.step(t + dt, dt)`, which retries a failed solve once
    and, if the retry fails too, leaves the state as it was apart from the cavity controls
    and the pending preconditioner refresh (see `Coupling.advance`).
    `record` reports, per cavity, the phase the step was *solved under* (the controller's own
    records hold the phase for the next step), and the volume, pressure, compliance pressure and
    outflow of that step, in SI units.
    """

    def __init__(
        self,
        params: Mapping[str, cycle.CycleParams],
        preconditioner_lag: int | None = None,
    ) -> None:
        self.params = dict(params)
        self.preconditioner_lag = preconditioner_lag
        self.controller: cycle.CycleController | None = None
        self._solved_under: dict[str, int] = {}

    def _controller(self) -> cycle.CycleController:
        if self.controller is None:
            raise RuntimeError("CycleCoupling used before attach()")
        return self.controller

    def cavities(self, mesh: dolfinx.mesh.Mesh) -> list[Cavity]:
        return [Cavity(marker=name, control=CavityControl(mesh)) for name in self.params]

    def problem_kwargs(self) -> dict[str, Any]:
        return {}

    def attach(self, problem: Any) -> None:
        self.controller = cycle.CycleController(
            problem,
            self.params,
            preconditioner_lag=self.preconditioner_lag,
        )

    def initialize(self, t0: float) -> None:
        controller = self._controller()
        controller.initialize(t0)
        self._solved_under = {name: int(c.phase) for name, c in controller.cycles.items()}

    def advance(self, t: float, dt: float) -> bool:
        controller = self._controller()
        under = {name: int(c.phase) for name, c in controller.cycles.items()}
        if not controller.step(t + dt, dt):
            return False
        self._solved_under = under
        return True

    def record(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, rec in self._controller().records.items():
            out[f"phase_{name}"] = float(self._solved_under[name])
            out[f"volume_{name}"] = rec.V
            out[f"pressure_{name}"] = rec.P
            out[f"Pc_{name}"] = rec.P_c
            out[f"Q_{name}"] = rec.Q
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller": self._controller().state_dict(),
            "solved_under": dict(self._solved_under),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._controller().load_state_dict(state["controller"])
        self._solved_under = {name: int(p) for name, p in state["solved_under"].items()}
