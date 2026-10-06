"""The Alya-style five-phase cardiac cycle, driven through `pulse.problem.CavityControl`.

A `CavityControl` lets one cavity's pressure unknown satisfy a volume
constraint, a prescribed pressure, or a pressure affine in the volume (a
Windkessel during ejection), switched between at run time. This module is the
policy that switches it: a state machine with five phases -- preload,
isovolumic contraction, ejection, isovolumic relaxation, filling -- each
imposing one of those constraints, and a controller that steps a
`pulse.problem.StaticProblem` (or `DynamicProblem`) through them.

It is a port of physcardems' ``cycle_controller.py`` (``biv_cavity_cycle_
controller.py`` before that) onto pulse's own `CavityControl`, restricted to
the subset simcardemsx needs: one Windkessel outflow per cavity, prescribed
(not legacy gain/fill-rate) filling, and no stall-detection phase counter.
Everything here is SI: seconds, pascals, cubic metres.

Phase -> constraint, from start-of-step state:
    PRELOAD                pressure: linear ramp (0 to t_zero, then to
                            t_end_diastole)
    ISOVOLUMIC_CONTRACTION volume:   V = end diastolic volume
    EJECTION                pressure: an implicit (affine-in-V) three-element
                            Windkessel:
                                Q   = -(V - V_n) / dt
                                P_c = (P_c_n + dt Q / C) / (1 + dt / (R_p C))
                                P_v = P_c + R_c Q
    ISOVOLUMIC_RELAXATION  volume:   V = end systolic volume
    FILLING                volume:   V = V_n + rate * dt

Valve opening (IVC -> EJECTION): the cavity pressure exceeds the Windkessel's
compliance pressure. Valve closing (EJECTION -> IVR): outflow has stopped, or
the cavity pressure has dropped below the compliance pressure, after
`min_ejection_duration`. With the valve closed, the compliance pressure drains
through the peripheral resistance once the cavity has ejected at least once.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx

from .geometry import HeartGeometry
from .problem import CavityControl, StaticProblem

logger = logging.getLogger(__name__)


class Phase(enum.IntEnum):
    """The five phases of one cardiac cycle, Alya-style."""

    PRELOAD = 0
    ISOVOLUMIC_CONTRACTION = 1
    EJECTION = 2
    ISOVOLUMIC_RELAXATION = 3
    FILLING = 4


#: Short names for log messages; not part of the public interface.
_PHASE_NAMES = {
    Phase.PRELOAD: "PRELOAD",
    Phase.ISOVOLUMIC_CONTRACTION: "ISOVOLUMIC_CONTRACTION",
    Phase.EJECTION: "EJECTION",
    Phase.ISOVOLUMIC_RELAXATION: "ISOVOLUMIC_RELAXATION",
    Phase.FILLING: "FILLING",
}


@dataclass(frozen=True)
class Windkessel:
    """A three-element Windkessel, stepped by backward Euler.

    Stateless: the compliance (arterial) pressure ``P_c`` lives in the
    caller's `CavityCycle`, not here, so one `Windkessel` can describe several
    cavities. ``characteristic_impedance`` (the series resistance ``R_c``) is
    0 by default, giving a two-element model.
    """

    p_init: float
    compliance: float
    resistance: float
    characteristic_impedance: float = 0.0

    def ejection_law(self, P_c: float, V_n: float, dt: float) -> tuple[float, float]:
        """Coefficients of ``P_v = A + B V`` an ejecting cavity's Newton solve enforces.

        This is the same backward-Euler compliance update as `advance`, solved
        together with the cavity's own ``Q = -(V - V_n) / dt``, but expressed
        as an affine function of the (still unknown) volume `V` rather than
        evaluated at a known one. Once Newton converges on a volume ``V``,
        calling `advance` with that ``V`` reproduces the same ``(P_c, Q)``
        this implies -- `A + B V == P_c + R_c Q` to round-off.
        """
        D = 1.0 + dt / (self.resistance * self.compliance)
        Rc = self.characteristic_impedance
        A = P_c / D + V_n / (self.compliance * D) + Rc * V_n / dt
        B = -(1.0 / (self.compliance * D) + Rc / dt)
        return A, B

    def advance(
        self,
        P_c: float,
        V: float,
        V_n: float,
        dt: float,
        *,
        ejecting: bool,
        has_ejected: bool,
    ) -> tuple[float, float]:
        """Advance the compliance pressure over a converged step; return ``(P_c, Q)``.

        While ejecting, ``Q`` is the outflow implied by the converged volume
        change and ``P_c`` is the backward-Euler compliance update. Outside
        ejection there is no outflow (``Q = 0``); the compliance pressure
        holds at its last value until the cavity has ejected at least once, at
        which point it drains through the peripheral resistance.
        """
        D = 1.0 + dt / (self.resistance * self.compliance)
        if ejecting:
            Q = -(V - V_n) / dt
            P_c_new = (P_c + dt * Q / self.compliance) / D
        else:
            Q = 0.0
            P_c_new = P_c / D if has_ejected else P_c
        return P_c_new, Q


@dataclass(frozen=True)
class PrescribedInflow:
    """Filling by a fixed volumetric rate: ``V_target = V_n + rate * dt``."""

    rate: float  # m^3/s

    def target_volume(self, V_n: float, dt: float) -> float:
        return V_n + self.rate * dt


@dataclass(frozen=True)
class CycleParams:
    """One cavity's cycle timing and Windkessel/filling laws, all SI."""

    t_zero: float
    preload_pressure: float
    t_end_diastole: float
    p_end_diastole: float
    p_fill: float
    period: float
    windkessel: Windkessel
    filling: PrescribedInflow
    min_ejection_duration: float = 0.01
    dvol_eps: float = 1e-10


@dataclass
class CavityCycle:
    """One cavity's phase and Windkessel state -- the ported subset of physcardems' ``CavityState``.

    Mutable, and mutated only by `CycleController.step` on a converged step:
    a failed step (`step` returning ``False``) leaves every field here
    exactly as it was.
    """

    phase: Phase = Phase.PRELOAD
    n_beats: int = 0
    last_phase_change: float = 0.0
    volume_n: float = 0.0
    pressure_n: float = 0.0
    dvol_n: float = 0.0
    wdk_pressure_n: float = 0.0
    outflow_n: float = 0.0
    has_ejected: bool = False
    end_dia_vol: float = 0.0
    end_sys_vol: float = 0.0


@dataclass
class CavityRecord:
    """A snapshot of one cavity, written by `CycleController.step` after each converged step.

    ``V``, ``P``, ``P_c`` and ``Q`` are the step just solved. ``phase`` is not
    the phase they were solved under: it is the phase *after* that step's
    transition, i.e. the one in force for the *next* step. The two differ on
    exactly the step at which the phase switches; the phase a step was solved
    under is the ``phase`` of the record before it.
    """

    phase: Phase
    V: float
    P: float
    P_c: float
    Q: float


#: How each field type of `CavityCycle`/`CavityRecord` goes to JSON and back. The
#: annotations are strings here (``from __future__ import annotations``).
_TO_JSON = {"Phase": int, "int": int, "float": float, "bool": bool}
_FROM_JSON = {"Phase": Phase, "int": int, "float": float, "bool": bool}

_Fields = TypeVar("_Fields", CavityCycle, CavityRecord)


def _fields_to_json(obj: CavityCycle | CavityRecord) -> dict[str, Any]:
    return {f.name: _TO_JSON[str(f.type)](getattr(obj, f.name)) for f in dataclasses.fields(obj)}


def _fields_from_json(cls: type[_Fields], data: Mapping[str, Any]) -> _Fields:
    return cls(**{f.name: _FROM_JSON[str(f.type)](data[f.name]) for f in dataclasses.fields(cls)})


def _preload_pressure(cyc: CavityCycle, params: CycleParams, t: float) -> float:
    """The PRELOAD ramp: 0 to `t_zero`, then on to `p_end_diastole` at `t_end_diastole`."""
    if t <= params.t_zero:
        return params.preload_pressure * t / params.t_zero
    if params.t_end_diastole > params.t_zero:
        return (params.p_end_diastole - params.preload_pressure) * (
            t - cyc.n_beats * params.period - params.t_zero
        ) / (params.t_end_diastole - params.t_zero) + params.preload_pressure
    return cyc.pressure_n


def _apply_control(
    control: CavityControl,
    cyc: CavityCycle,
    params: CycleParams,
    t: float,
    dt: float,
) -> None:
    """Set this cavity's constraint for the step ending at `t`, from start-of-step state."""
    if cyc.phase == Phase.PRELOAD:
        control.set_pressure(_preload_pressure(cyc, params, t))
    elif cyc.phase == Phase.ISOVOLUMIC_CONTRACTION:
        control.set_volume(cyc.end_dia_vol)
    elif cyc.phase == Phase.EJECTION:
        A, B = params.windkessel.ejection_law(cyc.wdk_pressure_n, cyc.volume_n, dt)
        control.set_affine_pressure(A, B)
    elif cyc.phase == Phase.ISOVOLUMIC_RELAXATION:
        control.set_volume(cyc.end_sys_vol)
    elif cyc.phase == Phase.FILLING:
        control.set_volume(params.filling.target_volume(cyc.volume_n, dt))
    else:
        raise ValueError(f"Unknown phase {cyc.phase}")  # pragma: no cover - IntEnum is exhaustive


def _advance_phase(cyc: CavityCycle, params: CycleParams, t: float) -> bool:
    """Transition `cyc.phase` from its freshly committed state; return whether it changed.

    Ported from physcardems ``cycle.py`` lines 161-186, without the
    ``ejection_pressure`` valve-opening override (the valve always opens at
    the Windkessel's own compliance pressure here) and without the
    stall-detection ``phase_counter`` branch at lines 187-192 (FILLING here
    only ever ends at the next beat's `t_zero`).
    """
    old = cyc.phase
    since = t - cyc.last_phase_change

    if (
        cyc.phase == Phase.PRELOAD
        and t >= cyc.n_beats * params.period + params.t_end_diastole
        and t >= params.t_zero
    ):
        cyc.phase, cyc.last_phase_change = Phase.ISOVOLUMIC_CONTRACTION, t
    elif cyc.phase == Phase.ISOVOLUMIC_CONTRACTION:
        if cyc.pressure_n > cyc.wdk_pressure_n:
            cyc.phase, cyc.last_phase_change = Phase.EJECTION, t
            cyc.has_ejected = True
    elif cyc.phase == Phase.EJECTION and since > params.min_ejection_duration:
        outflow_stopped = cyc.dvol_n > -params.dvol_eps
        pressure_below_arterial = cyc.pressure_n < cyc.wdk_pressure_n
        if outflow_stopped or pressure_below_arterial:
            cyc.phase, cyc.last_phase_change = Phase.ISOVOLUMIC_RELAXATION, t
    elif cyc.phase == Phase.ISOVOLUMIC_RELAXATION and cyc.pressure_n < params.p_fill:
        cyc.phase, cyc.last_phase_change = Phase.FILLING, t
    elif cyc.phase == Phase.FILLING:
        if params.period > 0.0 and t >= (cyc.n_beats + 1) * params.period + params.t_zero:
            cyc.phase, cyc.last_phase_change = Phase.PRELOAD, t
            cyc.n_beats += 1

    changed = cyc.phase != old
    if changed:
        logger.info(
            f"{_PHASE_NAMES[old]} -> {_PHASE_NAMES[cyc.phase]} at t={t:.4f} s "
            f"(V={cyc.volume_n * 1e6:.4f} mL, P={cyc.pressure_n / 1e3:.4f} kPa, "
            f"P_c={cyc.wdk_pressure_n / 1e3:.4f} kPa)",
        )
    return changed


def _set_preconditioner_lag(problem: StaticProblem, lag: int) -> None:
    """Set ``snes_lag_preconditioner`` through the PETSc options database.

    petsc4py has no direct setter for this option, so it goes through
    `PETSc.Options()` and `SNES.setFromOptions`, exactly as physcardems'
    ``cavity.py:set_preconditioner_lag`` does.
    """
    snes = problem.problem.solver
    key = f"{snes.getOptionsPrefix() or ''}snes_lag_preconditioner"
    opts = PETSc.Options()
    opts.setValue(key, lag)
    try:
        snes.setFromOptions()
    finally:
        # `PETSc.Options()` is process-global: a key left behind would reach
        # every other SNES with this prefix.
        opts.delValue(key)


class CycleController:
    """Step one or more `CavityControl`-carrying cavities of `problem` through the five phases.

    Construction resolves each name in `params` to the matching controlled
    `Cavity` of `problem` (`KeyError` if it is not a cavity of `problem`, or is
    one without a `control`), and precompiles each cavity's volume form so
    `step` does not rebuild it every call.

    `preconditioner_lag`, when given, is the steady-state
    ``snes_lag_preconditioner``. A solve is instead run with the
    preconditioner rebuilt every iteration (lag 1), then this value is
    restored, when it is: the first solve; the solve after any phase change;
    the retry of a failed solve; and the first solve of the step after a
    failed `step`. `None` (the default) leaves the solver's own
    preconditioner-lag setting alone.

    `records` maps each cavity to its `CavityRecord` from the last converged
    step: that step's ``V``/``P``/``P_c``/``Q``, with the ``phase`` for the
    *next* step (see `CavityRecord`).
    """

    def __init__(
        self,
        problem: StaticProblem,
        params: dict[str, CycleParams],
        preconditioner_lag: int | None = None,
    ) -> None:
        self.problem = problem
        self.params = dict(params)
        self.preconditioner_lag = preconditioner_lag

        by_marker = {cavity.marker: cavity for cavity in problem.cavities}
        self._controls: dict[str, CavityControl] = {}
        for name in self.params:
            cavity = by_marker.get(name)
            if cavity is None or cavity.control is None:
                controlled = sorted(c.marker for c in problem.cavities if c.control is not None)
                raise KeyError(
                    f"{name!r} is not a controlled cavity of this problem. Controlled "
                    f"cavities: {controlled}",
                )
            self._controls[name] = cavity.control

        self._cavity_index = {cavity.marker: i for i, cavity in enumerate(problem.cavities)}

        geometry = problem.geometry
        if not isinstance(geometry, HeartGeometry):
            raise RuntimeError("CycleController needs a HeartGeometry, for its volume forms")
        self._volume_forms = {
            name: dolfinx.fem.form(
                geometry.volume_form(problem.u) * geometry.ds(geometry.markers[name][0]),
            )
            for name in self.params
        }

        self.cycles: dict[str, CavityCycle] = {name: CavityCycle() for name in self.params}
        self.records: dict[str, CavityRecord] = {}
        self._initialized = False
        # True, as physcardems' `_refactor`: the first solve runs at lag 1, so
        # `preconditioner_lag` is in force from the second solve on rather
        # than only after the first refresh.
        self._refresh_pending = True

    def _volume(self, name: str) -> float:
        comm: MPI.Comm = self.problem.geometry.mesh.comm
        return comm.allreduce(
            dolfinx.fem.assemble_scalar(self._volume_forms[name]),
            op=MPI.SUM,
        )

    def _pressure(self, name: str) -> float:
        return float(self.problem.cavity_pressures[self._cavity_index[name]].x.array[0])

    def initialize(self, t0: float) -> None:
        """Read each cavity's current volume/pressure and start every phase at PRELOAD."""
        for name, cyc in self.cycles.items():
            V = self._volume(name)
            P = self._pressure(name)
            cyc.phase = Phase.PRELOAD
            cyc.n_beats = 0
            cyc.last_phase_change = t0
            cyc.volume_n = V
            cyc.pressure_n = P
            cyc.dvol_n = 0.0
            cyc.wdk_pressure_n = self.params[name].windkessel.p_init
            cyc.outflow_n = 0.0
            cyc.has_ejected = False
            cyc.end_dia_vol = V
            cyc.end_sys_vol = 0.0
        self._initialized = True
        self.records = {
            name: CavityRecord(
                phase=cyc.phase,
                V=cyc.volume_n,
                P=cyc.pressure_n,
                P_c=cyc.wdk_pressure_n,
                Q=cyc.outflow_n,
            )
            for name, cyc in self.cycles.items()
        }

    def state_dict(self) -> dict[str, Any]:
        """Everything `step` carries from one call to the next, JSON-able.

        ``{"initialized", "refresh_pending", "cycles", "records"}``, with each
        `CavityCycle` and `CavityRecord` as a dict of its fields and the phase
        as an int. It survives ``json.loads(json.dumps(...))`` unchanged.
        """
        return {
            "initialized": self._initialized,
            "refresh_pending": self._refresh_pending,
            "cycles": {name: _fields_to_json(cyc) for name, cyc in self.cycles.items()},
            "records": {name: _fields_to_json(rec) for name, rec in self.records.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore what `state_dict` returned, for the same cavities.

        Raises `KeyError`, changing nothing, naming every cavity that is in
        only one of `state` and this controller. A pending preconditioner
        refresh is never cleared: a fresh solver has no factorization to
        reuse, and a rolled-back failure keeps its refresh pending.
        """
        names = set(state["cycles"]) | set(state["records"])
        mismatch = sorted(names ^ set(self.params))
        if mismatch:
            raise KeyError(
                f"Cavities {mismatch} are in only one of the state "
                f"({sorted(names)}) and this controller ({sorted(self.params)})",
            )
        cycles = {
            name: _fields_from_json(CavityCycle, state["cycles"][name]) for name in self.params
        }
        records = {
            name: _fields_from_json(CavityRecord, state["records"][name])
            for name in self.params
            if name in state["records"]
        }
        self.cycles = cycles
        self.records = records
        self._initialized = bool(state["initialized"])
        self._refresh_pending = self._refresh_pending or bool(state["refresh_pending"])

    def _solve_once(self) -> bool:
        """One Newton solve, refreshing the preconditioner first if one is pending.

        Never raises on non-convergence, whatever ``parameters["raise_on_failure"]``
        says: `step` has to see the failure to roll back. A pending refresh is
        cleared only by a converged solve.
        """
        lag = self.preconditioner_lag
        if self._refresh_pending and lag is not None:
            _set_preconditioner_lag(self.problem, 1)
            try:
                ok = self.problem.solve(raise_on_failure=False)
            finally:
                _set_preconditioner_lag(self.problem, lag)
        else:
            ok = self.problem.solve(raise_on_failure=False)
        if ok:
            self._refresh_pending = False
        return ok

    def step(self, t: float, dt: float) -> bool:
        """Solve one step ending at `t`, of duration `dt`; return whether it converged.

        On success, every `CavityCycle` in `self.cycles` and `self.records` is
        updated. On failure `problem.reset_states()` has put the mechanics
        state back exactly as it was before this call, and nothing in
        `self.cycles`/`self.records` has been touched; a preconditioner
        refresh stays pending, so a retry (with a smaller `dt`, say) starts
        from a fresh one.
        """
        if not self._initialized:
            raise RuntimeError("CycleController.step() called before initialize()")

        for name, cyc in self.cycles.items():
            _apply_control(self._controls[name], cyc, self.params[name], t, dt)

        ok = self._solve_once()
        if not ok:
            self.problem.reset_states()
            self._refresh_pending = True
            ok = self._solve_once()
            if not ok:
                self.problem.reset_states()
                return False

        changed = False
        records: dict[str, CavityRecord] = {}
        for name, cyc in self.cycles.items():
            params = self.params[name]
            V = self._volume(name)
            P = self._pressure(name)
            ejecting = cyc.phase == Phase.EJECTION

            cyc.wdk_pressure_n, cyc.outflow_n = params.windkessel.advance(
                cyc.wdk_pressure_n,
                V,
                cyc.volume_n,
                dt,
                ejecting=ejecting,
                has_ejected=cyc.has_ejected,
            )
            cyc.dvol_n = V - cyc.volume_n
            cyc.volume_n = V
            cyc.pressure_n = P
            if cyc.phase == Phase.PRELOAD:
                cyc.end_dia_vol = V

            if _advance_phase(cyc, params, t):
                changed = True
                if cyc.phase == Phase.ISOVOLUMIC_CONTRACTION:
                    cyc.end_dia_vol = V
                elif cyc.phase == Phase.ISOVOLUMIC_RELAXATION:
                    cyc.end_sys_vol = V

            records[name] = CavityRecord(
                phase=cyc.phase,
                V=V,
                P=P,
                P_c=cyc.wdk_pressure_n,
                Q=cyc.outflow_n,
            )

        self.records = records
        if changed:
            self._refresh_pending = True
        return True
