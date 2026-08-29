"""Putting the 3D ventricle and the 0D circuit at the same operating point.

The Regazzoni circuit's parameters were tuned against its own left-ventricular
elastance, not against a mesh. Coupling a mesh to it without doing anything
about that gives a loop that is arithmetically correct and physiologically
absurd: on the ellipsoid the demos use, the ventricle fills to about 197 mL.

The reason is worth stating precisely, because the obvious reading of it is
wrong. At the volumes the circuit works at, the mesh develops about a fifth of
the pressure the circuit's elastance would -- 4 mmHg where it expects 19 -- so
it looks like a much softer ventricle. It is not: its resting `dp/dV` is the
steeper of the two. What differs is the unstressed volume. The circuit's chamber
carries no pressure at 42 mL and this mesh carries none until about 125 mL, so
the circuit spends the whole of diastole pushing against a chamber that has not
started to resist yet. Softening the material would not fix that, and stiffening
it would make matters worse.

Fixing that needs a cheap stand-in for the mesh, so the loading can be tuned
without a 3D solve in the loop. The stand-in here is not a fit: because the
mechanics problem is quasi-static, its solution is determined by the cavity
volume and the activation alone, so the cavity pressure is an honest function
``p(V, Ta)``. Sampling it on a grid and interpolating approximates only the
sampling, not the physics. A 0D run driven by that table should therefore land
on the same loop as the coupled 3D-0D run, which is worth checking and is what
:func:`Calibration.check_against` does.

That argument has two conditions, and both are worth stating because neither is
checked automatically. The mechanics must carry no history -- no inertia, no
viscoelasticity, no internal variables -- so a `DynamicProblem`, or a model with
a `Viscous` term, is outside what this can represent. And the problem must stay
on one equilibrium branch; a hyperelastic cavity can have more than one, and
`check_against` is what would catch a run that changed branches.

The calibration itself then has three knobs and three targets:

======================  ====================================================
``Ta_scale``            contractility -- multiplies the activation trace
``R_AR_SYS``            afterload -- systemic arterial resistance
``extra_volume_mL``     preload -- blood added to the systemic venous pool
======================  ====================================================

against an end-diastolic volume, an ejection fraction and a peak systolic
pressure -- and the end-diastolic volume is itself read off the measured mesh,
at the volume where filling reaches a normal pressure, rather than chosen in
advance. The circuit is closed and conserves volume, so adding blood once at
the start is enough; it is applied to ``p_VEN_SYS``, the most compliant
compartment, where it perturbs the pressures least.

The result is written to a JSON file rather than recomputed, so that runs which
are meant to be compared -- a different coupling scheme, a different time step --
start from the same operating point and differ only in what is under test.

Skipping
--------
The 3D sampling costs a few dozen static solves. Set ``PULSE_CALIBRATE=0`` to
skip it and run against the circuit's published parameters instead, accepting
the operating point that produces. An existing artifact is always reused;
``PULSE_RECALIBRATE=1`` forces it to be rebuilt.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

__all__ = [
    "Calibration",
    "PVSurface",
    "Targets",
    "RawSamples",
    "sample_pv_surface",
    "calibrate",
    "load_or_calibrate",
]

mL = 1e-6
mmHg = 133.322387415


@dataclass(slots=True)
class Targets:
    """The operating point the loading is tuned to reach.

    End-diastolic volume rather than filling pressure, because it is monotone
    in the preload knob and easy to read off a loop; ejection fraction rather
    than end-systolic volume, because it does not presume a chamber size on top
    of the one `EDV` already sets.

    Leave `EDV` as `None`, which is the default, and it is read off the mesh:
    the volume at which the measured resting curve reaches `filling_pressure`.
    That is worth preferring to a number chosen in advance. An idealized
    ellipsoid is not a person and its unloaded cavity need be nowhere near a
    person's, so a textbook end-diastolic volume asks it to fill to a pressure
    no ventricle fills to -- and the preload knob then runs to its stop trying
    to drain a circuit that was never the problem.
    """

    EDV: float | None = None  # mL; None means "ask the mesh"
    EF: float = 0.45
    p_max: float = 120.0  # mmHg
    filling_pressure: float = 8.0  # mmHg, used only when EDV is None

    def resolve(self, surface: "PVSurface") -> "Targets":
        """Fill in an `EDV` read off the resting curve of `surface`."""
        if self.EDV is not None:
            return self
        rest = surface.pressures[:, 0]
        if not (rest[0] <= self.filling_pressure <= rest[-1]):
            logger.warning(
                f"A filling pressure of {self.filling_pressure:.1f} mmHg is outside the "
                f"measured resting range ({rest[0]:.1f} to {rest[-1]:.1f} mmHg); "
                "the end-diastolic volume target is an extrapolation",
            )
        EDV = float(np.interp(self.filling_pressure, rest, surface.volumes))
        logger.info(
            f"End-diastolic volume target read off the mesh: {EDV:.1f} mL, "
            f"where the resting curve reaches {self.filling_pressure:.1f} mmHg",
        )
        return Targets(
            EDV=EDV,
            EF=self.EF,
            p_max=self.p_max,
            filling_pressure=self.filling_pressure,
        )

    @property
    def ESV(self) -> float:
        assert self.EDV is not None, "call resolve() first"
        return self.EDV * (1.0 - self.EF)


# ---------------------------------------------------------------------------
# The 3D part: measure p(V, Ta)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PVSurface:
    """Cavity pressure as a function of cavity volume and activation.

    Parameters
    ----------
    volumes : np.ndarray
        Sampled cavity volumes, ascending, in mL.
    activations : np.ndarray
        Sampled activation levels, ascending, in Pa.
    pressures : np.ndarray
        Cavity pressure in mmHg, shape ``(len(volumes), len(activations))``.
    """

    volumes: np.ndarray
    activations: np.ndarray
    pressures: np.ndarray

    def __post_init__(self) -> None:
        self.volumes = np.asarray(self.volumes, dtype=float)
        self.activations = np.asarray(self.activations, dtype=float)
        self.pressures = np.asarray(self.pressures, dtype=float)
        if not np.all(np.isfinite(self.pressures)):
            raise ValueError("The sampled pressure surface has holes in it")
        if self.pressures.shape != (len(self.volumes), len(self.activations)):
            raise ValueError("Pressure grid does not match the volume and activation axes")

    def __call__(self, volume: Any, activation: Any) -> Any:
        """Pressure in mmHg at a volume in mL and an activation in Pa.

        Bilinear, and deliberately not clamped at the edges of the grid: the
        bracketing cell is chosen with clipped indices, so a query outside the
        sampled box continues along the slope of the nearest cell. The 0D
        solver does overshoot early in a run, and a clamped surface would make
        the chamber infinitely compliant exactly where it is misbehaving.

        This is a hot path -- it runs once per right-hand side evaluation --
        so it is written out rather than delegated to an interpolator object.
        """
        V = np.asarray(volume, dtype=float)
        A = np.asarray(activation, dtype=float)
        Vg, Ag, P = self.volumes, self.activations, self.pressures

        i = np.clip(np.searchsorted(Vg, V) - 1, 0, len(Vg) - 2)
        j = np.clip(np.searchsorted(Ag, A) - 1, 0, len(Ag) - 2)
        tv = (V - Vg[i]) / (Vg[i + 1] - Vg[i])
        ta = (A - Ag[j]) / (Ag[j + 1] - Ag[j])

        out = (
            (1 - tv) * (1 - ta) * P[i, j]
            + tv * (1 - ta) * P[i + 1, j]
            + (1 - tv) * ta * P[i, j + 1]
            + tv * ta * P[i + 1, j + 1]
        )
        return float(out) if np.ndim(out) == 0 else out

    def stiffness(self, Ta_ref: float) -> dict[str, float]:
        """How stiff the chamber is, in the terms the circuit's own is stated in.

        Reported as secant slopes across the sampled volume range, at rest and
        at ``Ta_ref``, against the published ``EB`` and ``EA + EB``. Fitting the
        circuit's ``p = (EA*e + EB)*(V - V0)`` form to the surface instead is
        tempting and is a trap: the measured surface is nearly flat in volume
        under activation, and the only way that form reproduces a flat surface
        is to send ``V0`` to a few hundred negative millilitres and the
        stiffnesses to nearly zero. The fit succeeds, the residual is
        unremarkable, and the resulting ``EA`` is not comparable to the
        published one at all.

        A secant slope has no such freedom. It is the number the comparison
        wanted in the first place.
        """
        V = self.volumes
        rest = self.pressures[:, 0]
        j = int(np.argmin(np.abs(self.activations - Ta_ref)))
        active = self.pressures[:, j]
        span = V[-1] - V[0]
        return {
            "E_rest": float((rest[-1] - rest[0]) / span),
            "E_active": float((active[-1] - active[0]) / span),
            # Where the resting curve crosses zero: the chamber's own unstressed
            # volume, and the counterpart of the circuit's `V0`. Worth reporting
            # next to the stiffnesses because the two discrepancies look alike
            # from a pressure reading and are not alike at all -- a chamber can
            # develop far less pressure than expected at a given volume while
            # being the stiffer of the two, if it is simply unstressed at a much
            # larger volume.
            "V0": float(np.interp(0.0, rest, V)),
            "Ta_ref": float(Ta_ref),
            "Ta_used": float(self.activations[j]),
            "V_lo": float(V[0]),
            "V_hi": float(V[-1]),
            "p_rest_lo": float(rest[0]),
            "p_rest_hi": float(rest[-1]),
        }


@dataclass(slots=True)
class RawSamples:
    """What the sampler measured, holes and all.

    Kept apart from :class:`PVSurface` because the two answer different
    questions. This is the measurement: every grid point that was attempted,
    with a NaN wherever the mesh would not hold that volume at that tension.
    A `PVSurface` is a rectangle carved out of it, and which rectangle is a
    decision -- a lower activation ceiling buys back the dilated volumes that
    would otherwise be dropped for failing only at the top of the range.

    Caching this rather than the carved rectangle means changing that decision
    costs nothing, instead of another sweep of static solves.
    """

    volumes: np.ndarray
    activations: np.ndarray
    pressures: np.ndarray

    def trim(self, max_activation: float | None = None) -> PVSurface:
        """Carve out the largest hole-free rectangle up to `max_activation`."""
        A, P = self.activations, self.pressures
        if max_activation is not None:
            keep_A = A <= max_activation + 1e-9
            if keep_A.sum() < 2:
                raise ValueError("An activation ceiling that low leaves nothing to interpolate")
            A, P = A[keep_A], P[:, keep_A]

        keep_V = np.all(np.isfinite(P), axis=1)
        if not keep_V.all():
            logger.warning(
                f"Dropping {(~keep_V).sum()} volume(s) the mesh would not hold at every "
                f"activation up to {A[-1] * 1e-3:.0f} kPa: "
                f"{np.round(self.volumes[~keep_V], 1).tolist()} mL",
            )
        if keep_V.sum() < 2:
            raise RuntimeError(
                f"Fewer than two volumes survived. Lower the activation ceiling: {self.coverage()}",
            )
        return PVSurface(
            volumes=self.volumes[keep_V],
            activations=A,
            pressures=P[keep_V],
        )

    def coverage(self) -> str:
        """What each possible activation ceiling would buy, as a table."""
        lines = ["activation ceiling -> volumes retained"]
        for j, Ta in enumerate(self.activations):
            n = int(np.all(np.isfinite(self.pressures[:, : j + 1]), axis=1).sum())
            if n >= 2:
                span = self.volumes[np.all(np.isfinite(self.pressures[:, : j + 1]), axis=1)]
                lines.append(
                    f"  {Ta * 1e-3:6.0f} kPa -> {n:2d} volumes, {span[0]:.0f}-{span[-1]:.0f} mL",
                )
        return "\n".join(lines)


def sample_pv_surface(
    problem,
    volume_constant,
    activation,
    volumes: Sequence[float],
    activations: Sequence[float],
    *,
    comm,
    substeps: int = 10,
) -> RawSamples:
    """Measure the cavity pressure over a grid of volumes and activations.

    Parameters
    ----------
    problem : pulse.problem.StaticProblem
        A volume-controlled problem on the geometry to be coupled, built with
        the same model and boundary conditions the coupled problem will use.
        It is handed back at rest, in the middle of the reachable volume
        range, so the caller can ramp on from a converged state.
    volume_constant : dolfinx.fem.Constant
        The constant holding the prescribed cavity volume, in m^3.
    activation : pulse.Variable
        The activation the model was built with, in Pa.
    volumes : Sequence[float]
        Cavity volumes to *try*, in mL, sorted ascending on the way in. Which
        of them are actually reachable is discovered, not assumed.
    activations : Sequence[float]
        Activation levels to visit, in Pa. Sorted ascending on the way in.
    substeps : int
        How finely to ramp when a direct step between grid points fails.

    Notes
    -----
    The requested volume range is a wish. A ventricle whose unloaded cavity
    holds 125 mL cannot be made to hold 45 mL with nothing contracting -- that
    is suction hard enough to invert the mesh, and no amount of continuation
    gets there. So the first activation level is walked outward from wherever
    the problem currently sits, in both directions, and stops at the first
    volume that does not solve. That establishes the reachable range, and the
    remaining activation levels are sampled only over it.

    The rest of the grid is then filled a column at a time: each volume is
    restored to its own resting solution and the activation is marched up from
    there, so the volume never changes while the tension is up. A step that
    fails outright is retried as a ramp before being given up on, and a failure
    puts the problem back exactly where it was rather than leaving it stranded
    between two grid points.
    """
    volumes = np.sort(np.asarray(volumes, dtype=float))
    activations = np.sort(np.asarray(activations, dtype=float))
    requested = len(volumes)

    def read_pressure() -> float:
        return float(problem.cavity_pressures[0].x.array[0]) / mmHg

    def snapshot() -> dict[str, Any]:
        """Everything needed to put the problem back where it was.

        `reset_states` alone is not enough. It restores the fields from the
        previous solve, but leaves the volume and activation constants at
        whatever the failed step set them to -- so the mesh is at one operating
        point and the constants claim another. Every subsequent continuation
        then ramps from a position the problem is not actually in, which is
        what turns one hard grid point into a row of spurious failures.
        """
        state = {
            "u": problem.u.x.array.copy(),
            "cavity": [p.x.array.copy() for p in problem.cavity_pressures],
            "V": float(volume_constant.value),
            "Ta": float(activation.value.value),
        }
        if problem.is_incompressible:
            state["p"] = problem.p.x.array.copy()
        return state

    def restore(state: dict[str, Any]) -> None:
        problem.u.x.array[:] = state["u"]
        for pressure, values in zip(problem.cavity_pressures, state["cavity"]):
            pressure.x.array[:] = values
        if problem.is_incompressible:
            problem.p.x.array[:] = state["p"]
        volume_constant.value = state["V"]
        activation.assign(state["Ta"])

    def step_to(V_mL: float, Ta_Pa: float) -> bool:
        """Move to a grid point, ramping if the direct step does not converge."""
        good = snapshot()
        volume_constant.value = V_mL * mL
        activation.assign(Ta_Pa)
        if problem.solve():
            return True

        logger.debug(f"Direct step to V={V_mL:.1f} mL, Ta={Ta_Pa:.0f} Pa failed; ramping")
        restore(good)
        for frac in np.linspace(0.0, 1.0, substeps + 1)[1:]:
            volume_constant.value = good["V"] + frac * (V_mL * mL - good["V"])
            activation.assign(good["Ta"] + frac * (Ta_Pa - good["Ta"]))
            if not problem.solve():
                restore(good)
                return False
        return True

    # --- find the reachable volume range at the lowest activation ------------
    #
    # Down from the current volume first, then up from wherever that stopped,
    # in one continuous march. Doing it as two sweeps out of the middle would
    # need a jump back across the whole range between them, and deflating an
    # inflated ventricle in one step is exactly what does not converge -- which
    # would make the range look narrower than it is, for a solver reason rather
    # than a physical one.
    here = float(volume_constant.value) / mL
    start = int(np.argmin(np.abs(volumes - here)))
    Ta0 = float(activations[0])
    reachable: dict[int, float] = {}
    rest: dict[int, dict[str, Any]] = {}

    lo = start
    for i in range(start, -1, -1):
        if not step_to(float(volumes[i]), Ta0):
            logger.info(f"  V={volumes[i]:.1f} mL is below what the mesh will hold unloaded")
            break
        reachable[i] = read_pressure()
        rest[i] = snapshot()
        lo = i

    for i in range(lo, len(volumes)):
        if not step_to(float(volumes[i]), Ta0):
            logger.info(f"  V={volumes[i]:.1f} mL is above what the mesh will hold")
            break
        reachable[i] = read_pressure()
        rest[i] = snapshot()

    kept = sorted(reachable)
    if len(kept) < 2:
        raise RuntimeError(
            f"Only {len(kept)} of the requested volumes were reachable. "
            "The requested range is nowhere near the unloaded volume.",
        )
    volumes = volumes[kept]
    pressures = np.full((len(volumes), len(activations)), np.nan)
    pressures[:, 0] = [reachable[i] for i in kept]
    logger.info(
        f"Reachable volume range: {volumes[0]:.1f} to {volumes[-1]:.1f} mL "
        f"({len(volumes)} of {requested} requested points)",
    )

    # --- march the activation up at each volume in turn ----------------------
    #
    # One activation sweep per volume, each starting from that volume's own
    # resting solution, and the volume never moves while the tension is up.
    # Snaking across the grid instead -- volumes forward at one activation,
    # backward at the next -- looks tidier and costs the same, but it jumps to
    # the far end of the volume range every time the activation changes, and a
    # contracting ventricle asked to change volume by a fifth in one step does
    # not converge. That produced whole rows of failures that had nothing to do
    # with what the mesh can actually hold.
    total = len(volumes) * (len(activations) - 1)
    done = 0
    for i in range(len(volumes)):
        restore(rest[kept[i]])
        for j, Ta in enumerate(activations[1:], start=1):
            if step_to(float(volumes[i]), float(Ta)):
                pressures[i, j] = read_pressure()
            else:
                # Higher tension at the same volume is only a longer reach from
                # here, so stop climbing this column rather than pretend the
                # next rung is easier.
                logger.warning(
                    f"No solution at V={volumes[i]:.1f} mL, Ta={Ta * 1e-3:.0f} kPa; "
                    "abandoning this volume",
                )
                done += len(activations) - j
                break
            done += 1
        if done % 10 < len(activations) or done == total:
            logger.info(f"  sampled {done}/{total} points")

    # Hand the problem back at rest, in the middle of the reachable range,
    # rather than wherever the last column happened to stop. The caller's next
    # move is a ramp to some operating volume, and starting that from a fully
    # contracted, maximally dilated state means unwinding a hundred millilitres
    # and eighty kilopascals at once -- which does not converge, and would make
    # a successful sweep look like a failed one. A snapshot costs nothing and
    # every reachable volume has a resting solution already.
    middle = kept[len(kept) // 2]
    restore(rest[middle])
    logger.info(f"Leaving the problem at rest at {volumes[len(kept) // 2]:.1f} mL")

    # `volume` on a distributed mesh is a partial sum, so make sure everyone
    # agrees on what was measured before it is written down.
    pressures = comm.bcast(pressures, root=0)
    volumes = comm.bcast(volumes, root=0)
    samples = RawSamples(volumes=volumes, activations=activations, pressures=pressures)

    # Reported, not enforced. A cavity that does not stiffen with volume has a
    # falling pressure-volume relation over part of its range, which is a real
    # property of some ventricle-and-active-model combinations and not a
    # sampling mistake -- but it is worth knowing about before a closed loop is
    # driven by it.
    slope = np.diff(pressures, axis=0)
    measured = np.isfinite(slope)
    falling = measured & (slope <= 0)
    if falling.any():
        worst = int(np.nanargmin(np.where(measured, slope, np.inf)) % slope.shape[1])
        logger.warning(
            f"Pressure falls with volume at {falling.sum()} of {measured.sum()} sampled "
            f"intervals, worst around {activations[worst] * 1e-3:.0f} kPa. The chamber "
            "has a non-monotone pressure-volume relation, which is a real property of "
            "some ventricle-and-active-model pairs rather than a sampling mistake -- "
            "but a closed loop driven by it will not behave like a textbook one.",
        )
    logger.info("Sampling coverage:\n" + samples.coverage())
    return samples


# ---------------------------------------------------------------------------
# The 0D part: tune the loading against the measured surface
# ---------------------------------------------------------------------------


def _run_circuit(
    surface: PVSurface,
    activation_at: Callable[[float], float],
    parameters: dict[str, Any],
    *,
    Ta_scale: float,
    R_AR_SYS: float,
    extra_volume_mL: float,
    num_beats: int,
    dt: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Run the circuit with the measured chamber to its limit cycle.

    Returns the state at the start of the final beat, and the final beat.
    """
    import copy

    from circulation import base, regazzoni2020

    params = copy.deepcopy(parameters)
    params["circulation"]["SYS"]["R_AR"] = R_AR_SYS

    def p_LV(V, t):
        return surface(V, Ta_scale * activation_at(float(t)))

    model = regazzoni2020.Regazzoni2020(parameters=params, p_LV=p_LV, add_units=False)

    initial = base.remove_units(model.default_initial_conditions())
    initial = {k: float(v) for k, v in initial.items()}
    initial["p_VEN_SYS"] += extra_volume_mL / params["circulation"]["SYS"]["C_VEN"]

    model.solve(num_beats=max(num_beats - 1, 1), dt=dt, initial_state=initial)
    state = dict(zip(model.state_names(), (float(x) for x in model.state)))
    final_beat = model.solve(num_beats=1, dt=dt)
    return state, final_beat


def _beat_metrics(beat: dict[str, np.ndarray]) -> dict[str, float]:
    V = np.asarray(beat["V_LV"], dtype=float)
    p = np.asarray(beat["p_LV"], dtype=float)
    i_ED = int(np.argmax(V))
    return {
        "EDV": float(V[i_ED]),
        "ESV": float(np.min(V)),
        "EF": float((V[i_ED] - np.min(V)) / V[i_ED]),
        "p_max": float(np.max(p)),
        # End-diastole is taken at maximum volume, which is mitral valve
        # closure. Contraction has begun by then, so `p_ED` is not the filling
        # pressure -- `p_min`, the low point of diastole, is closer to that.
        "p_ED": float(p[i_ED]),
        "p_min": float(np.min(p)),
    }


@dataclass
class Calibration:
    """Everything a coupled run needs to start at a sensible operating point."""

    Ta_scale: float
    R_AR_SYS: float
    extra_volume_mL: float
    initial_state: dict[str, float]
    achieved: dict[str, float]
    targets: dict[str, float]
    stiffness: dict[str, float]
    published_chamber: dict[str, float]
    surface_volumes: list[float]
    surface_activations: list[float]
    surface_pressures: list[list[float]]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def p_LV_ED(self) -> float:
        """End-diastolic pressure of the calibrated loop, in mmHg.

        Reported, not consumed. The prestress target that defines the unloaded
        reference configuration has to be fixed *before* the mesh can be
        sampled, so it cannot come from here without an outer iteration.
        """
        return float(self.achieved["p_ED"])

    @property
    def surface(self) -> PVSurface:
        return PVSurface(
            volumes=np.asarray(self.surface_volumes),
            activations=np.asarray(self.surface_activations),
            pressures=np.asarray(self.surface_pressures),
        )

    def circulation_parameters(self, base: dict[str, Any]) -> dict[str, Any]:
        """`base` with the calibrated loading applied."""
        import copy

        params = copy.deepcopy(base)
        params["circulation"]["SYS"]["R_AR"] = self.R_AR_SYS
        return params

    def save(self, path: Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        logger.info(f"Wrote calibration to {path}")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        data.pop("_interp", None)
        return cls(**data)

    def summary(self) -> str:
        t, a = self.targets, self.achieved
        st, pub = self.stiffness, self.published_chamber
        lines = [
            "Calibrated operating point:",
            f"  EDV    {a['EDV']:7.1f} mL   (target {t['EDV']:.1f})",
            f"  ESV    {a['ESV']:7.1f} mL   (implied {t['ESV']:.1f})",
            f"  EF     {a['EF'] * 100:7.1f} %    (target {t['EF'] * 100:.1f})",
            f"  p_max  {a['p_max']:7.1f} mmHg (target {t['p_max']:.1f})",
            f"  p_ED   {a['p_ED']:7.1f} mmHg (at mitral valve closure)",
            f"  p_min  {a['p_min']:7.1f} mmHg (filling pressure)",
            "Knobs:",
            f"  Ta_scale        {self.Ta_scale:.3f}",
            f"  R_AR_SYS        {self.R_AR_SYS:.3f} mmHg s/mL",
            f"  extra blood     {self.extra_volume_mL:+.1f} mL",
            f"Chamber stiffness over {st['V_lo']:.0f}-{st['V_hi']:.0f} mL, "
            "measured vs the circuit's own:",
            f"  dp/dV at rest    {st['E_rest']:7.3f} vs {pub['EB']:7.3f} mmHg/mL",
            f"  dp/dV activated  {st['E_active']:7.3f} vs {pub['EA'] + pub['EB']:7.3f} mmHg/mL",
            f"  unstressed volume{st['V0']:7.1f} vs {pub['V0']:7.1f} mL",
            f"  resting pressure spans {st['p_rest_lo']:.1f} to {st['p_rest_hi']:.1f} mmHg",
        ]
        return "\n".join(lines)

    def check_against(
        self,
        volumes: Sequence[float],
        activations: Sequence[float],
        pressures: Sequence[float],
    ) -> dict[str, float]:
        """Compare the surrogate against a coupled 3D-0D run, point by point.

        The mechanics problem is quasi-static, so its cavity pressure really is
        a function of volume and activation. A coupled run's own ``(V, Ta, p)``
        triples therefore have to sit on the sampled surface, and how far off
        they are is interpolation error and nothing else. A large number here
        means either the grid is too coarse or the two runs are not solving the
        same mechanics problem.

        Returns the worst and rms error in mmHg, and how far the loop strayed
        outside the sampled volume range, where the surface is extrapolated.
        """
        V = np.asarray(volumes, dtype=float)
        A = np.asarray(activations, dtype=float)
        p = np.asarray(pressures, dtype=float)
        err = np.asarray(self.surface(V, A)) - p
        lo, hi = self.surface_volumes[0], self.surface_volumes[-1]
        return {
            "max_mmHg": float(np.max(np.abs(err))),
            "rms_mmHg": float(np.sqrt(np.mean(err**2))),
            "outside_mL": float(max(0.0, lo - V.min(), V.max() - hi)),
        }


def calibrate(
    surface: PVSurface,
    activation_at: Callable[[float], float],
    base_parameters: dict[str, Any],
    *,
    targets: Targets | None = None,
    Ta_ref: float,
    num_beats: int = 12,
    dt: float = 1e-3,
    provenance: dict[str, Any] | None = None,
) -> Calibration:
    """Tune contractility, afterload and preload against `targets`.

    Pure 0D: the mesh enters only through `surface`, so this costs seconds and
    no static solves.
    """
    targets = (targets or Targets()).resolve(surface)
    published = base_parameters["chambers"]["LV"]

    # Contractility and afterload are scale-like and strictly positive, so they
    # are searched in logarithms; the blood volume offset is signed and is not.
    # It is carried in hundreds of millilitres so that all three knobs are
    # order one, which is what the finite-difference Jacobian steps assume.
    x0 = np.array(
        [
            np.log(1.0),
            np.log(base_parameters["circulation"]["SYS"]["R_AR"]),
            0.0,
        ],
    )
    Ta_scale_max = float(surface.activations[-1]) / Ta_ref
    bounds = (
        np.array([np.log(0.2), np.log(0.05), -8.0]),
        np.array([np.log(max(Ta_scale_max, 1.1)), np.log(10.0), 8.0]),
    )

    calls = {"n": 0}

    def unpack(x):
        return float(np.exp(x[0])), float(np.exp(x[1])), 100.0 * float(x[2])

    def residual(x):
        Ta_scale, R_AR_SYS, extra = unpack(x)
        _, beat = _run_circuit(
            surface,
            activation_at,
            base_parameters,
            Ta_scale=Ta_scale,
            R_AR_SYS=R_AR_SYS,
            extra_volume_mL=extra,
            num_beats=num_beats,
            dt=dt,
        )
        m = _beat_metrics(beat)
        calls["n"] += 1
        logger.info(
            f"  [{calls['n']:3d}] Ta x{Ta_scale:.2f}  R_AR={R_AR_SYS:.3f}  "
            f"dV={extra:+7.1f} mL  ->  EDV={m['EDV']:6.1f}  EF={m['EF'] * 100:5.1f}%  "
            f"p_max={m['p_max']:6.1f}",
        )
        return np.array(
            [
                (m["EDV"] - targets.EDV) / targets.EDV,
                (m["EF"] - targets.EF) / targets.EF,
                (m["p_max"] - targets.p_max) / targets.p_max,
            ],
        )

    logger.info("Calibrating the loading against the measured chamber...")
    sol = least_squares(residual, x0, bounds=bounds, diff_step=0.05, xtol=1e-4, ftol=1e-4)
    Ta_scale, R_AR_SYS, extra = unpack(sol.x)

    # A knob resting on its bound means the targets were not reachable and the
    # optimizer stopped at the edge of what it was allowed to try, which is a
    # different situation from having converged.
    for name, x, lo, hi in zip(
        ("Ta_scale", "R_AR_SYS", "extra_volume"),
        sol.x,
        bounds[0],
        bounds[1],
    ):
        span = hi - lo
        if x - lo < 0.01 * span or hi - x < 0.01 * span:
            logger.warning(
                f"{name} settled on its bound. The targets are out of reach for this "
                "mesh with this loading; widen the range or accept the operating point.",
            )

    state, beat = _run_circuit(
        surface,
        activation_at,
        base_parameters,
        Ta_scale=Ta_scale,
        R_AR_SYS=R_AR_SYS,
        extra_volume_mL=extra,
        num_beats=num_beats,
        dt=dt,
    )

    return Calibration(
        Ta_scale=Ta_scale,
        R_AR_SYS=R_AR_SYS,
        extra_volume_mL=extra,
        initial_state=state,
        achieved=_beat_metrics(beat),
        targets={
            "EDV": targets.EDV,
            "EF": targets.EF,
            "ESV": targets.ESV,
            "p_max": targets.p_max,
        },
        stiffness=surface.stiffness(Ta_ref),
        published_chamber={k: float(published[k]) for k in ("EA", "EB", "V0")},
        surface_volumes=[float(v) for v in surface.volumes],
        surface_activations=[float(a) for a in surface.activations],
        surface_pressures=[[float(p) for p in row] for row in surface.pressures],
        provenance=dict(provenance or {}, num_beats=num_beats, dt=dt),
    )


# ---------------------------------------------------------------------------
# What the demos call
# ---------------------------------------------------------------------------


def wanted() -> bool:
    """Whether calibration is switched on. See the module docstring."""
    value = os.getenv("PULSE_CALIBRATE", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def forced() -> bool:
    value = os.getenv("PULSE_RECALIBRATE", "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def resample() -> bool:
    value = os.getenv("PULSE_RESAMPLE", "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _cached_surface(path: Path, build: Callable[[], RawSamples], comm) -> RawSamples:
    """Measure the mesh once.

    The two halves of a calibration cost about two orders of magnitude apart --
    dozens of static solves against a couple of minutes of 0D runs -- so they
    are cached apart. Re-tuning the loading, or moving the targets, should not
    re-measure the ventricle.
    """
    if path.exists() and not resample():
        logger.info(f"Reusing the sampled surface in {path}")
        data = np.load(path)
        return RawSamples(
            volumes=data["volumes"],
            activations=data["activations"],
            pressures=data["pressures"],
        )
    surface = build()
    if comm.rank == 0:
        np.savez(
            path,
            volumes=surface.volumes,
            activations=surface.activations,
            pressures=surface.pressures,
        )
        logger.info(f"Wrote the sampled surface to {path}")
    comm.barrier()
    return surface


def load_or_calibrate(
    path: Path,
    *,
    build_surface: Callable[[], RawSamples],
    activation_at: Callable[[float], float],
    base_parameters: dict[str, Any],
    Ta_ref: float,
    comm,
    targets: Targets | None = None,
    max_activation: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> Calibration | None:
    """Read the cached calibration, or build it, or decline to.

    Returns `None` if calibration is switched off and nothing is cached, which
    the caller should read as "run against the published parameters and expect
    the operating point to be off".
    """
    path = Path(path)
    if path.exists() and not (forced() or resample()):
        logger.info(f"Reusing the calibration in {path}")
        return Calibration.load(path)
    if not wanted():
        logger.warning(
            "PULSE_CALIBRATE=0 and no cached calibration: running against the "
            "circuit's published parameters. The mesh and the circuit will not "
            "share an operating point.",
        )
        return None

    samples = _cached_surface(path.with_suffix(".surface.npz"), build_surface, comm)
    surface = samples.trim(max_activation)
    calibration = None
    if comm.rank == 0:
        calibration = calibrate(
            surface,
            activation_at,
            base_parameters,
            targets=targets,
            Ta_ref=Ta_ref,
            provenance=provenance,
        )
        calibration.save(path)
    comm.barrier()
    return Calibration.load(path)
