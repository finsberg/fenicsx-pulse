"""The Alya-style five-phase cardiac cycle (`pulse.cycle`), on top of `CavityControl`.

`test_ejection_law_is_backward_euler_windkessel` needs no FEM: it checks that
`Windkessel.ejection_law`'s affine coefficients, evaluated at the volume Newton
eventually converges to, reproduce exactly the `(P_c, Q)` `Windkessel.advance`
computes from that same volume -- the two have to agree, since `ejection_law`
is what the mechanics solve enforces during ejection and `advance` is how the
cycle then updates its own Windkessel state from the result.

`test_phases_run_in_order_on_lv_ellipsoid` is the real gate: a one-cavity
`DynamicProblem` on pulse's own LV ellipsoid, driven through a full em_tref7
LV cycle by `CycleController`, checked against the five-phase order the whole
design is for.

The other tests use a small unit-cube problem (the same ENDO/FIXED tagging
`test_cavity_control.py` uses, for the same reason: it keeps the cavity's rim
in the x = 0 plane through the origin, where the divergence-theorem volume is
exactly the enclosed volume) -- fast enough to not need `@pytest.mark.slow`.
"""

from __future__ import annotations

import dataclasses
import json
import math

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import pytest

import pulse
import pulse.cycle as cycle
from pulse.circulation import mL, mmHg

cardiac_geometries = pytest.importorskip("cardiac_geometries")

#: em_tref7's LV cycle (third-party/physcardems/configs/elife/em_tref7.toml,
#: [circulation.lv] and [circulation.lv.windkessel], `period` from `[time]`
#: `pcl_ms`), converted to SI with pulse's own `mL`/`mmHg`.
LV_PERIOD = 0.8


def lv_cycle_params() -> cycle.CycleParams:
    return cycle.CycleParams(
        t_zero=0.05,
        preload_pressure=500.0,
        t_end_diastole=0.12,
        p_end_diastole=1000.0,
        p_fill=500.0,
        period=LV_PERIOD,
        windkessel=cycle.Windkessel(
            p_init=9000.0,
            compliance=1.5 * mL / mmHg,
            resistance=1.1 * mmHg / mL,
            characteristic_impedance=0.03 * mmHg / mL,
        ),
        filling=cycle.PrescribedInflow(rate=0.046 * mL / 1e-3),
    )


def test_ejection_law_is_backward_euler_windkessel():
    C = 1.5 * mL / mmHg
    R_p = 1.1 * mmHg / mL
    R_c = 0.03 * mmHg / mL
    P_c = 9e3
    V_n = 120 * mL
    V = 118 * mL
    dt = 2e-3

    # The test's own backward-Euler Windkessel update, independent of `Windkessel`.
    D = 1.0 + dt / (R_p * C)
    Q = -(V - V_n) / dt
    P_c_new = (P_c + dt * Q / C) / D
    P_v = P_c_new + R_c * Q

    wk = cycle.Windkessel(p_init=P_c, compliance=C, resistance=R_p, characteristic_impedance=R_c)
    A, B = wk.ejection_law(P_c, V_n, dt)
    assert A + B * V == pytest.approx(P_v, rel=1e-12)

    result = wk.advance(P_c, V, V_n, dt, ejecting=True, has_ejected=True)
    assert result == pytest.approx((P_c_new, Q), rel=1e-12)


# --- A small, fast controlled-cavity problem (ENDO on 5 faces of a unit cube,
# FIXED on the x = 0 face -- see test_cavity_control.py's module docstring for
# why this tagging, rather than HeartGeometry's usual Marker/locate_entities
# path, is what makes the divergence-theorem volume exact). ---


@pytest.fixture
def cube_geometry():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    mesh.topology.create_connectivity(2, 3)
    exterior = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, 2, exterior)
    is_fixed = np.isclose(midpoints[:, 0], 0.0)
    values = np.where(is_fixed, 2, 1).astype(np.int32)
    order = np.argsort(exterior)
    facet_tags = dolfinx.mesh.meshtags(
        mesh,
        2,
        exterior[order].astype(np.int32),
        values[order],
    )
    return pulse.HeartGeometry(
        mesh=mesh,
        facet_tags=facet_tags,
        markers={"ENDO": (1, 2), "FIXED": (2, 2)},
    )


def _cube_dirichlet_bc(geometry):
    def bc(V):
        facets = geometry.facet_tags.find(2)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

    return bc


def _cube_dynamic_problem(geometry, raise_on_failure=False):
    model = pulse.CardiacModel(
        material=pulse.NeoHookean(mu=pulse.Variable(10.0, "kPa")),
        active=pulse.Passive(),
        compressibility=pulse.compressibility.Compressible2(),
    )
    control = pulse.problem.CavityControl(geometry.mesh)
    parameters: dict = {
        "mesh_unit": "m",
        "rho": pulse.Variable(1e3, "kg/m^3"),
        "dt": pulse.Variable(2e-3, "s"),
    }
    if raise_on_failure:
        parameters["raise_on_failure"] = True
    problem = pulse.problem.DynamicProblem(
        model=model,
        geometry=geometry,
        bcs=pulse.BoundaryConditions(dirichlet=(_cube_dirichlet_bc(geometry),)),
        cavities=[pulse.problem.Cavity(marker="ENDO", control=control)],
        parameters=parameters,
    )
    return problem


def _lag_key(problem) -> str:
    return f"{problem.problem.solver.getOptionsPrefix() or ''}snes_lag_preconditioner"


def _record_lag_calls(monkeypatch) -> list[int]:
    """Record every lag `CycleController` sets, still setting it for real."""
    calls: list[int] = []
    real = cycle._set_preconditioner_lag

    def recording(problem, lag):
        calls.append(lag)
        real(problem, lag)

    monkeypatch.setattr(cycle, "_set_preconditioner_lag", recording)
    return calls


def _take_converged_steps(controller, n: int, dt: float = 2e-3) -> float:
    """`n` converged PRELOAD steps from t = 0, so the state is no longer all zeros."""
    t = 0.0
    for _ in range(n):
        t += dt
        assert controller.step(t=t, dt=dt) is True
    return t


def _snapshot(problem, controller):
    return {
        "u": problem.u.x.array.copy(),
        "u_old": problem.u_old.x.array.copy(),
        "v_old": problem.v_old.x.array.copy(),
        "a_old": problem.a_old.x.array.copy(),
        "p": [p.x.array.copy() for p in problem.cavity_pressures],
        "cycles": {name: dataclasses.replace(c) for name, c in controller.cycles.items()},
        "records": {name: dataclasses.replace(r) for name, r in controller.records.items()},
    }


def _demand_infeasible_ivc(problem, controller) -> None:
    """One Newton iteration, to reach an IVC volume of half the current one."""
    problem.problem.solver.setTolerances(max_it=1)
    cyc = controller.cycles["ENDO"]
    cyc.phase = cycle.Phase.ISOVOLUMIC_CONTRACTION
    cyc.end_dia_vol = 0.5 * cyc.volume_n


def _assert_restored(problem, controller, before) -> None:
    assert np.array_equal(problem.u.x.array, before["u"])
    assert np.array_equal(problem.u_old.x.array, before["u_old"])
    assert np.array_equal(problem.v_old.x.array, before["v_old"])
    assert np.array_equal(problem.a_old.x.array, before["a_old"])
    for p_before, p in zip(before["p"], problem.cavity_pressures):
        assert np.array_equal(p.x.array, p_before)
    assert controller.cycles == before["cycles"]
    assert controller.records == before["records"]


def test_failed_step_restores_state_bit_for_bit(cube_geometry):
    """`step` must return `False` and leave the mechanics state and every
    `CavityCycle` field exactly as they were, when a demanded IVC volume the
    Newton solve is given only one iteration to reach cannot be met.

    Two converged steps come first, so that everything compared is non-zero
    beforehand: from an all-zero state a rollback that zeroed the fields
    instead of restoring them would pass too.
    """
    problem = _cube_dynamic_problem(cube_geometry)
    controller = cycle.CycleController(problem, {"ENDO": lv_cycle_params()})
    controller.initialize(t0=0.0)
    t = _take_converged_steps(controller, 2)

    _demand_infeasible_ivc(problem, controller)
    before = _snapshot(problem, controller)
    for name in ("u", "u_old", "v_old", "a_old"):
        assert np.any(before[name] != 0.0), name
    assert all(np.all(p != 0.0) for p in before["p"])

    ok = controller.step(t=t + 2e-3, dt=2e-3)

    assert ok is False
    _assert_restored(problem, controller, before)


def test_failed_step_does_not_raise_or_leak_the_lag_under_raise_on_failure(
    cube_geometry,
    monkeypatch,
):
    """With `parameters["raise_on_failure"]`, `problem.solve()` would raise on
    the failed solve; `step` must still return `False` with the state restored,
    must leave no temporary lag key in the (process-global) options database,
    and must keep the refresh pending, so the next step starts from a fresh
    preconditioner.
    """
    lag = 5
    problem = _cube_dynamic_problem(cube_geometry, raise_on_failure=True)
    controller = cycle.CycleController(
        problem,
        {"ENDO": lv_cycle_params()},
        preconditioner_lag=lag,
    )
    controller.initialize(t0=0.0)
    calls = _record_lag_calls(monkeypatch)
    t = _take_converged_steps(controller, 1)

    _demand_infeasible_ivc(problem, controller)
    before = _snapshot(problem, controller)
    calls.clear()

    ok = controller.step(t=t + 2e-3, dt=2e-3)

    assert ok is False
    _assert_restored(problem, controller, before)
    # The first attempt runs at the steady lag; only the retry refreshes.
    assert calls == [1, lag]
    assert _lag_key(problem) not in PETSc.Options()

    # Retry the step, now feasibly: back in PRELOAD with Newton's budget back.
    problem.problem.solver.setTolerances(max_it=50)
    controller.cycles["ENDO"].phase = cycle.Phase.PRELOAD
    calls.clear()
    assert controller.step(t=t + 1e-3, dt=1e-3) is True
    assert calls == [1, lag]
    assert _lag_key(problem) not in PETSc.Options()


def test_step_before_initialize_raises(cube_geometry):
    problem = _cube_dynamic_problem(cube_geometry)
    controller = cycle.CycleController(problem, {"ENDO": lv_cycle_params()})
    with pytest.raises(RuntimeError):
        controller.step(t=2e-3, dt=2e-3)


def test_unknown_cavity_name_raises_key_error(cube_geometry):
    problem = _cube_dynamic_problem(cube_geometry)
    with pytest.raises(KeyError):
        cycle.CycleController(problem, {"NOT_A_CAVITY": lv_cycle_params()})


def test_preconditioner_lag_is_refreshed_then_restored_and_not_leaked(cube_geometry, monkeypatch):
    """The first solve, and the first solve after a phase switch, run at lag 1
    and then restore `preconditioner_lag`; any other solve leaves the lag
    alone. The temporary options key must never be left in the options
    database (`PETSc.Options()` is process-global, so a leaked key would reach
    every other SNES solve in the process, not just this one).

    PRELOAD is shortened to two steps here, so the switch into IVC comes at
    t = 4 ms; IVC then holds (the cube's passive pressure stays far below the
    Windkessel's 9 kPa).
    """
    lag = 5
    params = dataclasses.replace(lv_cycle_params(), t_zero=2e-3, t_end_diastole=4e-3)
    problem = _cube_dynamic_problem(cube_geometry)
    controller = cycle.CycleController(problem, {"ENDO": params}, preconditioner_lag=lag)
    controller.initialize(t0=0.0)
    calls = _record_lag_calls(monkeypatch)
    key = _lag_key(problem)
    assert key not in PETSc.Options()

    expected = [
        # (t, lag calls during the step, phase after the step)
        (2e-3, [1, lag], cycle.Phase.PRELOAD),  # the first solve
        (4e-3, [], cycle.Phase.ISOVOLUMIC_CONTRACTION),  # a plain solve; switches
        (6e-3, [1, lag], cycle.Phase.ISOVOLUMIC_CONTRACTION),  # the solve after the switch
        (8e-3, [], cycle.Phase.ISOVOLUMIC_CONTRACTION),  # a plain solve
    ]
    for t, expected_calls, expected_phase in expected:
        calls.clear()
        assert controller.step(t=t, dt=2e-3) is True
        assert calls == expected_calls, t
        assert controller.cycles["ENDO"].phase == expected_phase, t
        assert key not in PETSc.Options()


# --- Restart: the problem's restart Functions and metadata, and the controller's state. ---


def test_restart_function_names_of_a_controlled_cavity_problem(cube_geometry):
    """The cube is compressible, with one controlled cavity: no ``p``, no rigid
    body, no circuit. The Functions are the problem's own, not copies, since a
    restart writes into them in place.
    """
    problem = _cube_dynamic_problem(cube_geometry)

    functions = problem.restart_functions()

    assert [name for name, _ in functions] == [
        "mechanics_u",
        "mechanics_u_old",
        "mechanics_cavity_pressure_ENDO",
        "mechanics_cavity_pressure_ENDO_old",
        "mechanics_v_old",
        "mechanics_a_old",
    ]
    expected = [
        problem.u,
        problem.u_old,
        problem.cavity_pressures[0],
        problem.cavity_pressures_old[0],
        problem.v_old,
        problem.a_old,
    ]
    assert all(f is g for (_, f), g in zip(functions, expected))
    assert problem.restart_metadata() == {}


def test_load_restart_metadata_refuses_a_circulation_the_problem_lacks(cube_geometry):
    problem = _cube_dynamic_problem(cube_geometry)
    with pytest.raises(ValueError, match="circulation"):
        problem.load_restart_metadata({"circulation_steps": 2})


def test_cycle_restart_is_bit_identical(cube_geometry):
    """Restoring into a fresh problem and controller and stepping on must match
    stepping on uninterrupted, bit for bit.

    PRELOAD is shortened to two steps, so the restart point is the switch into
    IVC: the saved phase is not the initial one, and the steps after the
    restart run under a different constraint from the steps before it.
    """
    params = {"ENDO": dataclasses.replace(lv_cycle_params(), t_zero=2e-3, t_end_diastole=4e-3)}
    dt = 2e-3

    a = _cube_dynamic_problem(cube_geometry)
    a_ctl = cycle.CycleController(a, params)
    a_ctl.initialize(0.0)
    t = _take_converged_steps(a_ctl, 2, dt=dt)
    assert a_ctl.cycles["ENDO"].phase == cycle.Phase.ISOVOLUMIC_CONTRACTION

    arrays = [f.x.array.copy() for _, f in a.restart_functions()]
    saved = json.loads(json.dumps(a_ctl.state_dict()))
    metadata = json.loads(json.dumps(a.restart_metadata()))

    b = _cube_dynamic_problem(cube_geometry)
    b_ctl = cycle.CycleController(b, params)
    b_ctl.initialize(0.0)
    for (_, f), values in zip(b.restart_functions(), arrays):
        f.x.array[:] = values
    b_ctl.load_state_dict(saved)
    b.load_restart_metadata(metadata)

    for _ in range(2):
        t += dt
        assert a_ctl.step(t=t, dt=dt) is True
        assert b_ctl.step(t=t, dt=dt) is True

    for (name, f), (_, g) in zip(a.restart_functions(), b.restart_functions()):
        assert np.any(f.x.array != 0.0), name
        assert np.array_equal(f.x.array, g.x.array), name
    assert a_ctl.state_dict() == b_ctl.state_dict()


def test_state_dict_is_json_and_covers_every_field(cube_geometry):
    """Every `CavityCycle`/`CavityRecord` field, the phase as an int, and a JSON
    round trip that changes nothing: floats stay floats, ints stay ints."""
    problem = _cube_dynamic_problem(cube_geometry)
    controller = cycle.CycleController(problem, {"ENDO": lv_cycle_params()})
    controller.initialize(0.0)
    _take_converged_steps(controller, 1)

    state = controller.state_dict()

    assert set(state) == {"initialized", "refresh_pending", "cycles", "records"}
    cyc, record = state["cycles"]["ENDO"], state["records"]["ENDO"]
    assert set(cyc) == {f.name for f in dataclasses.fields(cycle.CavityCycle)}
    assert set(record) == {f.name for f in dataclasses.fields(cycle.CavityRecord)}
    assert type(cyc["phase"]) is int and type(record["phase"]) is int
    assert type(cyc["last_phase_change"]) is float
    assert type(cyc["n_beats"]) is int
    assert type(cyc["has_ejected"]) is bool

    round_tripped = json.loads(json.dumps(state))
    assert round_tripped == state
    assert all(type(round_tripped["cycles"]["ENDO"][k]) is type(v) for k, v in cyc.items())

    controller.load_state_dict(round_tripped)
    assert controller.cycles["ENDO"].phase is cycle.Phase.PRELOAD
    assert isinstance(controller.records["ENDO"], cycle.CavityRecord)
    assert controller.records["ENDO"].phase is cycle.Phase.PRELOAD
    assert controller.state_dict() == state


def test_load_state_dict_refuses_other_cavities(cube_geometry):
    problem = _cube_dynamic_problem(cube_geometry)
    controller = cycle.CycleController(problem, {"ENDO": lv_cycle_params()})
    controller.initialize(0.0)
    before = controller.state_dict()

    other = json.loads(json.dumps(before))
    other["cycles"] = {"LV": other["cycles"].pop("ENDO")}
    other["records"] = {"LV": other["records"].pop("ENDO")}

    with pytest.raises(KeyError, match=r"\['ENDO', 'LV'\]"):
        controller.load_state_dict(other)
    # Refused before anything was changed.
    assert controller.state_dict() == before


def test_load_state_dict_keeps_a_pending_refresh(cube_geometry):
    """A pending refresh is never cleared by a load: a fresh solver has no
    factorization to reuse. A pending refresh in the loaded state is kept too."""
    problem = _cube_dynamic_problem(cube_geometry)
    controller = cycle.CycleController(problem, {"ENDO": lv_cycle_params()})
    controller.initialize(0.0)
    _take_converged_steps(controller, 1)
    settled = controller.state_dict()
    assert settled["refresh_pending"] is False

    fresh = cycle.CycleController(_cube_dynamic_problem(cube_geometry), {"ENDO": lv_cycle_params()})
    assert fresh.state_dict()["refresh_pending"] is True
    fresh.load_state_dict(settled)
    assert fresh.state_dict()["refresh_pending"] is True

    controller.load_state_dict({**settled, "refresh_pending": True})
    assert controller.state_dict()["refresh_pending"] is True


# --- The real gate: a full LV cycle on pulse's own ellipsoid. ---


@pytest.fixture(scope="module")
def ellipsoid_geo(tmp_path_factory):
    """Same fixture parameters as `test_circulation_coupling.py`'s `geo`."""
    geodir = tmp_path_factory.mktemp("lv_cycle")
    comm = MPI.COMM_WORLD
    if not (geodir / "mesh.xdmf").exists():
        comm.barrier()
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=geodir,
            create_fibers=True,
            fiber_space="Quadrature_6",
            r_short_endo=0.025,
            r_short_epi=0.035,
            r_long_endo=0.09,
            r_long_epi=0.097,
            psize_ref=0.05,
            mu_apex_endo=-math.pi,
            mu_base_endo=-math.acos(5 / 17),
            mu_apex_epi=-math.pi,
            mu_base_epi=-math.acos(5 / 20),
            comm=comm,
            fiber_angle_epi=-60,
            fiber_angle_endo=60,
        )
    return cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)


def _twitch(t: float) -> float:
    """Unit twitch shape (simcardemsx's `tests/conftest.py:_twitch`), rescaled to seconds:

    0 until 5 ms after onset, peaking at 1 at 25 ms after onset.
    """
    tau = max(t - 0.005, 0.0)
    return (tau / 0.02) * math.exp(1.0 - tau / 0.02)


#: kPa. The Laplace estimate P ~= 0.8 Ta for this ellipsoid puts a 60 kPa
#: peak well above what IVC needs to open the valve at the Windkessel's
#: 9 kPa (Ta ~= 11.25 kPa). Measured: max P 15.5 kPa, EDV 181.0 mL, ESV
#: 148.9 mL, FILLING from t = 0.270 s.
TMAX_KPA = 60.0


@pytest.mark.slow
def test_phases_run_in_order_on_lv_ellipsoid(ellipsoid_geo):
    geo = ellipsoid_geo
    geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore[arg-type]

    Ta = pulse.Variable(dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    # Damped, as a heart model is. Undamped, the wall rings after the switch
    # into IVR's volume constraint: the cavity pressure saw-tooths by up to
    # 1.6 kPa per step, and at dt = 2 ms Newton needs up to 72 iterations on
    # one IVR step and diverges on it on some PETSc builds. With `Viscous`
    # the IVR pressure falls monotonically and no step needs more than 4.
    model = pulse.CardiacModel(
        material=material,
        active=pulse.ActiveStress(geo.f0, activation=Ta),
        compressibility=pulse.Compressible(),
        viscoelasticity=pulse.Viscous(),
    )
    control = pulse.problem.CavityControl(geo.mesh)

    dt = 2e-3
    problem = pulse.problem.DynamicProblem(
        model=model,
        geometry=geometry,
        cavities=[pulse.problem.Cavity(marker="ENDO", control=control)],
        parameters={
            "base_bc": pulse.problem.BaseBC.fixed,
            "mesh_unit": "m",
            "rho": pulse.Variable(1e3, "kg/m^3"),
            "dt": pulse.Variable(dt, "s"),
        },
    )

    controller = cycle.CycleController(problem, {"ENDO": lv_cycle_params()})
    controller.initialize(t0=0.0)

    # One entry per step: (t, the phase in force *during* the step, V, P).
    # The phase is read before `step`, which transitions it after solving.
    history: list[tuple[float, cycle.Phase, float, float]] = []
    t = 0.0
    n_steps = int(round(LV_PERIOD / dt))
    for _ in range(n_steps):
        t_new = t + dt
        Ta.assign(TMAX_KPA * _twitch(t_new - 0.12))
        solved_under = controller.cycles["ENDO"].phase
        ok = controller.step(t_new, dt)
        assert ok, f"step failed to converge at t={t_new:.4f} s"
        t = t_new

        cyc = controller.cycles["ENDO"]
        history.append((t, solved_under, cyc.volume_n, cyc.pressure_n))
        if cyc.phase == cycle.Phase.FILLING:
            break

    # The phases steps were solved under, then the one the run ended in.
    phases_seen = [h[1] for h in history] + [controller.cycles["ENDO"].phase]
    distinct_phases: list[cycle.Phase] = [phases_seen[0]]
    for phase in phases_seen[1:]:
        if phase != distinct_phases[-1]:
            distinct_phases.append(phase)

    assert distinct_phases == [
        cycle.Phase.PRELOAD,
        cycle.Phase.ISOVOLUMIC_CONTRACTION,
        cycle.Phase.EJECTION,
        cycle.Phase.ISOVOLUMIC_RELAXATION,
        cycle.Phase.FILLING,
    ]

    for phase in (cycle.Phase.ISOVOLUMIC_CONTRACTION, cycle.Phase.ISOVOLUMIC_RELAXATION):
        volumes = [h[2] for h in history if h[1] == phase]
        assert volumes, f"no step solved under {phase!r}"
        assert max(volumes) - min(volumes) <= 1e-6 * abs(volumes[0])

    # A constraint switch may change the pressure's rate -- the pressure rises
    # steeply through IVC and then far less once the valve opens -- but must
    # not make it jump. At each switch step k, the first solved under the new
    # phase, the second difference P_k - 2 P_(k-1) + P_(k-2) is what a jump
    # shows up in, whatever the rate either side of it.
    max_P = max(h[3] for h in history)
    switches = [k for k in range(1, len(history)) if history[k][1] != history[k - 1][1]]
    assert len(switches) == 3
    for k in switches:
        assert k >= 2
        d2P = abs(history[k][3] - 2.0 * history[k - 1][3] + history[k - 2][3])
        assert d2P < 0.1 * max_P, (
            f"switch {history[k - 1][1].name} -> {history[k][1].name} at "
            f"t={history[k][0]:.4f} s: |P_k - 2 P_(k-1) + P_(k-2)| = {d2P:.1f} Pa "
            f">= 0.1 * max P = {0.1 * max_P:.1f} Pa"
        )
