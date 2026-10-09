"""`pulse.coupling`: the couplings that own a mechanics step, on a coarse LV ellipsoid."""

import gc
import json
import math
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import pulse
from pulse import cycle
from pulse.circulation import ChamberCoupling, mL, mmHg
from pulse.coupling import Coupling, CycleCoupling, MonolithicCoupling, Phase, SplitCoupling
from pulse.problem import Cavity, CavityControl

cardiac_geometries = pytest.importorskip("cardiac_geometries")

DT = 2e-3


@pytest.fixture(scope="module")
def lv(tmp_path_factory):
    """A coarse LV ellipsoid in metres: (cardiac_geometries Geometry, pulse.HeartGeometry)."""
    comm = MPI.COMM_WORLD
    geodir = comm.bcast(tmp_path_factory.mktemp("lv_coupling"), root=0)
    if comm.rank == 0:
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=geodir,
            create_fibers=True,
            fiber_space="P_1",
            r_short_endo=0.025,
            r_short_epi=0.035,
            r_long_endo=0.09,
            r_long_epi=0.097,
            psize_ref=0.05,
            mu_apex_endo=-math.pi,
            mu_base_endo=-math.acos(5 / 17),
            mu_apex_epi=-math.pi,
            mu_base_epi=-math.acos(5 / 20),
            comm=MPI.COMM_SELF,
            fiber_angle_epi=-60,
            fiber_angle_endo=60,
        )
    comm.barrier()
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
    geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})
    return geo, geometry


@pytest.fixture(autouse=True)
def _collective_cleanup():
    """Destroy PETSc-owning objects at the same point on every rank, or a lagging garbage
    collection on one rank deadlocks the collective destroy on the other."""
    yield
    gc.collect()
    MPI.COMM_WORLD.barrier()


def _model(geo, viscous=False):
    material = pulse.HolzapfelOgden(
        f0=geo.f0,
        s0=geo.s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    Ta = pulse.Variable(dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    return pulse.CardiacModel(
        material=material,
        active=pulse.ActiveStress(geo.f0, activation=Ta),
        compressibility=pulse.Compressible(),
        viscoelasticity=pulse.Viscous() if viscous else pulse.viscoelasticity.NoneViscoElasticity(),
    )


def _cycle_params():
    """`howto/restart.py`'s compressed LV cycle: PRELOAD ends after two 2 ms steps."""
    return cycle.CycleParams(
        t_zero=2e-3,
        preload_pressure=500.0,
        t_end_diastole=4e-3,
        p_end_diastole=1000.0,
        p_fill=500.0,
        period=0.8,
        windkessel=cycle.Windkessel(
            p_init=9000.0,
            compliance=1.5 * mL / mmHg,
            resistance=1.1 * mmHg / mL,
            characteristic_impedance=0.03 * mmHg / mL,
        ),
        filling=cycle.PrescribedInflow(rate=0.046 * mL / 1e-3),
    )


def _dynamic_problem(geometry, model, cavities, **kwargs):
    return pulse.problem.DynamicProblem(
        model=model,
        geometry=geometry,
        cavities=cavities,
        parameters={
            "base_bc": pulse.problem.BaseBC.fixed,
            "mesh_unit": "m",
            "rho": pulse.Variable(1e3, "kg/m^3"),
            "dt": pulse.Variable(DT, "s"),
        },
        **kwargs,
    )


def _cycle_coupled(lv):
    geo, geometry = lv
    coupling = CycleCoupling({"ENDO": _cycle_params()})
    problem = _dynamic_problem(
        geometry,
        _model(geo, viscous=True),
        coupling.cavities(geo.mesh),
        **coupling.problem_kwargs(),
    )
    coupling.attach(problem)
    coupling.initialize(0.0)
    return coupling, problem


def _restart_state(problem):
    """Copies of every restart Function's array, by name, and the restart metadata."""
    arrays = {name: f.x.array.copy() for name, f in problem.restart_functions()}
    return arrays, json.loads(json.dumps(problem.restart_metadata()))


def _assert_same_restart_state(before, after):
    arrays_before, meta_before = before
    arrays_after, meta_after = after
    assert meta_after == meta_before
    assert arrays_after.keys() == arrays_before.keys()
    for name, values in arrays_before.items():
        assert np.array_equal(arrays_after[name], values), name


def _without_solver_hints(state):
    """`state_dict()` minus the controller's pending-refresh hint (see `Coupling.advance`)."""
    state = json.loads(json.dumps(state))
    state["controller"].pop("refresh_pending")
    return state


def test_phase_input():
    assert Phase(0.8)(1.0) == pytest.approx(0.2)
    assert Phase(1.0)(0.25) == pytest.approx(0.25)


def test_cycle_coupling_is_a_coupling():
    assert isinstance(CycleCoupling({"ENDO": _cycle_params()}), Coupling)


def test_cycle_coupling_steps_exactly_like_the_controller(lv):
    geo, geometry = lv
    params = {"ENDO": _cycle_params()}
    direct = _dynamic_problem(
        geometry,
        _model(geo, viscous=True),
        [Cavity(marker="ENDO", control=CavityControl(geo.mesh))],
    )
    controller = cycle.CycleController(direct, params)
    controller.initialize(t0=0.0)
    coupling, problem = _cycle_coupled(lv)

    t = 0.0
    phases = []
    for _ in range(4):
        phases.append(int(controller.cycles["ENDO"].phase))
        assert controller.step(t + DT, DT)
        assert coupling.advance(t, DT)
        t += DT
    record = coupling.record()
    ref = controller.records["ENDO"]
    pairs = [
        ("volume_ENDO", ref.V),
        ("pressure_ENDO", ref.P),
        ("Pc_ENDO", ref.P_c),
        ("Q_ENDO", ref.Q),
    ]
    if MPI.COMM_WORLD.size == 1:
        assert np.array_equal(problem.u.x.array, direct.u.x.array)
        for key, value in pairs:
            assert record[key] == value
    else:  # two independent Newton solves may differ in the last bits under MPI
        np.testing.assert_allclose(problem.u.x.array, direct.u.x.array, rtol=1e-9, atol=1e-14)
        for key, value in pairs:
            assert record[key] == pytest.approx(value, rel=1e-9, abs=1e-14)
    assert record["phase_ENDO"] == phases[-1]  # the phase the last step was solved under
    assert set(record) == {"phase_ENDO", "volume_ENDO", "pressure_ENDO", "Pc_ENDO", "Q_ENDO"}


def test_cycle_coupling_failed_advance_changes_nothing(lv, monkeypatch):
    coupling, problem = _cycle_coupled(lv)
    assert coupling.advance(0.0, DT)
    before_state = _without_solver_hints(coupling.state_dict())
    before = _restart_state(problem)

    def fail(*args, **kwargs):
        problem.update_old_states()
        problem.u.x.array[:] += 1.0
        return False

    monkeypatch.setattr(problem, "solve", fail)
    assert coupling.advance(DT, DT) is False
    assert coupling.state_dict()["controller"]["refresh_pending"] is True
    assert _without_solver_hints(coupling.state_dict()) == before_state
    _assert_same_restart_state(before, _restart_state(problem))


def test_cycle_coupling_state_round_trips_through_json(lv):
    coupling, _ = _cycle_coupled(lv)
    assert coupling.advance(0.0, DT)
    state = json.loads(json.dumps(coupling.state_dict()))
    other, _ = _cycle_coupled(lv)
    other.load_state_dict(state)
    assert _without_solver_hints(other.state_dict()) == _without_solver_hints(coupling.state_dict())


WINDKESSEL = Path(__file__).parent / "data" / "windkessel.ode"
INITIAL = {"p_AR": 70.0}


def _static_problem(geometry, model, cavities, **kwargs):
    parameters = {"base_bc": pulse.problem.BaseBC.fixed, "mesh_unit": "m"}
    parameters.update(kwargs.pop("parameters", {}))
    return pulse.problem.StaticProblem(
        model=model,
        geometry=geometry,
        cavities=cavities,
        parameters=parameters,
        **kwargs,
    )


def _numpy_circuit():
    pytest.importorskip("gotranx")
    from pulse.circulation import GotranxNumpyCirculation

    return GotranxNumpyCirculation(ode_file=WINDKESSEL, drop_components=("timing", "LV"))


def _cavity_volume(geometry, u) -> float:
    return MPI.COMM_WORLD.allreduce(geometry.volume("ENDO", u=u), op=MPI.SUM)


def _split_coupled(lv, record=()):
    geo, geometry = lv
    coupling = SplitCoupling(
        _numpy_circuit(),
        [ChamberCoupling(marker="ENDO", volume_state="V_LV", pressure_missing="p_LV")],
        inputs={"beat_phase": Phase(1.0)},
        initial_state=INITIAL,
        record=record,
    )
    problem = _static_problem(
        geometry,
        _model(geo),
        coupling.cavities(geo.mesh),
        **coupling.problem_kwargs(),
    )
    coupling.attach(problem)
    coupling.initialize(0.0)
    return coupling, problem


def _demo_loop(lv, n):
    """`land_circulation_biv.py`'s loop, n steps: solve at V_k -> p_k, then
    y_{k+1} = y_k + dt rhs(t_k, y_k, p_k). Returns y_n."""
    geo, geometry = lv
    circuit = _numpy_circuit()
    volume = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0))
    problem = _static_problem(geometry, _model(geo), [Cavity(marker="ENDO", volume=volume)])
    i_volume = circuit.state_index("V_LV")
    y = circuit.initial_states_with(INITIAL)
    y[i_volume] = _cavity_volume(geometry, None) / mL
    t = 0.0
    for _ in range(n):
        volume.value = y[i_volume] * mL
        assert problem.solve()
        missing = np.zeros(len(circuit.missing_names))
        missing[circuit.missing_index("p_LV")] = (
            float(problem.cavity_pressures[0].x.array[0]) / mmHg
        )
        missing[circuit.missing_index("beat_phase")] = t % 1.0
        y = y + DT * circuit.rhs(t, y, missing)
        t += DT
    return y


def test_split_coupling_equals_the_demo_loop(lv):
    _, geometry = lv
    coupling, problem = _split_coupled(lv)
    t = 0.0
    for _ in range(3):
        assert coupling.advance(t, DT)
        t += DT
    expected = _demo_loop(lv, 3)
    if MPI.COMM_WORLD.size == 1:
        np.testing.assert_array_equal(coupling.y, expected)  # bit for bit
    else:  # MUMPS differs in the last bits between two problem instances under MPI
        np.testing.assert_allclose(coupling.y, expected, rtol=1e-9, atol=1e-14)
    record = coupling.record()
    # The volume rows of the Newton system are in mL, so the mesh volume is converged to
    # within `snes_atol` = 1e-6 mL of the circuit's.
    assert record["volume_ENDO"] == pytest.approx(
        _cavity_volume(geometry, problem.u),
        rel=0,
        abs=1e-6 * mL,
    )
    assert record["circ_V_LV"] * mL == record["volume_ENDO"]


def test_split_coupling_failed_advance_changes_nothing(lv, monkeypatch):
    coupling, problem = _split_coupled(lv)
    assert coupling.advance(0.0, DT)
    assert coupling.advance(DT, DT)
    # u_old lags u, so a failed attempt's `update_old_states` would show
    assert not np.array_equal(problem.u.x.array, problem.u_old.x.array)
    y, p = coupling.y.copy(), dict(coupling.p)
    before = _restart_state(problem)
    volume = float(problem.cavities[0].volume.value)

    def fail(*args, **kwargs):
        problem.update_old_states()
        problem.u.x.array[:] += 1.0
        return False

    monkeypatch.setattr(problem, "solve", fail)
    assert coupling.advance(2 * DT, DT) is False
    assert np.array_equal(coupling.y, y)
    assert coupling.p == p
    _assert_same_restart_state(before, _restart_state(problem))
    assert float(problem.cavities[0].volume.value) == volume


def test_split_coupling_state_round_trips_through_json(lv):
    coupling, _ = _split_coupled(lv)
    assert coupling.advance(0.0, DT)
    state = json.loads(json.dumps(coupling.state_dict()))
    other, problem = _split_coupled(lv)
    other.load_state_dict(state)
    assert np.array_equal(other.y, coupling.y)  # floats survive JSON exactly
    assert other.p == coupling.p
    assert other.t == coupling.t
    i_volume = list(coupling.model.state_names).index("V_LV")
    assert float(problem.cavities[0].volume.value) == coupling.y[i_volume] * mL


def test_split_coupling_records_monitors(lv):
    coupling, _ = _split_coupled(lv, record=("Q_in", "Q_out"))
    record = coupling.record()
    assert {"volume_ENDO", "pressure_ENDO", "circ_V_LV", "circ_p_AR"} <= set(record)
    assert {"circ_Q_in", "circ_Q_out"} <= set(record)


def test_split_coupling_initial_solve_failure_raises(lv, monkeypatch):
    geo, geometry = lv
    coupling = SplitCoupling(
        _numpy_circuit(),
        [ChamberCoupling(marker="ENDO", volume_state="V_LV", pressure_missing="p_LV")],
        inputs={"beat_phase": Phase(1.0)},
    )
    problem = _static_problem(geometry, _model(geo), coupling.cavities(geo.mesh))
    coupling.attach(problem)
    monkeypatch.setattr(problem, "solve", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="t=0"):
        coupling.initialize(0.0)


def _monolithic_coupled(lv, record=(), scheme="backward_euler"):
    pytest.importorskip("gotranx")
    from pulse.circulation import GotranxCirculation

    geo, geometry = lv
    coupling = MonolithicCoupling(
        GotranxCirculation(ode_file=WINDKESSEL, drop_components=("timing", "LV")),
        [ChamberCoupling(marker="ENDO", volume_state="V_LV", pressure_missing="p_LV")],
        inputs={"beat_phase": Phase(1.0)},
        initial_state={"p_AR": 70.0},
        monitor_model=_numpy_circuit(),
        record=record,
        scheme=scheme,
    )
    problem = _static_problem(
        geometry,
        _model(geo),
        coupling.cavities(geo.mesh),
        **coupling.problem_kwargs(),
    )
    coupling.attach(problem)
    coupling.initialize(0.0)
    return coupling, problem


def test_monolithic_coupling_holds_the_constraint(lv):
    _, geometry = lv
    coupling, problem = _monolithic_coupled(lv, record=("Q_in",))
    t = 0.0
    for _ in range(3):
        assert coupling.advance(t, DT)
        t += DT
    record = coupling.record()
    assert record["volume_ENDO"] == pytest.approx(_cavity_volume(geometry, problem.u), rel=1e-8)
    assert record["circ_V_LV"] * mL == record["volume_ENDO"]
    assert "circ_Q_in" in record
    assert coupling.state_dict() == {"t": pytest.approx(3 * DT), "dt": pytest.approx(DT)}
    assert float(problem.circulation_time.value) == pytest.approx(3 * DT)


@pytest.mark.parametrize("scheme", ["backward_euler", "bdf2"])
def test_monolithic_coupling_failed_advance_rolls_back(lv, monkeypatch, scheme):
    """A failed advance leaves every restart Function and the restart metadata as they were,
    even when the failed attempt converged (and so shifted the BDF2 history)."""
    coupling, problem = _monolithic_coupled(lv, scheme=scheme)
    assert coupling.advance(0.0, DT)
    assert coupling.advance(DT, DT)
    before = _restart_state(problem)
    stencil = [float(c.value) for c in problem._circulation_stencil]
    real_solve = problem.solve

    def fail(*args, **kwargs):
        real_solve(*args, **kwargs)
        return False

    monkeypatch.setattr(problem, "solve", fail)
    assert coupling.advance(2 * DT, DT) is False
    _assert_same_restart_state(before, _restart_state(problem))
    assert [float(c.value) for c in problem._circulation_stencil] == stencil
    assert coupling.state_dict() == {"t": pytest.approx(2 * DT), "dt": pytest.approx(DT)}


def test_no_coupling_failed_advance_rolls_back(lv, monkeypatch):
    geo, geometry = lv
    volume = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0))
    volume.value = 1.05 * _cavity_volume(geometry, None)
    coupling = pulse.coupling.NoCoupling()
    problem = _static_problem(geometry, _model(geo), [Cavity(marker="ENDO", volume=volume)])
    coupling.attach(problem)
    assert coupling.advance(0.0, DT)
    assert not np.array_equal(problem.u.x.array, problem.u_old.x.array)
    before = _restart_state(problem)

    def fail(*args, **kwargs):
        problem.update_old_states()
        problem.u.x.array[:] += 1.0
        return False

    monkeypatch.setattr(problem, "solve", fail)
    assert coupling.advance(DT, DT) is False
    _assert_same_restart_state(before, _restart_state(problem))


def test_monolithic_bdf2_takes_backward_euler_at_a_new_dt(lv, monkeypatch):
    """BDF2's fixed stencil assumes a uniform step. A step at a new dt (a halved retry, or the
    full step after it) is taken as backward Euler, which makes the history uniform again."""
    coupling, problem = _monolithic_coupled(lv, scheme="bdf2")
    stencils: list[tuple[float, ...]] = []
    real_solve = problem.solve
    fail_next = [False]

    def solve(*args, **kwargs):
        stencils.append(tuple(float(c.value) for c in problem._circulation_stencil))
        ok = real_solve(*args, **kwargs)
        if fail_next[0]:
            fail_next[0] = False
            return False
        return ok

    monkeypatch.setattr(problem, "solve", solve)
    be, bdf2 = pulse.problem.BACKWARD_EULER_STENCIL, pulse.problem.BDF2_STENCIL

    assert coupling.advance(0.0, 2 * DT)  # first step: one past level only
    assert coupling.advance(2 * DT, 2 * DT)  # same dt: BDF2
    fail_next[0] = True
    assert coupling.advance(4 * DT, 2 * DT) is False  # BDF2 attempt, rolled back
    assert coupling.advance(4 * DT, DT)  # the halves: a new dt, so backward Euler...
    assert coupling.advance(5 * DT, DT)  # ...then BDF2 at the uniform dt
    assert coupling.advance(6 * DT, 2 * DT)  # back to the full step: a new dt again
    assert coupling.advance(8 * DT, 2 * DT)
    assert stencils == [be, bdf2, bdf2, be, bdf2, be, bdf2]
    assert coupling.state_dict() == {"t": pytest.approx(10 * DT), "dt": pytest.approx(2 * DT)}


def test_monolithic_coupling_dt_round_trips(lv):
    coupling, _ = _monolithic_coupled(lv, scheme="bdf2")
    assert coupling.state_dict() == {"t": 0.0, "dt": None}
    assert coupling.advance(0.0, DT)
    state = json.loads(json.dumps(coupling.state_dict()))
    other, _ = _monolithic_coupled(lv, scheme="bdf2")
    other.load_state_dict(state)
    assert other.state_dict() == coupling.state_dict()
    other.load_state_dict({"t": DT})  # a checkpoint from before "dt" existed
    assert other.state_dict() == {"t": DT, "dt": None}
