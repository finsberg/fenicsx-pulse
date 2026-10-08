"""`pulse.coupling`: the couplings that own a mechanics step, on a coarse LV ellipsoid."""

import gc
import json
import math

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import pulse
from pulse import cycle
from pulse.circulation import mL, mmHg
from pulse.coupling import Coupling, CycleCoupling, Phase
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


def _durable(state):
    """`state_dict()` minus the controller's preconditioner-refresh flag, which is a solver-side
    hint (a failed step or a fresh solver always leaves it pending), not 0D state."""
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
    # bitwise equal in serial; under MPI the two Newton solves may differ in the last bits
    np.testing.assert_allclose(problem.u.x.array, direct.u.x.array, rtol=1e-9, atol=1e-14)
    record = coupling.record()
    assert record["volume_ENDO"] == pytest.approx(controller.records["ENDO"].V, rel=1e-9, abs=1e-14)
    assert record["pressure_ENDO"] == pytest.approx(
        controller.records["ENDO"].P,
        rel=1e-9,
        abs=1e-14,
    )
    assert record["Pc_ENDO"] == pytest.approx(controller.records["ENDO"].P_c, rel=1e-9, abs=1e-14)
    assert record["Q_ENDO"] == pytest.approx(controller.records["ENDO"].Q, rel=1e-9, abs=1e-14)
    assert record["phase_ENDO"] == phases[-1]  # the phase the last step was solved under
    assert set(record) == {"phase_ENDO", "volume_ENDO", "pressure_ENDO", "Pc_ENDO", "Q_ENDO"}


def test_cycle_coupling_failed_advance_changes_nothing(lv, monkeypatch):
    coupling, problem = _cycle_coupled(lv)
    assert coupling.advance(0.0, DT)
    before_state = _durable(coupling.state_dict())
    before_u = problem.u.x.array.copy()

    def fail(*args, **kwargs):
        problem.update_old_states()
        problem.u.x.array[:] += 1.0
        return False

    monkeypatch.setattr(problem, "solve", fail)
    assert coupling.advance(DT, DT) is False
    assert _durable(coupling.state_dict()) == before_state
    assert np.array_equal(problem.u.x.array, before_u)


def test_cycle_coupling_state_round_trips_through_json(lv):
    coupling, _ = _cycle_coupled(lv)
    assert coupling.advance(0.0, DT)
    state = json.loads(json.dumps(coupling.state_dict()))
    other, _ = _cycle_coupled(lv)
    other.load_state_dict(state)
    assert _durable(other.state_dict()) == _durable(coupling.state_dict())
