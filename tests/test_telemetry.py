import json
import logging

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import pytest

import pulse
from pulse.telemetry import NullMonitor, PerformanceMonitor


class FakeKSP:
    def getConvergedReason(self):
        return 2


class FakeSNES:
    def __init__(self, iterations, linear, reason):
        self._it, self._lin, self._reason = iterations, linear, reason

    def getIterationNumber(self):
        return self._it

    def getLinearSolveIterations(self):
        return self._lin

    def getConvergedReason(self):
        return self._reason

    def getKSP(self):
        return FakeKSP()


def test_track_time_and_counters_accumulate():
    monitor = PerformanceMonitor()
    for _ in range(2):
        with monitor.track_time("a"):
            pass
    monitor.count("halvings")
    monitor.count("halvings", 2)
    assert monitor.timings["a"] >= 0.0
    assert monitor.counters == {"halvings": 3}


def test_record_snes_counts_iterations_and_failures():
    monitor = PerformanceMonitor()
    monitor.record_snes(FakeSNES(3, 3, 2))
    monitor.record_snes(FakeSNES(5, 7, -3))  # diverged
    assert monitor.newton_total_iterations == 8
    assert monitor.newton_max_iterations == 5
    assert monitor.newton_last_iterations == 5
    assert monitor.newton_failures == 1
    assert monitor.ksp_total_iterations == 10
    assert monitor.ksp_max_iterations == 7


def test_advance_step_logs_every_log_frequency(caplog):
    caplog.set_level(logging.INFO, logger="pulse.telemetry")
    monitor = PerformanceMonitor(log_frequency=2)
    monitor.advance_step(0.0, 0.1)
    assert "step timing" not in caplog.text
    monitor.advance_step(0.1, 0.2)
    assert "step timing step=2" in caplog.text


def test_save_summary(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    monitor = PerformanceMonitor()
    monitor.record_snes(FakeSNES(2, 4, 2))
    monitor.count("halvings")
    with monitor.track_time("x"):
        pass
    monitor.advance_step(0.0, 1.0)
    monitor.save_summary(tmp_path / "perf.json")
    MPI.COMM_WORLD.barrier()
    data = json.loads((tmp_path / "perf.json").read_text())
    assert data["total_steps"] == 1
    assert data["newton"]["total_iterations"] == 2
    assert data["ksp"]["total_iterations"] == 4
    assert data["counters"] == {"halvings": 1}
    assert "x" in data["timings"]


def _problem(monitor=None):
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    boundaries = [
        pulse.Marker(name="X0", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0.0)),
        pulse.Marker(name="X1", marker=2, dim=2, locator=lambda x: np.isclose(x[0], 1.0)),
    ]
    geometry = pulse.HeartGeometry(mesh=mesh, boundaries=boundaries)
    model = pulse.CardiacModel(
        material=pulse.NeoHookean(),
        active=pulse.Passive(),
        compressibility=pulse.Compressible(),
    )
    traction = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), "kPa")

    def clamp(V):
        facets = geometry.facet_tags.find(1)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        return [dolfinx.fem.dirichletbc(dolfinx.fem.Function(V), dofs)]

    bcs = pulse.BoundaryConditions(
        neumann=(pulse.NeumannBC(traction=traction, marker=2),),
        dirichlet=(clamp,),
    )
    kwargs = {} if monitor is None else {"monitor": monitor}
    return pulse.StaticProblem(
        model=model, geometry=geometry, bcs=bcs, parameters={"u_space": "P_1"}, **kwargs
    )


def test_problem_defaults_to_null_monitor():
    assert isinstance(_problem().monitor, NullMonitor)


def test_problem_solve_is_monitored():
    monitor = PerformanceMonitor()
    problem = _problem(monitor)
    assert problem.solve()
    assert monitor.timings["newton_solve"] > 0.0
    from packaging.version import Version

    if Version(dolfinx.__version__) >= Version("0.10"):
        assert monitor.newton_total_iterations >= 1
        assert monitor.newton_failures == 0
    else:  # scifem's Newton solver exposes no SNES
        pytest.skip("iteration counts need dolfinx >= 0.10")


def _hopeless_problem(monitor):
    """A problem Newton cannot solve in one iteration, with `raise_on_failure=True`.

    Mirrors `tests/test_solve_convergence_reporting.py`'s `_hopeless`: a huge load
    and one allowed iteration make this a genuine convergence failure rather than
    a mesh that inverts for an unrelated reason. Unlike that helper, `solve` is
    configured to raise, so this exercises the failure path where `record_snes`
    must still run despite PETSc raising out of `self.problem.solve()`.
    """
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    boundaries = [
        pulse.Marker(name="X0", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0.0)),
        pulse.Marker(name="X1", marker=2, dim=2, locator=lambda x: np.isclose(x[0], 1.0)),
    ]
    geometry = pulse.Geometry(mesh=mesh, boundaries=boundaries, metadata={"quadrature_degree": 4})

    f0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((1.0, 0.0, 0.0)))
    s0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 1.0, 0.0)))
    model = pulse.CardiacModel(
        material=pulse.HolzapfelOgden(
            f0=f0,
            s0=s0,
            **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
        ),
        active=pulse.ActiveStress(f0, activation=dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))),
        compressibility=pulse.Compressible(),
    )

    def dirichlet_bc(V):
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, geometry.facet_tags.find(1))
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

    t = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-1e7))
    bcs = pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
        neumann=(pulse.NeumannBC(traction=t, marker=2),),
    )
    petsc_options = dict(pulse.StaticProblem.default_parameters()["petsc_options"])
    petsc_options["snes_max_it"] = 1
    return pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters={
            "petsc_options": petsc_options,
            "raise_on_failure": True,
        },
        monitor=monitor,
    )


def test_record_snes_runs_even_when_solve_raises():
    from packaging.version import Version

    if Version(dolfinx.__version__) < Version("0.10"):
        pytest.skip("iteration counts need dolfinx >= 0.10")

    monitor = PerformanceMonitor()
    problem = _hopeless_problem(monitor)
    with pytest.raises((PETSc.Error, RuntimeError)):
        problem.solve()
    assert monitor.newton_failures == 1
