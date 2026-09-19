"""`StaticProblem.solve` has to be able to say that it failed.

It documents a `bool`, and callers throughout the demos are written as
`if not problem.solve():`. Until this was fixed the default PETSc options set
`snes_error_if_not_converged`, so on dolfinx >= 0.10 PETSc raised before the
return value was ever computed and every one of those guards was unreachable.
The legacy branch was worse: it returned scifem's iteration count, which is
truthy whether or not Newton got anywhere.

The tests here pin both directions -- that a failed solve reports itself, and
that `raise_on_failure` still gets the exception back.
"""

import logging

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import pytest

import pulse


def _problem(mesh, parameters=None, traction=0.0):
    """A unit cube, fixed on one face, pulled on the opposite one."""
    boundaries = [
        ("X0", 1, 2, lambda x: np.isclose(x[0], 0)),
        ("X1", 2, 2, lambda x: np.isclose(x[0], 1)),
    ]
    geo = pulse.Geometry(mesh=mesh, boundaries=boundaries, metadata={"quadrature_degree": 4})

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
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, geo.facet_tags.find(1))
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

    t = dolfinx.fem.Constant(mesh, PETSc.ScalarType(traction))
    bcs = pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
        neumann=(pulse.NeumannBC(traction=t, marker=2),),
    )
    return pulse.StaticProblem(model=model, geometry=geo, bcs=bcs, parameters=parameters or {})


def _hopeless(mesh, **parameters):
    """A problem Newton cannot solve: a huge load and one iteration to do it in.

    Capping the iteration count rather than only raising the load keeps this
    quick and keeps it a *convergence* failure rather than a mesh that inverts
    somewhere and fails for an unrelated reason.
    """
    petsc_options = dict(pulse.StaticProblem.default_parameters()["petsc_options"])
    petsc_options["snes_max_it"] = 1
    return _problem(
        mesh,
        parameters={"petsc_options": petsc_options, **parameters},
        traction=-1e7,
    )


def test_converged_solve_returns_true(mesh):
    assert _problem(mesh, traction=-1.0).solve() is True


def test_failed_solve_returns_false_instead_of_raising(mesh):
    assert _hopeless(mesh).solve() is False


def test_failed_solve_warns_even_if_the_caller_ignores_the_result(mesh):
    # A caller that drops the return value used to get a crash and would now
    # get silence, so non-convergence has to announce itself some other way.
    problem = _hopeless(mesh)
    logger = logging.getLogger("pulse")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        problem.solve()
    finally:
        logger.removeHandler(handler)

    assert any(r.levelno >= logging.WARNING for r in records), (
        "a non-converged solve must be logged at warning level"
    )


def test_raise_on_failure_argument_restores_the_exception(mesh):
    with pytest.raises((PETSc.Error, RuntimeError)):
        _hopeless(mesh).solve(raise_on_failure=True)


def test_raise_on_failure_parameter_restores_the_exception(mesh):
    # The same, set once for the problem rather than per call, so an existing
    # script with many call sites can keep the old behaviour with one edit.
    with pytest.raises((PETSc.Error, RuntimeError)):
        _hopeless(mesh, raise_on_failure=True).solve()


def test_petsc_option_does_not_override_raise_on_failure(mesh):
    # `snes_error_if_not_converged` is set on the solver by `solve` itself, so
    # passing it through `petsc_options` must not quietly bring the exception
    # back and turn the documented return value into unreachable code again.
    petsc_options = dict(pulse.StaticProblem.default_parameters()["petsc_options"])
    petsc_options["snes_max_it"] = 1
    petsc_options["snes_error_if_not_converged"] = True
    problem = _problem(
        mesh,
        parameters={"petsc_options": petsc_options},
        traction=-1e7,
    )
    assert problem.solve() is False


def test_a_failed_solve_leaves_no_trace_after_reset(mesh):
    # `reset_states` is how a caller recovers, and it has to actually recover:
    # a continuation loop that steps, fails, rolls back and tries a smaller
    # step must not be starting from the failed iterate.
    problem = _hopeless(mesh)
    before = problem.u.x.array.copy()
    assert problem.solve() is False
    problem.reset_states()
    assert np.allclose(problem.u.x.array, before)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial is enough for a contract check")
def test_solve_still_reports_success_for_a_sequence_of_load_steps(mesh):
    problem = _problem(mesh, traction=0.0)
    traction = problem.bcs.neumann[0].traction
    for value in np.linspace(0.0, -2.0, 5):
        traction.value = value
        assert problem.solve() is True
