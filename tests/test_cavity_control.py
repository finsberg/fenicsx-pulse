"""A cavity whose constraint is chosen at run time, through `CavityControl`.

A plain `Cavity(volume=...)` always prescribes the volume. A controlled cavity
carries one pressure unknown whose equation is picked by `Constant`s: the
volume, the pressure, or a pressure affine in the volume. Switching between
them changes only those constants, so the compiled problem is reused.

Each mode is checked against the formulation it should reproduce: volume mode
against a plain prescribed-volume cavity, pressure mode against a Neumann
traction. The controlled rows are scaled (volume in mL, pressure in kPa) where
the Lagrangian rows are not, so Newton stops at a different iterate and the two
agree to solver tolerance rather than to round-off.

The cavity surface is every face of the cube but the fixed one. A cavity
pressure loads the wall through the derivative of the divergence-theorem volume
V(u) = (-1/3) int x . n da, which equals a pressure traction only when the
surface is closed off by a rim lying in a plane through the origin and not
moving out of it, as an LV endocardium is at a fixed base. Here that rim is the
edge of the fixed face x = 0; on the x = 1 face alone it moves freely, and the
load picks up edge terms a Neumann pressure does not have.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import pulse
from pulse.problem import Cavity, CavityControl

#: Solver-tolerance agreement between two formulations of the same problem.
RTOL = 1e-7


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)


@pytest.fixture
def geometry(mesh):
    def endo(x):
        # Every face but the fixed one x = 0 (see the module docstring). No
        # facet of x = 0 has all its vertices on any of these planes.
        return (
            np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
            | np.isclose(x[2], 0.0)
            | np.isclose(x[2], 1.0)
        )

    def fixed_face(x):
        return np.isclose(x[0], 0.0)

    boundaries = [
        pulse.Marker(name="ENDO", marker=1, dim=2, locator=endo),
        pulse.Marker(name="FIXED", marker=2, dim=2, locator=fixed_face),
    ]
    return pulse.HeartGeometry(mesh=mesh, boundaries=boundaries)


@pytest.fixture
def dirichlet_bc(geometry):
    def bc(V):
        facets = geometry.facet_tags.find(2)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

    return bc


def _model():
    return pulse.CardiacModel(
        material=pulse.NeoHookean(mu=pulse.Variable(10.0, "kPa")),
        active=pulse.Passive(),
        compressibility=pulse.compressibility.Compressible2(),
    )


def _problem(geometry, dirichlet_bc, cavities=(), neumann=()):
    return pulse.problem.StaticProblem(
        model=_model(),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=tuple(neumann)),
        cavities=list(cavities),
        parameters={"mesh_unit": "m"},
    )


def _volume(problem, geometry):
    return geometry.mesh.comm.allreduce(geometry.volume("ENDO", u=problem.u), op=MPI.SUM)


def _pressure(problem):
    return float(problem.cavity_pressures[0].x.array[0])


def _relative_difference(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_volume_mode_matches_prescribed_volume(mesh, geometry, dirichlet_bc):
    """Volume mode is the prescribed-volume constraint, scaled to mL."""
    target = 1.05 * mesh.comm.allreduce(geometry.volume("ENDO"), op=MPI.SUM)

    volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(target))
    reference = _problem(geometry, dirichlet_bc, [Cavity("ENDO", volume=volume)])
    assert reference.solve()

    control = CavityControl(mesh)
    control.set_volume(target)
    controlled = _problem(geometry, dirichlet_bc, [Cavity("ENDO", control=control)])
    assert controlled.solve()

    assert _relative_difference(controlled.u.x.array, reference.u.x.array) < RTOL
    assert _pressure(controlled) == pytest.approx(_pressure(reference), rel=RTOL)
    assert _volume(controlled, geometry) == pytest.approx(target, rel=RTOL)


def test_pressure_mode_matches_neumann_pressure(mesh, geometry, dirichlet_bc):
    """Pressure mode loads the cavity exactly as a Neumann pressure would."""
    endo = geometry.markers["ENDO"][0]
    neumann = pulse.NeumannBC(traction=pulse.Variable(1.0, "kPa"), marker=endo)
    reference = _problem(geometry, dirichlet_bc, neumann=[neumann])
    assert reference.solve()

    control = CavityControl(mesh)
    control.set_pressure(1000.0)
    controlled = _problem(geometry, dirichlet_bc, [Cavity("ENDO", control=control)])
    assert controlled.solve()

    assert _relative_difference(controlled.u.x.array, reference.u.x.array) < RTOL
    assert _pressure(controlled) == pytest.approx(1000.0, rel=1e-10)


def test_affine_mode_holds_and_switching_needs_no_rebuild(mesh, geometry, dirichlet_bc):
    """p = A + B V(u) holds at convergence, and a switch reuses the problem."""
    # B is sized so that B V(u) is comparable to A on this 1 m^3 cavity, as a
    # Windkessel's is for a heart. The -1e8 Pa/m^3 of an LV would put ~1e8 Pa
    # on a 10 kPa material here.
    A, B = 500.0, -1e3
    control = CavityControl(mesh)
    control.set_affine_pressure(A, B)
    problem = _problem(geometry, dirichlet_bc, [Cavity("ENDO", control=control)])
    solver = problem.problem

    assert problem.solve()
    V = _volume(problem, geometry)
    assert _pressure(problem) == pytest.approx(A + B * V, rel=1e-8)

    control.set_volume(0.98 * V)
    assert problem.solve()
    assert _volume(problem, geometry) == pytest.approx(0.98 * V, rel=1e-8)
    assert problem.problem is solver


def test_cavity_needs_exactly_one_constraint(mesh, geometry, dirichlet_bc):
    """Both a volume and a control, or neither and no chamber, is refused."""
    volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    control = CavityControl(mesh)

    with pytest.raises(ValueError, match="ENDO"):
        _problem(geometry, dirichlet_bc, [Cavity("ENDO", volume=volume, control=control)])

    with pytest.raises(ValueError, match="ENDO"):
        _problem(geometry, dirichlet_bc, [Cavity("ENDO")])
