"""A cavity whose constraint is chosen at run time, through `CavityControl`.

A plain `Cavity(volume=...)` always prescribes the volume. A controlled cavity
carries one pressure unknown whose equation is picked by `Constant`s: the
volume, the pressure, or a pressure affine in the volume. Switching between
them changes only those constants, so the compiled problem is reused.

Each mode is checked against the formulation it should reproduce: volume mode
against a plain prescribed-volume cavity, pressure mode against a Neumann
traction. A controlled cavity's pressure row is scaled to kPa where a Neumann
traction has no row at all, so Newton stops at a different iterate and the two
agree to solver tolerance rather than to round-off; the volume rows are both
in mL, and are compared to the same tolerance.

The cavity surface is every face of the cube but the fixed one. A cavity
pressure loads the wall through the derivative of the divergence-theorem volume
V(u) = (-1/3) int x . n da, which equals a pressure traction only when the
surface is closed off by a rim lying in a plane through the origin and not
moving out of it, as an LV endocardium is at a fixed base. Here that rim is the
edge of the fixed face x = 0; on the x = 1 face alone it moves freely, and the
load picks up edge terms a Neumann pressure does not have.

The ENDO/FIXED meshtags are built by hand from the mesh's exterior facets,
classified by facet midpoint -- not from `HeartGeometry`'s usual
`Marker`/`locate_entities` path. `locate_entities` tags a facet whenever every
one of its vertices *individually* satisfies the locator, which is not the
same as the facet lying on the surface the locator describes: a union of five
plane conditions lets a facet's three vertices each satisfy a *different*
clause (e.g. a corner facet of the fixed x = 0 face can have one vertex on
y = 0, one on y = 1, one on z = 1), and `locate_entities` does not restrict to
boundary facets either. On this mesh that tags two boundary facets of the
fixed face as ENDO too (`ENDO` and `FIXED` on the same facet) and tags 27
interior facets that are not on the boundary at all, inflating `ENDO`'s area
to 5.25 instead of 5. Tagging by midpoint over `exterior_facet_indices` alone
avoids both: every exterior facet gets exactly one tag and no interior facet
gets any.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import pulse
from pulse.circulation import ChamberCoupling, mL
from pulse.problem import Cavity, CavityControl, volume_scale

#: Solver-tolerance agreement between two formulations of the same problem.
RTOL = 1e-7


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)


@pytest.fixture
def geometry(mesh):
    # See the module docstring: tagging by midpoint over the mesh's own
    # exterior facets, rather than through a `Marker`/`locate_entities`
    # locator, is what makes ENDO's area exactly 5 (not 5.25) and keeps every
    # facet single-tagged.
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


def _problem(geometry, dirichlet_bc, cavities=(), neumann=(), **overrides):
    kwargs = dict(
        model=_model(),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=tuple(neumann)),
        cavities=list(cavities),
        parameters={"mesh_unit": "m"},
    )
    kwargs.update(overrides)
    return pulse.problem.StaticProblem(**kwargs)


class _StubCirculation:
    """Just enough of `CirculationModel` to mark a chamber coupled.

    `_check_cavities` only needs `self.circulation is not None` and each
    chamber's `marker`; it never calls into the circuit itself, so nothing
    here has to do anything real.
    """

    @property
    def state_names(self) -> tuple[str, ...]:
        return ()

    @property
    def missing_names(self) -> tuple[str, ...]:
        return ()

    @property
    def initial_states(self) -> np.ndarray:
        return np.zeros(0)

    def rhs(self, t, states, missing):
        raise NotImplementedError


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


def test_prescribed_volume_follows_a_sub_milliliter_step(mesh, geometry, dirichlet_bc):
    """A volume change below `snes_atol` in m^3 (1 mL) is still solved for.

    The volume row is measured in mL, as a controlled cavity's is; in m^3 a
    0.5 mL step starts Newton below its absolute tolerance, and the solve
    returns without moving the wall.
    """
    target = 1.05 * mesh.comm.allreduce(geometry.volume("ENDO"), op=MPI.SUM)
    volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(target))
    problem = _problem(geometry, dirichlet_bc, [Cavity("ENDO", volume=volume)])
    assert problem.solve()

    volume.value = target + 0.5 * mL
    assert problem.solve()
    assert _volume(problem, geometry) == pytest.approx(target + 0.5 * mL, rel=0, abs=1e-6 * mL)


@pytest.mark.parametrize("mesh_unit, scale", [("m", 1e6), ("cm", 1.0), ("mm", 1e-3)])
def test_volume_scale_converts_mesh_volumes_to_milliliters(mesh_unit, scale):
    assert volume_scale(mesh_unit) == pytest.approx(scale, rel=1e-12)


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


def test_coupled_cavity_refuses_a_volume_too(mesh, geometry, dirichlet_bc):
    """A coupled cavity that also carries an explicit volume is refused.

    Without this, the circuit rewrite (`_init_circulation_spaces`) silently
    replaces that volume with the chamber's own volume state -- the cavity
    would still solve, just not against the volume it was given.
    """
    volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    chamber = ChamberCoupling(marker="ENDO", volume_state="V", pressure_missing="p")

    with pytest.raises(ValueError, match="ENDO"):
        _problem(
            geometry,
            dirichlet_bc,
            [Cavity("ENDO", volume=volume)],
            circulation=_StubCirculation(),
            chambers=[chamber],
        )


def test_coupled_cavity_refuses_a_control_too(mesh, geometry, dirichlet_bc):
    """A coupled cavity that also carries a control is refused (existing rule)."""
    control = CavityControl(mesh)
    chamber = ChamberCoupling(marker="ENDO", volume_state="V", pressure_missing="p")

    with pytest.raises(ValueError, match="ENDO"):
        _problem(
            geometry,
            dirichlet_bc,
            [Cavity("ENDO", control=control)],
            circulation=_StubCirculation(),
            chambers=[chamber],
        )


def test_controlled_cavity_needs_mesh_unit_m(mesh, geometry, dirichlet_bc):
    """A control's V_target/A/B are SI; any other mesh_unit is refused.

    `geometry.volume_form` is in mesh units, so a controlled cavity's rows
    would silently compare a millimeter-scaled V(u) against a V_target/A/B
    meant in metres/pascals if this weren't refused.
    """
    control = CavityControl(mesh)

    with pytest.raises(ValueError, match="ENDO"):
        _problem(
            geometry,
            dirichlet_bc,
            [Cavity("ENDO", control=control)],
            parameters={"mesh_unit": "mm"},
        )
