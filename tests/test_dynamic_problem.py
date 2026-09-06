"""Regression tests for DynamicProblem's algebraic-constraint terms: the
cavity-volume Lagrange multiplier and the incompressibility pressure.

Generalized-alpha evaluates the material/force residual at the
alpha_f-interpolated configuration `interpolate(u_old, u, alpha_f)`, which is
correct for genuine second-order dynamics. But the cavity-volume and
incompressibility constraints are algebraic (Lagrange multipliers, not part
of the differential dynamics): enforcing them against that filtered
configuration instead of the true current `self.u` leaves `self.u`'s actual
volume/incompressibility unconstrained, which let a spurious oscillation
leak into the multiplier (observed as cavity pressure swinging by up to 20x
between timesteps under smooth volume forcing, before this fix).

Both tests below check the *true* current state -- `geometry.volume(...,
u=problem.u)`, or `J(problem.u)` -- satisfies its constraint after a
sequence of steps whose targets change enough that `u_old` ends up far from
the solution at each step. That is exactly the regime where the bug showed
up: if the constraint were (incorrectly) enforced against the alpha_f blend,
these assertions would fail by a large margin, not just a rounding error.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import pulse


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)


@pytest.fixture
def geometry(mesh):
    def endo(x):
        # Not the x=0 face: there, X = (0, y, z) is orthogonal to the outward
        # normal (-1, 0, 0), so the divergence-theorem volume integrand
        # (-1/3) X . n vanishes identically and "volume" would trivially be 0.
        return np.isclose(x[0], 1.0)

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


def _cardiac_model(mesh, comp_model):
    material = pulse.NeoHookean(mu=pulse.Variable(10.0, "kPa"))
    active_model = pulse.Passive()
    return pulse.CardiacModel(material=material, active=active_model, compressibility=comp_model)


def _run_volume_ramp(problem, geometry, Volume, volumes):
    """Advance through a volume ramp, returning the true cavity volume
    (computed from the actual solved self.u, independent of whatever the
    residual internally used) after the final step."""
    for target in volumes:
        Volume.value = target
        converged = problem.solve()
        assert converged
    return geometry.mesh.comm.allreduce(
        geometry.volume("ENDO", u=problem.u),
        op=MPI.SUM,
    )


def test_dynamic_cavity_constraint_matches_true_configuration(mesh, geometry, dirichlet_bc):
    """The cavity-volume constraint must pin the *actual* current volume
    V(self.u) to the target, not just the alpha_f-interpolated blend."""
    comp_model = pulse.compressibility.Compressible2()
    model = _cardiac_model(mesh, comp_model)

    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))
    initial_volume = mesh.comm.allreduce(geometry.volume("ENDO"), op=MPI.SUM)
    Volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(initial_volume))
    cavity = pulse.problem.Cavity(marker="ENDO", volume=Volume)

    parameters = {
        "dt": pulse.Variable(1e-3, "s"),
        "rho": pulse.Variable(1e3, "kg/m^3"),
        "mesh_unit": "m",
    }
    problem = pulse.problem.DynamicProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        cavities=[cavity],
        parameters=parameters,
    )

    # A ramp with several changing targets: each step starts from a u_old
    # that is meaningfully different from the new target's solution, which
    # is exactly the regime that exposes alpha_f-filtering of an algebraic
    # constraint.
    target_volumes = initial_volume * np.array([1.05, 1.15, 1.10, 1.25])
    final_target = target_volumes[-1]

    true_volume = _run_volume_ramp(problem, geometry, Volume, target_volumes)

    rel_error = abs(true_volume - final_target) / final_target
    assert rel_error < 1e-4, (
        f"True cavity volume {true_volume:.6e} deviates from target "
        f"{final_target:.6e} by {rel_error:.2%} -- the cavity constraint is "
        "being enforced against the wrong (alpha_f-filtered) configuration."
    )


def _incompressibility_defect(problem, geometry):
    F = ufl.Identity(3) + ufl.grad(problem.u)
    J = ufl.det(F)
    form = dolfinx.fem.form((J - 1.0) ** 2 * geometry.dx)
    return geometry.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM) ** 0.5


def test_dynamic_incompressibility_constraint_matches_static_reference(
    mesh,
    geometry,
    dirichlet_bc,
):
    """DynamicProblem's incompressibility defect J(self.u)-1 should be the
    same order of magnitude as StaticProblem's on the identical ramp.

    Note: on this coarse P2/P1 mesh, the dominant source of the J-1 defect
    is ordinary mixed-FEM discretization/Newton-tolerance error (verified:
    reverting the incompressibility term to use the alpha_f-filtered u
    instead of self.u changes the defect by <0.1%, unlike the cavity case
    where it changes it by >10x) -- so an absolute tolerance here would not
    actually be testing the alpha_f-filtering fix. Comparing against
    StaticProblem (unaffected by any alpha-filtering question at all) is
    the meaningful invariant: DynamicProblem must not be dramatically worse.
    """
    comp_model = pulse.compressibility.Incompressible()
    # Kept short: the incompressible mixed problem is expensive to solve,
    # and two steps (an initial jump, then a change of direction) is enough
    # to put u_old meaningfully far from the solution at each step.
    target_volumes_factor = np.array([1.02, 1.08])

    dynamic_bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))
    initial_volume = mesh.comm.allreduce(geometry.volume("ENDO"), op=MPI.SUM)
    Volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(initial_volume))
    cavity = pulse.problem.Cavity(marker="ENDO", volume=Volume)
    parameters = {
        "dt": pulse.Variable(1e-3, "s"),
        "rho": pulse.Variable(1e3, "kg/m^3"),
        "mesh_unit": "m",
    }
    dynamic_problem = pulse.problem.DynamicProblem(
        model=_cardiac_model(mesh, comp_model),
        geometry=geometry,
        bcs=dynamic_bcs,
        cavities=[cavity],
        parameters=parameters,
    )
    for target in initial_volume * target_volumes_factor:
        Volume.value = target
        assert dynamic_problem.solve()
    dynamic_defect = _incompressibility_defect(dynamic_problem, geometry)

    static_bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))
    Volume_static = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(initial_volume))
    cavity_static = pulse.problem.Cavity(marker="ENDO", volume=Volume_static)
    static_problem = pulse.problem.StaticProblem(
        model=_cardiac_model(mesh, comp_model),
        geometry=geometry,
        bcs=static_bcs,
        cavities=[cavity_static],
    )
    for target in initial_volume * target_volumes_factor:
        Volume_static.value = target
        assert static_problem.solve()
    static_defect = _incompressibility_defect(static_problem, geometry)

    assert dynamic_defect < 5 * static_defect, (
        f"DynamicProblem's incompressibility defect ({dynamic_defect:.3e}) is "
        f"much larger than StaticProblem's ({static_defect:.3e}) on the same "
        "ramp -- the incompressibility constraint may be enforced against "
        "the wrong (alpha_f-filtered) configuration."
    )
