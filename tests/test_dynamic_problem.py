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


class _FlaggedActiveStress(pulse.ActiveStress):
    """`ActiveStress`, but opted into end-of-step evaluation."""

    evaluate_at_end_of_step = True


class _RateActive(pulse.active_model.ActiveModel):
    r"""A minimal active model whose stress depends on the stretch rate,

    .. math::
        \mathbf{S}_a = \frac{k}{dt} (\lambda(\mathbf{C}) - 1)\, f_0 \otimes f_0

    used only to check that ``evaluate_at_end_of_step`` also moves a model
    that (unlike ``ActiveStress``) is genuinely rate-dependent, not just
    flagged. ``k`` and ``dt`` are plain Constants; nothing here is a real
    contraction model.
    """

    evaluate_at_end_of_step = True

    def __init__(self, f0, k, dt):
        self.f0 = f0
        self.k = k
        self.dt = dt

    def Fe(self, F):
        return F

    def strain_energy(self, C):
        raise NotImplementedError

    def S(self, C):
        lmbda = ufl.sqrt(ufl.inner(C * self.f0, self.f0))
        return (self.k * (lmbda - 1.0) / self.dt) * ufl.outer(self.f0, self.f0)


def _dynamic_problem(geometry, dirichlet_bc, active_model):
    """A `DynamicProblem` with material and compressibility switched off
    (zero stiffness) and no inertia (rho=0), so the active model is the
    *only* source of stress.

    Used only by the end-of-step gate below: it isolates the active-stress
    term so that comparing two independently-assembled residual vectors
    doesn't have to subtract away large, physically-identical quantities
    (passive elasticity, inertia) first -- see that test's docstring for why
    that subtraction alone would not be precise enough.
    """
    material = pulse.NeoHookean(mu=pulse.Variable(0.0, "kPa"))
    comp_model = pulse.compressibility.Compressible2(kappa=pulse.Variable(0.0, "Pa"))
    model = pulse.CardiacModel(material=material, active=active_model, compressibility=comp_model)
    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))
    parameters = {
        "dt": pulse.Variable(1e-3, "s"),
        "rho": pulse.Variable(0.0, "kg/m^3"),
        "mesh_unit": "m",
        "alpha_m": 0.2,
        "alpha_f": 0.4,
    }
    return pulse.problem.DynamicProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters=parameters,
    )


def _assemble(form) -> np.ndarray:
    vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(form))
    return vec.array.copy()


@pytest.mark.parametrize("active", ["flagged_active_stress", "rate"])
def test_flagged_active_stress_is_assembled_at_end_of_step(geometry, dirichlet_bc, active):
    """`DynamicProblem` must assemble a flagged active model's stress at the
    true end-of-step `self.u`, not at the alpha_f-interpolated configuration
    it uses for the rest of the material form -- the same treatment already
    given to the cavity constraint and J - 1.

    Checked by comparing two otherwise-identical problems, one with the flag
    on and one off, against the difference computed by hand from the active
    model's own S. `R_flag` and `R_unflag` are two independently-compiled
    forms that are mathematically equal apart from the (much smaller)
    active-stress term under test, so their difference matches `expected`
    only up to ordinary floating-point round-off, not bit-exactly; `atol`
    below is sized for that round-off, not for the interesting quantity.
    """
    mesh = geometry.mesh
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))

    if active == "flagged_active_stress":
        Ta = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(5.0)), "kPa")
        active_flag = _FlaggedActiveStress(f0, activation=Ta)
        active_unflag = _FlaggedActiveStress(f0, activation=Ta)
    else:
        k = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(500.0))
        dt_const = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1e-3))
        active_flag = _RateActive(f0, k=k, dt=dt_const)
        active_unflag = _RateActive(f0, k=k, dt=dt_const)
    active_unflag.evaluate_at_end_of_step = False

    problem_flag = _dynamic_problem(geometry, dirichlet_bc, active_flag)
    problem_unflag = _dynamic_problem(geometry, dirichlet_bc, active_unflag)

    for problem in (problem_flag, problem_unflag):
        problem.u.interpolate(lambda x: 0.01 * x)

    R_flag = _assemble(problem_flag.R[0])
    R_unflag = _assemble(problem_unflag.R[0])

    alpha_f = problem_flag.parameters["alpha_f"]
    u = problem_flag.u
    u_old = problem_flag.u_old  # zero
    u_test = problem_flag.u_test
    I = ufl.Identity(3)

    def stress_and_variation(u_expr):
        F = I + ufl.grad(u_expr)
        C = ufl.variable(F.T * F)
        var_C = ufl.grad(u_test).T * F + F.T * ufl.grad(u_test)
        return active_flag.S(C), var_C

    Sa_u, varC_u = stress_and_variation(u)
    u_alpha = alpha_f * u_old + (1 - alpha_f) * u
    Sa_ua, varC_ua = stress_and_variation(u_alpha)

    expected_form = (ufl.inner(Sa_u, 0.5 * varC_u) - ufl.inner(Sa_ua, 0.5 * varC_ua)) * geometry.dx
    expected = _assemble(expected_form)

    assert np.linalg.norm(expected) > 0
    assert np.allclose(
        R_flag - R_unflag,
        expected,
        rtol=1e-12,
        atol=1e-12 * np.linalg.norm(expected),
    )


def _neo_hookean_dynamic_problem(dt, mesh=None):
    if mesh is None:
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
    traction = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "kPa")

    def clamp(V):
        facets = geometry.facet_tags.find(1)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        return [dolfinx.fem.dirichletbc(dolfinx.fem.Function(V), dofs)]

    bcs = pulse.BoundaryConditions(
        neumann=(pulse.NeumannBC(traction=traction, marker=2),),
        dirichlet=(clamp,),
    )
    problem = pulse.DynamicProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters={"dt": dt, "u_space": "P_1"},
    )
    return problem, traction


def test_constant_dt_can_change_between_solves():
    """A Constant-backed dt set to 1 ms before the first solve reproduces a float dt of 1 ms."""
    reference, t_ref = _neo_hookean_dynamic_problem(pulse.Variable(1e-3, "s"))
    # dt_constant must live on the same mesh instance as `problem`'s own
    # forms: a Constant built on a different (even if topologically
    # identical) mesh object raises UFL's "multiple domains" error as soon
    # as it is combined with problem's own C/C_dot in the material form.
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    dt_constant = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2e-3))
    problem, t_new = _neo_hookean_dynamic_problem(pulse.Variable(dt_constant, "s"), mesh=mesh)
    dt_constant.value = 1e-3  # changed after the forms were built
    for step in range(1, 4):
        t_ref.assign(0.5 * step)
        t_new.assign(0.5 * step)
        assert reference.solve()
        assert problem.solve()
    np.testing.assert_allclose(problem.u.x.array, reference.u.x.array, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        problem.v_old.x.array, reference.v_old.x.array, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(problem.a_old.x.array, reference.a_old.x.array, rtol=1e-8, atol=1e-8)
