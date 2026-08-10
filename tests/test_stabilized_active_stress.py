"""
Tests for `StabilizedActiveStress` and `ActiveStressFormulation.stretch`.

The stabilization term added here is meant to be a *numerical* device: it
must change the discrete scheme's stability without changing the continuous
problem being solved. Two properties encode that, and they are the two tests
worth reading:

- `test_vanishes_at_fixed_point`: when lmbda_prev equals the current stretch,
  the stress must reduce exactly to the unstabilized one. If it does not, the
  term is biasing the solution rather than stabilizing it.
- `test_tangent_matches_finite_difference`: the assembled Jacobian must match
  a finite-difference directional derivative of the residual. This is what
  guarantees Newton actually *sees* the active stiffness; without it the term
  would be inert and the instability would remain.

Everything else checks the algebra of the two formulations against closed
forms derived by hand from the strain energy.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import pulse
from pulse.active_stress import (
    ActiveStressFormulation,
    StabilizedActiveStress,
    fiber_stretch,
)
from pulse.units import Variable


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)


@pytest.fixture
def f0(mesh):
    return dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))


def _uniaxial_u(mesh, stretch):
    """Displacement field of a uniform stretch `stretch` along x."""
    V = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(
        lambda x: np.vstack([(stretch - 1.0) * x[0], np.zeros_like(x[1]), np.zeros_like(x[2])])
    )
    return u


def _assemble(expr, mesh):
    """Integrate a scalar UFL expression over the mesh."""
    form = dolfinx.fem.form(expr * ufl.dx)
    return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


# ---------------------------------------------------------------------------
# The two properties that make the stabilization legitimate
# ---------------------------------------------------------------------------


def test_vanishes_at_fixed_point(mesh, f0):
    """With lmbda_prev == lmbda(u), the stabilized stress must equal the
    plain stretch-formulation stress exactly -- no added bias."""
    stretch = 1.15
    u = _uniaxial_u(mesh, stretch)
    C = ufl.variable((ufl.Identity(3) + ufl.grad(u)).T * (ufl.Identity(3) + ufl.grad(u)))

    Vs = dolfinx.fem.functionspace(mesh, ("DG", 1))
    lmbda_prev = dolfinx.fem.Function(Vs)

    stabilized = StabilizedActiveStress(
        f0=f0,
        activation=Variable(30.0, "kPa"),
        active_stiffness=Variable(500.0, "kPa"),  # deliberately large
        lmbda_prev=lmbda_prev,
    )
    stabilized.update_prev(u)  # lmbda_prev := lmbda(u)

    plain = pulse.ActiveStress(
        f0=f0,
        activation=Variable(30.0, "kPa"),
        formulation=ActiveStressFormulation.stretch,
    )

    diff = _assemble(ufl.inner(stabilized.S(C) - plain.S(C), stabilized.S(C) - plain.S(C)), mesh)
    assert diff < 1e-18, f"stabilization biased the stress at its own fixed point (||dS||^2={diff})"


def _tangent_and_fd(mesh, f0, Ka, u, direction, lmbda_prev):
    """Return (J[direction], finite-difference dR[direction]) for the active
    contribution alone, at the given active stiffness."""
    V = u.function_space
    active = StabilizedActiveStress(
        f0=f0,
        activation=Variable(25.0, "kPa"),
        active_stiffness=Variable(Ka, "kPa"),
        lmbda_prev=lmbda_prev,
    )

    v = ufl.TestFunction(V)
    F_grad = ufl.Identity(3) + ufl.grad(u)
    C = F_grad.T * F_grad
    # Residual of the active contribution alone: inner(S, 0.5 * dC/du[v])
    R = ufl.inner(active.S(C), 0.5 * ufl.derivative(C, u, v)) * ufl.dx
    J = ufl.derivative(R, u, ufl.TrialFunction(V))

    J_mat = dolfinx.fem.assemble_matrix(dolfinx.fem.form(J))
    J_mat.scatter_reverse()
    analytic = J_mat.to_dense() @ direction.x.array

    eps = 1e-7
    u0 = u.x.array.copy()
    R_form = dolfinx.fem.form(R)
    u.x.array[:] = u0 + eps * direction.x.array
    R_plus = dolfinx.fem.assemble_vector(R_form).array.copy()
    u.x.array[:] = u0 - eps * direction.x.array
    R_minus = dolfinx.fem.assemble_vector(R_form).array.copy()
    u.x.array[:] = u0

    return analytic, (R_plus - R_minus) / (2 * eps)


def test_tangent_matches_finite_difference(mesh, f0):
    """The assembled Jacobian must be the true derivative of the residual, and
    it must actually contain the active stiffness contribution.

    The second half is the point of the class. A stabilization term that
    Newton cannot see is inert: the solve would converge to the same
    unstabilized answer and the oscillatory instability would remain. So it is
    not enough that J be self-consistent -- J must also *change* when Ka does.
    """
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
    u = dolfinx.fem.Function(V)
    rng = np.random.default_rng(0)
    u.x.array[:] = 0.05 * rng.standard_normal(u.x.array.size)

    direction = dolfinx.fem.Function(V)
    direction.x.array[:] = rng.standard_normal(direction.x.array.size)

    Vs = dolfinx.fem.functionspace(mesh, ("DG", 0))
    lmbda_prev = dolfinx.fem.Function(Vs)
    lmbda_prev.x.array[:] = 1.02

    analytic, numeric = _tangent_and_fd(mesh, f0, 400.0, u, direction, lmbda_prev)
    rel_err = np.linalg.norm(analytic - numeric) / max(np.linalg.norm(analytic), 1e-30)
    assert rel_err < 1e-5, (
        f"assembled tangent disagrees with finite differences (rel err {rel_err})"
    )

    # ...and the stiffness is genuinely in there.
    without_Ka, _ = _tangent_and_fd(mesh, f0, 0.0, u, direction, lmbda_prev)
    contribution = np.linalg.norm(analytic - without_Ka) / np.linalg.norm(without_Ka)
    assert contribution > 1e-3, (
        "the active stiffness makes no difference to the Jacobian, so Newton "
        f"cannot see it (relative change {contribution})"
    )


# ---------------------------------------------------------------------------
# Algebra of the two formulations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stretch", [0.9, 1.0, 1.12, 1.3])
def test_stretch_formulation_first_piola_is_normalized(mesh, f0, stretch):
    """P f0 has magnitude exactly Ta, independent of stretch.

    Since P = Ta (F f0 x f0)/|F f0|, contracting with f0 gives Ta times the
    unit vector along the deformed fiber. That stretch-independence is
    precisely what distinguishes this convention from the invariant one,
    where the same quantity would come out as Ta * lmbda.
    """
    u = _uniaxial_u(mesh, stretch)
    F_grad = ufl.Identity(3) + ufl.grad(u)
    C = F_grad.T * F_grad

    Ta = 42.0
    active = pulse.ActiveStress(
        f0=f0,
        activation=Variable(Ta, "kPa"),
        formulation=ActiveStressFormulation.stretch,
    )
    P = F_grad * active.S(C)
    traction = _assemble(ufl.sqrt(ufl.inner(P * f0, P * f0)), mesh)

    # Ta is given in kPa; base units are Pa
    np.testing.assert_allclose(traction, Ta * 1e3, rtol=1e-10)


@pytest.mark.parametrize("stretch", [0.9, 1.12, 1.3])
def test_two_formulations_differ_by_the_stretch(mesh, f0, stretch):
    """S_invariant = lmbda * S_stretch, exactly. Documents the factor a user
    changing `formulation` is opting into."""
    u = _uniaxial_u(mesh, stretch)
    F_grad = ufl.Identity(3) + ufl.grad(u)
    C = F_grad.T * F_grad

    kwargs = dict(f0=f0, activation=Variable(37.0, "kPa"))
    S_inv = pulse.ActiveStress(formulation=ActiveStressFormulation.invariant, **kwargs).S(C)
    S_str = pulse.ActiveStress(formulation=ActiveStressFormulation.stretch, **kwargs).S(C)

    resid = S_inv - fiber_stretch(C, f0) * S_str
    assert _assemble(ufl.inner(resid, resid), mesh) < 1e-18


def test_invariant_formulation_is_unchanged(mesh, f0):
    """The default must still be the historical behaviour, S = Ta f0 x f0."""
    u = _uniaxial_u(mesh, 1.2)
    F_grad = ufl.Identity(3) + ufl.grad(u)
    C = F_grad.T * F_grad

    active = pulse.ActiveStress(f0=f0, activation=Variable(11.0, "kPa"))
    assert active.formulation == ActiveStressFormulation.invariant

    expected = (11.0 * 1e3) * ufl.outer(f0, f0)
    resid = active.S(C) - expected
    assert _assemble(ufl.inner(resid, resid), mesh) < 1e-18


def test_stress_matches_differentiated_strain_energy(mesh, f0):
    """S given in closed form must equal 2 dPsi/dC, for both classes.

    Guards against the closed form and the potential drifting apart, since
    only the closed form is used in assembly.
    """
    u = _uniaxial_u(mesh, 1.18)
    F_grad = ufl.Identity(3) + ufl.grad(u)
    C = ufl.variable(F_grad.T * F_grad)

    Vs = dolfinx.fem.functionspace(mesh, ("DG", 0))
    lmbda_prev = dolfinx.fem.Function(Vs)
    lmbda_prev.x.array[:] = 1.05

    models = [
        pulse.ActiveStress(
            f0=f0,
            activation=Variable(23.0, "kPa"),
            formulation=ActiveStressFormulation.stretch,
        ),
        StabilizedActiveStress(
            f0=f0,
            activation=Variable(23.0, "kPa"),
            active_stiffness=Variable(310.0, "kPa"),
            lmbda_prev=lmbda_prev,
        ),
    ]

    for model in models:
        resid = model.S(C) - 2.0 * ufl.diff(model.strain_energy(C), C)
        err = _assemble(ufl.inner(resid, resid), mesh)
        assert err < 1e-16, f"{type(model).__name__}: S != 2 dPsi/dC (residual {err})"


def test_zero_stiffness_recovers_plain_active_stress(mesh, f0):
    """Ka = 0 must reproduce the unstabilized scheme -- the configuration the
    instability tests in downstream packages need."""
    u = _uniaxial_u(mesh, 1.2)
    F_grad = ufl.Identity(3) + ufl.grad(u)
    C = F_grad.T * F_grad

    Vs = dolfinx.fem.functionspace(mesh, ("DG", 0))
    lmbda_prev = dolfinx.fem.Function(Vs)
    lmbda_prev.x.array[:] = 1.07  # deliberately != lmbda(u)

    stabilized = StabilizedActiveStress(
        f0=f0,
        activation=Variable(19.0, "kPa"),
        active_stiffness=Variable(0.0, "kPa"),
        lmbda_prev=lmbda_prev,
    )
    plain = pulse.ActiveStress(
        f0=f0,
        activation=Variable(19.0, "kPa"),
        formulation=ActiveStressFormulation.stretch,
    )

    resid = stabilized.S(C) - plain.S(C)
    assert _assemble(ufl.inner(resid, resid), mesh) < 1e-18


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_defaults_to_reference_configuration(mesh, f0):
    """Without lmbda_prev, the model must start from the undeformed state."""
    active = StabilizedActiveStress(f0=f0)
    assert isinstance(active.lmbda_prev, dolfinx.fem.Constant)
    np.testing.assert_allclose(float(active.lmbda_prev.value), 1.0)


def test_update_prev_rejects_constant(mesh, f0):
    """A Constant cannot hold a spatially varying stretch; failing loudly beats
    silently freezing lmbda_prev at 1.0 for a whole simulation."""
    active = StabilizedActiveStress(f0=f0)
    with pytest.raises(TypeError, match="lmbda_prev"):
        active.update_prev(_uniaxial_u(mesh, 1.1))


def test_update_prev_records_current_stretch(mesh, f0):
    stretch = 1.23
    Vs = dolfinx.fem.functionspace(mesh, ("DG", 0))
    lmbda_prev = dolfinx.fem.Function(Vs)
    active = StabilizedActiveStress(f0=f0, lmbda_prev=lmbda_prev)

    active.update_prev(_uniaxial_u(mesh, stretch))
    np.testing.assert_allclose(lmbda_prev.x.array, stretch, rtol=1e-9)


def test_stretch_formulation_rejects_transverse_activation(mesh, f0):
    active = pulse.ActiveStress(
        f0=f0,
        activation=Variable(10.0, "kPa"),
        eta=0.3,
        formulation=ActiveStressFormulation.stretch,
    )
    u = _uniaxial_u(mesh, 1.1)
    F_grad = ufl.Identity(3) + ufl.grad(u)
    with pytest.raises(NotImplementedError, match="eta"):
        active.S(F_grad.T * F_grad)


def test_usable_in_a_cardiac_model(mesh, f0):
    """The whole point is that it drops into CardiacModel like any other
    active model."""
    material = pulse.HolzapfelOgden(
        f0=f0,
        s0=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 1.0, 0.0))),
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    model = pulse.CardiacModel(
        material=material,
        active=StabilizedActiveStress(f0=f0, activation=Variable(20.0, "kPa")),
        # Compressible, not Incompressible: the latter needs a registered
        # pressure field, which is irrelevant to what this test checks.
        compressibility=pulse.Compressible(),
    )
    u = _uniaxial_u(mesh, 1.1)
    F_grad = ufl.Identity(3) + ufl.grad(u)
    # The material derives its stress by differentiating w.r.t. C, so C must
    # be a ufl.variable here.
    C = ufl.variable(F_grad.T * F_grad)
    assert _assemble(ufl.inner(model.S(C), ufl.Identity(3)), mesh) != 0.0
