import itertools

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

from pulse import Variable, kinematics
from pulse.active_model import Passive
from pulse.active_stress import ActiveStress, FrankStarlingActiveStress


def W_fun(Ta: float, eta: float = 0.0) -> float:
    return 0.5 * Ta * ((4 - 1) + eta * ((12 - 3) - (4 - 1)))


@pytest.mark.parametrize(
    "eta, Ta",
    itertools.product(
        (0.0, 0.2, 0.5, 1.0),
        (0.0, 1.0, 2.0),
    ),
)
def test_transversely_active_stress(eta, Ta, mesh, u) -> None:
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    active_model = ActiveStress(f0, eta=eta)

    u.interpolate(lambda x: x)
    F = kinematics.DeformationGradient(u)
    C = F.T * F

    W = active_model.strain_energy(C)

    assert active_model.Fe(F) is F
    active_model.activation.assign(Ta)

    assert np.isclose(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(W * ufl.dx)),
        W_fun(Ta=Ta, eta=eta),
    )

    assert np.isclose(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(active_model.Ta * ufl.dx)),
        Ta,
    )
    active_model.T_ref.value = 2.0
    assert np.isclose(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(active_model.Ta * ufl.dx)),
        2 * Ta,
    )


def test_Passive(u) -> None:
    active_model = Passive()

    u.interpolate(lambda x: x)
    F = kinematics.DeformationGradient(u)
    C = F.T * F

    assert active_model.Fe(F) is F
    W = active_model.strain_energy(C)
    assert np.isclose(dolfinx.fem.assemble_scalar(dolfinx.fem.form(W * ufl.dx)), 0.0)


@pytest.fixture
def base_active_model(mesh):
    """Initializes the active stress model with standard test parameters."""
    f0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))
    Ta = Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(5.0)), "kPa")

    # Simple parameters for easy mental math:
    # slope = (2.0 - 0.5) / (1.2 - 1.0) = 7.5
    model = FrankStarlingActiveStress(
        f0=f0,
        activation=Ta,
        amp_min=0.5,
        amp_max=2.0,
        stretch_threshold=1.0,
        stretch_optimal=1.2,
    )
    return model


def test_initialization(mesh):
    """Test that the dataclass initializes correctly with default/custom values."""
    # f0 = ufl.as_vector([1.0, 0.0, 0.0])
    # gamma = ufl.as_ufl(1.0)
    f0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))
    gamma = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))

    # Test Defaults
    model_default = FrankStarlingActiveStress(f0=f0, activation=gamma)
    assert model_default.amp_min == 0.0
    assert model_default.amp_max == 1.0

    # Test Overrides
    model_custom = FrankStarlingActiveStress(f0=f0, activation=gamma, amp_max=3.0)
    assert model_custom.amp_max == 3.0


def test_unregistered_u_raises_error(base_active_model):
    """Test that evaluating the multiplier without a registered displacement raises an error."""
    with pytest.raises(ValueError, match="Displacement 'u' has not been registered"):
        base_active_model.frank_starling_multiplier()


@pytest.mark.parametrize(
    "stretch, expected_multiplier",
    [
        (0.90, 0.5),  # Below stretch_threshold -> amp_min
        (1.00, 0.5),  # Exactly at stretch_threshold -> amp_min
        (1.10, 1.25),  # Midway on ascending limb: 0.5 + 7.5*(1.1 - 1.0) = 1.25
        (1.20, 2.0),  # Exactly at stretch_optimal -> amp_max
        (1.30, 2.0),  # Above stretch_optimal -> amp_max
    ],
)
def test_frank_starling_multiplier_evaluation(
    mesh,
    base_active_model,
    stretch,
    expected_multiplier,
):
    """
    Test the piecewise stretch mathematical logic.
    We prescribe a uniform displacement field, integrate the UFL expression
    over the unit volume, and check the result.
    """
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (mesh.topology.dim,)))
    u = dolfinx.fem.Function(V)

    # Prescribe uniform stretch in X direction: u(x) = (stretch - 1.0) * X
    # This guarantees lambda_f = stretch uniformly across the whole domain.
    u.interpolate(lambda x: ((stretch - 1.0) * x[0], 0.0 * x[1], 0.0 * x[2]))

    # Register displacement
    base_active_model.register(u)

    # Get the UFL expression
    g_lam_expr = base_active_model.frank_starling_multiplier()

    # Integrate expression over unit domain. (Integral of a constant C over volume 1.0 == C)
    form = dolfinx.fem.form(g_lam_expr * ufl.dx)
    computed_multiplier = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)

    # Assert floating point near equality
    np.testing.assert_allclose(computed_multiplier, expected_multiplier, rtol=1e-5)


def test_active_tension_evaluation(mesh, base_active_model):
    """Test that the property Ta correctly multiplies baseline activation with the multiplier."""
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (mesh.topology.dim,)))
    u = dolfinx.fem.Function(V)

    # Stretch = 1.10 -> Multiplier should be 1.25 (from the math in previous test)
    # Baseline activation = 5.0 (from fixture)
    # Expected Ta = 5.0 * 1.25 = 6.25 kPa = 6250 Pa
    u.interpolate(lambda x: ((1.10 - 1.0) * x[0], 0.0 * x[1], 0.0 * x[2]))

    base_active_model.register(u)
    Ta_expr = base_active_model.Ta

    form = dolfinx.fem.form(Ta_expr * ufl.dx)
    computed_Ta = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)

    np.testing.assert_allclose(computed_Ta, 6250.0, rtol=1e-5)
