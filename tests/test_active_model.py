import itertools

import dolfinx
import numpy as np
import pytest
import ufl

from fenicsx_pulse import kinematics
from fenicsx_pulse.active_model import Passive
from fenicsx_pulse.active_stress import ActiveStress


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
    W = active_model.strain_energy(F)

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

    assert active_model.Fe(F) is F
    W = active_model.strain_energy(F)
    assert np.isclose(dolfinx.fem.assemble_scalar(dolfinx.fem.form(W * ufl.dx)), 0.0)
