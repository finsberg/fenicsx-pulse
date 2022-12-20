import math

import dolfinx
import pulsex
import pytest
import ufl
import utils


@pytest.mark.parametrize("obj_str", ("float", "Constant", "Function"))
def test_linear_elastic_model(obj_str, mesh, P1, u):
    E = 2.0
    _nu = 0.2
    nu = utils.float2object(f=_nu, obj_str=obj_str, mesh=mesh, V=P1)
    model = pulsex.LinearElastic(E=E, nu=nu)

    u.interpolate(lambda x: x)
    F = pulsex.kinematics.DeformationGradient(u)
    # F = 2I, e = I, tr(e) = 3
    # sigma = (E / (1 + nu)) * (e + (nu / (1 - 2 * nu)) * tr(e) * I
    # sigma = (E / (1 + nu)) * (1 + (nu / (1 - 2 * nu)) * 3) * I
    sigma = model.sigma(F)
    I = ufl.Identity(3)
    zero = sigma - (E / (1 + _nu)) * (1 + (_nu / (1 - 2 * _nu)) * 3) * I
    assert utils.matrix_is_zero(zero)


@pytest.mark.parametrize("obj_str", ("float", "Constant", "Function"))
def test_linear_elastic_model_with_invalid_range(obj_str, mesh, P1):
    E = 2.0
    _nu = 0.5
    nu = utils.float2object(f=_nu, obj_str=obj_str, mesh=mesh, V=P1)

    with pytest.raises(pulsex.exceptions.InvalidRangeError):
        pulsex.LinearElastic(E=E, nu=nu)


@pytest.mark.parametrize(
    "params_func, expected_value",
    (
        (pulsex.HolzapfelOgden.orthotropic_parameters, 1.2352937267),
        (pulsex.HolzapfelOgden.partly_orthotropic_parameters, 1.435870273157),
        (pulsex.HolzapfelOgden.transversely_isotropic_parameters, 53.6468124607508),
    ),
)
def test_holzapfel_ogden(params_func, expected_value, mesh, u):
    params = params_func()
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    model = pulsex.HolzapfelOgden(f0=f0, s0=s0, **params)

    u.interpolate(lambda x: x / 10)
    F = pulsex.kinematics.DeformationGradient(u)
    # F = I + 0.1 I, C = 1.21 I
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    assert math.isclose(value, expected_value)
