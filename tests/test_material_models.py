import math

import dolfinx
import fenicsx_pulse
import pytest
import ufl
import utils


@pytest.mark.parametrize("obj_str", ("float", "Constant", "Function"))
def test_linear_elastic_model(obj_str, mesh, P1, u) -> None:
    E = 2.0
    _nu = 0.2
    nu = utils.float2object(f=_nu, obj_str=obj_str, mesh=mesh, V=P1)
    model = fenicsx_pulse.LinearElastic(E=E, nu=nu)

    u.interpolate(lambda x: x)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    # F = 2I, e = I, tr(e) = 3
    # sigma = (E / (1 + nu)) * (e + (nu / (1 - 2 * nu)) * tr(e) * I
    # sigma = (E / (1 + nu)) * (1 + (nu / (1 - 2 * nu)) * 3) * I
    sigma = model.sigma(F)
    I = ufl.Identity(3)
    zero = sigma - (E / (1 + _nu)) * (1 + (_nu / (1 - 2 * _nu)) * 3) * I
    assert utils.matrix_is_zero(zero)


@pytest.mark.parametrize("obj_str", ("float", "Constant", "Function"))
def test_linear_elastic_model_with_invalid_range(obj_str, mesh, P1) -> None:
    E = 2.0
    _nu = 0.5
    nu = utils.float2object(f=_nu, obj_str=obj_str, mesh=mesh, V=P1)

    with pytest.raises(fenicsx_pulse.exceptions.InvalidRangeError):
        fenicsx_pulse.LinearElastic(E=E, nu=nu)


@pytest.mark.parametrize(
    "params_func, expected_value",
    (
        (fenicsx_pulse.HolzapfelOgden.orthotropic_parameters, 1.2352937267),
        (fenicsx_pulse.HolzapfelOgden.partly_orthotropic_parameters, 1.435870273157),
        (
            fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters,
            53.6468124607508,
        ),
    ),
)
def test_holzapfel_ogden(params_func, expected_value, mesh, u) -> None:
    params = params_func()
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    model = fenicsx_pulse.HolzapfelOgden(f0=f0, s0=s0, **params)

    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    # F = I + 0.1 I, C = 1.21 I
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    assert math.isclose(value, expected_value)


def test_holzapfel_ogden_invalid_range():
    with pytest.raises(fenicsx_pulse.exceptions.InvalidRangeError):
        fenicsx_pulse.HolzapfelOgden(a=-1.0)


@pytest.mark.parametrize(
    "params, attr",
    (
        ({"a_f": 1}, "f0"),
        ({"a_s": 1}, "s0"),
        ({"a_fs": 1}, "f0 and/or s0"),
    ),
)
def test_holzapfel_ogden_raises_MissingModelAttribute(params, attr):
    with pytest.raises(fenicsx_pulse.exceptions.MissingModelAttribute) as e:
        fenicsx_pulse.HolzapfelOgden(**params)
    assert e.value == fenicsx_pulse.exceptions.MissingModelAttribute(
        attr=attr,
        model="HolzapfelOgden",
    )


def test_holzapfel_ogden_neohookean(u):
    model = fenicsx_pulse.HolzapfelOgden(a=1.0)
    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I, I1= 3*1.21
    # psi = (a / 2) * (I1 - 3) = 0.5 (3 * 1.21 - 3) = 0.315
    assert math.isclose(value, 0.315)


def test_holzapfel_ogden_pure_fiber(u, mesh):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    model = fenicsx_pulse.HolzapfelOgden(a_f=1.0, f0=f0)
    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I, I4f = 1.21
    # psi = (a_f / 2) * (I4 - 1)**2 = 0.5 * 0.21**2
    assert math.isclose(value, 0.5 * 0.21**2)


def test_holzapfel_ogden_pure_fiber_sheets(u, mesh):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    model = fenicsx_pulse.HolzapfelOgden(a_fs=1.0, f0=f0, s0=f0)
    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, = 1.1 -> I8fs = 1.21
    # psi = (a_f / 2) * I8fs**2 = 0.5 * 1.21**2
    assert math.isclose(value, 0.5 * 1.21**2)
