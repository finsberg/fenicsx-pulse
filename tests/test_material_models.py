import math

import dolfinx
import pytest
import ufl
import utils

import fenicsx_pulse


@pytest.mark.parametrize("obj_str", ("float", "Constant", "Function"))
def test_linear_elastic_model(obj_str, mesh, P1, u) -> None:
    E = 2.0
    _nu = 0.2
    nu = utils.float2object(f=_nu, obj_str=obj_str, mesh=mesh, V=P1)
    model = fenicsx_pulse.LinearElastic(E=fenicsx_pulse.Variable(E, "Pa"), nu=nu)

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
    E = fenicsx_pulse.Variable(2.0, "kPa")
    _nu = 0.5
    nu = utils.float2object(f=_nu, obj_str=obj_str, mesh=mesh, V=P1)

    with pytest.raises(fenicsx_pulse.exceptions.InvalidRangeError):
        fenicsx_pulse.LinearElastic(E=E, nu=nu)


@pytest.mark.parametrize(
    "params_func, expected_value",
    (
        (fenicsx_pulse.HolzapfelOgden.orthotropic_parameters, 1235.2937267586497),
        (fenicsx_pulse.HolzapfelOgden.partly_orthotropic_parameters, 1435.8702731579442),
        (
            fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters,
            53646.81246074524,
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
    # Convert to Pa

    assert math.isclose(value, 315.0)


def test_holzapfel_ogden_pure_fiber(u, mesh):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    model = fenicsx_pulse.HolzapfelOgden(a_f=1.0, f0=f0)
    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I, I4f = 1.21
    # psi = (a_f / 2) * (I4 - 1)**2 = 0.5 * 0.21**2
    assert math.isclose(value, 1000 * 0.5 * 0.21**2)


def test_holzapfel_ogden_pure_fiber_sheets(u, mesh):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    model = fenicsx_pulse.HolzapfelOgden(a_fs=1.0, f0=f0, s0=f0)
    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, = 1.1 -> I8fs = 1.21
    # psi = (a_f / 2) * I8fs**2 = 0.5 * 1.21**2
    assert math.isclose(value, 1000 * 0.5 * 1.21**2)


def test_neo_hookean(u, mesh):
    model = fenicsx_pulse.NeoHookean(mu=1.0)
    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I
    # psi = (mu / 2) * (I1 - 3) = 0.5 * (3.63 - 3)
    assert math.isclose(value, 1000 * 0.5 * 0.63)


def test_saint_venant_kirchhoff(u):
    lmbda = 1.0
    mu = 1.0
    model = fenicsx_pulse.SaintVenantKirchhoff(lmbda=lmbda, mu=mu)
    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # gradu = 0.1 I, epsilon = 0.5 (gradu + gradu.T) = 0.1 I
    # tr(epsilon) = 0.1 * 3 = 0.3, tr(epsilon * epsilon) = 0.1**2 * 3 = 0.03
    # psi = lmbda / 2 * tr(epsilon)**2 + mu * tr(epsilon * epsilon)
    expected = 0.5 * lmbda * 0.3**2 + mu * 0.03
    assert math.isclose(value, expected)


def test_guccione_isotropic(u):
    C = 10.0
    bf = bt = bfs = 1.0
    model = fenicsx_pulse.Guccione(C=C, bf=bf, bt=bt, bfs=bfs)
    assert model.is_isotropic()

    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I, E = 0.5 * (C - I) = 0.105 I
    # E * E = 0.105**2 I, tr(E * E) = 0.105**2 * 3
    # psi = 0.5 * C * (exp(E * E) - 1) = 0.5 * 10.0 * (exp(0.105 ** 2) - 1)
    assert math.isclose(value, 1000 * 0.5 * 10.0 * (math.exp(3 * (0.105**2)) - 1))


def test_guccione_anisotropic(u, mesh):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    n0 = dolfinx.fem.Constant(mesh, (0.0, 0.0, 1.0))

    C = 10.0
    bf = 1.0
    bt = 2.0
    bfs = 3.0
    model = fenicsx_pulse.Guccione(C=C, bf=bf, bt=bt, bfs=bfs, f0=f0, s0=s0, n0=n0)
    assert not model.is_isotropic()

    u.interpolate(lambda x: x / 10)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I, E = 0.5 * (C - I) = 0.105 I
    # E11 = E22 = E33 = 0.105, E12 = E13 = E23 = 0
    # Q = bf * E11**2 + bt * (E22**2 + E33**2 + 2 * E23**2) + bfs * (2 * E12**2 + 2 * E13**2)
    # Q = E11**2 (bf + 2 bt)
    Q = (0.105**2) * (1 + 2 * 2)
    assert math.isclose(value, 1000 * 0.5 * 10.0 * (math.exp(Q) - 1))
