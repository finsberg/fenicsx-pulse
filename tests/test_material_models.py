import math

import dolfinx
import pytest
import ufl

import pulse


@pytest.mark.parametrize(
    "params_func, expected_value",
    (
        (pulse.HolzapfelOgden.orthotropic_parameters, 1235.2937267586497),
        (pulse.HolzapfelOgden.partly_orthotropic_parameters, 1435.8702731579442),
        (
            pulse.HolzapfelOgden.transversely_isotropic_parameters,
            53646.81246074524,
        ),
    ),
)
def test_holzapfel_ogden(params_func, expected_value, mesh, u) -> None:
    params = params_func()
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    model = pulse.HolzapfelOgden(f0=f0, s0=s0, **params)

    u.interpolate(lambda x: x / 10)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    # F = I + 0.1 I, C = 1.21 I
    psi = model.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    assert math.isclose(value, expected_value)


def test_holzapfel_ogden_invalid_range():
    with pytest.raises(pulse.exceptions.InvalidRangeError):
        pulse.HolzapfelOgden(a=-1.0)


@pytest.mark.parametrize(
    "params, attr",
    (
        ({"a_f": 1}, "f0"),
        ({"a_s": 1}, "s0"),
        ({"a_fs": 1}, "f0 and/or s0"),
    ),
)
def test_holzapfel_ogden_raises_MissingModelAttribute(params, attr):
    with pytest.raises(pulse.exceptions.MissingModelAttribute) as e:
        pulse.HolzapfelOgden(**params)
    assert e.value == pulse.exceptions.MissingModelAttribute(
        attr=attr,
        model="HolzapfelOgden",
    )


def test_holzapfel_ogden_neohookean(u):
    model = pulse.HolzapfelOgden(a=1.0)
    u.interpolate(lambda x: x / 10)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I, I1= 3*1.21
    # psi = (a / 2) * (I1 - 3) = 0.5 (3 * 1.21 - 3) = 0.315
    # Convert to Pa

    assert math.isclose(value, 315.0)


def test_holzapfel_ogden_pure_fiber(u, mesh):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    model = pulse.HolzapfelOgden(a_f=1.0, f0=f0)
    u.interpolate(lambda x: x / 10)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I, I4f = 1.21
    # psi = (a_f / 2) * (I4 - 1)**2 = 0.5 * 0.21**2
    assert math.isclose(value, 1000 * 0.5 * 0.21**2)


def test_holzapfel_ogden_pure_fiber_sheets(u, mesh):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    model = pulse.HolzapfelOgden(a_fs=1.0, f0=f0, s0=f0)
    u.interpolate(lambda x: x / 10)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, = 1.1 -> I8fs = 1.21
    # psi = (a_f / 2) * I8fs**2 = 0.5 * 1.21**2
    assert math.isclose(value, 1000 * 0.5 * 1.21**2)


def test_neo_hookean(u, mesh):
    model = pulse.NeoHookean(mu=1.0)
    u.interpolate(lambda x: x / 10)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I
    # psi = (mu / 2) * (I1 - 3) = 0.5 * (3.63 - 3)
    assert math.isclose(value, 1000 * 0.5 * 0.63)


def test_saint_venant_kirchhoff(u):
    lmbda = 1.0
    mu = 1.0
    model = pulse.SaintVenantKirchhoff(lmbda=lmbda, mu=mu)
    u.interpolate(lambda x: x / 10)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # gradu = 0.1 I, C = 1.21 I , E = 0.5 * (C - I) = 0.105 I
    # tr(E) = 0.105 * 3 = 0.315, tr(E * E) = 0.105**2 * 3
    # psi = lmbda / 2 * tr(E)**2 + mu * tr(E * E)
    expected = 0.5 * lmbda * 0.315**2 + mu * 3 * (0.105**2)

    assert math.isclose(value, expected)


def test_guccione_isotropic(u):
    C = 10.0
    bf = bt = bfs = 1.0
    model = pulse.Guccione(C=C, bf=bf, bt=bt, bfs=bfs)
    assert model.is_isotropic()

    u.interpolate(lambda x: x / 10)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)
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
    model = pulse.Guccione(C=C, bf=bf, bt=bt, bfs=bfs, f0=f0, s0=s0, n0=n0)
    assert not model.is_isotropic()

    u.interpolate(lambda x: x / 10)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    # F = I + 0.1 I, C = 1.21 I, E = 0.5 * (C - I) = 0.105 I
    # E11 = E22 = E33 = 0.105, E12 = E13 = E23 = 0
    # Q = bf * E11**2 + bt * (E22**2 + E33**2 + 2 * E23**2) + bfs * (2 * E12**2 + 2 * E13**2)
    # Q = E11**2 (bf + 2 bt)
    Q = (0.105**2) * (1 + 2 * 2)
    assert math.isclose(value, 1000 * 0.5 * 10.0 * (math.exp(Q) - 1))
