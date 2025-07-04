import dolfinx
import numpy as np
import pytest
import ufl

from pulse import invariants, kinematics


@pytest.mark.parametrize(
    "cls, expected",
    (
        (kinematics.DeformationGradient, 4 * 3),
        (kinematics.IsochoricDeformationGradient_from_u, pow(8, -2 / 3) * 4 * 3),
    ),
)
def test_I1(cls, expected, u) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    C = F.T * F
    I1 = invariants.I1(C)

    assert np.isclose(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(I1 * ufl.dx)),
        expected,
    )


@pytest.mark.parametrize(
    "cls, expected",
    (
        (kinematics.DeformationGradient, 0.5 * ((4 * 3) ** 2 - (4**2) * 3)),
        (
            kinematics.IsochoricDeformationGradient_from_u,
            0.5 * (((pow(8, -2 / 3) * 4 * 3) ** 2) - ((pow(8, -2 / 3) * 4) ** 2) * 3),
        ),
    ),
)
def test_I2(cls, expected, u) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    C = F.T * F
    I2 = invariants.I2(C)
    assert np.isclose(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(I2 * ufl.dx)),
        expected,
    )


@pytest.mark.parametrize(
    "cls, expected",
    (
        (kinematics.DeformationGradient, 4**3),
        (kinematics.IsochoricDeformationGradient_from_u, (pow(8, -2 / 3) * 4) ** 3),
    ),
)
def test_I3(cls, expected, u) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    C = F.T * F
    I3 = invariants.I3(C)

    assert np.isclose(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(I3 * ufl.dx)),
        expected,
    )


@pytest.mark.parametrize(
    "cls, expected",
    (
        (kinematics.DeformationGradient, 4),
        (kinematics.IsochoricDeformationGradient_from_u, pow(8, -2 / 3) * 4),
    ),
)
def test_I4(cls, expected, u, mesh) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    C = F.T * F
    a0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    I4 = invariants.I4(C, a0)

    assert np.isclose(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(I4 * ufl.dx)),
        expected,
    )


@pytest.mark.parametrize(
    "cls, expected",
    (
        (kinematics.DeformationGradient, 4**2),
        (kinematics.IsochoricDeformationGradient_from_u, (pow(8, -2 / 3) * 4) ** 2),
    ),
)
def test_I5(cls, expected, u, mesh) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    C = F.T * F
    a0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    I5 = invariants.I5(C, a0)

    assert np.isclose(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(I5 * ufl.dx)),
        expected,
    )


@pytest.mark.parametrize(
    "cls, expected",
    ((kinematics.DeformationGradient, 0), (kinematics.IsochoricDeformationGradient_from_u, 0)),
)
def test_I8(cls, expected, u, mesh) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    C = F.T * F
    a0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    b0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    I8 = invariants.I8(C, a0, b0)
    assert np.isclose(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(I8 * ufl.dx)),
        expected,
    )
