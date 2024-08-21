import pytest
import ufl
import utils

from fenicsx_pulse import kinematics


def test_SecondOrderIdentity(u) -> None:
    assert kinematics.SecondOrderIdentity(u) == ufl.Identity(3)


@pytest.mark.parametrize(
    "cls, factor",
    (
        (kinematics.DeformationGradient, 2),
        (utils.IsochoricDeformationGradient, pow(8, -1 / 3) * 2),
    ),
)
def test_DeformationGradient(cls, factor, u) -> None:
    r"""Test deformation gradient for a linear displacement field, i.e

    .. math::
        u(x) = x

    Notes
    -----
    If not isochoric then

    .. math::
        \mathbf{F} = I + \nabla u = I + I = 2I

    If isochoric then

    .. math::
        \mathbf{F}_{\mathrm{iso}} = J^{-1/3} F = 8^{-1/3} F = 8^{-1/3} 2 I

    """
    u.interpolate(lambda x: x)
    F = cls(u)
    assert F.ufl_shape == (3, 3)
    # Since u is linear (u = x),  F = I + I = 2 I
    zero = F - factor * ufl.Identity(3)
    assert utils.matrix_is_zero(zero)


@pytest.mark.parametrize(
    "cls, factor",
    (
        (kinematics.DeformationGradient, 4),
        (utils.IsochoricDeformationGradient, pow(8, -2 / 3) * 4),
    ),
)
def test_RightCauchyGreen(cls, factor, u) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    C = kinematics.RightCauchyGreen(F)
    assert C.ufl_shape == (3, 3)
    zero = C - factor * ufl.Identity(3)
    assert utils.matrix_is_zero(zero)


@pytest.mark.parametrize(
    "cls, factor",
    (
        (kinematics.DeformationGradient, 4),
        (utils.IsochoricDeformationGradient, pow(8, -2 / 3) * 4),
    ),
)
def test_LeftCauchyGreen(cls, factor, u) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    B = kinematics.LeftCauchyGreen(F)
    assert B.ufl_shape == (3, 3)
    zero = B - factor * ufl.Identity(3)
    assert utils.matrix_is_zero(zero)


@pytest.mark.parametrize(
    "cls, factor",
    (
        (kinematics.DeformationGradient, 1),
        (utils.IsochoricDeformationGradient, pow(8, -1 / 3) * 2 - 1),
    ),
)
def test_EngineeringStrain(cls, factor, u) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    E = kinematics.EngineeringStrain(F)
    assert E.ufl_shape == (3, 3)
    zero = E - factor * ufl.Identity(3)
    assert utils.matrix_is_zero(zero)


@pytest.mark.parametrize(
    "cls, factor",
    (
        (kinematics.DeformationGradient, 1.5),
        (utils.IsochoricDeformationGradient, 0.5 * (pow(8, -2 / 3) * 4 - 1)),
    ),
)
def test_GreenLagrangeStrain(cls, factor, u) -> None:
    u.interpolate(lambda x: x)
    F = cls(u)
    E = kinematics.GreenLagrangeStrain(F)
    assert E.ufl_shape == (3, 3)
    zero = E - factor * ufl.Identity(3)
    assert utils.matrix_is_zero(zero)


def test_PiolaTransform(u):
    u.interpolate(lambda x: x)
    F = kinematics.DeformationGradient(u)
    A = F
    B = kinematics.PiolaTransform(A, F)
    C = kinematics.InversePiolaTransform(B, F)
    zero = A - C
    assert utils.matrix_is_zero(zero)
