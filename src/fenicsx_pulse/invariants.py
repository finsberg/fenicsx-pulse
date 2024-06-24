r"""This module defines invariants of the deformation gradient.

The deformation gradient :math:`\mathbf{F}` is a tensor field that describes the
deformation of a material. The invariants of the deformation gradient are scalar
quantities that are invariant under rigid body motion.

The invariants are defined as follows:

- First principal invariant
    :math:`I_1 = \text{tr}(\mathbf{C})`
- Second principal invariant
    :math:`I_2 = \left( I_1^2 - \text{tr}(\mathbf{C}\cdot\mathbf{C})\right)`
- Third principal invariant
    :math:`I_3 = \text{det}(\mathbf{C})`
- Fourth quasi invariant
    :math:`I_{4\mathbf{a_0}} = \mathbf{C}\mathbf{a}_0 \cdot \mathbf{a}_0`
- Fifth quasi invariant
    :math:`I_{5\mathbf{a_0}} = \mathbf{C}\mathbf{a}_0 \cdot \mathbf{C}\mathbf{a}_0`
- Eight quasi invariant
    :math:`I_{8\mathbf{a_0}\mathbf{b_0}} = \mathbf{F}\mathbf{a}_0 \cdot \mathbf{F}\mathbf{b}_0`

where :math:`\mathbf{C} = \mathbf{F}^T\mathbf{F}` is the right Cauchy-Green tensor.
"""

import ufl

from . import kinematics
from ._types import FuncConst


def I1(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""First principal invariant

    .. math::
        I_1 = \mathrm{tr}(\mathbf{C})

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        First principal invariant

    """
    C = kinematics.RightCauchyGreen(F)
    I1 = ufl.tr(C)
    return I1


def I2(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""Second principal invariant

    .. math::
        I_2 = \left( I_1^2 - \mathrm{tr}(\mathbf{C}\cdot\mathbf{C})\right)

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        Second principal invariant

    """
    C = kinematics.RightCauchyGreen(F)
    return 0.5 * (I1(F) * I1(F) - ufl.tr(C * C))


def I3(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""Third principal invariant

    .. math::
        I_3 = \mathrm{det}(\mathbf{C})

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        Third principal invariant

    """
    C = kinematics.RightCauchyGreen(F)
    return ufl.det(C)


def I4(F: ufl.core.expr.Expr, a0: FuncConst) -> ufl.core.expr.Expr:
    r"""Fourth quasi invariant

    .. math::
        I_{4\mathbf{a_0}} = \mathbf{C}\mathbf{a}_0 \cdot \mathbf{a}_0

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient
    a0 : FuncConst
        Some direction

    Returns
    -------
    ufl.core.expr.Expr
        Fourth quasi invariant in the direction a0

    """
    C = kinematics.RightCauchyGreen(F)
    return ufl.inner(C * a0, a0)


def I5(F: ufl.core.expr.Expr, a0: FuncConst) -> ufl.core.expr.Expr:
    r"""Fifth quasi invariant

    .. math::
        I_{5\mathbf{a_0}} = \mathbf{C}\mathbf{a}_0 \cdot \mathbf{C}\mathbf{a}_0

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient
    a0 : FuncConst
        Some direction

    Returns
    -------
    ufl.core.expr.Expr
        Fifth quasi invariant in the direction a0

    """
    C = kinematics.RightCauchyGreen(F)
    return ufl.inner(C * a0, C * a0)


def I8(F: ufl.core.expr.Expr, a0: FuncConst, b0: FuncConst) -> ufl.core.expr.Expr:
    r"""Eight quasi invariant

    .. math::
        I_{8\mathbf{a_0}\mathbf{b_0}}
        = \mathbf{F}\mathbf{a}_0 \cdot \mathbf{F}\mathbf{b}_0

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient
    a0 : FuncConst
        Some direction
    b0 : FuncConst
        Another direction

    Returns
    -------
    ufl.core.expr.Expr
        Eight quasi invariant in the direction a0

    """
    return ufl.inner(F * a0, F * b0)
