import dolfinx
import ufl


# Second order identity tensor
def SecondOrderIdentity(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """Return identity with same dimension as input"""
    dim = ufl.domain.find_geometric_dimension(F)
    return ufl.Identity(dim)


def DeformationGradient(
    u: dolfinx.fem.Function,
) -> ufl.core.expr.Expr:
    r"""Return deformation gradient from displacement.

    Parameters
    ----------
    u : dolfinx.fem.Function
        The displacement field
    isochoric : bool, optional
        If true return the isochoric deformation gradient, by default False

    Returns
    -------
    ufl.core.expr.Expr
        The deformation gradient

    Notes
    -----

    Given a displacement field :math:`\mathbf{u}`, the deformation gradient
    is given by

    .. math::
        \mathbf{F} = \mathbf{I} + \nabla \mathbf{u}

    """
    I = SecondOrderIdentity(u)
    return I + ufl.grad(u)


def IsochoricDeformationGradient(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""Return the isochoric deformation gradient

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        The isochoric deformation gradient

    Notes
    -----
    We can decompose the the deformation gradient multiplicatively into
    the volumetric (:math:`\mathbf{F}_{\mathrm{vol}}`) and isochoric
    (:math:`\mathbf{F}_{\mathrm{iso}}`) components

    .. math::
        \mathbf{F} = \mathbf{F}_{\mathrm{iso}} \cdot \mathbf{F}_{\mathrm{vol}}

    such that :math:`\mathrm{det}(\mathbf{F}_{\mathrm{iso}}) = 1`. In this case,
    we can work out that

    .. math::
        \mathbf{F}_{\mathrm{vol}} = J^{1/3}\mathbf{I}

    and consequently

    .. math::
        \mathbf{F}_{\mathrm{vol}} = J^{-1/3}\mathbf{F}

    with :math:`J = \mathrm{det}(\mathbf{F})`.

    """
    J = Jacobian(F)
    dim = ufl.domain.find_geometric_dimension(F)
    return pow(J, -1.0 / float(dim)) * F


def Jacobian(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""Determinant of the deformation gradient

    .. math::
        J = \mathrm{det}(\mathbf{F})

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        The jacobian


    """
    return ufl.det(F)


def LeftCauchyGreen(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""Left Cauchy-Green deformation tensor

    .. math::
        \mathbf{C} = \mathbf{F}\mathbf{F}^T


    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        Left Cauchy-Green deformation tensor

    """
    return F * F.T


def RightCauchyGreen(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""Right Cauchy-Green deformation tensor

    .. math::
        \mathbf{C} = \mathbf{F}^T\mathbf{F}

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        Right Cauchy-Green deformation tensor

    """
    return F.T * F


def EngineeringStrain(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""Engineering strain

    .. math::
        \mathbf{\varepsilon} = \frac{1}{2}\left( \nabla u + (\nabla u)^T \right)

    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        Engineering strain

    """
    I = SecondOrderIdentity(F)
    gradu = F - I
    return 0.5 * (gradu + gradu.T)


def GreenLagrangeStrain(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""Green-Lagrange strain tensor

    .. math::
        \mathbf{E} = \frac{1}{2}\left( \mathbf{C} - \mathbf{I}\right)


    Parameters
    ----------
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        Green-Lagrange strain tensor

    """
    I = SecondOrderIdentity(F)
    C = RightCauchyGreen(F)
    return 0.5 * (C - I)


def PiolaTransform(A, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    r"""Pull-back of a two-tensor from the current to the reference
    configuration

    Parameters
    ----------
    A : ufl.core.expr.Expr
        The tensor you want to push forward
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        The pull-back

    Notes
    -----
    A pull-back is a transformation of a rank-2 tensor from
    the current configuration to the reference configuration.
    A common example is the pull-back of the Cauchy
    stress tensor to the reference configuration which yields the
    First Piola-Kirchhoff stress tensor, i.e

    .. math::
        \mathbf{P} = J \sigma \mathbf{F}^{-T}

    """
    J = Jacobian(F)
    B = J * A * ufl.inv(F).T
    return B


def InversePiolaTransform(
    A: ufl.core.expr.Expr,
    F: ufl.core.expr.Expr,
) -> ufl.core.expr.Expr:
    r"""Push-forward of a rank two-tensor from the reference to the current
    configuration

    Parameters
    ----------
    A : ufl.core.expr.Expr
        The tensor you want to push forward
    F : ufl.core.expr.Expr
        The deformation gradient

    Returns
    -------
    ufl.core.expr.Expr
        The push-forward

    Notes
    -----
    A push-forward is a transformation of a rank-2 tensor from
    the reference configuration to the current configuration.
    A common example is the push-forward of the First Piola-Kirchhoff
    stress tensor to the current configuration which yields the
    Cauchy stress tensor, i.e

    .. math::
        \sigma = \frac{1}{J} \mathbf{P} \mathbf{F}^T

    """
    J = Jacobian(F)
    B = (1 / J) * A * F.T
    return B
