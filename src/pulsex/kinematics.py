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
        \mathbf{\varepsilon} = \mathbf{F} - \mathbf{I}

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
    return F - I


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
    """Pull-back of a two-tensor from the current to the reference
    configuration"""
    J = Jacobian(F)
    B = J * A * ufl.inv(F).T
    return B


def InversePiolaTransform(A, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """Push-forward of a two-tensor from the reference to the current
    configuration"""
    J = Jacobian(F)
    B = (1 / J) * A * F.T
    return B
