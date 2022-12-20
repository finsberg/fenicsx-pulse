import ufl


def subplus(x):
    r"""
    Ramp function

    .. math::
       \max\{x,0\}
    """

    return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)


def heaviside(x):
    r"""
    Heaviside function
    .. math::
       \mathcal{H}(x) = \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}
    """

    return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)
