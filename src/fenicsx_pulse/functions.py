import ufl


def subplus(x):
    r"""
    Ramp function

    .. math::
       \max\{x,0\}
    """

    return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)


def heaviside(
    x: ufl.Coefficient,
    k: float = 1.0,
    use_exp: bool = False,
) -> ufl.Coefficient:
    r"""
    Heaviside function

    .. math::
       \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}

    or

    .. math::
        \frac{1}{1 + e^{-k (x - 1)}}
    """

    if use_exp:
        return 1 / (1 + ufl.exp(-k * (x - 1)))
    else:
        return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)
