import logging
import typing
from dataclasses import dataclass, field
from functools import partial

import dolfinx
import numpy as np
import ufl

from .. import exceptions, functions, invariants
from ..material_model import HyperElasticMaterial
from ..units import Variable

Invariant = typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr]

logger = logging.getLogger(__name__)


def heaviside(x: ufl.core.expr.Expr, use_heaviside: bool) -> ufl.core.expr.Expr:
    if use_heaviside:
        return functions.heaviside(x)
    return ufl.as_ufl(1.0)


@dataclass(slots=True)
class HolzapfelOgden(HyperElasticMaterial):
    r"""
    Orthotropic model by Holzapfel and Ogden

    Parameters
    ----------
    f0: dolfinx.fem.Function | dolfinx.fem.Constant | None
        Function representing the direction of the fibers
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None
        Function representing the direction of the sheets
    a: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 0.0
    b: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 0.0
    a_f: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 0.0
    b_f: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 0.0
    a_s: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 0.0
    b_s: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 0.0
    a_fs: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 0.0
    b_fs: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 0.0
    use_subplus: bool
        Use subplus function when computing anisotropic contribution,
        by default True
    use_heaviside: bool
        Use heaviside function when computing anisotropic contribution,
        by default True

    Notes
    -----
    Original model from Holzapfel and Ogden [1]_.
    The strain energy density function is given by

    .. math::
        \Psi(I_1, I_{4\mathbf{f}_0}, I_{4\mathbf{s}_0}, I_{8\mathbf{f}_0\mathbf{s}_0})
        = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
        + \frac{a_f}{2 b_f} \mathcal{H}(I_{4\mathbf{f}_0} - 1)
        \left( e^{ b_f (I_{4\mathbf{f}_0} - 1)_+^2} -1 \right)
        + \frac{a_s}{2 b_s} \mathcal{H}(I_{4\mathbf{s}_0} - 1)
        \left( e^{ b_s (I_{4\mathbf{s}_0} - 1)_+^2} -1 \right)
        + \frac{a_{fs}}{2 b_{fs}} \left( e^{ b_{fs}
        I_{8 \mathbf{f}_0 \mathbf{s}_0}^2} -1 \right)

    where

    .. math::
        (x)_+ = \max\{x,0\}

    and

    .. math::
        \mathcal{H}(x) = \begin{cases}
            1, & \text{if $x > 0$} \\
            0, & \text{if $x \leq 0$}
        \end{cases}

    is the Heaviside function.

    .. [1] Holzapfel, Gerhard A., and Ray W. Ogden.
        "Constitutive modelling of passive myocardium:
        a structurally based framework for material characterization.
        "Philosophical Transactions of the Royal Society of London A:
        Mathematical, Physical and Engineering Sciences 367.1902 (2009):
        3445-3475.

    """

    f0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    a: Variable = field(default_factory=lambda: Variable(0.0, "kPa"))
    b: Variable = field(default_factory=lambda: Variable(0.0, "dimensionless"))
    a_f: Variable = field(default_factory=lambda: Variable(0.0, "kPa"))
    b_f: Variable = field(default_factory=lambda: Variable(0.0, "dimensionless"))
    a_s: Variable = field(default_factory=lambda: Variable(0.0, "kPa"))
    b_s: Variable = field(default_factory=lambda: Variable(0.0, "dimensionless"))
    a_fs: Variable = field(default_factory=lambda: Variable(0.0, "kPa"))
    b_fs: Variable = field(default_factory=lambda: Variable(0.0, "dimensionless"))
    use_subplus: bool = field(default=True, repr=False)
    use_heaviside: bool = field(default=True, repr=False)

    _W1_func: Invariant = field(
        init=False,
        repr=False,
    )
    _W4f_func: Invariant = field(
        init=False,
        repr=False,
    )
    _W4s_func: Invariant = field(
        init=False,
        repr=False,
    )
    _W8fs_func: Invariant = field(
        init=False,
        repr=False,
    )
    _I4f: Invariant = field(
        init=False,
        repr=False,
    )
    _I4s: Invariant = field(
        init=False,
        repr=False,
    )
    _I8fs: Invariant = field(
        init=False,
        repr=False,
    )

    def __post_init__(self):
        # Check that all values are positive
        for attr in ["a", "b", "a_f", "b_f", "a_s", "a_s", "a_fs", "b_fs"]:
            p = getattr(self, attr)
            if not isinstance(p, Variable):
                unit = "dimensionless" if attr.startswith("b") else "kPa"
                if attr.startswith("a"):
                    logger.warning("Setting %s to %s %s", attr, p, unit)
                p = Variable(p, unit)
                setattr(self, attr, p)

            if not exceptions.check_value_greater_than(
                getattr(self, attr).value,
                0.0,
                inclusive=True,
            ):
                raise exceptions.InvalidRangeError(
                    name=attr,
                    expected_range=(0.0, np.inf),
                )

        if self.f0 is not None:
            self._I4f = partial(invariants.I4, a0=self.f0)
        else:
            self._I4f = lambda x: 0.0

        if self.s0 is not None:
            self._I4s = partial(invariants.I4, a0=self.s0)
        else:
            self._I4s = lambda x: 0.0

        if self.f0 is not None and self.s0 is not None:
            self._I8fs = partial(invariants.I8, a0=self.f0, b0=self.s0)
        else:
            self._I8fs = lambda x: 0.0

        a_f = self.a_f
        b_f = self.b_f
        a_s = self.a_s
        b_s = self.b_s

        self._W1_func = self._resolve_W1()
        self._W4f_func = self._resolve_W4(a=a_f, b=b_f, required_attr="f0")
        self._W4s_func = self._resolve_W4(a=a_s, b=b_s, required_attr="s0")
        self._W8fs_func = self._resolve_W8fs()

    def _resolve_W1(self) -> Invariant:
        a = self.a.to_base_units()
        b = self.b.to_base_units()

        if exceptions.check_value_greater_than(self.a.value, 1e-10):
            if exceptions.check_value_greater_than(self.b.value, 1e-10):
                return lambda I1: (a / (2.0 * b)) * (ufl.exp(b * (I1 - 3)) - 1.0)
            else:
                return lambda I1: (a / 2.0) * (I1 - 3)
        else:
            return lambda I1: 0.0

    def _resolve_W4(self, a: Variable, b: Variable, required_attr: str) -> Invariant:
        subplus = functions.subplus if self.use_subplus else lambda x: x

        if exceptions.check_value_greater_than(a.value, 1e-10):
            a0 = getattr(self, required_attr)
            if a0 is None:
                raise exceptions.MissingModelAttribute(
                    attr=required_attr,
                    model=type(self).__name__,
                )

            if exceptions.check_value_greater_than(b.value, 1e-10):
                return (
                    lambda I4: (a.to_base_units() / (2.0 * b.to_base_units()))
                    * heaviside(I4 - 1, use_heaviside=self.use_heaviside)
                    * (ufl.exp(b.to_base_units() * subplus(I4 - 1) ** 2) - 1.0)
                )
            else:
                return (
                    lambda I4: (a.to_base_units() / 2.0)
                    * heaviside(I4 - 1, use_heaviside=self.use_heaviside)
                    * subplus(I4 - 1) ** 2
                )
        else:
            return lambda I4: 0.0

    def _resolve_W8fs(self) -> Invariant:
        a_fs = self.a_fs.to_base_units()
        b_fs = self.b_fs.to_base_units()
        if exceptions.check_value_greater_than(self.a_fs.value, 1e-10):
            if self.f0 is None or self.s0 is None:
                raise exceptions.MissingModelAttribute(
                    attr="f0 and/or s0",
                    model=type(self).__name__,
                )
            if exceptions.check_value_greater_than(self.b_fs.value, 1e-10):
                return lambda I8: a_fs / (2.0 * b_fs) * (ufl.exp(b_fs * I8**2) - 1.0)
            else:
                return lambda I8: a_fs / 2.0 * I8**2
        else:
            return lambda I8: 0.0

    @staticmethod
    def transversely_isotropic_parameters() -> dict[str, Variable]:
        """
        Material parameters for the Holzapfel Ogden model
        Taken from Table 1 row 3 in the main paper
        """

        return {
            "a": Variable(2.280, "kPa"),
            "b": Variable(9.726, "dimensionless"),
            "a_f": Variable(1.685, "kPa"),
            "b_f": Variable(15.779, "dimensionless"),
            "a_s": Variable(0.0, "kPa"),
            "b_s": Variable(0.0, "dimensionless"),
            "a_fs": Variable(0.0, "kPa"),
            "b_fs": Variable(0.0, "dimensionless"),
        }

    @staticmethod
    def partly_orthotropic_parameters() -> dict[str, Variable]:
        """
        Material parameters for the Holzapfel Ogden model
        Taken from Table 1 row 1 in the main paper
        """

        return {
            "a": Variable(0.057, "kPa"),
            "b": Variable(8.094, "dimensionless"),
            "a_f": Variable(21.503, "kPa"),
            "b_f": Variable(15.819, "dimensionless"),
            "a_s": Variable(6.841, "kPa"),
            "b_s": Variable(6.959, "dimensionless"),
            "a_fs": Variable(0.0, "kPa"),
            "b_fs": Variable(0.0, "dimensionless"),
        }

    @staticmethod
    def orthotropic_parameters() -> dict[str, Variable]:
        """
        Material parameters for the Holzapfel Ogden model
        Taken from Table 1 row 2 in the main paper
        """

        return {
            "a": Variable(0.059, "kPa"),
            "b": Variable(8.023, "dimensionless"),
            "a_f": Variable(18.472, "kPa"),
            "b_f": Variable(16.026, "dimensionless"),
            "a_s": Variable(2.481, "kPa"),
            "b_s": Variable(11.120, "dimensionless"),
            "a_fs": Variable(0.216, "kPa"),
            "b_fs": Variable(11.436, "dimensionless"),
        }

    def _W1(self, I1: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self._W1_func(I1)

    def _W4f(self, I4f: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self._W4f_func(I4f)

    def _W4s(self, I4s: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self._W4s_func(I4s)

    def _W8fs(self, I8fs: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self._W8fs_func(I8fs)

    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        I1 = invariants.I1(F)
        I4f = self._I4f(F)
        I4s = self._I4s(F)
        I8fs = self._I8fs(F)
        return self._W1(I1) + self._W4f(I4f) + self._W4s(I4s) + self._W8fs(I8fs)

    def __str__(self) -> str:
        return (
            "a/2b (exp(b(I1 - 3)) - 1) + "
            "af/2bf H(I4f - 1) (exp(bf (I4f - 1)_+^2) - 1) + "
            "as/2bs H(I4s - 1) (exp(bs (I4s - 1)_+^2) - 1) + "
            "afs/2bfs (exp(bfs I8fs^2) - 1)"
        )
