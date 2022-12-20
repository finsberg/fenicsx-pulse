r"""
Orthotropic model by Holzapfel and Ogden
.. math::
    \mathcal{W}(I_1, I_{4f_0})
    = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
    + \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4f_0} - 1)_+^2} -1 \right)
    + \frac{a_s}{2 b_s} \left( e^{ b_s (I_{4s_0} - 1)_+^2} -1 \right)
    + \frac{a_fs}{2 b_fs} \left( e^{ b_fs I_{8fs}^2} -1 \right)

where

.. math::
    (\cdot)_+ = \max\{x,0\}
.. rubric:: Reference
    Holzapfel, Gerhard A., and Ray W. Ogden. "Constitutive modelling of
    passive myocardium: a structurally based framework for material characterization.
    "Philosophical Transactions of the Royal Society of London A:
    Mathematical, Physical and Engineering Sciences 367.1902 (2009): 3445-3475.

"""
import typing
from dataclasses import dataclass
from dataclasses import field

import dolfinx
import numpy as np
import ufl

from . import exceptions
from . import functions
from . import invariants
from .material_model import HyperElasticMaterialModel


@dataclass(slots=True)
class HolzapfelOgden(HyperElasticMaterialModel):
    f0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    a: float | dolfinx.fem.Function | dolfinx.fem.Constant = 0.0
    b: float | dolfinx.fem.Function | dolfinx.fem.Constant = 0.0
    a_f: float | dolfinx.fem.Function | dolfinx.fem.Constant = 0.0
    b_f: float | dolfinx.fem.Function | dolfinx.fem.Constant = 0.0
    a_s: float | dolfinx.fem.Function | dolfinx.fem.Constant = 0.0
    b_s: float | dolfinx.fem.Function | dolfinx.fem.Constant = 0.0
    a_fs: float | dolfinx.fem.Function | dolfinx.fem.Constant = 0.0
    b_fs: float | dolfinx.fem.Function | dolfinx.fem.Constant = 0.0
    use_subplus: bool = field(default=True, repr=False)
    use_heaviside: bool = field(default=True, repr=False)

    _W1_func: typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr] = field(
        init=False,
        repr=False,
    )
    _W4f_func: typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr] = field(
        init=False,
        repr=False,
    )
    _W4s_func: typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr] = field(
        init=False,
        repr=False,
    )
    _W8fs_func: typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self):
        # Check that all values are positive
        for attr in ["a", "b", "a_f", "b_f", "a_s", "a_s", "a_fs", "b_fs"]:
            if not exceptions.check_value_greater_than(
                getattr(self, attr),
                0.0,
                inclusive=True,
            ):
                raise exceptions.InvalidRangeError(
                    name=attr,
                    expected_range=(0.0, np.inf),
                )

        self._W1_func = self._resolve_W1()
        self._W4f_func = self._resolve_W4(a=self.a_f, b=self.b_f)

        self._W4s_func = self._resolve_W4(a=self.a_s, b=self.b_s)
        self._W8fs_func = self._resolve_W8fs()

    def _resolve_W1(self):
        if exceptions.check_value_greater_than(self.a, 1e-10):
            if exceptions.check_value_greater_than(self.b, 1e-10):
                return lambda I1: (self.a / (2.0 * self.b)) * (
                    ufl.exp(self.b * (I1 - 3)) - 1.0
                )
            else:
                return lambda I1: (self.a / 2.0) * (I1 - 3)
        else:
            return lambda I1: 0.0

    def _resolve_W4(self, a, b):
        subplus = functions.subplus if self.use_subplus else lambda x: x
        heaviside = functions.heaviside if self.use_heaviside else lambda x: 1
        if exceptions.check_value_greater_than(a, 1e-10):
            if exceptions.check_value_greater_than(b, 1e-10):
                return (
                    lambda I4: (a / (2.0 * b))
                    * heaviside(I4 - 1)
                    * (ufl.exp(b * subplus(I4 - 1) ** 2) - 1.0)
                )
            else:
                return lambda I4: (a / 2.0) * heaviside(I4 - 1) * subplus(I4 - 1) ** 2
        else:
            return lambda I4: 0.0

    def _resolve_W8fs(self):
        if exceptions.check_value_greater_than(self.a_fs, 1e-10):
            if exceptions.check_value_greater_than(self.b_fs, 1e-10):
                return (
                    lambda I8: self.a_fs
                    / (2.0 * self.b_fs)
                    * (ufl.exp(self.b_fs * I8**2) - 1.0)
                )
            else:
                return lambda I8: self.a_fs / 2.0 * I8**2
        else:
            return lambda I8: 0.0

    @staticmethod
    def transversely_isotropic_parameters() -> dict[str, float]:
        """
        Material parameters for the Holzapfel Ogden model
        Taken from Table 1 row 1 in the main paper
        """

        return {
            "a": 2.280,
            "b": 9.726,
            "a_f": 1.685,
            "b_f": 15.779,
            "a_s": 0.0,
            "b_s": 0.0,
            "a_fs": 0.0,
            "b_fs": 0.0,
        }

    @staticmethod
    def partly_orthotropic_parameters() -> dict[str, float]:
        """
        Material parameters for the Holzapfel Ogden model
        Taken from Table 1 row 1 in the main paper
        """

        return {
            "a": 0.057,
            "b": 8.094,
            "a_f": 21.503,
            "b_f": 15.819,
            "a_s": 6.841,
            "b_s": 6.959,
            "a_fs": 0.0,
            "b_fs": 0.0,
        }

    @staticmethod
    def orthotropic_parameters() -> dict[str, float]:
        """
        Material parameters for the Holzapfel Ogden model
        Taken from Table 1 row 2 in the main paper
        """

        return {
            "a": 0.059,
            "b": 8.023,
            "a_f": 18.472,
            "b_f": 16.026,
            "a_s": 2.481,
            "b_s": 11.120,
            "a_fs": 0.216,
            "b_fs": 11.436,
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
        I4f = invariants.I4(F, self.f0)
        I4s = invariants.I4(F, self.s0)
        I8fs = invariants.I8(F, self.f0, self.s0)

        return self._W1(I1) + self._W4f(I4f) + self._W4s(I4s) + self._W8fs(I8fs)
