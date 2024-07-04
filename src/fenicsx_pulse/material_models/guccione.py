from dataclasses import dataclass

import dolfinx
import numpy as np
import ufl

from .. import exceptions, kinematics
from ..material_model import HyperElasticMaterial


@dataclass(slots=True)
class Guccione(HyperElasticMaterial):
    r"""
    Orthotropic model by Holzapfel and Ogden

    Parameters
    ----------
    f0: dolfinx.fem.Function | dolfinx.fem.Constant | None
        Function representing the direction of the fibers
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None
        Function representing the direction of the sheets
    n0: dolfinx.fem.Function | dolfinx.fem.Constant | None
        Function representing the direction of the sheet normal
    C: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 2.0
    bf: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 8.0
    bt: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 2.0
    bfs: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 4.0

    Notes
    -----
    Original model from Guccione [2]_.
    The strain energy density function is given by

    .. math::
        \Psi = \frac{C}{2} \left( \mathrm{exp}^{Q} - 1 \right)

    where

    .. math::
        Q = b_f E_{11}^2 + b_t \left( E_{22}^2 + E_{33}^2 + E_{23}^2 + E_{32}^2 \right)
            + b_{fs} \left( E_{12}^2 + E_{21}^2 + E_{13}^2 + E_{31}^2 \right)


    .. [2] Guccione, Julius M., Andrew D. McCulloch, and L. K. Waldman.
        "Passive material properties of intact ventricular myocardium determined
        from a cylindrical model." (1991): 42-55.


    """

    f0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    n0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    C: float | dolfinx.fem.Function | dolfinx.fem.Constant = dolfinx.default_scalar_type(2.0)
    bf: float | dolfinx.fem.Function | dolfinx.fem.Constant = dolfinx.default_scalar_type(8.0)
    bt: float | dolfinx.fem.Function | dolfinx.fem.Constant = dolfinx.default_scalar_type(2.0)
    bfs: float | dolfinx.fem.Function | dolfinx.fem.Constant = dolfinx.default_scalar_type(4.0)

    def __post_init__(self):
        # Check that all values are positive
        for attr in ["C", "bf", "bt", "bfs"]:
            if not exceptions.check_value_greater_than(
                getattr(self, attr),
                0.0,
                inclusive=True,
            ):
                raise exceptions.InvalidRangeError(
                    name=attr,
                    expected_range=(0.0, np.inf),
                )

    @staticmethod
    def default_parameters() -> dict:
        return {"C": 2.0, "bf": 8.0, "bt": 2.0, "bfs": 4.0}

    def is_isotropic(self) -> bool:
        """
        Return True if the material is isotropic.
        """

        try:
            return float(self.bt) == 1.0 and float(self.bf) == 1.0 and float(self.bfs) == 1.0
        except TypeError:
            return False

    def _Q(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        E = kinematics.GreenLagrangeStrain(F)

        if self.is_isotropic():
            # isotropic case
            return ufl.inner(E, E)

        else:
            E11, E12, E13 = (
                ufl.inner(E * self.f0, self.f0),
                ufl.inner(E * self.f0, self.s0),
                ufl.inner(E * self.f0, self.n0),
            )
            _, E22, E23 = (
                ufl.inner(E * self.s0, self.f0),
                ufl.inner(E * self.s0, self.s0),
                ufl.inner(E * self.s0, self.n0),
            )
            _, _, E33 = (
                ufl.inner(E * self.n0, self.f0),
                ufl.inner(E * self.n0, self.s0),
                ufl.inner(E * self.n0, self.n0),
            )

            return (
                self.bf * E11**2
                + self.bt * (E22**2 + E33**2 + 2 * E23**2)
                + self.bfs * (2 * E12**2 + 2 * E13**2)
            )

    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return 0.5 * self.C * (ufl.exp(self._Q(F)) - 1.0)
