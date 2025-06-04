import logging
from dataclasses import dataclass, field

import dolfinx
import numpy as np
import ufl

from .. import exceptions, kinematics
from ..material_model import HyperElasticMaterial
from ..units import Variable

logger = logging.getLogger(__name__)


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
    C: Variable = field(default_factory=lambda: Variable(2.0, "kPa"))
    bf: Variable = field(default_factory=lambda: Variable(8.0, "dimensionless"))
    bt: Variable = field(default_factory=lambda: Variable(2.0, "dimensionless"))
    bfs: Variable = field(default_factory=lambda: Variable(4.0, "dimensionless"))

    def __post_init__(self):
        # Check that all values are positive
        for attr in ["C", "bf", "bt", "bfs"]:
            p = getattr(self, attr)
            if not isinstance(p, Variable):
                unit = "dimensionless" if attr.startswith("b") else "kPa"
                if attr == "C":
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

    @staticmethod
    def default_parameters() -> dict[str, Variable]:
        return {
            "C": Variable(2.0, "kPa"),
            "bf": Variable(8.0, "dimensionless"),
            "bt": Variable(2.0, "dimensionless"),
            "bfs": Variable(4.0, "dimensionless"),
        }

    def is_isotropic(self) -> bool:
        """
        Return True if the material is isotropic.
        """
        bt = self.bt.to_base_units()
        bf = self.bf.to_base_units()
        bfs = self.bfs.to_base_units()

        try:
            return float(bt) == 1.0 and float(bf) == 1.0 and float(bfs) == 1.0
        except TypeError:
            return False

    def _Q(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        E = kinematics.GreenLagrangeStrain(F)

        bt = self.bt.to_base_units()
        bf = self.bf.to_base_units()
        bfs = self.bfs.to_base_units()

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
                bf * E11**2 + bt * (E22**2 + E33**2 + 2 * E23**2) + bfs * (2 * E12**2 + 2 * E13**2)
            )

    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        C = self.C.to_base_units()
        return 0.5 * C * (ufl.exp(self._Q(F)) - 1.0)

    def __str__(self):
        return "0.5C (exp(Q) - 1)"
