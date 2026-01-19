import logging
from dataclasses import dataclass, field

import dolfinx
import numpy as np
import ufl

from .. import exceptions
from ..material_model import HyperElasticMaterial
from ..units import Variable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Usyk(HyperElasticMaterial):
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
    Original model from Usyk [2]_.
    The strain energy density function is given by

    .. math::
        \Psi = \frac{C}{2} \left( \mathrm{exp}^{Q} - 1 \right)

    where

    .. math::
        Q = b_f E_{11}^2 + b_t \left( E_{22}^2 + E_{33}^2 + E_{23}^2 + E_{32}^2 \right)
            + b_{fs} \left( E_{12}^2 + E_{21}^2 + E_{13}^2 + E_{31}^2 \right)


    .. [3] Usyk, Taras P., Ian J. LeGrice, and Andrew D. McCulloch.
        "Computational model of three-dimensional cardiac electromechanics."
        Computing and visualization in science 4.4 (2002): 249-257..


    """

    f0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    n0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    C: Variable = field(default_factory=lambda: Variable(0.88, "kPa"))
    bf: Variable = field(default_factory=lambda: Variable(8.0, "dimensionless"))
    bs: Variable = field(default_factory=lambda: Variable(6.0, "dimensionless"))
    bn: Variable = field(default_factory=lambda: Variable(3.0, "dimensionless"))
    bfs: Variable = field(default_factory=lambda: Variable(12.0, "dimensionless"))
    bfn: Variable = field(default_factory=lambda: Variable(3.0, "dimensionless"))
    bsn: Variable = field(default_factory=lambda: Variable(3.0, "dimensionless"))

    def __post_init__(self):
        # Check that all values are positive
        for attr in ["C", "bf", "bs", "bn", "bfs", "bfn", "bsn"]:
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
        logger.debug(f"Created material model: {type(self).__name__}")
        logger.debug(f"Material parameters: {self.parameters}")

    @property
    def parameters(self) -> dict[str, Variable]:
        return {
            "C": self.C,
            "bf": self.bf,
            "bs": self.bs,
            "bn": self.bn,
            "bfs": self.bfs,
            "bfn": self.bfn,
            "bsn": self.bsn,
        }

    @staticmethod
    def default_parameters() -> dict[str, Variable]:
        return {
            "C": Variable(2.0, "kPa"),
            "bf": Variable(8.0, "dimensionless"),
            "bs": Variable(6.0, "dimensionless"),
            "bn": Variable(3.0, "dimensionless"),
            "bfs": Variable(12.0, "dimensionless"),
            "bfn": Variable(3.0, "dimensionless"),
            "bsn": Variable(3.0, "dimensionless"),
        }

    def is_isotropic(self) -> bool:
        """
        Return True if the material is isotropic.
        """
        bf = self.bf.to_base_units()
        bs = self.bs.to_base_units()
        bn = self.bn.to_base_units()
        bfs = self.bfs.to_base_units()
        bfn = self.bfn.to_base_units()
        bsn = self.bsn.to_base_units()

        try:
            return (
                float(bs) == 1.0
                and float(bn) == 1.0
                and float(bf) == 1.0
                and float(bfs) == 1.0
                and float(bfn) == 1.0
                and float(bsn) == 1.0
            )
        except TypeError:
            return False

    def _Q(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        dim = C.ufl_shape[0]
        E = 0.5 * (C - ufl.Identity(dim))

        bf = self.bf.to_base_units()
        bs = self.bs.to_base_units()
        bn = self.bn.to_base_units()
        bfs = self.bfs.to_base_units()
        bfn = self.bfn.to_base_units()
        bsn = self.bsn.to_base_units()

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
                bf * E11**2
                + bs * E22**2
                + bn * E33**2
                + bfs * 2 * E12**2
                + bfn * 2 * E13**2
                + bsn * 2 * E23**2
            )

    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        C_ = self.C.to_base_units()
        return 0.5 * C_ * (ufl.exp(self._Q(C)) - 1.0)

    def __str__(self):
        return "0.5C (exp(Q) - 1)"
