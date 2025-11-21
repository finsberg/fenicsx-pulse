from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import ufl

from .. import exceptions
from ..material_model import HyperElasticMaterial
from ..units import Variable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SaintVenantKirchhoff(HyperElasticMaterial):
    r"""
    Class for Saint Venant-Kirchhoff material

    Let

    .. math::
        \epsilon =  \frac{1}{2} ( \nabla u + \nabla u^T  +  \nabla u + \nabla u^T )

    Then

    .. math
        \psi(F) = \frac{\lambda}{2} \mathrm{tr} \left( \epsilon \right)^2
        + \mu \mathrm{tr} \left( \epsilon \cdot \epsilon \right)

    Parameters
    ----------
    mu: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Shear modulus
    lmbda: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Lame parameter

    """

    mu: Variable
    lmbda: Variable

    def __post_init__(self):
        # Check that all values are positive
        if not isinstance(self.mu, Variable):
            self.mu = Variable(self.mu, "dimensionless")
        if not isinstance(self.lmbda, Variable):
            self.lmbda = Variable(self.lmbda, "dimensionless")

        if not exceptions.check_value_greater_than(
            self.mu.value,
            0.0,
            inclusive=True,
        ):
            raise exceptions.InvalidRangeError(
                name="mu",
                expected_range=(0.0, np.inf),
            )
        logger.debug(f"Created material model: {type(self).__name__}")
        logger.debug(f"Material parameters: {self.parameters}")

    @property
    def parameters(self) -> dict[str, Variable]:
        return {"mu": self.mu, "lmbda": self.lmbda}

    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        dim = C.ufl_shape[0]
        E = 0.5 * (C - ufl.Identity(dim))

        mu = self.mu.to_base_units()
        lmbda = self.lmbda.to_base_units()

        return lmbda / 2 * (ufl.tr(E) ** 2) + mu * ufl.tr(E * E)
