import logging
from dataclasses import dataclass, field

import numpy as np
import ufl

from .. import exceptions, invariants
from ..material_model import HyperElasticMaterial
from ..units import Variable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NeoHookean(HyperElasticMaterial):
    r"""Neo-Hookean material model

    The strain energy density function is given by

    .. math::
        \Psi(F) = \frac{\mu}{2} \left( \text{tr}(C) - 3 \right)


    Parameters
    ----------
    mu: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Material parameter, by default 15.0


    Raises
    ------
    exceptions.InvalidRangeError
        If the material parameter is not positive
    """

    mu: Variable = field(default_factory=lambda: Variable(15.0, "kPa"))

    def __post_init__(self):
        if not isinstance(self.mu, Variable):
            unit = "kPa"
            logger.warning("Setting mu to %s %s", self.mu, unit)
            self.mu = Variable(self.mu, unit)
        # Check that value are positive
        if not exceptions.check_value_greater_than(
            self.mu.value,
            0.0,
            inclusive=True,
        ):
            raise exceptions.InvalidRangeError(
                name="mu",
                expected_range=(0.0, np.inf),
            )

    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        I1 = invariants.I1(F)
        dim = ufl.domain.find_geometric_dimension(F)
        mu = self.mu.to_base_units()
        return 0.5 * mu * (I1 - dim)

    def __str__(self):
        return "0.5\u03bc (I1 - 3)"
