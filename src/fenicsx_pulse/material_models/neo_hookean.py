from dataclasses import dataclass

import dolfinx
import numpy as np
import ufl

from .. import exceptions, invariants
from ..material_model import HyperElasticMaterial


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

    mu: float | dolfinx.fem.Function | dolfinx.fem.Constant = dolfinx.default_scalar_type(15.0)

    def __post_init__(self):
        # Check that all values are positive
        if not exceptions.check_value_greater_than(
            self.mu,
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
        return 0.5 * self.mu * (I1 - dim)
