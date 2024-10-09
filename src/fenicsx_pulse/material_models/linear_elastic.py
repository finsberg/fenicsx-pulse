import logging
from dataclasses import dataclass, field

import ufl

from .. import exceptions, kinematics
from ..material_model import Material
from ..units import Variable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LinearElastic(Material):
    """Linear elastic material



    Parameters
    ----------
    E: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Youngs module
    nu: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Poisson's ratio
    """

    E: Variable = field(default_factory=lambda: Variable(10, "kPa"))
    nu: Variable = field(default_factory=lambda: Variable(0.3, "dimensionless"))

    def __post_init__(self):
        if not isinstance(self.E, Variable):
            unit = "kPa"
            logger.warning("Setting mu to %s %s", self.E, unit)
            self.E = Variable(self.E, unit)
        if not isinstance(self.nu, Variable):
            self.nu = Variable(self.nu, "dimensionless")

        # The poisson ratio has to be between -1.0 and 0.5
        nu = self.nu.to_base_units()
        if not exceptions.check_value_between(nu, -1.0, 0.5):
            raise exceptions.InvalidRangeError(name="mu", expected_range=(-1.0, 0.5))

    def sigma(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Cauchy stress for linear elastic material

        .. math::
            \sigma = \frac{E}{1 + \nu} \left( \varepsilon +
            \frac{\nu}{1 + \nu} \mathrm{tr}(\varepsilon) \mathbf{I} \right)


        Parameters
        ----------
        F : ufl.core.expr.Expr
            The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            _description_
        """
        e = kinematics.EngineeringStrain(F)
        I = kinematics.SecondOrderIdentity(F)
        nu = self.nu.to_base_units()
        E = self.E.to_base_units()
        return (E / (1 + nu)) * (e + (nu / (1 - 2 * nu)) * ufl.tr(e) * I)
