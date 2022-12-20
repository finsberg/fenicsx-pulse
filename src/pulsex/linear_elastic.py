from dataclasses import dataclass

import dolfinx
import ufl

from . import kinematics
from .material_model import Material


@dataclass(frozen=True, slots=True)
class LinearElastic(Material):
    """Linear elastic material

    Parameters
    ----------
    E: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Youngs module
    nu: float | dolfinx.fem.Function | dolfinx.fem.Constant
        Poisson's ratio
    """

    E: float | dolfinx.fem.Function | dolfinx.fem.Constant
    nu: float | dolfinx.fem.Function | dolfinx.fem.Constant

    def sigma(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Cauchy stress for linear elastic material

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
        return (self.E / (1 + self.nu)) * (
            e + (self.nu / (1 - 2 * self.nu)) * ufl.tr(e) * I
        )
