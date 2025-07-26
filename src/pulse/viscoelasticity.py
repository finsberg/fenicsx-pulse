import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import ufl

from .units import Variable

logger = logging.getLogger(__name__)


@dataclass
class ViscoElasticity(ABC):
    @abstractmethod
    def strain_energy(self, C_dot) -> ufl.Form:
        """Strain energy density function.

        Parameters
        ----------
        C_dot : ufl.core.expr.Expr
            The time derivative of the deformation gradient (strain rate)

        Returns
        -------
        ufl.Form
            The strain energy density function
        """
        ...

    def S(self, C_dot: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Cauchy stress tensor for the viscoelasticity model.

        Parameters
        ----------
        C_dot : ufl.core.expr.Expr
            The time derivative of the deformation gradient (strain rate)

        Returns
        -------
        ufl.core.expr.Expr
            The Cauchy stress tensor
        """
        return 2.0 * ufl.diff(self.strain_energy(C_dot), C_dot)

    def P(self, F_dot: ufl.core.expr.Expr | None = None) -> ufl.core.expr.Expr:
        """First Piola-Kirchhoff stress for the viscoelasticity model."""
        if F_dot is None:
            raise ValueError("F_dot must be provided for P calculation.")
        C_dot = F_dot.T * F_dot
        return ufl.diff(self.strain_energy(C_dot), F_dot)


class NoneViscoElasticity(ViscoElasticity):
    def strain_energy(self, C_dot: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return 0.0


@dataclass
class Viscous(ViscoElasticity):
    eta: Variable = field(default_factory=lambda: Variable(1e2, "Pa s"))

    def __post_init__(self):
        if not isinstance(self.eta, Variable):
            unit = "Pa s"
            logger.warning("Setting eta to %s %s", self.eta, unit)
            self.eta = Variable(self.eta, unit)

    def strain_energy(self, C_dot) -> ufl.Form:
        E_dot = 0.5 * C_dot
        eta = self.eta.to_base_units()
        return 0.5 * eta * ufl.tr(E_dot * E_dot)

    def __str__(self):
        return "0.5\u03b7 tr (E_dot E_dot)"
