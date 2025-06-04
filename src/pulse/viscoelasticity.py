import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import ufl

from .units import Variable

logger = logging.getLogger(__name__)


@dataclass
class ViscoElasticity(ABC):
    @abstractmethod
    def strain_energy(self, E_dot) -> ufl.Form:
        """Strain energy density function.

        Parameters
        ----------
        E_dot : ufl.Coefficient
            The strain rate tensor

        Returns
        -------
        ufl.Form
            The strain energy density function
        """
        ...


class NoneViscoElasticity(ViscoElasticity):
    def strain_energy(self, E_dot) -> ufl.Form:
        return 0.0


@dataclass
class Viscous(ViscoElasticity):
    eta: Variable = field(default_factory=lambda: Variable(1e2, "Pa s"))

    def __post_init__(self):
        if not isinstance(self.eta, Variable):
            unit = "Pa s"
            logger.warning("Setting eta to %s %s", self.eta, unit)
            self.eta = Variable(self.eta, unit)

    def strain_energy(self, E_dot) -> ufl.Form:
        eta = self.eta.to_base_units()
        return 0.5 * eta * ufl.tr(E_dot * E_dot)

    def __str__(self):
        return "0.5\u03b7 tr (E_dot E_dot)"
