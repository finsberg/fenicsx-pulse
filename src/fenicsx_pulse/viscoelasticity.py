from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import ufl

from .units import Variable


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


class Viscous(ViscoElasticity):
    eta: Variable = field(default_factory=lambda: Variable(1e2, "Pa s"))

    def strain_energy(self, E_dot) -> ufl.Form:
        eta = self.eta.to_base_units()
        return 0.5 * eta * ufl.tr(E_dot * E_dot)
