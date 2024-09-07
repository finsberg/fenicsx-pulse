from abc import ABC, abstractmethod

import dolfinx
import ufl


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
    eta: dolfinx.fem.Constant = dolfinx.default_scalar_type(0.1)

    def strain_energy(self, E_dot) -> ufl.Form:
        return 0.5 * self.eta * ufl.tr(E_dot * E_dot)
