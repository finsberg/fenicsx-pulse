import abc
from dataclasses import dataclass

import dolfinx
import ufl


class Compressibility(abc.ABC):
    @abc.abstractmethod
    def strain_energy(self, J: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        ...


@dataclass
class Incompressible(Compressibility):
    p: dolfinx.fem.Function

    def strain_energy(self, J: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self.p * (J - 1.0)


@dataclass
class Compressible(Compressibility):
    kappa: float | dolfinx.fem.Function | dolfinx.fem.Constant = 1e3

    def strain_energy(self, J: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self.kappa * (J * ufl.ln(J) - J + 1)
