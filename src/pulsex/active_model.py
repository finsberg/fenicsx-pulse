from __future__ import annotations

import abc

import ufl


class ActiveModel(abc.ABC):
    @abc.abstractmethod
    def F(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        ...

    @abc.abstractmethod
    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        ...


class Passive(ActiveModel):
    """No active component"""

    def F(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return F

    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return ufl.Constant(0.0)
