"""This module defines the cardiac model.

The cardiac model is a combination of a material model,
an active model, and a compressibility model.
"""

from dataclasses import dataclass, field
from typing import Protocol

import dolfinx
import ufl

from .viscoelasticity import NoneViscoElasticity


class ActiveModel(Protocol):
    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def S(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def P(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...


class Compressibility(Protocol):
    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def S(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def P(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def is_compressible(self) -> bool: ...

    def register(self, p: dolfinx.fem.Function | None) -> None: ...


class HyperElasticMaterial(Protocol):
    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def P(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def S(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...


class ViscoElasticity(Protocol):
    def strain_energy(self, C_dot: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def P(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def S(self, C_dot: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...


@dataclass(frozen=True, slots=True)
class CardiacModel:
    material: HyperElasticMaterial
    active: ActiveModel
    compressibility: Compressibility
    viscoelasticity: ViscoElasticity = field(default_factory=NoneViscoElasticity)

    def strain_energy(
        self,
        C: ufl.core.expr.Expr,
        C_dot: ufl.core.expr.Expr | None = None,
    ) -> ufl.core.expr.Expr:
        psi = (
            self.material.strain_energy(C)
            + self.active.strain_energy(C)
            + self.compressibility.strain_energy(C)
        )
        if C_dot is not None:
            psi += self.viscoelasticity.strain_energy(C_dot)

        return psi

    def S(
        self,
        C: ufl.core.expr.Expr,
        C_dot: ufl.core.expr.Expr | None = None,
    ) -> ufl.core.expr.Expr:
        """Cauchy stress for the cardiac model."""

        S = self.material.S(C) + self.active.S(C) + self.compressibility.S(C)
        if C_dot is not None:
            S += self.viscoelasticity.S(C_dot)
        return S

    def P(
        self,
        F: ufl.core.expr.Expr,
        F_dot: ufl.core.expr.Expr | None = None,
    ) -> ufl.core.expr.Expr:
        """First Piola-Kirchhoff stress for the cardiac model."""
        P = self.material.P(F) + self.active.P(F) + self.compressibility.P(F)
        if F_dot is not None:
            P += self.viscoelasticity.P(F_dot)
        return P
