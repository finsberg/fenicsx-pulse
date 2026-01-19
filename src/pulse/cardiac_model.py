"""This module defines the cardiac model.

The cardiac model is a combination of a material model,
an active model, and a compressibility model.
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

import dolfinx
import ufl

from . import kinematics
from .viscoelasticity import NoneViscoElasticity

logger = logging.getLogger(__name__)


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

    def P(self, F: ufl.core.expr.Expr, dev: bool) -> ufl.core.expr.Expr: ...

    def S(self, C: ufl.core.expr.Expr, dev: bool) -> ufl.core.expr.Expr: ...


class ViscoElasticity(Protocol):
    def strain_energy(self, C_dot: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def P(self, F_dot: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...

    def S(self, C_dot: ufl.core.expr.Expr) -> ufl.core.expr.Expr: ...


@dataclass(frozen=True, slots=True)
class CardiacModel:
    material: HyperElasticMaterial
    active: ActiveModel
    compressibility: Compressibility
    viscoelasticity: ViscoElasticity = field(default_factory=NoneViscoElasticity)

    def __post_init__(self):
        logger.debug("Created CardiacModel with components:")
        logger.debug(f"  Material: {type(self.material).__name__}")
        logger.debug(f"  Active Model: {type(self.active).__name__}")
        logger.debug(f"  Compressibility: {type(self.compressibility).__name__}")
        logger.debug(f"  Viscoelasticity: {type(self.viscoelasticity).__name__}")

    def strain_energy(
        self,
        C: ufl.core.expr.Expr,
        C_dot: ufl.core.expr.Expr | None = None,
    ) -> ufl.core.expr.Expr:
        """Total strain energy for the cardiac model.

        Parameters
        ----------
        C : ufl.core.expr.Expr
            Right Cauchy-Green deformation tensor
        C_dot : ufl.core.expr.Expr | None, optional
            Time derivative of the right Cauchy-Green deformation tensor, by default None

        Returns
        -------
        ufl.core.expr.Expr
            The total strain energy density
        """
        if self.compressibility.is_compressible():
            Cdev = kinematics.Cdev(C)
        else:
            Cdev = C

        psi = (
            self.material.strain_energy(Cdev)
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
        dev = self.compressibility.is_compressible()

        S = self.material.S(C, dev=dev) + self.active.S(C) + self.compressibility.S(C)
        if C_dot is not None:
            S += self.viscoelasticity.S(C_dot)
        return S

    def P(
        self,
        F: ufl.core.expr.Expr,
        F_dot: ufl.core.expr.Expr | None = None,
    ) -> ufl.core.expr.Expr:
        """First Piola-Kirchhoff stress for the cardiac model."""
        dev = self.compressibility.is_compressible()

        P = self.material.P(F, dev=dev) + self.active.P(F) + self.compressibility.P(F)
        if F_dot is not None:
            P += self.viscoelasticity.P(F_dot)
        return P

    def sigma(
        self,
        F: ufl.core.expr.Expr,
        F_dot: ufl.core.expr.Expr | None = None,
    ) -> ufl.core.expr.Expr:
        r"""Cauchy stress tensor

        Parameters
        ----------
        F : ufl.core.expr.Expr
           The deformation gradient
        F_dot : ufl.core.expr.Expr | None

        Returns
        -------
        ufl.core.expr.Expr
            The Cauchy stress tensor
        """
        from .kinematics import InversePiolaTransform

        return InversePiolaTransform(self.P(F, F_dot), F)
