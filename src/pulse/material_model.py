"""This module defines the material model interface and some common material models.

The material model describes the mechanical behavior of a material. The material
model is used to compute the stress tensor given the deformation gradient.

The material model interface defines two methods:

- `sigma(F)`: The Cauchy stress tensor
- `P(F)`: The first Piola-Kirchhoff stress tensor

The `sigma` method computes the Cauchy stress tensor given the deformation gradient.
The `P` method computes the first Piola-Kirchhoff stress tensor given the deformation gradient.

The `HyperElasticMaterial` class is a base class for hyperelastic material models.
Hyperelastic materials are materials that have a strain energy density function that
depends only on the deformation gradient. The `strain_energy` method computes the
strain energy density function given the deformation gradient.
"""

import abc

import ufl

from . import kinematics


class Material(abc.ABC):
    @abc.abstractmethod
    def sigma(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Cauchy stress tensor

        Parameters
        ----------
        F : ufl.core.expr.Expr
           The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            The Cauchy stress tensor
        """

    def P(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""First Piola-Kirchhoff stress tensor

        Parameters
        ----------
        F : ufl.core.expr.Expr
           The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            The first Piola-Kirchhoff stress tensor
        """
        return kinematics.PiolaTransform(self.sigma(F), F)


class HyperElasticMaterial(Material, abc.ABC):
    @abc.abstractmethod
    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Strain energy density function

        Parameters
        ----------
        F : ufl.core.expr.Expr
            The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            The strain energy density
        """

    def P(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""First Piola-Kirchhoff stress tensor

        Parameters
        ----------
        F : ufl.core.expr.Expr
           The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            The first Piola-Kirchhoff stress tensor

        Notes
        -----
        For a hyperelastic material model with strain energy density
        function :math:`\Psi = \Psi(\mathbf{F})`, the first
        Piola-Kirchhoff stress tensor is given by

        .. math::
            \mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
        """
        F = ufl.variable(F)
        return ufl.diff(self.strain_energy(F), F)

    def sigma(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return kinematics.InversePiolaTransform(self.P(F), F)
