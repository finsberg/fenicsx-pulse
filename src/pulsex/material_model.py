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
        return ufl.diff(self.strain_energy(F), F)

    def sigma(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return kinematics.InversePiolaTransform(self.P(F), F)
