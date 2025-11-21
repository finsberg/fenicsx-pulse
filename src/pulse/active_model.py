r"""This module defines the ActiveModel class which is an abstract
class for active models. Active models are used to incorporate
active stress or active strain in the material model.

The ActiveModel class defines two methods:

- `Fe(F)`: Transforming the deformation gradient to an active deformation gradient
- `strain_energy(F)`: Active strain energy density function

The `Fe` method transforms the deformation gradient to an active deformation gradient.
For example in the active strain approach we perform a multiplicative decomposition of
the deformation gradient into an elastic and an active part, i.e

.. math::
    \mathbf{F} = \mathbf{F}_e \mathbf{F}_a

In which case the active model can be incorporated by transforming the full deformation
gradient into a pure elastic component.

The `strain_energy` method defines the active strain energy density function. For example
in the active stress approach, the active stress is added as an extra stress component

.. math::
    \mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}} + \mathbf{P}_a

where :math:`\mathbf{P}_a`. Now we can instead rewrite this as a total strain energy
by considering the following equation

.. math::
    \mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
    =  \frac{\partial \Psi_p}{\partial \mathbf{F}}
    +  \frac{\partial \Psi_a}{\partial \mathbf{F}}

where :math:`\Psi_p` is the passive (classical) strain energy density function
and :math:`\Psi_a` is the corresponding active strain energy density function.

The `Passive` class is a simple active model with no active component.
This model could for example be used if you want to use a pure passive model.

"""

from __future__ import annotations

import abc
import logging

import dolfinx
import ufl

logger = logging.getLogger(__name__)


class ActiveModel(abc.ABC):
    @abc.abstractmethod
    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Method to transforming the deformation
        gradient to an an active deformation gradient.
        For example in the active strain approach we
        perform a multiplicative decomposition of the deformation
        gradient into an elastic and an active part, i.e

        .. math::
            \mathbf{F} = \mathbf{F}_e \mathbf{F}_a

        In which case the active model can be incorporated by
        transforming the full deformation gradient into
        a pure elastic component

        Parameters
        ----------
        F : ufl.core.expr.Expr
            The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            The elastic deformation gradient
        """

    @abc.abstractmethod
    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Active strain energy function. For example in the
        active stress approach, the active stress is added
        as an extra stress component

        .. math::
            \mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}} + \mathbf{P}_a

        where :math:`\mathbf{P}_a`. Now we can instead rewrite this
        as a total strain energy by considering the following equation

        .. math::
            \mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
            =  \frac{\partial \Psi_p}{\partial \mathbf{F}}
            +  \frac{\partial \Psi_a}{\partial \mathbf{F}}

        where :math:`\Psi_p` is the passive (classical) strain energy
        density function and :math:`\Psi_a` is the corresponding active
        strain energy density function.

        Parameters
        ----------
        C : ufl.core.expr.Expr
            The right Cauchy-Green deformation tensor

        Returns
        -------
        ufl.core.expr.Expr
            The active strain energy density function
        """

    def S(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Cauchy stress tensor for the active model.

        Parameters
        ----------
        F : ufl.core.expr.Expr
            The right Cauchy-Green deformation tensor

        Returns
        -------
        ufl.core.expr.Expr
            The Cauchy stress tensor
        """
        return 2.0 * ufl.diff(self.strain_energy(C), C)

    def P(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """First Piola-Kirchhoff stress tensor for the active model.

        Parameters
        ----------
        F : ufl.core.expr.Expr
            The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            The first Piola-Kirchhoff stress tensor
        """
        return ufl.diff(self.strain_energy(F.T * F), F)


class Passive(ActiveModel):
    """Active model with no active component.
    This model could for example be used if you
    want to use a pure passive model.
    """

    def __init__(self) -> None:
        logger.debug("Created Passive active model")

    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return F

    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return dolfinx.fem.Constant(ufl.domain.extract_unique_domain(C), 0.0)
