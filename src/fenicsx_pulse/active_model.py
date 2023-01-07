from __future__ import annotations

import abc

import dolfinx
import ufl


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
    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Active strain energy function. For example in the
        active stress approach, the active stress is added
        as an extra stress component

        .. math::
            \\mathbf{P} = \frac{\\partial \\Psi}{\\partial \\mathbf{F}} + \\mathbf{P}_a

        where :math:`\\mathbf{P}_a`. Now we can instead rewrite this
        as a total strain energy by considering the following equation

        .. math::
            \\mathbf{P} = \frac{\\partial \\Psi}{\\partial \\mathbf{F}}
            =  \frac{\\partial \\Psi_p}{\\partial \\mathbf{F}}
            +  \frac{\\partial \\Psi_a}{\\partial \\mathbf{F}}

        where :math:`\\Psi_p` is the passive (classical) strain energy
        density function and :math:`\\Psi_a` is the corresponding active
        strain energy density function.

        Parameters
        ----------
        F : ufl.core.expr.Expr
            The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            The active strain energy density function
        """


class Passive(ActiveModel):
    """Active model with no active component.
    This model could for example be used if you
    want to use a pure passive model.
    """

    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return F

    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return dolfinx.fem.Constant(F.ufl_domain(), 0.0)
