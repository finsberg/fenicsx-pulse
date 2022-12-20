from dataclasses import dataclass
from enum import Enum

import dolfinx
import ufl

from .active_model import ActiveModel


class ActiveStressModels(str, Enum):
    transversely = "transversely"
    orthotropic = "orthotropic"
    fully_anisotropic = "fully_anisotropic"


@dataclass(slots=True)
class ActiveStress(ActiveModel):
    """Active stress model

    f0: dolfinx.fem.Function | dolfinx.fem.Constant
        The cardiac fiber direction
    activation: dolfinx.fem.Function | dolfinx.fem.Constant | None
        A function or constant representing the activation.
        If not provided a constant will be created.
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None
        The sheets orientation. Only needed for orthotropic
        active stress models
    n0: dolfinx.fem.Function | dolfinx.fem.Constant | None
        The sheet-normal orientation. Only needed for orthotropic
        active stress models
    T_ref: float = 1.0
        Reference active stress, by default 1.0
    eta: float = 0.0
        Amount of transverse active stress, by default 0.0.
        A value of zero means that all active stress is along
        the fiber direction. If the value is 1.0 then all
        active stress will be in the transverse direction.
    isotropy: ActiveStressModels
        What kind of active stress model to use, by
        default 'transversely'
    """

    f0: dolfinx.fem.Function | dolfinx.fem.Constant
    activation: dolfinx.fem.Function | dolfinx.fem.Constant = None
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    n0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    T_ref: dolfinx.fem.Constant = 1.0
    eta: dolfinx.fem.Constant = 0.0
    isotropy: ActiveStressModels = ActiveStressModels.transversely

    def __post_init__(self) -> None:
        if self.activation is None:
            self.activation = dolfinx.fem.Constant(self.f0.ufl_domain(), 0.0)

        self.T_ref = dolfinx.fem.Constant(self.f0.ufl_domain(), self.T_ref)
        self.eta = dolfinx.fem.Constant(self.f0.ufl_domain(), self.eta)

    @property
    def Ta(self) -> ufl.core.expr.Expr:
        """The active stress"""
        return self.T_ref * self.activation

    def F(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return F

    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Active strain energy density

        Parameters
        ----------
        F : ufl.core.expr.Expr
            _description_

        Returns
        -------
        ufl.core.expr.Expr
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        C = F.T * F
        if self.isotropy == ActiveStressModels.transversely:
            return transversely_active_stress(Ta=self.Ta, C=C, f0=self.f0, eta=self.eta)
        else:
            raise NotImplementedError


def transversely_active_stress(Ta, C, f0, eta=0.0):
    """
    Return active strain energy when activation is only
    working along the fibers, with a possible transverse
    component defined by eta

    Arguments
    ---------
    Ta : dolfinx.fem.Function or dolfinx.femConstant
        A scalar function representing the magnitude of the
        active stress in the reference configuration (first Piola)
    C : ufl.Form
        The right Cauchy-Green deformation tensor
    f0 : dolfinx.fem.Function
        A vector function representing the direction of the
        active stress
    eta : float
        Amount of active stress in the transverse direction
        (relative to f0)
    """

    I4f = ufl.inner(C * f0, f0)
    I1 = ufl.tr(C)
    return 0.5 * Ta * ((I4f - 1) + eta * ((I1 - 3) - (I4f - 1)))
