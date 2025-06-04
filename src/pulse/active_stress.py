"""This module contains the active stress model for the cardiac
mechanics problem. The active stress model is used to describe
the active contraction of the heart. The active stress model
is used to compute the active stress given the deformation gradient.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import dolfinx
import numpy as np
import ufl

from .active_model import ActiveModel
from .units import Variable

logger = logging.getLogger(__name__)


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
    activation: Variable = field(default_factory=lambda: Variable(0.0, "kPa"))
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    n0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    T_ref: dolfinx.fem.Constant = dolfinx.default_scalar_type(1.0)
    eta: dolfinx.fem.Constant = dolfinx.default_scalar_type(0.0)
    isotropy: ActiveStressModels = ActiveStressModels.transversely

    def __post_init__(self) -> None:
        if not isinstance(self.activation, Variable):
            unit = "kPa"
            logger.warning("Activation is not a Variable, defaulting to kPa")
            self.activation = Variable(self.activation, unit)

        Ta = self.activation.to_base_units()

        if Ta is None:
            Ta = 0.0

        if isinstance(Ta, (float, int)) or np.isscalar(Ta):
            self.activation = Variable(
                dolfinx.fem.Constant(
                    ufl.domain.extract_unique_domain(self.f0),
                    dolfinx.default_scalar_type(Ta),
                ),
                self.activation.unit,
            )

        self.T_ref = dolfinx.fem.Constant(ufl.domain.extract_unique_domain(self.f0), self.T_ref)
        self.eta = dolfinx.fem.Constant(ufl.domain.extract_unique_domain(self.f0), self.eta)

    @property
    def Ta(self) -> ufl.core.expr.Expr:
        """The active stress"""
        Ta = self.activation.to_base_units()
        return self.T_ref * Ta

    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return F

    def strain_energy(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Active strain energy density

        Parameters
        ----------
        F : ufl.core.expr.Expr
            The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            The active strain energy density

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

    def __str__(self) -> str:
        return "Ta (I4f - 1 + \u03b7 ((I1 - 3) - (I4f - 1)))"


def transversely_active_stress(Ta, C, f0, eta=0.0):
    r"""
    Return active strain energy when activation is only
    working along the fibers, with a possible transverse
    component defined by :math:`\eta` with :math:`\eta = 0`
    meaning that all active stress is along the fiber and
    :math:`\eta = 1` meaning that all active stress is in the
    transverse direction. The active strain energy is given by

    .. math::
        W = \frac{1}{2} T_a \left( I_{4f} - 1 + \eta ((I_1 - 3) - (I_{4f} - 1)) \right)

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
