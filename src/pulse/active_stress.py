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
        logger.debug(f"Created ActiveStress model with Isotropy: {self.isotropy}")

    @property
    def Ta(self) -> ufl.core.expr.Expr:
        """The active stress"""
        Ta = self.activation.to_base_units()
        return self.T_ref * Ta

    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return F

    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Active strain energy density

        Parameters
        ----------
        C : ufl.core.expr.Expr
            The right Cauchy-Green deformation tensor

        Returns
        -------
        ufl.core.expr.Expr
            The active strain energy density

        Raises
        ------
        NotImplementedError
            _description_
        """
        if self.isotropy == ActiveStressModels.transversely:
            return transversely_active_stress_strain_energy(
                Ta=self.Ta,
                C=C,
                f0=self.f0,
                eta=self.eta,
            )
        else:
            raise NotImplementedError

    def S(self, C: ufl.core.expr.Expr, dev: bool = False) -> ufl.core.expr.Expr:
        """Cauchy stress tensor for the active stress model.

        Parameters
        ----------
        C : ufl.core.expr.Expr
            The right Cauchy-Green deformation tensor
        dev : bool
            Whether to compute the stress for the deviatoric part only

        Returns
        -------
        ufl.core.expr.Expr
            The Cauchy stress tensor
        """

        if self.isotropy == ActiveStressModels.transversely:
            return transversely_active_stress(Ta=self.Ta, f0=self.f0, eta=self.eta)
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return "Ta (I4f - 1 + \u03b7 ((I1 - 3) - (I4f - 1)))"


@dataclass(slots=True)
class FrankStarlingActiveStress(ActiveStress):
    """
    Active stress model incorporating the Frank-Starling mechanism.
    Multiplies the baseline time-dependent activation by a stretch-dependent factor.
    """

    amp_min: float = 0.0
    amp_max: float = 1.0
    lam_threslo: float = 0.85
    lam_maxlo: float = 1.15

    # Internal field to store the displacement.
    # init=False ensures it is not requested in the class constructor.
    _u: dolfinx.fem.Function | None = field(default=None, init=False, repr=False)

    def register(self, u: dolfinx.fem.Function) -> None:
        """
        Register the displacement field.
        This must be called before the active stress is evaluated.
        """
        self._u = u

    def frank_starling_multiplier(self) -> ufl.core.expr.Expr:
        """Computes the stretch-dependent multiplier g(lambda)."""
        if self._u is None:
            # return 1.0
            raise ValueError("Displacement 'u' has not been registered. Call register(u) first.")

        # Reconstruct kinematics from the registered displacement
        dim = self._u.ufl_shape[0]
        I = ufl.Identity(dim)
        F = I + ufl.grad(self._u)
        C = F.T * F

        # Calculate fiber stretch: lambda_f = sqrt(f0 * C * f0)
        I4 = ufl.inner(C * self.f0, self.f0)
        lam = ufl.sqrt(I4)

        # Slope for the linear ascending limb
        slope = (self.amp_max - self.amp_min) / (self.lam_maxlo - self.lam_threslo)

        # Piecewise linear Frank-Starling curve
        g_lam = ufl.conditional(
            ufl.le(lam, self.lam_threslo),
            self.amp_min,
            ufl.conditional(
                ufl.le(lam, self.lam_maxlo),
                self.amp_min + slope * (lam - self.lam_threslo),
                self.amp_max,
            ),
        )
        return g_lam

    @property
    def Ta(self) -> ufl.core.expr.Expr:
        """
        Overrides the base active tension property.
        The parent class methods (like S and stress_tensor) will automatically
        use this stretch-dependent tension.
        """
        Ta = self.activation.to_base_units()
        return self.T_ref * Ta * self.frank_starling_multiplier()


def transversely_active_stress_strain_energy(Ta, C, f0, eta=0.0):
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


def transversely_active_stress(Ta, f0, eta=0.0):
    r"""
    Return the Cauchy stress tensor for the active stress model
    when activation is only working along the fibers, with a
    possible transverse component defined by :math:`\eta` with
    :math:`\eta = 0` meaning that all active stress is along the
    fiber and :math:`\eta = 1` meaning that all active stress is in
    the transverse direction. The Cauchy stress tensor is given by

    .. math::
        \sigma = T_a \left( I_{4f} - 1 + \eta ((I_1 - 3) - (I_{4f} - 1)) \right) f_0

    Arguments
    ---------
    Ta : dolfinx.fem.Function or dolfinx.femConstant
        A scalar function representing the magnitude of the
        active stress in the reference configuration (first Piola)
    f0 : dolfinx.fem.Function
        A vector function representing the direction of the
        active stress
    eta : float
        Amount of active stress in the transverse direction
        (relative to f0)
    """
    S = Ta * ufl.outer(f0, f0)
    if not np.isclose(float(eta), 0.0):
        S += Ta * eta * (ufl.Identity(len(f0)) - ufl.outer(f0, f0))
    return S
