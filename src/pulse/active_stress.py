"""This module contains the active stress model for the cardiac
mechanics problem. The active stress model is used to describe
the active contraction of the heart. The active stress model
is used to compute the active stress given the deformation gradient.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import dolfinx
import numpy as np
import ufl

from . import kinematics
from .active_model import ActiveModel
from .units import Variable

logger = logging.getLogger(__name__)


class ActiveStressModels(str, Enum):
    transversely = "transversely"
    orthotropic = "orthotropic"
    fully_anisotropic = "fully_anisotropic"


class ActiveStressFormulation(str, Enum):
    r"""Which power of the fiber stretch the active energy is linear in.

    Both formulations put the active stress along the fiber, and differ only
    in what :math:`T_a` is taken to multiply -- by a factor of the fiber
    stretch :math:`\lambda`. Both appear in the literature, so the choice
    must be explicit rather than implied.

    ``invariant``
        :math:`\Psi_a = \frac{1}{2} T_a (I_{4f} - 1)`, giving
        :math:`\mathbf{S}_a = T_a\, f_0 \otimes f_0` and
        :math:`\mathbf{P}_a = T_a\, \mathbf{F} f_0 \otimes f_0`.
        Adding :math:`T_a f_0 \otimes f_0` to the second Piola stress is the
        most widespread form of the active stress approach in cardiac
        mechanics, and it remains the default here for that reason and for
        backwards compatibility. Note the fibre traction it delivers scales
        with the stretch: :math:`|\mathbf{P}_a f_0| = T_a \lambda`.

    ``stretch``
        :math:`\Psi_a = T_a (\lambda - 1)` with
        :math:`\lambda = \sqrt{I_{4f}}`, giving
        :math:`\mathbf{S}_a = \frac{T_a}{\lambda} f_0 \otimes f_0` and
        :math:`\mathbf{P}_a = T_a \frac{\mathbf{F} f_0 \otimes f_0}
        {|\mathbf{F} f_0|}`.
        This is the convention used by Regazzoni & Quarteroni
        :cite:`regazzoni2021oscillation`, and the one
        :class:`StabilizedActiveStress` is derived in. Choose it when
        :math:`T_a` comes from a force-generation model whose active
        stiffness is defined as :math:`\partial \dot{T_a}/\partial
        \dot\lambda`, so that tension and stiffness refer to the same
        kinematic variable.
    """

    invariant = "invariant"
    stretch = "stretch"


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
    formulation: ActiveStressFormulation
        Which active-stress convention to use, by default 'invariant'.
        The two differ by a factor of the fiber stretch, so this changes
        results rather than just their derivation -- see
        :class:`ActiveStressFormulation` for how to choose.
    """

    f0: dolfinx.fem.Function | dolfinx.fem.Constant
    activation: Variable = field(default_factory=lambda: Variable(0.0, "kPa"))
    s0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    n0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None
    T_ref: dolfinx.fem.Constant | float = 1.0
    eta: dolfinx.fem.Constant | float = 0.0
    isotropy: ActiveStressModels = ActiveStressModels.transversely
    formulation: ActiveStressFormulation = ActiveStressFormulation.invariant

    def __post_init__(self) -> None:
        if not isinstance(self.activation, Variable):
            unit = "kPa"
            logger.warning("Activation is not a Variable, defaulting to kPa")
            self.activation = Variable(self.activation, unit)

        Ta = self.activation.to_base_units()

        if Ta is None:
            Ta = 0.0

        domain = ufl.domain.extract_unique_domain(self.f0)
        assert isinstance(domain, ufl.Mesh)

        if isinstance(Ta, (float, int)) or np.isscalar(Ta):
            self.activation = Variable(
                dolfinx.fem.Constant(
                    domain,
                    dolfinx.default_scalar_type(cast(Any, Ta)),
                ),
                self.activation.unit,
            )

        if not isinstance(self.T_ref, dolfinx.fem.Constant):
            self.T_ref = dolfinx.fem.Constant(
                domain,
                self.T_ref,
            )
        if not isinstance(self.eta, dolfinx.fem.Constant):
            self.eta = dolfinx.fem.Constant(domain, self.eta)
        logger.debug(f"Created ActiveStress model with Isotropy: {self.isotropy}")

    @property
    def Ta(self) -> ufl.core.expr.Expr:
        """The active stress"""
        Ta = self.activation.to_base_units()
        return cast(ufl.core.expr.Expr, self.T_ref * Ta)

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
        if self.isotropy != ActiveStressModels.transversely:
            raise NotImplementedError

        if self.formulation == ActiveStressFormulation.stretch:
            _check_no_transverse(self.eta)
            return stretch_active_stress_strain_energy(Ta=self.Ta, C=C, f0=self.f0)

        return transversely_active_stress_strain_energy(
            Ta=self.Ta,
            C=C,
            f0=self.f0,
            eta=self.eta,
        )

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

        if self.isotropy != ActiveStressModels.transversely:
            raise NotImplementedError

        if self.formulation == ActiveStressFormulation.stretch:
            _check_no_transverse(self.eta)
            return stretch_active_stress(Ta=self.Ta, C=C, f0=self.f0)

        return transversely_active_stress(Ta=self.Ta, f0=self.f0, eta=self.eta)

    def __str__(self) -> str:
        if self.formulation == ActiveStressFormulation.stretch:
            return "Ta (\u03bb - 1)"
        return "Ta (I4f - 1 + \u03b7 ((I1 - 3) - (I4f - 1)))"


@dataclass(slots=True)
class StabilizedActiveStress(ActiveModel):
    r"""Active stress with the consistent stabilization term of Regazzoni &
    Quarteroni, for use when :math:`T_a` comes from an *external*
    force-generation solver.

    Why you probably want this
    --------------------------
    The usual way to drive :class:`ActiveStress` is to advance some cell-level
    contraction model, write its tension into ``activation``, and solve
    mechanics with that value held fixed. This is a segregated (staggered)
    scheme, and it has a failure mode that is easy to hit and hard to
    diagnose: whenever the *active* stiffness of the tissue exceeds its
    passive stiffness -- routine in contracting myocardium -- the scheme
    develops non-physical oscillations in :math:`T_a` and :math:`\lambda`.
    Regazzoni & Quarteroni :cite:`regazzoni2021oscillation` show it is then not
    merely inaccurate but not
    convergent, its amplification factor tending to :math:`-K_a/K_p < -1` as
    :math:`\Delta t \to 0`. **Reducing the time step makes it worse**, so the
    problem cannot be tuned away.

    The cause is that a staggered scheme treats active tension as a dead load
    over the mechanics solve, when physically it is a population of
    crossbridges behaving as springs. Restoring that gives

    .. math::
        \mathbf{P}_{act} = \left[T_a + K_a(\lambda - \lambda_{prev})\right]
            \frac{\mathbf{F} f_0 \otimes f_0}{|\mathbf{F} f_0|}

    which is the gradient of

    .. math::
        \Psi_a = T_a (\lambda - \lambda_{prev})
               + \tfrac{1}{2} K_a (\lambda - \lambda_{prev})^2

    and is what this class implements. The extra term is
    :math:`\mathcal{O}(\Delta t)` and vanishes in the limit, so the scheme
    remains consistent with the same continuous problem -- it is a numerical
    device, not a change of model -- while becoming unconditionally stable.

    Usage
    -----
    Each time step, in this order:

    1. advance the force-generation model using :math:`\lambda_{prev}`,
    2. assign the resulting tension and stiffness to ``activation`` and
       ``active_stiffness``,
    3. solve mechanics,
    4. call :meth:`update_prev` with the new displacement.

    Step 4 matters: :math:`\lambda_{prev}` must be the *same* stretch that was
    fed to the force-generation model in step 1. If the two drift apart the
    added term is no longer a consistent perturbation and can itself
    destabilize the solve.

    Parameters
    ----------
    f0 : dolfinx.fem.Function | dolfinx.fem.Constant
        The cardiac fiber direction
    activation : Variable
        The active tension :math:`T_a`, from the force-generation model
    active_stiffness : Variable
        The active stiffness :math:`K_a = \partial \dot{T_a} /
        \partial \dot\lambda`, from the same model, in the same units as
        ``activation`` and per unit of the *same* stretch variable. Setting
        it to zero recovers the plain staggered scheme -- useful for
        demonstrating the instability, not for production.
    lmbda_prev : dolfinx.fem.Function | dolfinx.fem.Constant | None
        Fiber stretch at the previous time step. Defaults to a constant 1.0,
        i.e. the reference configuration. Pass a ``Function`` (and use
        :meth:`update_prev`) for anything beyond a single step.

    Notes
    -----
    Unlike :class:`ActiveStress` this takes no ``T_ref`` or ``eta``. A
    reference scaling applied to :math:`T_a` but not :math:`K_a` would
    silently break the consistency of the stabilization, and there is no
    accepted transverse generalization of an energy written in
    :math:`\lambda`. Scale both quantities before assigning them instead.
    """

    f0: dolfinx.fem.Function | dolfinx.fem.Constant
    activation: Variable = field(default_factory=lambda: Variable(0.0, "kPa"))
    active_stiffness: Variable = field(default_factory=lambda: Variable(0.0, "kPa"))
    lmbda_prev: dolfinx.fem.Function | dolfinx.fem.Constant | None = None

    def __post_init__(self) -> None:
        mesh = ufl.domain.extract_unique_domain(self.f0)
        assert isinstance(mesh, ufl.Mesh)

        self.activation = _as_constant_variable(self.activation, mesh, "activation")
        self.active_stiffness = _as_constant_variable(
            self.active_stiffness,
            mesh,
            "active_stiffness",
        )

        if self.lmbda_prev is None:
            self.lmbda_prev = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))

        logger.debug("Created StabilizedActiveStress model")

    @property
    def Ta(self) -> ufl.core.expr.Expr:
        """Active tension, in base units."""
        return cast(ufl.core.expr.Expr, self.activation.to_base_units())

    @property
    def Ka(self) -> ufl.core.expr.Expr:
        """Active stiffness, in base units."""
        return cast(ufl.core.expr.Expr, self.active_stiffness.to_base_units())

    def dlmbda(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Increment in fiber stretch since the previous time step."""
        return fiber_stretch(C, self.f0) - self.lmbda_prev

    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return F

    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r""":math:`\Psi_a = T_a \Delta\lambda + \frac{1}{2} K_a \Delta\lambda^2`"""
        dl = self.dlmbda(C)
        return self.Ta * dl + 0.5 * self.Ka * dl**2

    def S(self, C: ufl.core.expr.Expr, dev: bool = False) -> ufl.core.expr.Expr:
        r"""Second Piola-Kirchhoff stress,

        .. math::
            \mathbf{S} = \frac{T_a + K_a \Delta\lambda}{\lambda}
                         f_0 \otimes f_0

        Given in closed form rather than by differentiating
        :meth:`strain_energy`; the two agree, which
        ``test_stabilized_active_stress.py`` checks.
        """
        lmbda = fiber_stretch(C, self.f0)
        return ((self.Ta + self.Ka * self.dlmbda(C)) / lmbda) * ufl.outer(self.f0, self.f0)

    def update_prev(self, u: dolfinx.fem.Function) -> None:
        """Record the fiber stretch of displacement ``u`` as :math:`\\lambda_{prev}`.

        Call once per time step, after the mechanics solve. Requires
        ``lmbda_prev`` to be a ``Function``; a ``Constant`` cannot hold a
        spatially varying stretch.
        """
        if not isinstance(self.lmbda_prev, dolfinx.fem.Function):
            raise TypeError(
                "update_prev requires lmbda_prev to be a dolfinx.fem.Function, got "
                f"{type(self.lmbda_prev).__name__}. Construct StabilizedActiveStress "
                "with lmbda_prev=dolfinx.fem.Function(V) to step it in time.",
            )
        F = kinematics.DeformationGradient(u)
        self.lmbda_prev.interpolate(
            dolfinx.fem.Expression(
                fiber_stretch(F.T * F, self.f0),
                self.lmbda_prev.function_space.element.interpolation_points,
            ),
        )

    def __str__(self) -> str:
        return "Ta Δλ + ½ Ka Δλ²"


def _as_constant_variable(value, mesh, name: str) -> Variable:
    """Coerce a raw number or Variable into a Variable holding a dolfinx object."""
    if not isinstance(value, Variable):
        logger.warning("%s is not a Variable, defaulting to kPa", name)
        value = Variable(value, "kPa")

    base: Any = value.to_base_units()
    if base is None:
        base = 0.0

    if isinstance(base, (float, int)) or np.isscalar(base):
        # cast: default_scalar_type is float or complex depending on how dolfinx
        # was built, so neither concrete scalar type type-checks on its own.
        return Variable(
            dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(cast(Any, base))),
            value.unit,
        )
    return value


def compute_frank_starling_multiplier(
    u: ufl.core.expr.Expr,
    f0: ufl.core.expr.Expr,
    amp_min: float,
    amp_max: float,
    stretch_threshold: float,
    stretch_optimal: float,
) -> ufl.core.expr.Expr:
    r"""
    Computes a stretch-dependent scalar multiplier for active tension to model
    the Frank-Starling mechanism using a piecewise linear ascending limb.

    Parameters
    ----------
    u : ufl.core.expr.Expr
        The macroscopic displacement vector field.
    f0 : ufl.core.expr.Expr
        The reference fiber direction vector field.
    amp_min : float
        The minimum amplification factor (used for tissue at resting or compressed lengths).
    amp_max : float
        The maximum amplification factor (used for tissue at or beyond the optimal stretch length).
    stretch_threshold : float
        The stretch ratio below which the active force remains at its minimum.
    stretch_optimal : float
        The optimal stretch ratio where the active force reaches its maximum plateau.

    Returns
    -------
    ufl.core.expr.Expr
        A symbolic UFL expression representing the spatial multiplier field g(lambda).

    Notes
    -----
    Mathematical Formulation:

    Let the right Cauchy-Green deformation tensor be :math:`\mathbf{C} = \mathbf{F}^T \mathbf{F}`
    where :math:`\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}` is the deformation gradient.

    The local fiber stretch :math:`\lambda` is computed as:

    .. math::

        \lambda = \sqrt{\mathbf{f}_0 \cdot (\mathbf{C} \mathbf{f}_0)}

    The multiplier :math:`g(\lambda)` is defined as a piecewise function:

    .. math::

        g(\lambda) =
        \begin{cases}
            a_{\min} & \text{if } \lambda \le \lambda_{\text{threshold}} \\
            a_{\min} + m (\lambda - \lambda_{\text{threshold}}) & \text{if }
            \lambda_{\text{threshold}} < \lambda \le \lambda_{\text{opt}} \\
            a_{\max} & \text{if } \lambda > \lambda_{\text{opt}}
        \end{cases}

    where the slope :math:`m` is calculated as:

    .. math::

        m = \frac{a_{\max} - a_{\min}}{\lambda_{\text{opt}} - \lambda_{\text{threshold}}}
    """
    dim = u.ufl_shape[0]
    I = ufl.Identity(dim)
    F = I + ufl.grad(u)
    C = F.T * F

    # Calculate fiber stretch: lambda_f = sqrt(f0 * C * f0)
    I4 = ufl.inner(C * f0, f0)
    lam = ufl.sqrt(I4)

    # Slope for the linear ascending limb
    slope = (amp_max - amp_min) / (stretch_optimal - stretch_threshold)

    # Piecewise linear ascending limb using UFL conditionals
    g_lam = ufl.conditional(
        ufl.le(lam, stretch_threshold),
        amp_min,
        ufl.conditional(
            ufl.le(lam, stretch_optimal),
            amp_min + slope * (lam - stretch_threshold),
            amp_max,
        ),
    )
    return g_lam


@dataclass(slots=True)
class FrankStarlingActiveStress(ActiveStress):
    """
    Active stress model incorporating the Frank-Starling mechanism.
    Multiplies the baseline time-dependent activation by a stretch-dependent factor.

    Parameters
    ----------
    amp_min : float, optional
        The minimum amplification factor, by default 0.0.
    amp_max : float, optional
        The maximum amplification factor, by default 1.0.
    stretch_threshold : float, optional
        The stretch ratio below which the active force
        remains at its minimum, by default 0.85.
    stretch_optimal : float, optional
        The optimal stretch ratio where the active force reaches
        its maximum plateau, by default 1.15.
    """

    amp_min: float = 0.0
    amp_max: float = 1.0
    stretch_threshold: float = 0.85
    stretch_optimal: float = 1.15

    # Internal field to store the displacement.
    # init=False ensures it is not requested in the class constructor.
    _u: dolfinx.fem.Function | None = field(default=None, init=False, repr=False)

    def register(self, u: dolfinx.fem.Function):
        """
        Registers the displacement field into the material model.
        This must be called before the active stress is evaluated so the model
        can calculate the dynamic stretch.

        Parameters
        ----------
        u : ufl.core.expr.Expr
            The displacement vector field to register.
        """
        self._u = u

    def frank_starling_multiplier(self) -> ufl.core.expr.Expr:
        """
        Class method wrapper that evaluates the standalone Frank-Starling multiplier
        function using the registered displacement and material properties.

        Returns
        -------
        ufl.core.expr.Expr
            A symbolic UFL expression of the multiplier.

        Raises
        ------
        ValueError
            If the displacement field `u` has not been registered yet.
        """
        if self._u is None:
            raise ValueError("Displacement 'u' has not been registered. Call register(u) first.")

        return compute_frank_starling_multiplier(
            u=self._u,
            f0=self.f0,
            amp_min=self.amp_min,
            amp_max=self.amp_max,
            stretch_threshold=self.stretch_threshold,
            stretch_optimal=self.stretch_optimal,
        )

    @property
    def Ta(self) -> ufl.core.expr.Expr:
        """
        Overrides the base active tension property from `ActiveStress`.
        The parent class methods (like S and stress_tensor) will automatically
        use this dynamically scaled active tension.

        Returns
        -------
        ufl.core.expr.Expr
            The total active tension (baseline activation * multiplier)
        """
        Ta = self.activation.to_base_units()
        return self.T_ref * Ta * self.frank_starling_multiplier()


def _check_no_transverse(eta) -> None:
    """The stretch formulation is defined for purely fiber-directed activation.

    There is no single accepted way to blend a transverse component into an
    energy written in :math:`\\lambda` rather than :math:`I_{4f}`, so rather
    than invent one, refuse it.
    """
    if not np.isclose(float(eta), 0.0):
        raise NotImplementedError(
            "ActiveStressFormulation.stretch is only defined for eta = 0 "
            f"(purely fiber-directed active stress), got eta = {float(eta)}. "
            "Use ActiveStressFormulation.invariant for transverse activation.",
        )


def fiber_stretch(C: ufl.core.expr.Expr, f0) -> ufl.core.expr.Expr:
    r"""Stretch along the fiber direction, :math:`\lambda = \sqrt{f_0 \cdot C f_0}`.

    Equal to :math:`|\mathbf{F} f_0|`, and to 1 in the reference configuration.

    Arguments
    ---------
    C : ufl.core.expr.Expr
        The right Cauchy-Green deformation tensor
    f0 : dolfinx.fem.Function or dolfinx.fem.Constant
        A vector function representing the fiber direction
    """
    return ufl.sqrt(ufl.inner(C * f0, f0))


def stretch_active_stress_strain_energy(Ta, C, f0):
    r"""Active strain energy that is linear in the fiber *stretch*,

    .. math::
        W = T_a (\lambda - 1), \qquad \lambda = \sqrt{I_{4f}}

    whose second Piola-Kirchhoff stress is :math:`T_a f_0 \otimes f_0 /
    \lambda` and whose first Piola-Kirchhoff stress is therefore
    :math:`T_a \mathbf{F} f_0 \otimes f_0 / |\mathbf{F} f_0|` -- the
    normalization used by Regazzoni & Quarteroni
    :cite:`regazzoni2021oscillation`.

    Compare :func:`transversely_active_stress_strain_energy`, which is linear
    in :math:`I_{4f} = \lambda^2` instead and hence differs by a factor of
    :math:`\lambda` in the resulting stress.

    Arguments
    ---------
    Ta : dolfinx.fem.Function or dolfinx.fem.Constant
        A scalar function representing the magnitude of the active tension
    C : ufl.Form
        The right Cauchy-Green deformation tensor
    f0 : dolfinx.fem.Function
        A vector function representing the fiber direction
    """
    return Ta * (fiber_stretch(C, f0) - 1.0)


def stretch_active_stress(Ta, C, f0):
    r"""Second Piola-Kirchhoff stress for :func:`stretch_active_stress_strain_energy`,

    .. math::
        \mathbf{S} = \frac{T_a}{\lambda} f_0 \otimes f_0

    Arguments
    ---------
    Ta : dolfinx.fem.Function or dolfinx.fem.Constant
        A scalar function representing the magnitude of the active tension
    C : ufl.Form
        The right Cauchy-Green deformation tensor
    f0 : dolfinx.fem.Function
        A vector function representing the fiber direction
    """
    return (Ta / fiber_stretch(C, f0)) * ufl.outer(f0, f0)


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
    Ta : dolfinx.fem.Function or dolfinx.fem.Constant
        A scalar function representing the magnitude of the active stress.
        Note that with this (``invariant``) formulation the resulting fibre
        traction is :math:`|\mathbf{P}_a f_0| = T_a \lambda`, not
        :math:`T_a` -- it is :class:`ActiveStressFormulation` ``stretch``
        under which :math:`T_a` is itself the first Piola fibre traction.
        Which one to supply depends on what your activation model's
        :math:`T_a` was calibrated to mean; see
        :class:`ActiveStressFormulation`.
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
    Ta : dolfinx.fem.Function or dolfinx.fem.Constant
        A scalar function representing the magnitude of the active stress.
        With this (``invariant``) formulation the resulting fibre traction is
        :math:`|\mathbf{P}_a f_0| = T_a \lambda`; see
        :class:`ActiveStressFormulation`.
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
