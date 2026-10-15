"""This module defines boundary conditions.

Boundary conditions are used to specify the behavior of the solution on the boundary of the domain.
The boundary conditions can be Dirichlet, Neumann, or Robin boundary conditions.

Dirichlet boundary conditions are used to specify the solution on the boundary of the domain.
Neumann boundary conditions are used to specify the traction on the boundary of the domain.
Robin boundary conditions are used to specify a Robin type boundary condition
on the boundary of the domain. A pericardium boundary condition ties a surface to a fixed
surface around it, but lets it slide along that surface.

The boundary conditions are collected in a `BoundaryConditions` object.
"""

import logging
import math
import typing
from dataclasses import dataclass
from enum import Enum

import dolfinx
import ufl

from .surface import PericardialSurface, SampledSurface, TrackedSurface
from .units import Variable

logger = logging.getLogger(__name__)


class RobinNormal(str, Enum):
    r"""Which surface normal a :class:`RobinBC` spring or dashpot acts along.

    Both measure the spring's extension by projecting the displacement (or
    velocity) onto a normal, so neither is the true distance to the surface the
    spring is anchored to.

    ``current``
        The normal :math:`\mathbf{n}` and area :math:`da` of the current
        configuration, pushed forward with Nanson's formula,
        :math:`\mathbf{t} = k (\mathbf{u} \cdot \mathbf{n}) \mathbf{n}`. It
        derives from no energy. It follows the surface as it deforms, so it
        also resists displacement that turns normal as the wall rotates.
    ``reference``
        The normal :math:`\mathbf{N}` and area :math:`dA` of the reference
        configuration, :math:`\mathbf{t} = k (\mathbf{u} \cdot \mathbf{N}) \mathbf{N}`
        (Pfaller et al. 2019, eqs. 4-5). It is the derivative of the energy
        :math:`\frac{1}{2} \int k (\mathbf{u} \cdot \mathbf{N})^2 \, dA`, so its
        Jacobian is symmetric. It assumes small rotations of the surface. Under
        large deformation it lets a free base flare: in the fixed-point unloader
        demo, the same springs nearly double the inflation, and the unloaded
        base comes out wider than the loaded one.

    Leaving :attr:`RobinBC.normal` unset keeps what pulse did before the option
    existed, which the demos and templates were tuned with: springs act along
    ``current``, dashpots along ``reference``.

    On a curved surface both forms register pure tangential sliding as a
    change in the normal gap, of second order in the rotation. So a stiff
    epicardial spring holds back ventricular twist. ``reference`` does so
    more than ``current``.
    """

    reference = "reference"
    current = "current"


def nanson(F: ufl.core.expr.Expr, N: ufl.core.expr.Expr):
    r"""Push the unit normal ``N`` through ``F``: :math:`\mathbf{n}\, da = J \mathbf{F}^{-T}
    \mathbf{N}\, dA`. Returns the unit normal :math:`\mathbf{n}` and the area ratio
    :math:`da / dA`."""
    cof_N = ufl.det(F) * ufl.inv(F).T * N
    ratio = ufl.sqrt(ufl.dot(cof_N, cof_N))
    return cof_N / ratio, ratio


@dataclass(slots=True)
class NeumannBC:
    traction: Variable
    marker: int

    def __post_init__(self):
        if not isinstance(self.traction, Variable):
            unit = "kPa"
            logger.warning("Traction is not a Variable, defaulting to kPa")
            self.traction = Variable(self.traction, unit)
        logger.debug(f"Created NeumannBC on marker {self.marker} with traction {self.traction}")


@dataclass(slots=True)
class RobinBC:
    """A spring (``damping=False``) or dashpot (``damping=True``) on the facets ``marker``.

    It acts along the surface normal, or, with ``perpendicular=True``, in the tangent plane.
    ``normal`` chooses the current or the reference normal; see :class:`RobinNormal`. Left
    unset, it is ``current`` for a spring and ``reference`` for a dashpot, as before the
    option existed. The spring is at rest in the reference configuration.
    """

    value: Variable
    marker: int
    damping: bool = False
    perpendicular: bool = False
    normal: RobinNormal | None = None

    def __post_init__(self):
        if not isinstance(self.value, Variable):
            unit = "Pa s / m" if self.damping else "Pa / m"
            logger.warning(f"Value is not a Variable, defaulting to {unit}")
            self.value = Variable(self.value, unit)
        if self.normal is None:
            normal = RobinNormal.reference if self.damping else RobinNormal.current
        else:
            normal = RobinNormal(self.normal)
        self.normal = normal
        logger.debug(
            f"Created RobinBC on marker {self.marker} with value {self.value} "
            f"({'damping' if self.damping else 'stiffness'}, {normal.value} normal)",
        )

    def projection(
        self,
        N: ufl.core.expr.Expr,
        F: ufl.core.expr.Expr,
        mesh_is_reference: bool = True,
    ):
        """Return the projector this BC acts with and the area ratio from the mesh to it.

        ``N`` is the unit normal of the mesh. ``F`` is the deformation gradient from the mesh
        to the other configuration: the current one in a forward problem, and the reference
        one in the inverse (prestress) problem, where ``mesh_is_reference=False``. Integrate
        the traction against the mesh's ``ds`` times the returned ratio.
        """
        if (self.normal == RobinNormal.reference) == mesh_is_reference:
            n, ratio = N, 1.0
        else:
            n, ratio = nanson(F, N)
        nn = ufl.outer(n, n)
        if self.perpendicular:
            return ufl.Identity(nn.ufl_shape[0]) - nn, ratio
        return nn, ratio


@dataclass(slots=True)
class PericardiumBC:
    r"""Springs that hold the boundary ``marker`` at its distance from a fixed ``surface``,
    and let it slide along it without friction.

    The gap is the change in signed distance :math:`d` to the surface since the reference
    configuration, :math:`g = d(\mathbf{x}) - d(\mathbf{X})`, with
    :math:`\mathbf{x} = \mathbf{X} + \mathbf{u}`. The springs store
    :math:`W = \frac{1}{2} \int k g^2 \, dA`, so the traction is
    :math:`\mathbf{t} = k g \nabla d(\mathbf{x})` per reference area, and the Jacobian is
    symmetric.

    A :class:`RobinBC` measures the gap by projecting :math:`\mathbf{u}` onto a normal
    instead. On a curved surface that counts tangential sliding as a normal gap, so stiff
    springs hold back twist, or, along the current normal, let the boundary drift away from
    the surface. Here a boundary that slides along the surface stays at the same distance
    from it and feels no force; for example an LV rotating about the long axis of an
    :class:`~pulse.surface.EllipsoidSurface`. Subtracting :math:`d(\mathbf{X})` puts the
    springs at rest in the reference configuration, wherever the mesh's facets lie relative
    to the surface.

    ``surface`` is either a :class:`~pulse.surface.PericardialSurface`, which gives the
    distance as a UFL expression, like the ``EllipsoidSurface``, or a
    :class:`~pulse.surface.SampledSurface`, which computes it point by point, like the
    :class:`~pulse.surface.MeshSurface` through the marked facets of a mesh. With a sampled
    surface the integral is a sum over the nodes of a degree 2 Lagrange space on the
    boundary; see :class:`~pulse.surface.TrackedSurface`.

    So the springs do not hold a rotation that maps the surface onto itself. Hold it
    elsewhere, for example with springs in the plane of the base (``RobinBC`` with
    ``perpendicular=True``). Left free, the problem is singular: a solve may still converge,
    but the :class:`~pulse.unloading.FixedPointUnloader` does not.

    ``damping`` adds a dashpot :math:`c \dot{g} \nabla d(\mathbf{x})`, with
    :math:`\dot{g} = \nabla d(\mathbf{x}) \cdot \dot{\mathbf{u}}`, in a
    :class:`~pulse.problem.DynamicProblem`; other problems ignore it. ``unilateral`` makes
    the springs and dashpot act only where the boundary is outside its reference distance,
    :math:`g > 0`, so it can come away from the surface freely. In systole the heart moves
    inward, so a unilateral pericardium then does nothing.

    The unilateral switch :math:`\max(g, 0)` has a kink at :math:`g = 0`, and a boundary that
    crosses its reference distance there, as the epicardium does in early systole, can stall
    Newton. ``smoothing``, a length :math:`\epsilon`, rounds the kink off: the springs store
    :math:`W = \int k \Phi(g) \, dA` with :math:`\Phi' = r`, where :math:`r(g) = 0` for
    :math:`g \le 0`, :math:`g^2 / 2\epsilon` for :math:`0 < g < \epsilon` and
    :math:`g - \epsilon / 2` beyond. So the traction :math:`k\, r(g) \nabla d` is once
    differentiable, still zero in the reference configuration, and short of the unsmoothed
    one by :math:`k \epsilon / 2` once the gap passes :math:`\epsilon`. The dashpot is gated
    by :math:`r'(g)`, which goes from 0 to 1 across the same band. It applies only with
    ``unilateral=True``; ``None`` keeps the kink.

    In the inverse problem (:class:`~pulse.unloading.PrestressProblem`) the mesh is the
    loaded configuration, and the springs are still at rest in the unloaded one, as a
    :class:`RobinBC`'s are.
    """

    stiffness: Variable
    marker: int
    surface: PericardialSurface | SampledSurface
    damping: Variable | None = None
    unilateral: bool = False
    smoothing: Variable | None = None

    def __post_init__(self):
        if not isinstance(self.stiffness, Variable):
            logger.warning("Stiffness is not a Variable, defaulting to Pa / m")
            self.stiffness = Variable(self.stiffness, "Pa / m")
        if self.damping is not None and not isinstance(self.damping, Variable):
            logger.warning("Damping is not a Variable, defaulting to Pa s / m")
            self.damping = Variable(self.damping, "Pa s / m")
        if self.smoothing is not None:
            if not self.unilateral:
                raise ValueError("smoothing applies only to a unilateral PericardiumBC")
            if not isinstance(self.smoothing, Variable):
                logger.warning("Smoothing is not a Variable, defaulting to m")
                self.smoothing = Variable(self.smoothing, "m")
        logger.debug(
            f"Created PericardiumBC on marker {self.marker} with stiffness {self.stiffness}, "
            f"damping {self.damping}, unilateral={self.unilateral}, "
            f"smoothing {self.smoothing}, surface {self.surface}",
        )

    def traction(
        self,
        x: ufl.core.expr.Expr,
        X: ufl.core.expr.Expr,
        scale: float = 1.0,
        velocity: ufl.core.expr.Expr | None = None,
        surface: PericardialSurface | None = None,
    ) -> ufl.core.expr.Expr:
        """The traction per reference area on the material point at ``X``, now at ``x``.

        ``scale`` converts the mesh unit to metres, as for a :class:`RobinBC`. ``velocity``
        is the point's velocity, for the dashpot; without it, there is none. ``surface``
        gives the distance instead of the surface of this condition: its
        :class:`~pulse.surface.TrackedSurface`, if that is a sampled one (see
        :func:`pericardial_surface`).
        """
        if surface is None:
            if not hasattr(self.surface, "signed_distance"):
                raise TypeError(
                    f"{self.surface} gives no distance as a UFL expression; pass the "
                    "TrackedSurface that pericardial_surface makes of it",
                )
            surface = typing.cast(PericardialSurface, self.surface)
        xv = ufl.variable(x)
        d = surface.signed_distance(xv)
        n = ufl.diff(d, xv)
        g = d - surface.signed_distance(X)
        k = self.stiffness.to_base_units() * scale
        if self.unilateral and self.smoothing is not None:
            # In mesh units, as g is
            eps = self.smoothing.to_base_units() / scale
            spring = ufl.conditional(
                ufl.le(g, 0.0),
                0.0,
                ufl.conditional(ufl.ge(g, eps), g - eps / 2, g**2 / (2 * eps)),
            )
            gate = ufl.max_value(ufl.min_value(g / eps, 1.0), 0.0)
        elif self.unilateral:
            spring = ufl.max_value(g, 0.0)
            gate = ufl.conditional(ufl.gt(g, 0.0), 1.0, 0.0)
        else:
            spring = g
            gate = 1.0
        t = k * spring * n
        if self.damping is not None and velocity is not None:
            c = self.damping.to_base_units() * scale
            t += c * gate * ufl.dot(n, velocity) * n
        return t


def pericardial_surface(
    bc: PericardiumBC,
    mesh: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags | None,
    ds: ufl.Measure,
) -> tuple[PericardialSurface, ufl.Measure]:
    """What a problem on ``mesh`` builds the forms of ``bc`` with: the surface to pass to
    :meth:`PericardiumBC.traction`, and the measure to integrate the traction with over
    ``bc.marker``.

    They are the surface of ``bc`` and ``ds``, unless the surface is a sampled one. Then
    they are a new :class:`~pulse.surface.TrackedSurface` of it and its measure, and the
    problem has to update the tracked surface before it assembles anything.
    """
    if hasattr(bc.surface, "signed_distance"):
        return typing.cast(PericardialSurface, bc.surface), ds
    if facet_tags is None:
        raise ValueError("A PericardiumBC with a sampled surface needs facet tags")
    tracked = TrackedSurface(
        typing.cast(SampledSurface, bc.surface),
        mesh,
        facet_tags.find(bc.marker),
        ds,
    )
    return tracked, tracked.ds


def check_pericardium(
    bc: PericardiumBC,
    mesh: dolfinx.mesh.Mesh,
    ds: ufl.Measure,
    surface: PericardialSurface | None = None,
) -> None:
    """Warn if the surface of ``bc`` does not look like the boundary it holds: if the
    boundary is far from it, or its normal points elsewhere. A surface in the wrong unit, or
    around the wrong boundary, gives no error otherwise, since the springs are at rest
    wherever the boundary starts. ``surface`` and ``ds`` are those that
    :func:`pericardial_surface` gives."""
    from mpi4py import MPI

    if surface is None:
        surface = typing.cast(PericardialSurface, bc.surface)
    X = ufl.SpatialCoordinate(mesh)
    Xv = ufl.variable(X)
    d = surface.signed_distance(Xv)
    N = ufl.FacetNormal(mesh)

    def integrate(f):
        value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(f * ds(bc.marker)))
        return mesh.comm.allreduce(value, op=MPI.SUM)

    area = integrate(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)))
    if area == 0:
        raise ValueError(f"PericardiumBC marker {bc.marker} has no facets")
    rms = math.sqrt(integrate(d**2) / area)
    alignment = integrate(ufl.dot(ufl.diff(d, Xv), N)) / area
    logger.debug(
        f"PericardiumBC on marker {bc.marker}: RMS distance to the surface {rms:.3g}, "
        f"mean alignment of the normals {alignment:.3f}",
    )
    if rms > 0.1 * math.sqrt(area) or abs(alignment - 1) > 0.1:
        logger.warning(
            f"The surface of the PericardiumBC on marker {bc.marker} does not match the "
            f"boundary: the RMS distance between them is {rms:.3g} mesh units (area "
            f"{area:.3g}), and the mean dot product of their normals {alignment:.3f}. Is the "
            "surface in mesh units, and around this boundary?",
        )


class BoundaryConditions(typing.NamedTuple):
    neumann: typing.Sequence[NeumannBC] = ()
    dirichlet: typing.Sequence[
        typing.Callable[
            [dolfinx.fem.FunctionSpace],
            typing.Sequence[dolfinx.fem.bcs.DirichletBC],
        ]
    ] = ()
    robin: typing.Sequence[RobinBC] = ()
    body_force: typing.Sequence[float | dolfinx.fem.Constant | dolfinx.fem.Function] = ()
    pericardium: typing.Sequence[PericardiumBC] = ()
