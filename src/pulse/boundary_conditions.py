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
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import dolfinx
import ufl

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


class PericardialSurface(typing.Protocol):
    """A surface fixed in space that a :class:`PericardiumBC` holds a boundary against."""

    def signed_distance(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """The signed distance from the point ``x`` to the surface, positive outside, in
        mesh units. It must be smooth near the surface, and its gradient there of unit
        length and normal to the surface."""
        ...


@dataclass(slots=True)
class EllipsoidSurface:
    r"""An ellipsoid of revolution with semi-axis ``r_long`` along ``axis`` and ``r_short``
    across it, centred at ``center``. Lengths are in mesh units.

    The signed distance is the first-order one of the level set
    :math:`\phi = \sqrt{(s / r_\mathrm{long})^2 + \rho^2 / r_\mathrm{short}^2} - 1`, with
    :math:`s` and :math:`\rho` the axial and radial coordinates: :math:`d = \phi /
    |\nabla \phi|`. It is exact on the surface and for a sphere, and its error grows with
    the square of the distance over the radius of curvature. It is symmetric about the axis,
    so a rotation about the axis leaves it unchanged everywhere.
    """

    r_long: float
    r_short: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self):
        if self.r_long <= 0 or self.r_short <= 0:
            raise ValueError(f"Radii must be positive, got {self.r_long}, {self.r_short}")
        length = math.sqrt(sum(a * a for a in self.axis))
        if length == 0:
            raise ValueError("axis must be nonzero")
        self.axis = typing.cast(
            tuple[float, float, float],
            tuple(float(a) / length for a in self.axis),
        )

    @classmethod
    def from_cardiac_geometries(
        cls,
        geo: typing.Any,
        surface: str = "epi",
    ) -> "EllipsoidSurface":
        """The epicardial (``surface="epi"``) or endocardial (``"endo"``) ellipsoid of a
        cardiac-geometriesx ``lv_ellipsoid``, from its ``info`` (or the info dict itself).
        Its long axis is x and its centre the origin. If the mesh was scaled after it was
        made, scale the radii likewise."""
        info = geo if isinstance(geo, Mapping) else geo.info
        mesh_type = info.get("mesh_type")
        if mesh_type != "lv_ellipsoid":
            raise ValueError(
                f"Can only build an EllipsoidSurface from an 'lv_ellipsoid', got {mesh_type!r}",
            )
        if surface not in ("epi", "endo"):
            raise ValueError(f"surface must be 'epi' or 'endo', got {surface!r}")
        return cls(r_long=info[f"r_long_{surface}"], r_short=info[f"r_short_{surface}"])

    def signed_distance(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        a = ufl.as_vector(self.axis)
        y = x - ufl.as_vector(self.center)
        s = ufl.dot(y, a)
        y_radial = y - s * a
        psi = (s / self.r_long) ** 2 + ufl.dot(y_radial, y_radial) / self.r_short**2
        grad_psi = 2 * s / self.r_long**2 * a + 2 * y_radial / self.r_short**2
        # phi = sqrt(psi) - 1 and grad(phi) = grad(psi) / (2 sqrt(psi))
        return 2 * (psi - ufl.sqrt(psi)) / ufl.sqrt(ufl.dot(grad_psi, grad_psi))


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
    :class:`EllipsoidSurface`. Subtracting :math:`d(\mathbf{X})` puts the springs at rest in
    the reference configuration, wherever the mesh's facets lie relative to the surface.

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

    In the inverse problem (:class:`~pulse.unloading.PrestressProblem`) the mesh is the
    loaded configuration, and the springs are still at rest in the unloaded one, as a
    :class:`RobinBC`'s are.
    """

    stiffness: Variable
    marker: int
    surface: PericardialSurface
    damping: Variable | None = None
    unilateral: bool = False

    def __post_init__(self):
        if not isinstance(self.stiffness, Variable):
            logger.warning("Stiffness is not a Variable, defaulting to Pa / m")
            self.stiffness = Variable(self.stiffness, "Pa / m")
        if self.damping is not None and not isinstance(self.damping, Variable):
            logger.warning("Damping is not a Variable, defaulting to Pa s / m")
            self.damping = Variable(self.damping, "Pa s / m")
        logger.debug(
            f"Created PericardiumBC on marker {self.marker} with stiffness {self.stiffness}, "
            f"damping {self.damping}, unilateral={self.unilateral}, surface {self.surface}",
        )

    def traction(
        self,
        x: ufl.core.expr.Expr,
        X: ufl.core.expr.Expr,
        scale: float = 1.0,
        velocity: ufl.core.expr.Expr | None = None,
    ) -> ufl.core.expr.Expr:
        """The traction per reference area on the material point at ``X``, now at ``x``.

        ``scale`` converts the mesh unit to metres, as for a :class:`RobinBC`. ``velocity``
        is the point's velocity, for the dashpot; without it, there is none.
        """
        xv = ufl.variable(x)
        d = self.surface.signed_distance(xv)
        n = ufl.diff(d, xv)
        g = d - self.surface.signed_distance(X)
        k = self.stiffness.to_base_units() * scale
        if self.unilateral:
            t = k * ufl.max_value(g, 0.0) * n
        else:
            t = k * g * n
        if self.damping is not None and velocity is not None:
            c = self.damping.to_base_units() * scale
            rate = ufl.dot(n, velocity)
            if self.unilateral:
                rate = ufl.conditional(ufl.gt(g, 0.0), rate, 0.0)
            t += c * rate * n
        return t


def check_pericardium(bc: PericardiumBC, mesh: dolfinx.mesh.Mesh, ds: ufl.Measure) -> None:
    """Warn if ``bc.surface`` does not look like the boundary it holds: if the boundary is
    far from it, or its normal points elsewhere. A surface in the wrong unit, or around the
    wrong boundary, gives no error otherwise, since the springs are at rest wherever the
    boundary starts."""
    from mpi4py import MPI

    X = ufl.SpatialCoordinate(mesh)
    Xv = ufl.variable(X)
    d = bc.surface.signed_distance(Xv)
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
