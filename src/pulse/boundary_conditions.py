"""This module defines boundary conditions.

Boundary conditions are used to specify the behavior of the solution on the boundary of the domain.
The boundary conditions can be Dirichlet, Neumann, or Robin boundary conditions.

Dirichlet boundary conditions are used to specify the solution on the boundary of the domain.
Neumann boundary conditions are used to specify the traction on the boundary of the domain.
Robin boundary conditions are used to specify a Robin type boundary condition
on the boundary of the domain.

The boundary conditions are collected in a `BoundaryConditions` object.
"""

import logging
import typing
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
