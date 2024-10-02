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

import dolfinx

from .units import Variable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NeumannBC:
    traction: Variable
    marker: int

    def __post_init__(self):
        if not isinstance(self.traction, Variable):
            unit = "kPa"
            logger.warning("Traction is not a Variable, defaulting to kPa")
            self.traction = Variable(self.traction, unit)


@dataclass(slots=True)
class RobinBC:
    value: Variable
    marker: int
    damping: bool = False

    def __post_init__(self):
        if not isinstance(self.value, Variable):
            unit = "Pa / m" if self.damping else "Pa s / m"
            logger.warning(f"Value is not a Variable, defaulting to {unit}")
            self.value = Variable(self.value, unit)


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
