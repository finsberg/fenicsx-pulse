"""This module defines boundary conditions.

Boundary conditions are used to specify the behavior of the solution on the boundary of the domain.
The boundary conditions can be Dirichlet, Neumann, or Robin boundary conditions.

Dirichlet boundary conditions are used to specify the solution on the boundary of the domain.
Neumann boundary conditions are used to specify the traction on the boundary of the domain.
Robin boundary conditions are used to specify a Robin type boundary condition
on the boundary of the domain.

The boundary conditions are collected in a `BoundaryConditions` object.
"""

import typing

import dolfinx


class NeumannBC(typing.NamedTuple):
    traction: float | dolfinx.fem.Constant | dolfinx.fem.Function
    marker: int


class RobinBC(typing.NamedTuple):
    value: float | dolfinx.fem.Constant | dolfinx.fem.Function
    marker: int


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
