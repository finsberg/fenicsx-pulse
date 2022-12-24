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
            typing.Sequence[dolfinx.fem.bcs.DirichletBCMetaClass],
        ]
    ] = ()
    robin: typing.Sequence[RobinBC] = ()
    body_force: typing.Sequence[
        float | dolfinx.fem.Constant | dolfinx.fem.Function
    ] = ()
