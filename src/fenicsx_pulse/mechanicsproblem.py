"""This module defines the mechanics problem.

The mechanics problem is a combination of a cardiac model, a geometry, and boundary conditions.


"""

import typing
from dataclasses import dataclass, field

import basix
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl

from . import kinematics
from .boundary_conditions import BoundaryConditions
from .cardiac_model import CardiacModel


class Geometry(typing.Protocol):
    """Protocol for geometry objects used in mechanics problems."""

    dx: ufl.Measure
    ds: ufl.Measure
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags

    @property
    def facet_normal(self) -> ufl.FacetNormal: ...


@dataclass(slots=True)
class BaseMechanicsProblem:
    """Base class for mechanics problems."""

    model: CardiacModel
    geometry: Geometry
    bcs: BoundaryConditions = field(default_factory=BoundaryConditions)
    parameters: dict[str, typing.Any] = field(default_factory=dict)
    _problem: dolfinx.fem.petsc.NonlinearProblem = field(init=False, repr=False)
    _solver: dolfinx.fem.petsc.NonlinearProblem = field(init=False, repr=False)
    state_space: dolfinx.fem.FunctionSpace = field(init=False, repr=False)
    state: dolfinx.fem.Function = field(init=False, repr=False)
    test_state: dolfinx.fem.Function = field(init=False, repr=False)
    virtual_work: ufl.form.Form = field(init=False, repr=False)
    _dirichlet_bc: typing.Sequence[dolfinx.fem.bcs.DirichletBC] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def dirichlet_bc(self) -> typing.Sequence[dolfinx.fem.bcs.DirichletBC]:
        return self._dirichlet_bc or []

    @dirichlet_bc.setter
    def dirichlet_bc(self, value: typing.Sequence[dolfinx.fem.bcs.DirichletBC]) -> None:
        self._dirichlet_bc = value
        self._set_dirichlet_bc()
        self._init_solver()

    def __post_init__(self):
        self._init_space()
        self._init_form()
        self._init_solver()

    def _init_solver(self) -> None:
        self._problem = dolfinx.fem.petsc.NonlinearProblem(
            self.virtual_work,
            self.state,
            self.dirichlet_bc,
        )
        self._solver = dolfinx.nls.petsc.NewtonSolver(
            self.geometry.mesh.comm,
            self._problem,
        )
        # TODO: Make it possible for the user to set this
        self._solver.atol = 1e-8
        self._solver.rtol = 1e-8
        self._solver.convergence_criterion = "incremental"
        self._solver.report = True
        self._solver.max_it = 20

    def _external_work(self, u, v):
        F = kinematics.DeformationGradient(u)

        N = self.geometry.facet_normal
        ds = self.geometry.ds
        dx = self.geometry.dx

        external_work = []

        for neumann in self.bcs.neumann:
            n = neumann.traction * ufl.det(F) * ufl.inv(F).T * N
            external_work.append(ufl.inner(v, n) * ds(neumann.marker))

        for robin in self.bcs.robin:
            external_work.append(ufl.inner(robin.value * u, v) * ds(robin.marker))

        for body_force in self.bcs.body_force:
            external_work.append(
                -ufl.derivative(ufl.inner(body_force, u) * dx, u, v),
            )

        if len(external_work) > 0:
            return sum(external_work)

        return None

    def _set_dirichlet_bc(self) -> None:
        for dirichlet_bc in self.bcs.dirichlet:
            if callable(dirichlet_bc):
                try:
                    self._dirichlet_bc = dirichlet_bc(self.state_space)
                except Exception as ex:
                    print(ex)
                    raise ex
            else:
                raise NotImplementedError

    def solve(self):
        ret = self._solver.solve(self.state)
        self.state.x.scatter_forward()
        return ret


@dataclass(slots=True)
class MechanicsProblemMixed(BaseMechanicsProblem):
    """Mechanics problem for mixed formulation.

    This class is used to solve mechanics problems using a mixed formulation,
    where the displacement is the first component of the state and the pressure
    is the second component.

    Default spaces for the displacement and pressure are P_2 and P_1, respectively.

    You can set the order of the displacement and pressure spaces using the parameters
    `u_order` and `p_order`, respectively.
    """

    def _init_space(self) -> None:
        u_order = self.parameters.get("u_order", 2)
        P2 = basix.ufl.element(
            family="Lagrange",
            cell=self.geometry.mesh.ufl_cell().cellname(),
            degree=u_order,
            shape=(self.geometry.mesh.ufl_cell().topological_dimension(),),
        )
        p_order = self.parameters.get("p_order", 1)
        P1 = basix.ufl.element(
            family="Lagrange",
            cell=self.geometry.mesh.ufl_cell().cellname(),
            degree=p_order,
        )
        element = basix.ufl.mixed_element([P2, P1])

        self.state_space = dolfinx.fem.functionspace(self.geometry.mesh, element)
        self.state = dolfinx.fem.Function(self.state_space)
        self.test_state = ufl.TestFunction(self.state_space)

    def _init_form(self) -> None:
        u, p = ufl.split(self.state)
        v, _ = ufl.split(self.test_state)

        self.model.compressibility.register(p)

        F = kinematics.DeformationGradient(u)
        psi = self.model.strain_energy(F, p)
        self.virtual_work = ufl.derivative(
            psi * self.geometry.dx,
            coefficient=self.state,
            argument=self.test_state,
        )
        external_work = self._external_work(u, v)
        if external_work is not None:
            self.virtual_work += external_work

        self._set_dirichlet_bc()


@dataclass(slots=True)
class MechanicsProblem(BaseMechanicsProblem):
    """Mechanics problem for displacement-based formulation.

    This class is used to solve mechanics problems using a displacement-based formulation
    which is typically used for compressible or nearly-incompressible materials.

    Default space for the displacement is P_2.

    You can set the order of the displacement space using the parameter `u_order`.
    """

    def _init_space(self) -> None:
        u_order = self.parameters.get("u_order", 2)
        element = basix.ufl.element(
            family="Lagrange",
            cell=self.geometry.mesh.ufl_cell().cellname(),
            degree=u_order,
            shape=(self.geometry.mesh.ufl_cell().topological_dimension(),),
        )

        self.state_space = dolfinx.fem.functionspace(self.geometry.mesh, element)
        self.state = dolfinx.fem.Function(self.state_space)
        self.test_state = ufl.TestFunction(self.state_space)

    def _init_form(self) -> None:
        F = kinematics.DeformationGradient(self.state)
        psi = self.model.strain_energy(F)
        self.virtual_work = ufl.derivative(
            psi * self.geometry.dx,
            coefficient=self.state,
            argument=self.test_state,
        )
        external_work = self._external_work(self.state, self.test_state)
        if external_work is not None:
            self.virtual_work += external_work

        self._set_dirichlet_bc()
