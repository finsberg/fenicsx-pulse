import typing
from dataclasses import dataclass
from dataclasses import field

import dolfinx
import ufl

from . import kinematics
from .boundary_conditions import BoundaryConditions
from .cardiac_model import CardiacModel
from .geometry import Geometry


@dataclass(slots=True)
class MechanicsProblem:
    model: CardiacModel
    geometry: Geometry
    bcs: BoundaryConditions = field(default_factory=BoundaryConditions)
    _problem: dolfinx.fem.petsc.NonlinearProblem = field(init=False, repr=False)
    _solver: dolfinx.fem.petsc.NonlinearProblem = field(init=False, repr=False)
    state_space: dolfinx.fem.FunctionSpace = field(init=False, repr=False)
    state: dolfinx.fem.Function = field(init=False, repr=False)
    test_state: dolfinx.fem.Function = field(init=False, repr=False)
    _virtual_work: ufl.form.Form = field(init=False, repr=False)
    _dirichlet_bc: typing.Sequence[dolfinx.fem.bcs.DirichletBCMetaClass] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self):
        self._init_space()
        self._init_form()
        self._init_solver()

    def _init_space(self) -> None:
        P2 = ufl.VectorElement("Lagrange", self.geometry.mesh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", self.geometry.mesh.ufl_cell(), 1)

        self.state_space = dolfinx.fem.FunctionSpace(self.geometry.mesh, P2 * P1)
        self.state = dolfinx.fem.Function(self.state_space)
        self.test_state = ufl.TestFunction(self.state_space)

    def _init_form(self) -> None:
        u, p = ufl.split(self.state)
        v, _ = ufl.split(self.test_state)

        self.model.compressibility.register(p)

        F = kinematics.DeformationGradient(u)
        psi = self.model.strain_energy(F, p)
        self._virtual_work = ufl.derivative(
            psi * self.geometry.dx,
            coefficient=self.state,
            argument=self.test_state,
        )
        external_work = self._external_work(u, v)
        if external_work is not None:
            self._virtual_work += external_work

        self._set_dirichlet_bc()

    def _init_solver(self) -> None:
        self._problem = dolfinx.fem.petsc.NonlinearProblem(
            self._virtual_work,
            self.state,
            self._dirichlet_bc,
        )
        self._solver = dolfinx.nls.petsc.NewtonSolver(
            self.geometry.mesh.comm,
            self._problem,
        )
        # TODO: Make it possible for the user to set this
        self._solver.atol = 1e-8
        self._solver.rtol = 1e-8
        self._solver.convergence_criterion = "incremental"

    def _external_work(self, u, v):

        F = kinematics.DeformationGradient(u)

        N = self.geometry.facet_normal
        ds = self.geometry.ds
        dx = self.geometry.dx

        external_work = []

        for neumann in self.bcs.neumann:
            n = neumann.traction * ufl.cofac(F) * N
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
        return self._solver.solve(self.state)
