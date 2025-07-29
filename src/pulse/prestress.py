import typing
from dataclasses import dataclass, field

import basix
import dolfinx
import scifem
import ufl

from .boundary_conditions import BoundaryConditions
from .cardiac_model import CardiacModel
from .geometry import HeartGeometry
from .units import mesh_factor


@dataclass
class PrestressProblem:
    geometry: HeartGeometry
    model: CardiacModel
    bcs: BoundaryConditions
    parameters: dict[str, typing.Any] = field(default_factory=dict)

    def __post_init__(self):
        parameters = type(self).default_parameters()
        parameters.update(self.parameters)
        self.parameters = parameters
        self._init_spaces()
        self._init_forms()

    @property
    def states(self):
        return [self.u]

    @property
    def trial_functions(self):
        return [self.du]

    @property
    def test_functions(self):
        return [self.u_test]

    def K(self, R):
        K = []
        for i in range(self.num_states):
            K_row = [
                ufl.derivative(R[i], f, df) for f, df in zip(self.states, self.trial_functions)
            ]
            K.append(K_row)
        return K

    def _empty_form(self):
        return [ufl.as_ufl(0.0) for _ in range(self.num_states)]

    @property
    def num_states(self) -> int:
        return 1

    def _init_spaces(self):
        u_family, u_degree = self.parameters["u_space"].split("_")

        u_element = basix.ufl.element(
            family=u_family,
            cell=self.geometry.mesh.ufl_cell().cellname(),
            degree=int(u_degree),
            shape=(self.geometry.mesh.ufl_cell().topological_dimension(),),
        )
        self.u_space = dolfinx.fem.functionspace(self.geometry.mesh, u_element)
        self.u = dolfinx.fem.Function(self.u_space, name="u")
        self.u_test = ufl.TestFunction(self.u_space)
        self.du = ufl.TrialFunction(self.u_space)

    def _material_form(self, u: dolfinx.fem.Function):
        dim = self.u.ufl_shape[0]
        f = ufl.Identity(dim) + ufl.grad(self.u)  # Inverse tensor for inverse problem
        j = ufl.det(f)

        F = ufl.variable(ufl.inv(f))  # Compute original one to diff

        P = self.model.P(F)

        forms = self._empty_form()
        forms[0] += ufl.inner(j * P, ufl.grad(self.u_test) * ufl.inv(f)) * self.geometry.dx

        # TODO: Add incompressible version
        # if self.is_incompressible:
        #     forms[-1] += (J - 1.0) * self.p_test * self.geometry.dx

        return forms

    @property
    def R(self):
        R = self._empty_form()
        R_material = self._material_form(self.u)
        R_robin = self._robin_form(self.u)
        R_neumann = self._neumann_form(self.u)

        for i in range(self.num_states):
            R[i] += R_material[i]
            R[i] += R_robin[i]
            R[i] += R_neumann[i]

        return R

    def _init_forms(self):
        R = self.R
        u = self.states
        K = self.K(R)

        assert len(R) == self.num_states
        assert len(K) == self.num_states
        assert all(len(Ki) == self.num_states for Ki in K)

        bcs = self.base_dirichlet

        self._solver = scifem.NewtonSolver(
            R,
            K,
            u,
            bcs=bcs,
            max_iterations=25,
            petsc_options=self.parameters["petsc_options"],
        )

    def _robin_form(
        self,
        u: dolfinx.fem.Function,
        v: dolfinx.fem.Function | None = None,
    ) -> list[dolfinx.fem.Form]:
        form = ufl.as_ufl(0.0)
        N = self.geometry.facet_normal

        for robin in self.bcs.robin:
            if robin.damping:
                # Should be applied to the velocity
                continue
            k = robin.value.to_base_units() * mesh_factor(str(self.parameters["mesh_unit"]))

            if robin.perpendicular:
                nn = ufl.Identity(u.ufl_shape[0]) - ufl.outer(N, N)
            else:
                nn = ufl.outer(N, N)

            value = k * nn * u
            form += -ufl.dot(value, self.u_test) * self.geometry.ds(robin.marker)

        forms = self._empty_form()
        forms[0] += form
        return forms

    def _neumann_form(self, u: dolfinx.fem.Function) -> list[dolfinx.fem.Form]:
        N = self.geometry.facet_normal
        ds = self.geometry.ds

        form = ufl.as_ufl(0.0)
        for neumann in self.bcs.neumann:
            t = neumann.traction.to_base_units()
            form += t * ufl.inner(self.u_test, N) * ds(neumann.marker)

        forms = self._empty_form()
        forms[0] += form
        return forms

    @property
    def base_dirichlet(self):
        bcs = []
        # Now check if we have any for bcs

        for dirichlet_bc in self.bcs.dirichlet:
            if callable(dirichlet_bc):
                bcs += dirichlet_bc(self.u_space)

        return bcs

    @staticmethod
    def default_parameters():
        return {
            "u_space": "P_2",
            "p_space": "P_1",
            "mesh_unit": "m",
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        }

    def solve(self):
        self._solver.solve()
