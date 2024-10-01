import typing
from dataclasses import dataclass, field
from enum import Enum

import basix
import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import scifem
import ufl

from .boundary_conditions import BoundaryConditions
from .cardiac_model import CardiacModel
from .units import Variable, mesh_factor


class Geometry(typing.Protocol):
    """Protocol for geometry objects used in mechanics problems."""

    dx: ufl.Measure
    ds: ufl.Measure
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    markers: dict[str, tuple[int, int]]

    @property
    def facet_normal(self) -> ufl.FacetNormal: ...

    def volume_form(self, u: dolfinx.fem.Function) -> ufl.Coefficient: ...

    def surface_area(self, marker: str) -> float: ...

    def base_center(
        self,
        base: str = "BASE",
        u: dolfinx.fem.Function | None = None,
    ) -> tuple[float, float, float]: ...


class Cavity(typing.NamedTuple):
    marker: str
    volume: dolfinx.fem.Constant


class ControlMode(str, Enum):
    """Control mode for the problem"""

    pressure = "pressure"
    volume = "volume"


class BaseBC(str, Enum):
    """Base boundary condition"""

    fixed = "fixed"
    free = "free"
    fix_x = "fix_x"


@dataclass
class StaticProblem:
    model: CardiacModel
    geometry: Geometry
    parameters: dict[str, Variable] = field(default_factory=dict)
    bcs: BoundaryConditions = field(default_factory=BoundaryConditions)
    cavities: list[Cavity] = field(default_factory=list)

    def __post_init__(self):
        parameters = type(self).default_parameters()
        parameters.update(self.parameters)
        self.parameters = parameters
        self._init_spaces()
        self._init_forms()

    def _init_spaces(self):
        """Initialize function spaces"""
        self._init_u_space()
        self._init_p_space()
        self._init_base()

        self._init_cavity_pressure_spaces()
        if self.parameters["rigid_body_constraint"]:
            self._init_rigid_body()

    def _init_p_space(self):
        if self.is_incompressible:
            # Need lagrange multiplier for incompressible model
            p_family, p_degree = self.parameters["p_space"].split("_")
            p_element = basix.ufl.element(
                family=p_family,
                cell=self.geometry.mesh.ufl_cell().cellname(),
                degree=int(p_degree),
            )
            self.p_space = dolfinx.fem.functionspace(self.geometry.mesh, p_element)
            self.p = dolfinx.fem.Function(self.p_space)
            self.p_test = ufl.TestFunction(self.p_space)
            self.dp = ufl.TrialFunction(self.p_space)
        else:
            self.p_space = None
            self.p = None
            self.p_test = None
            self.dp = None

    def _init_u_space(self):
        u_family, u_degree = self.parameters["u_space"].split("_")

        u_element = basix.ufl.element(
            family=u_family,
            cell=self.geometry.mesh.ufl_cell().cellname(),
            degree=int(u_degree),
            shape=(self.geometry.mesh.ufl_cell().topological_dimension(),),
        )
        self.u_space = dolfinx.fem.functionspace(self.geometry.mesh, u_element)
        self.u = dolfinx.fem.Function(self.u_space)
        self.u_test = ufl.TestFunction(self.u_space)
        self.du = ufl.TrialFunction(self.u_space)

    def _init_base(self):
        self.base_center = dolfinx.fem.Constant(
            self.geometry.mesh,
            self.geometry.base_center(base=self.parameters["base_marker"]),
        )
        self.base_factor = dolfinx.fem.Constant(
            self.geometry.mesh,
            dolfinx.default_scalar_type(0.0),
        )

    @property
    def is_incompressible(self):
        return not self.model.compressibility.is_compressible()

    @staticmethod
    def default_parameters():
        return {
            "u_space": "P_2",
            "p_space": "P_1",
            "base_bc": BaseBC.fixed,
            "rigid_body_constraint": False,
            "mesh_unit": "m",
            "base_marker": "BASE",
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        }

    @property
    def top(self):
        return self.geometry.markers[self.parameters["base_marker"]][0]

    @property
    def num_cavity_pressure_states(self):
        return len(self.cavities)

    def _init_cavity_pressure_spaces(self):
        self.cavity_pressures = []
        self.cavity_pressures_old = []
        self.cavity_pressures_test = []
        self.cavity_pressures_trial = []
        self.real_space = scifem.create_real_functionspace(self.geometry.mesh)

        if self.num_cavity_pressure_states > 0:
            for _ in range(self.num_cavity_pressure_states):
                cavity_pressure = dolfinx.fem.Function(self.real_space)
                cavity_pressure_old = dolfinx.fem.Function(self.real_space)
                cavity_pressure_test = ufl.TestFunction(self.real_space)
                cavity_pressure_trial = ufl.TrialFunction(self.real_space)

                self.cavity_pressures.append(cavity_pressure)
                self.cavity_pressures_old.append(cavity_pressure_old)
                self.cavity_pressures_test.append(cavity_pressure_test)
                self.cavity_pressures_trial.append(cavity_pressure_trial)

    def _init_rigid_body(self):
        self.rigid_space = scifem.create_real_functionspace(self.geometry.mesh, value_shape=(6,))
        self.r = dolfinx.fem.Function(self.rigid_space)
        self.dr = ufl.TrialFunction(self.rigid_space)
        self.q = ufl.TestFunction(self.rigid_space)

    def _rigid_body_form(self, u: dolfinx.fem.Function):
        X = ufl.SpatialCoordinate(self.geometry.mesh)

        RM = [
            ufl.as_vector((1, 0, 0)),
            ufl.as_vector((0, 1, 0)),
            ufl.as_vector((0, 0, 1)),
            ufl.cross(X, ufl.as_vector((1, 0, 0))),
            ufl.cross(X, ufl.as_vector((0, 1, 0))),
            ufl.cross(X, ufl.as_vector((0, 0, 1))),
        ]
        return sum(ufl.inner(u, zi) * self.r[i] * ufl.dx for i, zi in enumerate(RM))

    def _material_form(self, u: dolfinx.fem.Function, u_test: ufl.TestFunction):
        F = ufl.grad(u) + ufl.Identity(3)
        internal_energy = self.model.strain_energy(F, p=self.p) * self.geometry.dx
        if self.is_incompressible:
            return [
                ufl.derivative(internal_energy, u, u_test),
                ufl.derivative(internal_energy, self.p, self.p_test),
            ]
        return [ufl.derivative(internal_energy, u, u_test)]

    def _robin_form(self, u: dolfinx.fem.Function, u_test: ufl.TestFunction):
        form = ufl.as_ufl(0.0)

        for robin in self.bcs.robin:
            k = robin.value.to_base_units() * mesh_factor(str(self.parameters["mesh_unit"]))
            form += ufl.inner(k * u, u_test) * self.geometry.ds(robin.marker)
        return form

    def _pressure_form(self, u, u_test):
        V_u = self.geometry.volume_form(u, b=self.base_center)
        form = ufl.as_ufl(0.0)
        for neumann in self.bcs.neumann:
            t = neumann.traction.to_base_units()
            form += -t * V_u * self.geometry.ds(neumann.marker)

        if self.num_cavity_pressure_states > 0:
            for i, (marker, cavity_volume) in enumerate(self.cavities):
                area = self.geometry.surface_area(marker)
                pendo = self.cavity_pressures[i]
                marker_id = self.geometry.markers[marker][0]
                form += pendo * (cavity_volume / area - V_u) * self.geometry.ds(marker_id)

        forms = []
        for s, s_test in [(self.u, self.u_test)] + list(
            zip(self.cavity_pressures, self.cavity_pressures_test),
        ):
            forms.append(ufl.derivative(form, s, s_test))

        return forms

    @property
    def base_dirichlet(self):
        bcs = []

        if self.parameters["base_bc"] == BaseBC.fixed:
            base_facets = self.geometry.facet_tags.find(self.top)
            dofs_base = dolfinx.fem.locate_dofs_topological(self.u_space, 2, base_facets)
            u_bc_base = dolfinx.fem.Function(self.u_space)
            u_bc_base.x.array[:] = 0
            bcs.append(dolfinx.fem.dirichletbc(u_bc_base, dofs_base))

        return bcs

    @property
    def num_states(self):
        return (
            1
            + self.num_cavity_pressure_states
            + int(self.parameters["rigid_body_constraint"])
            + int(self.is_incompressible)
        )

    @property
    def R(self):
        # Order is always (u, cavity pressures, rigid body, p)
        w = self.u_test
        u = self.u

        R = self._pressure_form(u, w)
        R_mat = self._material_form(u, w)
        R[0] += R_mat[0]
        R[0] += self._robin_form(u, w)

        if self.parameters["rigid_body_constraint"]:
            rigid_body = self._rigid_body_form(u)
            R[0] += ufl.derivative(rigid_body, self.u, self.u_test)
            for i in range(self.num_cavity_pressure_states):
                R[i + 1] += ufl.derivative(
                    rigid_body,
                    self.cavity_pressures[i],
                    self.cavity_pressures_test[i],
                )
            R += [ufl.derivative(rigid_body, self.r, self.q)]

        if self.is_incompressible:
            R += [R_mat[1]]

        return R

    @property
    def states(self):
        u = [self.u]
        if self.num_cavity_pressure_states > 0:
            u += self.cavity_pressures
        if self.parameters["rigid_body_constraint"]:
            u.append(self.r)
        if self.is_incompressible:
            u.append(self.p)
        return u

    def K(self, R):
        # Functions and trial functions
        functions = [(self.u, self.du)]
        if self.num_cavity_pressure_states > 0:
            functions += list(zip(self.cavity_pressures, self.cavity_pressures_trial))
        if self.parameters["rigid_body_constraint"]:
            functions.append((self.r, self.dr))
        if self.is_incompressible:
            functions.append((self.p, self.dp))

        K = []
        for i in range(self.num_states):
            K_row = [ufl.derivative(R[i], f, df) for f, df in functions]
            K.append(K_row)
        return K

    def _init_forms(self) -> None:
        """Initialize ufl forms"""

        # Markers
        if self.geometry.markers is None:
            raise RuntimeError("Missing markers in geometry")

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

    def solve(self) -> bool:
        """Solve the system"""
        ret = self._solver.solve()
        self.base_center.value[:] = self.geometry.base_center(u=self.u)

        return ret
