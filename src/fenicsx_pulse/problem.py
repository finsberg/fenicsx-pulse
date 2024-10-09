import typing
from dataclasses import dataclass, field
from enum import Enum

import basix
import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import numpy as np
import scifem
import ufl

from .boundary_conditions import BoundaryConditions
from .cardiac_model import CardiacModel
from .geometry import HeartGeometry
from .units import Variable, mesh_factor

T = typing.TypeVar("T", dolfinx.fem.Function, np.ndarray)


def interpolate(x0: T, x1: T, alpha: float):
    r"""Interpolate between :math:`x_0` and :math:`x_1`
    to find `math:`x_{1-\alpha}`

    Parameters
    ----------
    x0 : T
        First point
    x1 : T
        Second point
    alpha : float
        Amount of interpolate

    Returns
    -------
    T
        `math:`x_{1-\alpha}`
    """
    return alpha * x0 + (1 - alpha) * x1


class Geometry(typing.Protocol):
    """Protocol for geometry objects used in mechanics problems."""

    dx: ufl.Measure
    ds: ufl.Measure
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    markers: dict[str, tuple[int, int]]

    @property
    def facet_normal(self) -> ufl.FacetNormal: ...

    def surface_area(self, marker: str) -> float: ...


class Cavity(typing.NamedTuple):
    marker: str
    volume: dolfinx.fem.Constant


class BaseBC(str, Enum):
    """Base boundary condition"""

    fixed = "fixed"
    free = "free"


@dataclass
class StaticProblem:
    model: CardiacModel
    geometry: Geometry
    parameters: dict[str, typing.Any] = field(default_factory=dict)
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
        if isinstance(self.geometry, HeartGeometry):
            self._base_center_form = self.geometry.base_center_form(
                base=self.parameters["base_marker"],
                u=self.u,
            )
            self._base_area = self.geometry.surface_area(self.parameters["base_marker"])
            base_center = np.array(
                [dolfinx.fem.assemble_scalar(b) / self._base_area for b in self._base_center_form],
            )
            self.base_center = dolfinx.fem.Constant(
                self.geometry.mesh,
                base_center,
            )
        else:
            self.base_center = dolfinx.fem.Constant(
                self.geometry.mesh,
                np.zeros(self.geometry.mesh.geometry.dim),
            )

    def update_base(self):
        if isinstance(self.geometry, HeartGeometry):
            self.base_center.value[:] = np.array(
                [dolfinx.fem.assemble_scalar(b) / self._base_area for b in self._base_center_form],
            )

    @property
    def is_incompressible(self):
        return not self.model.compressibility.is_compressible()

    @staticmethod
    def default_parameters():
        return {
            "u_space": "P_2",
            "p_space": "P_1",
            "base_bc": BaseBC.free,
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
        self.cavity_pressures_test = []
        self.cavity_pressures_trial = []
        self.real_space = scifem.create_real_functionspace(self.geometry.mesh)

        if self.num_cavity_pressure_states > 0:
            for _ in range(self.num_cavity_pressure_states):
                cavity_pressure = dolfinx.fem.Function(self.real_space)
                cavity_pressure_test = ufl.TestFunction(self.real_space)
                cavity_pressure_trial = ufl.TrialFunction(self.real_space)

                self.cavity_pressures.append(cavity_pressure)
                self.cavity_pressures_test.append(cavity_pressure_test)
                self.cavity_pressures_trial.append(cavity_pressure_trial)

    def _init_rigid_body(self):
        if self.parameters["rigid_body_constraint"]:
            self.rigid_space = scifem.create_real_functionspace(
                self.geometry.mesh,
                value_shape=(6,),
            )
            self.r = dolfinx.fem.Function(self.rigid_space)
            self.dr = ufl.TrialFunction(self.rigid_space)
            self.q = ufl.TestFunction(self.rigid_space)

    def _create_residual_form(self, form: dolfinx.fem.Form) -> list[dolfinx.fem.Form]:
        return [
            ufl.derivative(form, f, f_test) for f, f_test in zip(self.states, self.test_functions)
        ]

    def _empty_form(self):
        return [ufl.as_ufl(0.0) for _ in range(self.num_states)]

    def _rigid_body_form(self, u: dolfinx.fem.Function) -> list[dolfinx.fem.Form]:
        if not self.parameters["rigid_body_constraint"]:
            return self._empty_form()
        X = ufl.SpatialCoordinate(self.geometry.mesh)

        RM = [
            ufl.as_vector((1, 0, 0)),
            ufl.as_vector((0, 1, 0)),
            ufl.as_vector((0, 0, 1)),
            ufl.cross(X, ufl.as_vector((1, 0, 0))),
            ufl.cross(X, ufl.as_vector((0, 1, 0))),
            ufl.cross(X, ufl.as_vector((0, 0, 1))),
        ]
        form = sum(ufl.inner(u, zi) * self.r[i] * ufl.dx for i, zi in enumerate(RM))
        forms = self._create_residual_form(form)
        return forms

    def _material_form(self, u: dolfinx.fem.Function, p: dolfinx.fem.Function):
        F = ufl.grad(u) + ufl.Identity(3)
        internal_energy = self.model.strain_energy(F, p=p) * self.geometry.dx

        return self._create_residual_form(internal_energy)

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
            value = ufl.inner(k * u, N)
            form += ufl.inner(value * self.u_test, N) * self.geometry.ds(robin.marker)

        forms = self._empty_form()
        forms[0] += form
        return forms

    def _neumann_form(self, u: dolfinx.fem.Function) -> list[dolfinx.fem.Form]:
        F = ufl.grad(u) + ufl.Identity(3)
        N = self.geometry.facet_normal
        ds = self.geometry.ds

        form = ufl.as_ufl(0.0)
        for neumann in self.bcs.neumann:
            t = neumann.traction.to_base_units()
            n = t * ufl.det(F) * ufl.inv(F).T * N
            form += ufl.inner(self.u_test, n) * ds(neumann.marker)

        forms = self._empty_form()
        forms[0] += form
        return forms

    def _body_force_form(self, u: dolfinx.fem.Function) -> list[dolfinx.fem.Form]:
        form = ufl.as_ufl(0.0)
        for body_force in self.bcs.body_force:
            form += -ufl.derivative(ufl.inner(body_force, u) * self.geometry.dx, u, self.u_test)

        forms = self._empty_form()
        forms[0] += form
        return forms

    def _cavity_pressure_form(
        self,
        u: dolfinx.fem.Function,
        cavity_pressures: list[dolfinx.fem.Function] | None = None,
    ):
        if self.num_cavity_pressure_states == 0:
            return self._empty_form()

        if not isinstance(self.geometry, HeartGeometry):
            raise RuntimeError("Cavity pressures are only supported for HeartGeometry")

        V_u = self.geometry.volume_form(u, b=self.base_center)
        form = ufl.as_ufl(0.0)

        assert cavity_pressures is not None
        assert len(self.cavities) == self.num_cavity_pressure_states
        for i, (marker, cavity_volume) in enumerate(self.cavities):
            area = self.geometry.surface_area(marker)
            pendo = cavity_pressures[i]
            marker_id = self.geometry.markers[marker][0]
            form += pendo * (cavity_volume / area - V_u) * self.geometry.ds(marker_id)

        return self._create_residual_form(form)

    @property
    def base_dirichlet(self):
        bcs = []

        # First add boundary conditions for the base
        if self.parameters["base_bc"] == BaseBC.fixed:
            base_facets = self.geometry.facet_tags.find(self.top)
            dofs_base = dolfinx.fem.locate_dofs_topological(self.u_space, 2, base_facets)
            u_bc_base = dolfinx.fem.Function(self.u_space)
            u_bc_base.x.array[:] = 0
            bcs.append(dolfinx.fem.dirichletbc(u_bc_base, dofs_base))

        # Now check if we have any for bcs

        for dirichlet_bc in self.bcs.dirichlet:
            if callable(dirichlet_bc):
                bcs += dirichlet_bc(self.u_space)

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

        R = self._empty_form()
        R_material = self._material_form(self.u, p=self.p)
        R_cavity = self._cavity_pressure_form(self.u, self.cavity_pressures)
        R_robin = self._robin_form(self.u)
        R_neumann = self._neumann_form(self.u)
        R_rigid = self._rigid_body_form(self.u)
        R_body_force = self._body_force_form(self.u)

        for i in range(self.num_states):
            R[i] += R_material[i]
            R[i] += R_cavity[i]
            R[i] += R_robin[i]
            R[i] += R_neumann[i]
            R[i] += R_rigid[i]
            R[i] += R_body_force[i]

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

    @property
    def test_functions(self):
        u = [self.u_test]
        if self.num_cavity_pressure_states > 0:
            u += self.cavity_pressures_test
        if self.parameters["rigid_body_constraint"]:
            u.append(self.q)
        if self.is_incompressible:
            u.append(self.p_test)
        return u

    @property
    def trial_functions(self):
        u = [self.du]
        if self.num_cavity_pressure_states > 0:
            u += self.cavity_pressures_trial
        if self.parameters["rigid_body_constraint"]:
            u.append(self.dr)
        if self.is_incompressible:
            u.append(self.dp)
        return u

    def K(self, R):
        K = []
        for i in range(self.num_states):
            K_row = [
                ufl.derivative(R[i], f, df) for f, df in zip(self.states, self.trial_functions)
            ]
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

    def update_fields(self):
        pass

    def solve(self) -> bool:
        """Solve the system"""
        ret = self._solver.solve()
        self.update_base()
        self.update_fields()

        return ret


class DynamicProblem(StaticProblem):
    def __post_init__(self):
        super().__post_init__()
        # Just make sure we have units on rho and dt
        for key, default_unit in [("rho", "kg/m^3"), ("dt", "s")]:
            if not isinstance(self.parameters[key], Variable):
                self.parameters[key] = Variable(self.parameters[key], default_unit)

    def _init_u_space(self):
        super()._init_u_space()
        self.u_old = dolfinx.fem.Function(self.u_space)
        self.v_old = dolfinx.fem.Function(self.u_space)
        self.a_old = dolfinx.fem.Function(self.u_space)

    def _init_p_space(self):
        super()._init_p_space()
        if self.is_incompressible:
            self.p_old = dolfinx.fem.Function(self.p_space)
        else:
            self.p_old = None

    def _init_cavity_pressure_spaces(self):
        super()._init_cavity_pressure_spaces()
        self.cavity_pressures_old = []
        for _ in range(self.num_cavity_pressure_states):
            cavity_pressure_old = dolfinx.fem.Function(self.real_space)
            self.cavity_pressures_old.append(cavity_pressure_old)

    def _material_form(self, u, v, p):
        F = ufl.grad(u) + ufl.Identity(3)
        F_dot = ufl.grad(v)
        l = F_dot * ufl.inv(F)  # Holzapfel eq: 2.139
        d = 0.5 * (l + l.T)  # Holzapfel 2.146
        E_dot = ufl.variable(F.T * d * F)  # Holzapfel 2.163
        # Viscous part of the material model
        forms = super()._material_form(u, p)

        P_v = F * ufl.diff(self.model.viscoelastic_strain_energy(E_dot), E_dot)
        forms[0] += ufl.inner(P_v, ufl.grad(self.u_test)) * self.geometry.dx
        return forms

    def _acceleration_form(self, a: dolfinx.fem.Function):
        rho = self.parameters["rho"].to_base_units()
        forms = self._empty_form()
        forms[0] += ufl.inner(rho * a, self.u_test) * self.geometry.dx
        return forms

    def _robin_form(self, u: dolfinx.fem.Function, v: dolfinx.fem.Function | None = None):
        forms = super()._robin_form(u)

        N = self.geometry.facet_normal

        for robin in self.bcs.robin:
            if not robin.damping:
                # Should be applied to the velocity
                continue
            assert v is not None
            k = robin.value.to_base_units() * mesh_factor(str(self.parameters["mesh_unit"]))
            value = ufl.inner(k * v, N)
            forms[0] += ufl.inner(value * self.u_test, N) * self.geometry.ds(robin.marker)
        return forms

    @property
    def R(self):
        # Order is always (u, cavity pressures, rigid body, p)

        alpha_m = self.parameters["alpha_m"]
        alpha_f = self.parameters["alpha_f"]

        a_new = self.a(u=self.u, u_old=self.u_old, v_old=self.v_old, a_old=self.a_old)
        v_new = self.v(a=a_new, v_old=self.v_old, a_old=self.a_old)

        u = interpolate(self.u_old, self.u, alpha_f)
        v = interpolate(self.v_old, v_new, alpha_f)
        a = interpolate(self.a_old, a_new, alpha_m)

        R = self._empty_form()
        R_material = self._material_form(u=u, v=v, p=self.p)
        R_cavity = self._cavity_pressure_form(u, self.cavity_pressures)
        R_robin = self._robin_form(u=u, v=v)
        R_neumann = self._neumann_form(u)
        R_rigid = self._rigid_body_form(u)
        R_acceleration = self._acceleration_form(a)

        for i in range(self.num_states):
            R[i] += R_material[i]
            R[i] += R_cavity[i]
            R[i] += R_robin[i]
            R[i] += R_neumann[i]
            R[i] += R_rigid[i]
            R[i] += R_acceleration[i]

        return R

    @staticmethod
    def default_parameters():
        parameters = StaticProblem.default_parameters()
        parameters.update(
            {
                "dt": Variable(1e-3, "s"),
                "rho": Variable(1e3, "kg/m^3"),
                "alpha_m": 0.2,
                "alpha_f": 0.4,
            },
        )
        return parameters

    def v(
        self,
        a: T,
        v_old: T,
        a_old: T,
    ) -> T:
        r"""
        Velocity computed using the generalized
        :math:`alpha`-method

        .. math::
            v_{i+1} = v_i + (1-\gamma) \Delta t a_i + \gamma \Delta t a_{i+1}

        Parameters
        ----------
        a : T
            Current acceleration
        v_old : T
            Previous velocity
        a_old: T
            Previous acceleration
        Returns
        -------
        T
            The current velocity
        """
        dt = self.parameters["dt"].to_base_units()
        return v_old + (1 - self._gamma) * dt * a_old + self._gamma * dt * a

    def a(
        self,
        u: T,
        u_old: T,
        v_old: T,
        a_old: T,
    ) -> T:
        r"""
        Acceleration computed using the generalized
        :math:`alpha`-method

        .. math::
            a_{i+1} = \frac{u_{i+1} - (u_i + \Delta t v_i +
            (0.5 - \beta) \Delta t^2 a_i)}{\beta \Delta t^2}

        Parameters
        ----------
        u : T
            Current displacement
        u_old : T
            Previous displacement
        v_old : T
            Previous velocity
        a_old: T
            Previous acceleration
        Returns
        -------
        T
            The current acceleration
        """
        dt = self.parameters["dt"].to_base_units()
        dt2 = dt**2
        beta = self._beta
        return (u - (u_old + dt * v_old + (0.5 - beta) * dt2 * a_old)) / (beta * dt2)

    def update_fields(self) -> None:
        """Update old values of displacement, velocity
        and acceleration
        """

        u = self.u.x.array.copy()
        u_old = self.u_old.x.array.copy()
        v_old = self.v_old.x.array.copy()
        a_old = self.a_old.x.array.copy()

        a = self.a(
            u=u,
            u_old=u_old,
            v_old=v_old,
            a_old=a_old,
        )

        v = self.v(a=a, v_old=v_old, a_old=a_old)

        self.a_old.x.array[:] = a
        self.v_old.x.array[:] = v
        self.u_old.x.array[:] = u

        # self.update_base_values()
        for i in range(self.num_cavity_pressure_states):
            self.cavity_pressures_old[i].x.array[:] = self.cavity_pressures[i].x.array.copy()

    @property
    def _gamma(self) -> float:
        """Parameter in the generalized alpha-method"""
        alpha_m = self.parameters["alpha_m"]
        alpha_f = self.parameters["alpha_f"]
        assert isinstance(alpha_m, float)
        assert isinstance(alpha_f, float)
        return 0.5 + alpha_f - alpha_m

    @property
    def _beta(self) -> float:
        """Parameter in the generalized alpha-method"""
        return (self._gamma + 0.5) ** 2 / 4.0
