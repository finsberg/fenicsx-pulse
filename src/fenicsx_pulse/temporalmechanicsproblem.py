r"""
Implementation of the mechanics problem

For time integration we employ the generalized :math:`\alpha`-method [1]_.

    .. [1] Silvano Erlicher, Luca Bonaventura, Oreste Bursi.
        The analysis of the Generalized-alpha method for
        non-linear dynamic problems. Computational Mechanics,
        Springer Verlag, 2002, 28, pp.83-104, doi:10.1007/s00466-001-0273-z
"""

import typing
from dataclasses import dataclass, field

import basix
import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import numpy as np
import ufl

from .boundary_conditions import BoundaryConditions
from .cardiac_model import CardiacModel


class Geometry(typing.Protocol):
    """Protocol for geometry objects used in mechanics problems."""

    dx: ufl.Measure
    ds: ufl.Measure
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    markers: dict[str, tuple[int, int]]

    @property
    def facet_normal(self) -> ufl.FacetNormal: ...


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


@dataclass
class Problem:
    model: CardiacModel
    geometry: Geometry
    parameters: dict[str, typing.Any] = field(default_factory=dict)
    bcs: BoundaryConditions = field(default_factory=BoundaryConditions)

    def __post_init__(self):
        parameters = type(self).default_parameters()
        parameters.update(self.parameters)
        self.parameters = parameters
        self._init_spaces()
        self._init_forms()

    def _init_spaces(self):
        """Initialize function spaces"""
        mesh = self.geometry.mesh

        family, degree = self.parameters["function_space"].split("_")

        element = basix.ufl.element(
            family=family,
            cell=self.geometry.mesh.ufl_cell().cellname(),
            degree=int(degree),
            shape=(self.geometry.mesh.ufl_cell().topological_dimension(),),
        )
        self.u_space = dolfinx.fem.functionspace(mesh, element)
        self.u = dolfinx.fem.Function(self.u_space)
        self.u_test = ufl.TestFunction(self.u_space)
        self.du = ufl.TrialFunction(self.u_space)

        self.u_old = dolfinx.fem.Function(self.u_space)
        self.v_old = dolfinx.fem.Function(self.u_space)
        self.a_old = dolfinx.fem.Function(self.u_space)

    def _acceleration_form(self, a: dolfinx.fem.Function, w: ufl.TestFunction):
        return ufl.inner(self.parameters["rho"] * a, w) * self.geometry.dx

    def _first_piola(self, F: ufl.Coefficient, v: dolfinx.fem.Function):
        F_dot = ufl.grad(v)
        l = F_dot * ufl.inv(F)  # Holzapfel eq: 2.139
        d = 0.5 * (l + l.T)  # Holzapfel 2.146
        E_dot = ufl.variable(F.T * d * F)  # Holzapfel 2.163

        return ufl.diff(self.model.material.strain_energy(F), F) + F * ufl.diff(
            self.model.viscoelastic_strain_energy(E_dot),
            E_dot,
        )

    def _form(self, u: dolfinx.fem.Function, v: dolfinx.fem.Function, w: ufl.TestFunction):
        F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
        P = self._first_piola(F, v)
        epi = ufl.dot(self.parameters["alpha_epi"] * u, self.N) + ufl.dot(
            self.parameters["beta_epi"] * v,
            self.N,
        )
        top = self.parameters["alpha_top"] * u + self.parameters["beta_top"] * v

        return (
            ufl.inner(P, ufl.grad(w)) * self.geometry.dx
            + self._pressure_term(F, w)
            + ufl.inner(epi * w, self.N) * self.geometry.ds(self.epi)
            + ufl.inner(top, w) * self.geometry.ds(self.top)
        )

    def _pressure_term(self, F, w):
        N = self.geometry.facet_normal
        ds = self.geometry.ds

        external_work = []

        for neumann in self.bcs.neumann:
            n = neumann.traction * ufl.det(F) * ufl.inv(F).T * N
            external_work.append(ufl.inner(w, n) * ds(neumann.marker))
        return sum(external_work)

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
        dt = self.parameters["dt"]
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
        dt = self.parameters["dt"]
        dt2 = dt**2
        beta = self._beta
        return (u - (u_old + dt * v_old + (0.5 - beta) * dt2 * a_old)) / (beta * dt2)

    def _update_fields(self) -> None:
        """Update old values of displacement, velocity
        and acceleration
        """
        a = self.a(
            u=self.u.x.array,
            u_old=self.u_old.x.array,
            v_old=self.v_old.x.array,
            a_old=self.a_old.x.array,
        )
        v = self.v(a=a, v_old=self.v_old.x.array, a_old=self.a_old.x.array)

        self.a_old.x.array[:] = a
        self.v_old.x.array[:] = v
        self.u_old.x.array[:] = self.u.x.array

    @property
    def epi(self):
        """Marker for the epicardium"""
        return self.geometry.markers["EPI"][0]

    @property
    def top(self):
        """Marker for the top or base"""
        return self.geometry.markers["BASE"][0]

    @property
    def N(self):
        """Facet Normal"""
        return ufl.FacetNormal(self.geometry.mesh)

    def _init_forms(self) -> None:
        """Initialize ufl forms"""
        w = self.u_test

        # Markers
        if self.geometry.markers is None:
            raise RuntimeError("Missing markers in geometry")

        alpha_m = self.parameters["alpha_m"]
        alpha_f = self.parameters["alpha_f"]

        a_new = self.a(u=self.u, u_old=self.u_old, v_old=self.v_old, a_old=self.a_old)
        v_new = self.v(a=a_new, v_old=self.v_old, a_old=self.a_old)

        virtual_work = self._acceleration_form(
            interpolate(self.a_old, a_new, alpha_m),
            w,
        ) + self._form(
            interpolate(self.u_old, self.u, alpha_f),
            interpolate(self.v_old, v_new, alpha_f),
            w,
        )

        self._problem = dolfinx.fem.petsc.NonlinearProblem(
            virtual_work,
            self.u,
            [],
        )
        self._solver = dolfinx.nls.petsc.NewtonSolver(
            self.geometry.mesh.comm,
            self._problem,
        )

    def von_Mises(self) -> ufl.Coefficient:
        r"""Compute the von Mises stress tensor :math`\sigma_v`, with

        .. math::

            \sigma_v^2 = \frac{1}{2} \left(
                (\mathrm{T}_{11} - \mathrm{T}_{22})^2 +
                (\mathrm{T}_{22} - \mathrm{T}_{33})^2 +
                (\mathrm{T}_{33} - \mathrm{T}_{11})^2 +
            \right) - 3 \left(
                \mathrm{T}_{12} + \mathrm{T}_{23} + \mathrm{T}_{31}
            \right)

        Returns
        -------
        ufl.Coefficient
            The von Mises stress tensor
        """
        u = self.u
        a = self.a(u=self.u, u_old=self.u_old, v_old=self.v_old, a_old=self.a_old)
        v = self.v(a=a, v_old=self.v_old, a_old=self.a_old)

        F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
        J = ufl.det(F)
        P = self._first_piola(F, v)

        # Cauchy
        T = pow(J, -1.0) * P * F.T
        von_Mises_squared = 0.5 * (
            (T[0, 0] - T[1, 1]) ** 2 + (T[1, 1] - T[2, 2]) ** 2 + (T[2, 2] - T[0, 0]) ** 2
        ) + 3 * (T[0, 1] + T[1, 2] + T[2, 0])

        return ufl.sqrt(abs(von_Mises_squared))

    @property
    def _gamma(self) -> float:
        """Parameter in the generalized alpha-method"""
        return 0.5 + self.parameters["alpha_f"] - self.parameters["alpha_m"]

    @property
    def _beta(self) -> float:
        """Parameter in the generalized alpha-method"""
        return (self._gamma + 0.5) ** 2 / 4.0

    def solve(self) -> bool:
        """Solve the system"""
        ret = self._solver.solve(self.u)
        self.u.x.scatter_forward()

        self._update_fields()
        return ret

    @staticmethod
    def default_parameters() -> typing.Dict[str, float | str]:
        return dict(
            alpha_top=1.0,
            alpha_epi=1e3,
            beta_top=5e-2,
            beta_epi=5e-2,
            p=0.0,
            rho=1e-3,
            dt=1e-3,
            alpha_m=0.2,
            alpha_f=0.4,
            function_space="P_2",
        )
