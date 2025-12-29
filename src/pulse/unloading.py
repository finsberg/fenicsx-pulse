import logging
import typing
from dataclasses import dataclass, field

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import scifem
import ufl

from .boundary_conditions import BoundaryConditions
from .cardiac_model import CardiacModel
from .geometry import HeartGeometry
from .problem import StaticProblem
from .units import Variable, assign, mesh_factor
from .utils import map_vector_field

logger = logging.getLogger(__name__)


class TargetPressure(typing.NamedTuple):
    traction: dolfinx.fem.Constant | Variable
    target: float
    name: str = ""


@dataclass
class FixedPointUnloader:
    r"""
    Class for finding the unloaded reference configuration using a fixed-point iteration
    (Backward Displacement Method).

    Given a target (loaded) geometry :math:`\Omega_t` and applied loads (e.g., pressure),
    this algorithm iteratively searches for a reference geometry :math:`\Omega_0` such that
    solving the forward mechanics problem on :math:`\Omega_0` yields :math:`\Omega_t`.

    The update rule for the reference coordinates :math:`\mathbf{X}` at iteration :math:`k` is:

    .. math::
        \mathbf{X}_{k+1} = \mathbf{x}_{target} - \mathbf{u}(\mathbf{X}_k)

    where :math:`\mathbf{x}_{target}` are the coordinates of the target geometry, and
    :math:`\mathbf{u}(\mathbf{X}_k)` is the displacement computed by solving the forward
    problem on the geometry defined by :math:`\mathbf{X}_k`.

    Parameters
    ----------
    geometry : HeartGeometry
        The target (loaded) geometry.
        Note: The mesh in this object will be modified in-place to become the reference geometry.
    model : CardiacModel
        The material model.
    bcs : BoundaryConditions
        The boundary conditions applied to the loaded state.
    parameters : dict
        Solver parameters.
        - max_iter (int): Maximum number of iterations (default: 10).
        - tol (float): Convergence tolerance for the geometry error (default: 1e-4).
        - Any other parameters accepted by `StaticProblem`.
    """

    geometry: HeartGeometry
    model: CardiacModel
    bcs: BoundaryConditions
    problem_parameters: dict[str, typing.Any] = field(default_factory=dict)
    unload_parameters: dict[str, typing.Any] = field(default_factory=dict)
    targets: typing.Sequence[TargetPressure] = field(default_factory=list)

    def __post_init__(self):
        # Set default parameters for the unloader
        defaults = self.default_parameters()
        defaults.update(self.unload_parameters)
        self.unload_parameters = defaults

        # Store the original target coordinates (owned + ghosts)
        # This serves as \mathbf{x}_{target}
        self._target_coords = self.geometry.mesh.geometry.x.copy()
        self._extract_model_fields()

    def _extract_model_fields(self) -> None:
        """
        Extract internal vector fields from the material and active models
        for deformation during unloading.
        """
        self._model_fields: dict[str, dict[str, dolfinx.fem.Function]] = {}
        for model_name in ("material", "active"):
            model = getattr(self.model, model_name)
            self._model_fields[model_name] = {}
            for fieldname in ("f0", "n0", "s0"):
                if hasattr(model, fieldname):
                    field = getattr(model, fieldname)
                    self._model_fields[model_name][fieldname] = field

    def _update_model_fields(self, u: dolfinx.fem.Function) -> None:
        """
        Update internal vector fields in the material and active models
        by deforming them according to the displacement field u.
        """
        for model_name, fields in self._model_fields.items():
            model = getattr(self.model, model_name)
            for fieldname, f in fields.items():
                if not isinstance(f, dolfinx.fem.Function):
                    msg = (
                        f"Cannot deform field '{fieldname}' of model '{model_name}': "
                        f"it is a {type(f).__name__},  not a Function."
                    )
                    logger.warning(msg)
                    continue
                logger.debug(f"Deforming field '{fieldname}' of model '{model_name}'.")
                deformed_field = map_vector_field(
                    f=f,
                    u=u,
                    normalize=True,
                    name=f"{fieldname}_deformed",
                )
                setattr(model, fieldname, deformed_field)

    def default_parameters(self) -> dict[str, typing.Any]:
        """
        Returns default solver parameters for the FixedPointUnloader.
        """
        return {
            "max_iter": 10,
            "tol": 1e-4,
            "ramp_steps": 20,
            "update_model_fields": True,
        }

    def unload(self) -> dolfinx.fem.Function:
        """
        Execute the fixed-point iteration to find the unloaded geometry.
        The `geometry.mesh` is updated in-place.
        """
        logger.info("Starting FixedPointUnloader...")

        max_iter = self.unload_parameters["max_iter"]
        tol = self.unload_parameters["tol"]
        update_model_fields = self.unload_parameters["update_model_fields"]
        comm = self.geometry.mesh.comm

        for i in range(max_iter):
            logger.info(f"Unloading Iteration {i + 1}/{max_iter}")

            # 1. Solve the Forward Problem on the current reference geometry (X_k)
            # We create a new StaticProblem instance to ensure forms are initialized
            # with the current mesh coordinates.
            problem = StaticProblem(
                model=self.model,
                geometry=self.geometry,
                bcs=self.bcs,
                parameters=self.problem_parameters,
            )

            for ramp in np.linspace(0.0, 1.0, self.unload_parameters["ramp_steps"]):
                for traction, target_pressure, name in self.targets:
                    value = ramp * target_pressure
                    assign(traction, value)
                    if name != "":
                        msg = f"Ramping {name} traction to {value:.4f}"
                    else:
                        msg = f"Ramping traction to {value:.4f}"
                    logger.debug(msg)
                problem.solve()
            u = problem.u

            # 2. Evaluate displacement u at the mesh nodes
            # u lives on a function space (e.g. P2), mesh nodes are coordinate element (e.g. P1)
            # This returns the values of u at the current nodal positions X_k
            u_at_nodes = scifem.evaluate_function(u, self.geometry.mesh.geometry.x, broadcast=True)

            # 3. Compute Error (Geometric Discrepancy)
            # The deformed position is X_k + u(X_k)
            # We want this to match X_target
            # Error vector R = (X_k + u) - X_target
            current_coords = self.geometry.mesh.geometry.x
            error_vec = (current_coords + u_at_nodes) - self._target_coords

            # Compute L2 norm of the error vector
            local_error_sq = np.sum(error_vec**2)
            global_error_sq = comm.allreduce(local_error_sq, op=MPI.SUM)
            error_norm = np.sqrt(global_error_sq)

            # Normalize error by number of nodes or domain size for consistency?
            # Usually simple L2 norm or relative norm is fine.
            # Let's use relative norm with respect to target coords norm.
            target_norm_sq = comm.allreduce(np.sum(self._target_coords**2), op=MPI.SUM)
            rel_error = error_norm / np.sqrt(target_norm_sq) if target_norm_sq > 0 else error_norm

            logger.info(f"  Geometric Error (rel): {rel_error:.2e} (abs: {error_norm:.2e})")

            if rel_error < tol:
                logger.info(f"FixedPointUnloader converged in {i + 1} iterations.")
                break

            # 4. Update Reference Geometry
            # Update rule: X_{k+1} = X_{target} - u(X_k)
            # This effectively moves the reference configuration 'backwards' by the displacement
            # required to reach the target from the current guess.
            self.geometry.mesh.geometry.x[:] = self._target_coords - u_at_nodes

            # Deforming the fields seems to cause issues with convergence, so we skip it for now.
            if update_model_fields:
                u.x.array[:] *= -1.0
                self._update_model_fields(u)

        else:
            logger.warning("FixedPointUnloader reached maximum iterations without converging.")

        # Reset mesh to original target coords for consistency
        self.geometry.mesh.geometry.x[:] = self._target_coords

        # Final displacement from loaded to unloaded
        u.x.array[:] *= -1.0
        return u


@dataclass
class PrestressProblem:
    r"""
    Class for solving the Inverse Elasticity Problem (IEP).

    The goal is to find the reference configuration :math:`\Omega_0` given the
    target (loaded) configuration :math:`\Omega_t` and the applied loads.

    We solve for the inverse displacement field :math:`\mathbf{u}` defined on
    :math:`\Omega_t` such that the reference coordinates are given by:

    .. math::
        \mathbf{X} = \mathbf{x} + \mathbf{u}(\mathbf{x})

    where :math:`\mathbf{x} \in \Omega_t`.

    The deformation gradient :math:`\mathbf{F}` (mapping reference to target) is
    computed as the inverse of the gradient of the mapping from target to reference:

    .. math::
        \mathbf{f} = \mathbf{I} + \\nabla_{\mathbf{x}} \mathbf{u} \\\\
        \mathbf{F} = \mathbf{f}^{-1}

    The weak form is derived by pulling back the equilibrium equations from the
    reference configuration to the target configuration.

    Parameters
    ----------
    geometry : HeartGeometry
        The geometry in the target (loaded) configuration.
    model : CardiacModel
        The material model.
    bcs : BoundaryConditions
        The boundary conditions applied to the target configuration.
    parameters : dict
        Solver parameters.
    targets : typing.Sequence[TargetPressure]
        Sequence of target pressures to be applied during the prestress procedure.
    ramp_steps : int
        Number of steps to ramp the target pressures.
    """

    geometry: HeartGeometry
    model: CardiacModel
    bcs: BoundaryConditions
    parameters: dict[str, typing.Any] = field(default_factory=dict)
    targets: typing.Sequence[TargetPressure] = field(default_factory=list)
    ramp_steps: int = 20

    def __post_init__(self):
        parameters = type(self).default_parameters()
        parameters.update(self.parameters)
        self.parameters = parameters
        self._init_spaces()
        self._init_forms()

    @property
    def is_incompressible(self) -> bool:
        return not self.model.compressibility.is_compressible()

    @property
    def states(self):
        u = [self.u]
        if self.is_incompressible:
            u.append(self.p)
        return u

    @property
    def trial_functions(self):
        u = [self.du]
        if self.is_incompressible:
            u.append(self.dp)
        return u

    @property
    def test_functions(self):
        u = [self.u_test]
        if self.is_incompressible:
            u.append(self.p_test)
        return u

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
        return 1 + int(self.is_incompressible)

    def _init_spaces(self):
        self._init_u_space()
        self._init_p_space()

    def _init_u_space(self):
        u_family, u_degree = self.parameters["u_space"].split("_")

        u_element = basix.ufl.element(
            family=u_family,
            cell=self.geometry.mesh.basix_cell(),
            degree=int(u_degree),
            shape=(self.geometry.mesh.topology.dim,),
        )
        self.u_space = dolfinx.fem.functionspace(self.geometry.mesh, u_element)
        self.u = dolfinx.fem.Function(self.u_space, name="u")
        self.u_test = ufl.TestFunction(self.u_space)
        self.du = ufl.TrialFunction(self.u_space)

    def _init_p_space(self):
        if self.is_incompressible:
            p_family, p_degree = self.parameters["p_space"].split("_")
            p_element = basix.ufl.element(
                family=p_family,
                cell=self.geometry.mesh.basix_cell(),
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
        self.model.compressibility.register(self.p)

    def _material_form(self, u: dolfinx.fem.Function):
        dim = self.u.ufl_shape[0]
        f = ufl.Identity(dim) + ufl.grad(self.u)  # Inverse tensor for inverse problem
        j = ufl.det(f)

        F = ufl.variable(ufl.inv(f))  # Compute original one to diff

        P = self.model.P(F)

        forms = self._empty_form()
        forms[0] += ufl.inner(j * P, ufl.grad(self.u_test) * ufl.inv(f)) * self.geometry.dx

        if self.is_incompressible:
            J = 1.0 / j
            # The constraint is (J - 1) * p_test * dX = (J - 1) * p_test * j * dx
            forms[-1] += (J - 1.0) * self.p_test * j * self.geometry.dx

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

    def unload(self) -> dolfinx.fem.Function:
        """
        Perform the unloading procedure to find the reference configuration.

        Returns
        -------
        dolfinx.fem.Function
            The displacement field mapping from the unloaded to the loaded configuration.
        """

        for ramp in np.linspace(0.0, 1.0, self.ramp_steps):
            for traction, target_pressure, name in self.targets:
                value = ramp * target_pressure
                assign(traction, value)
                if name != "":
                    msg = f"Ramping {name} traction to {value:.4f}"
                else:
                    msg = f"Ramping traction to {value:.4f}"
                logger.info(msg)
            self.solve()
        return self.u
