import logging
import typing
from dataclasses import dataclass, field

from mpi4py import MPI

import numpy as np
import scifem

from .boundary_conditions import BoundaryConditions
from .cardiac_model import CardiacModel
from .geometry import HeartGeometry
from .problem import StaticProblem

logger = logging.getLogger(__name__)


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
    parameters: dict[str, typing.Any] = field(default_factory=dict)

    _target_coords: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Set default parameters for the unloader
        defaults = {
            "max_iter": 10,
            "tol": 1e-4,
        }
        # Update with user parameters (user values override defaults)
        # Note: We keep all parameters in one dict to pass to StaticProblem
        full_params = defaults.copy()
        full_params.update(self.parameters)
        self.parameters = full_params

        # Store the original target coordinates (owned + ghosts)
        # This serves as \mathbf{x}_{target}
        self._target_coords = self.geometry.mesh.geometry.x.copy()

    def solve(self) -> None:
        """
        Execute the fixed-point iteration to find the unloaded geometry.
        The `geometry.mesh` is updated in-place.
        """
        logger.info("Starting FixedPointUnloader...")

        max_iter = self.parameters["max_iter"]
        tol = self.parameters["tol"]
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
                parameters=self.parameters,
            )

            # We allow the StaticProblem to fail if the guess is bad, but usually it should converge
            try:
                converged = problem.solve()
                if not converged:
                    logger.warning("Forward problem did not converge fully.")
            except Exception as e:
                logger.error(f"Forward problem failed: {e}")
                raise e

            u = problem.u

            # 2. Evaluate displacement u at the mesh nodes
            # u lives on a function space (e.g. P2), mesh nodes are coordinate element (e.g. P1)
            # This returns the values of u at the current nodal positions X_k
            u_at_nodes = scifem.evaluate_function(u, self.geometry.mesh.geometry.x)

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

        else:
            logger.warning("FixedPointUnloader reached maximum iterations without converging.")
