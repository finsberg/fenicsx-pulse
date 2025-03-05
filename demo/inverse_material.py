
from typing import Callable, Sequence
import logging
from pathlib import Path

from mpi4py import MPI
import dolfinx
import fenicsx_pulse
import cardiac_geometries
import cardiac_geometries.geometry
import numpy as np
import numpy.typing as npt
import ufl
import dolfinx.nls.petsc
import dolfinx.fem.petsc
import adios4dolfinx
import scifem


from scipy.optimize import minimize


logger = logging.getLogger(__name__)


def taylor_test(
    comm: MPI.Intracomm,
    cost: Callable[[npt.NDArray[float]], float],
    grad: Callable[[npt.NDArray[float]], npt.NDArray[float]] | None,
    m_0: npt.NDArray[float],
    p: float | npt.NDArray[float] = 1e-2,
    n: int = 5,
):
    """
    Compute a Taylor test for a cost function and gradient function from `m_0` in direction `p`

    """
    l0 = cost(m_0)
    if grad is None:
        local_gradient = np.zeros_like(m_0)
    else:
        local_gradient = grad(m_0)
    global_gradient = np.hstack(comm.allgather(local_gradient))

    if isinstance(p, float):
        p = np.full_like(m_0, p)
    p_global = np.hstack(comm.allgather(p[: len(local_gradient)]))
    dJdm = np.dot(global_gradient, p_global)
    remainder = []
    perturbance = []
    for i in range(0, n):
        step = 0.5**i
        l1 = cost(m_0 + step * p)
        remainder.append(l1 - l0 - step * dJdm)
        perturbance.append(step)
    conv_rate = convergence_rates(remainder, perturbance)
    return remainder, perturbance, conv_rate


def convergence_rates(r: Sequence[float], p: Sequence[float]):
    cr = []  # convergence rates
    for i in range(1, len(p)):
        cr.append(np.log(r[i] / r[i - 1]) / np.log(p[i] / p[i - 1]))
    return cr



def generate_data(geo, outdir):
    # In order to use the geometry with `pulse` we need to convert it to a `fenicsx_pulse.Geometry` object. We can do this by using the `from_cardiac_geometries` method. We also specify that we want to use a quadrature degree of 4
    geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

    # Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <fenicsx_pulse.holzapfelogden.HolzapfelOgden>`

    material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material_params["a"].assign(2.28)
    material_params["a_f"].assign(1.685)
    material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

    # We use an active stress approach with 30% transverse active stress (see {py:meth}`fenicsx_pulse.active_stress.transversely_active_stress`)

    Ta = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta, eta=0.3)

    # We use an incompressible model

    comp_model = fenicsx_pulse.Compressible()

    # and assembles the `CardiacModel`

    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    # We apply a traction in endocardium

    traction = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])

    # and finally combine all the boundary conditions

    bcs = fenicsx_pulse.BoundaryConditions(neumann=(neumann,))

    # and create a Mixed problem

    problem = fenicsx_pulse.StaticProblem(model=model, geometry=geometry, bcs=bcs, parameters={"base_bc": fenicsx_pulse.BaseBC.fixed})

    vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, f"{outdir}/lv_displacement.bp", [problem.u], engine="BP4")
    vtx.write(0.0)
    i = 1
    for plv in [0.1, 0.2, 0.3]:
        print(f"plv: {plv}")
        traction.assign(plv)
        problem.solve()

        vtx.write(float(i))
        i += 1

    adios4dolfinx.write_function_on_input_mesh(outdir / "u.bp", problem.u, time=0.0, name="u")




def run_taylor_test(comm, cost, grad):
    f_0 = np.array([1.8, 1.8])
    error, perturbance, rate = taylor_test(
        comm=comm,
        cost=cost,
        grad=None,
        m_0=f_0,
    )

    if comm.rank == 0:
        logger.info("\nRun Taylor test without gradient")
        logger.info(f"Error: {error}")
        logger.info(f"Perturbance: {perturbance}")
        logger.info(f"Convergence rate: {rate}")

    error, perturbance, rate = taylor_test(
        comm=comm,
        cost=cost,
        grad=grad,
        m_0=f_0,
    )
    if comm.rank == 0:
        logger.info("\nRun Taylor test with gradient")
        logger.info(f"Error: {error}")
        logger.info(f"Perturbance: {perturbance}")
        logger.info(f"Convergence rate: {rate}")


def run_optimization(comm, cost, grad):
    x0 = np.array([200.0])
    bounds = np.array([[100.0, 500.0]])
    tol = 1.0e-16
    res = minimize(
        fun=cost,
        x0=x0,
        method="L-BFGS-B",
        jac=grad,
        bounds=bounds,
        options={"maxiter": 400, "disp": True, "gtol": tol},
    )
    if comm.rank == 0:
        print(f"Found optimal value at {res.x}")


def main(loglevel=logging.DEBUG):
    if loglevel < logging.INFO:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    logging.basicConfig(level=loglevel)
    # mesh and function space
    comm = MPI.COMM_WORLD

    outdir = Path("inverse_material_lv_ellipsoid")
    outdir.mkdir(parents=True, exist_ok=True)
    geodir = outdir / "geometry"
    if not geodir.exists():
        cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, create_fibers=True, fiber_space="P_2", comm=comm)

    # If the folder already exist, then we just load the geometry

    geo = cardiac_geometries.geometry.Geometry.from_folder(
        comm=comm,
        folder=geodir,
    )

    if not (outdir / "u.bp").exists():
        generate_data(geo, outdir)

    V_control = scifem.create_real_functionspace(geo.mesh, value_shape=(2,))
    control = dolfinx.fem.Function(V_control, name="control")
    control.x.array[:] = np.array([2.28, 1.685])

    # In order to use the geometry with `pulse` we need to convert it to a `fenicsx_pulse.Geometry` object. We can do this by using the `from_cardiac_geometries` method. We also specify that we want to use a quadrature degree of 4
    geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

    # Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <fenicsx_pulse.holzapfelogden.HolzapfelOgden>`

    material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material_params["a"].assign(control[0])
    material_params["a_f"].assign(control[1])
    material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params, disable_check=True)  # type: ignore

    # We use an active stress approach with 30% transverse active stress (see {py:meth}`fenicsx_pulse.active_stress.transversely_active_stress`)

    Ta = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta, eta=0.3)

    # We use an incompressible model

    comp_model = fenicsx_pulse.Compressible()

    # and assembles the `CardiacModel`

    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    # We apply a traction in endocardium

    traction = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])

    # and finally combine all the boundary conditions

    bcs = fenicsx_pulse.BoundaryConditions(neumann=(neumann,))

    # and create a Mixed problem

    forward_problem = fenicsx_pulse.StaticProblem(model=model, geometry=geometry, bcs=bcs, parameters={"base_bc": fenicsx_pulse.BaseBC.fixed})


    bcs = forward_problem._solver.bcs
    u = forward_problem.u
    R = forward_problem.R[0]

    for plv in [0.1, 0.2, 0.3]:
        print(f"plv: {plv}")
        traction.assign(plv)
        forward_problem.solve()

    # Define desired u profile
    u_obs = dolfinx.fem.Function(forward_problem.u_space, name="u_obs")
    adios4dolfinx.read_function(outdir / "u.bp", u_obs, time=0.0, name="u")

    J = (1 / 2) * ufl.inner(u - u_obs, u - u_obs) * ufl.dx
    J_form = dolfinx.fem.form(J)

    # Define derivative of cost function
    # Bilinear and linear form of the adjoint problem
    adjoint_lhs = ufl.adjoint(ufl.derivative(R, u))
    adjoint_rhs = ufl.derivative(J, u)

    # Create adjoint problem solver
    lmbda = dolfinx.fem.Function(forward_problem.u_space, name="adjoint")
    lambda_0 = dolfinx.fem.Function(forward_problem.u_space)
    lambda_0.x.array[:] = 0

    if (bs := forward_problem.u_space.dofmap.bs) > 1:
        dofs = bcs[0].dof_indices()[0][::bs] // bs
    else:
        dofs = bcs[0].dof_indices()[0]

    # Homogenize bcs
    bcs_adjoint = [dolfinx.fem.dirichletbc(lambda_0, dofs) for bc in bcs]

    adjoint_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    if loglevel < logging.INFO:
        adjoint_options["ksp_monitor"] = None

    # Create Adjoint problem
    # Should use bc.set(0) instead - don't need lifting
    adjoint_problem = dolfinx.fem.petsc.LinearProblem(
        adjoint_lhs,
        adjoint_rhs,
        u=lmbda,
        bcs=bcs_adjoint,
    )

    # Compute sensitivity: dJ/dm
    # Partial derivative of J w.r.t. m
    dJdm = ufl.derivative(J, control)
    # partial derivative of R w.r.t. m
    dRdm = ufl.action(ufl.adjoint(ufl.derivative(R, control)), lmbda)
    dJdm = dolfinx.fem.form(dJdm - dRdm)

    def cost(f_data: np.ndarray):
        """Compute functional for a given control"""
        control.x.array[:] = f_data
        control.x.scatter_forward()
        forward_problem.solve()
        u.x.scatter_forward()
        value = comm.allreduce(dolfinx.fem.assemble_scalar(J_form), op=MPI.SUM)
        logger.debug(f"Evaluate cost J({f_data=})={value}")
        return value

    def grad(x):
        """
        Compute derivative of functional
        """
        J = cost(x)
        adjoint_problem.solve()
        lmbda.x.scatter_forward()
        dJdm_local = dolfinx.fem.assemble_vector(dJdm)
        dJdm_local.scatter_reverse(dolfinx.la.InsertMode.add)
        dJdm_local.scatter_forward()
        logger.debug(f"Evaluate derivate at {x=}, {J=} {dJdm_local.array=}")

        # FIXME: Not working in parallel with scipy.optimize.minimize
        arr = dJdm_local.array[
            : dJdm_local.index_map.size_local * dJdm_local.block_size
        ]

        return arr

    run_taylor_test(comm, cost, grad)
    run_optimization(comm, cost, grad)


if __name__ == "__main__":
    main()
