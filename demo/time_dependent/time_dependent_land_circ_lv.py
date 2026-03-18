# # LV ellipsoid coupled to a 0D circulatory model and a 0D cell model
#
# This example demonstrates a multiscale cardiac simulation coupling three distinct models:
# 1.  **3D Mechanics**: An idealized Left Ventricle (LV) ellipsoid model solving the balance of linear momentum with active contraction.
# 2.  **0D Circulation**: A lumped-parameter model of the closed-loop circulatory system {cite}`regazzoni2022cardiac`.
# 3.  **0D Electrophysiology**: A cellular model for human ventricular myocytes {cite}`tomek2019development` coupled with an excitation-contraction model {cite}`land2017model`.
#
# ## Coupling Strategy
#
# The coupling is achieved through a segregated approach:
#
# 1.  **Electrophysiology $\rightarrow$ Mechanics**:
#     The 0D cell model (TorOrd-Land) is pre-solved to generate a time-dependent active tension transient $T_a(t)$.
#     This $T_a(t)$ drives the active stress in the 3D mechanics model: $\mathbf{S}_{active} = T_a(t) \mathbf{f}_0 \otimes \mathbf{f}_0$.
#
# 2.  **Circulation $\leftrightarrow$ Mechanics**:
#     The 0D circulation model dictates the boundary conditions for the 3D mechanics model, and vice-versa.
#     We use a volume-based coupling strategy:
#     * The circulation model computes the new ventricular volume $V_{n+1}$ at the next time step.
#     * The 3D mechanics model solves a boundary value problem to find the cavity pressure $P_{n+1}$ required to achieve this volume $V_{n+1}$ (given the current activation $T_a(t_{n+1})$).
#     * This pressure $P_{n+1}$ is fed back to the circulation model.
#
# ## Mathematical Models
#
# ### 3D Mechanics
# * **Geometry**: Idealized LV ellipsoid with fiber architecture.
# * **Material**: Transversely isotropic Holzapfel-Ogden model.
# * **Active Stress**: $\mathbf{S}_{active} = T_a \mathbf{f}_0 \otimes \mathbf{f}_0$.
# * **Boundary Conditions**:
#     * **Epicardium & Base**: Robin BCs (springs) to mimic pericardial constraint and prevent rigid body motion.
#     * **Endocardium**: The pressure $P$ is a Lagrange multiplier enforcing the volume constraint $V(\mathbf{u}) = V_{target}$.
#
# ### 0D Circulation [Regazzoni et al. 2022]
# A closed-loop lumped-parameter network representing the systemic and pulmonary circulation.
# It includes 4 chambers (LA, LV, RA, RV) and systemic/pulmonary arteries and veins.
# In this example, we replace the 0D description of the LV with our 3D finite element model.
#
# ### 0D Cell Model [Tomek et al. 2019 + Land et al. 2017]
# * **TorOrd**: Detailed human ventricular action potential model.
# * **Land**: Mechanical model describing cross-bridge dynamics and calcium binding to Troponin-C.
#
# ---

from pathlib import Path
from mpi4py import MPI
import dolfinx
import logging
import circulation
import os
import math
from functools import lru_cache
from dolfinx import log
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import gotranx
import io4dolfinx
import pulse
import cardiac_geometries
import cardiac_geometries.geometry

# Next we set up the logging and the MPI communicator

circulation.log.setup_logging(logging.INFO)
logging.getLogger("scifem").setLevel(logging.WARNING)
logger = logging.getLogger("pulse")
comm = MPI.COMM_WORLD

# ## 1. Geometry Generation
# We create the idealized LV geometry with appropriate fibers using `cardiac_geometries`.

geodir = Path("lv_ellipsoid-time-dependent")
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="Quadrature_6",
        r_short_endo=0.025,
        r_short_epi=0.035,
        r_long_endo=0.09,
        r_long_epi=0.097,
        psize_ref=0.03,
        mu_apex_endo=-math.pi,
        mu_base_endo=-math.acos(5 / 17),
        mu_apex_epi=-math.pi,
        mu_base_epi=-math.acos(5 / 20),
        comm=comm,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

# Next we transform the geometry to a `HeartGeometry` object

geometry = pulse.HeartGeometry.from_cardiac_geometries(
    geo, metadata={"quadrature_degree": 6},
)

# ## 2. Constitutive Model
#
# We use the standard Holzapfel-Ogden model for passive material properties and an Active Stress model driven by a scalar parameter $T_a$.

material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

# We use an active stress approach with 30% transverse active stress (see {py:meth}`pulse.active_stress.transversely_active_stress`)

Ta = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa",
)
active_model = pulse.ActiveStress(geo.f0, activation=Ta)

# a compressible material model

comp_model = pulse.compressibility.Compressible2()

# and assembles the `CardiacModel`

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# ## 3. Boundary Conditions
#
# We apply Robin boundary conditions (springs) to the epicardium and base to represent the pericardium and tissue support.
#
# **Crucially**, for the endocardium, we use a **Cavity Volume Constraint**.
# Instead of applying a known pressure (Neumann BC), we enforce the cavity volume to match a target value provided by the circulation model. The Lagrange multiplier associated with this constraint is the cavity pressure.

alpha_epi = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e8)),
    "Pa / m",
)
robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
alpha_base = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)),
    "Pa / m",
)
robin_base = pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])

# For the pressure we use a Lagrange multiplier method to enforce a given volume. The resulting Lagrange multiplier will be the pressure in the cavity.
# To do this we create a `Cavity` object with a given volume, and specify which marker to use for the boundary condition.

initial_volume = geo.mesh.comm.allreduce(geometry.volume("ENDO"), op=MPI.SUM)
Volume = dolfinx.fem.Constant(
    geometry.mesh, dolfinx.default_scalar_type(initial_volume),
)
cavity = pulse.problem.Cavity(marker="ENDO", volume=Volume)

# We also specify the parameters for the problem and say that we want the base to move freely and that the units of the mesh is meters

parameters = {"base_bc": pulse.problem.BaseBC.free, "mesh_unit": "m"}

# Next we set up the problem.
outdir = Path("lv_ellipsoid_time_dependent_circulation_static")
bcs = pulse.BoundaryConditions(robin=(robin_epi, robin_base))
problem = pulse.problem.StaticProblem(
    model=model, geometry=geometry, bcs=bcs, cavities=[cavity], parameters=parameters,
)

outdir.mkdir(exist_ok=True)

# Now we can solve the problem

# log.set_log_level(log.LogLevel.INFO)
problem.solve()

# We also use the time step from the problem to set the time step for the 0D cell model

dt = 0.001
times = np.arange(0.0, 1.0, dt)

# ## 4. Electrophysiology (0D Cell Model)
#
# We pre-compute the active tension transient. We load the TorOrd-Land model using `gotranx`, solve it for 200 beats to reach a limit cycle (steady state), and save the final beat's active tension trace.

if comm.rank == 0:
    ode = gotranx.load_ode("TorOrdLand.ode")
    ode = ode.remove_singularities()
    code = gotranx.cli.gotran2py.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
        shape=gotranx.codegen.base.Shape.single,
    )
    Path("TorOrdLand.py").write_text(code)
comm.barrier()
import TorOrdLand

TorOrdLand_model = TorOrdLand.__dict__
Ta_index = TorOrdLand_model["monitor_index"]("Ta")
y = TorOrdLand_model["init_state_values"]()
# Get initial parameter values
p = TorOrdLand_model["init_parameter_values"]()
import numba

fgr = numba.njit(TorOrdLand_model["generalized_rush_larsen"])
mon = numba.njit(TorOrdLand_model["monitor_values"])
V_index = TorOrdLand_model["state_index"]("v")
Ca_index = TorOrdLand_model["state_index"]("cai")

# In the cell model the unit of time is in millisenconds, and we set the time step to 0.1 ms

# Time in milliseconds
dt_cell = 0.1

# Now we solve the cell model for 200 beats and save the state at the end of the simulation

state_file = outdir / "state.npy"
if not state_file.is_file():

    @numba.jit(nopython=True)
    def solve_beat(times, states, dt, p, V_index, Ca_index, Vs, Cais, Tas):
        for i, ti in enumerate(times):
            states[:] = fgr(states, ti, dt, p)
            Vs[i] = states[V_index]
            Cais[i] = states[Ca_index]
            monitor = mon(ti, states, p)
            Tas[i] = monitor[Ta_index]

    # Time in milliseconds

    nbeats = 200
    T = 1000.00

    times = np.arange(0, T, dt_cell)
    all_times = np.arange(0, T * nbeats, dt_cell)
    Vs = np.zeros(len(times) * nbeats)
    Cais = np.zeros(len(times) * nbeats)
    Tas = np.zeros(len(times) * nbeats)
    logger.info(f"Starting to solve {nbeats} beats of the cell model")
    for beat in range(nbeats):
        logger.debug(f"Solving beat {beat}")
        V_tmp = Vs[beat * len(times) : (beat + 1) * len(times)]
        Cai_tmp = Cais[beat * len(times) : (beat + 1) * len(times)]
        Ta_tmp = Tas[beat * len(times) : (beat + 1) * len(times)]
        solve_beat(times, y, dt_cell, p, V_index, Ca_index, V_tmp, Cai_tmp, Ta_tmp)

    fig, ax = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(10, 10))
    ax[0, 0].plot(all_times, Vs)
    ax[1, 0].plot(all_times, Cais)
    ax[2, 0].plot(all_times, Tas)
    ax[0, 1].plot(times, Vs[-len(times) :])
    ax[1, 1].plot(times, Cais[-len(times) :])
    ax[2, 1].plot(times, Tas[-len(times) :])
    ax[0, 0].set_ylabel("V")
    ax[1, 0].set_ylabel("Cai")
    ax[2, 0].set_ylabel("Ta")
    ax[2, 0].set_xlabel("Time [ms]")
    ax[2, 1].set_xlabel("Time [ms]")

    fig.savefig(outdir / "Ta_ORdLand.png")
    np.save(state_file, y)

# Next we load the state of the cell model and create an `ODEState` object. This is a simple wrapper around the cell model that allows us to step the model forward in time and get the active tension at a given time.
#

y = np.load(state_file)


class ODEState:
    def __init__(self, y, dt_cell, p, t=0.0):
        self.y = y
        self.dt_cell = dt_cell
        self.p = p
        self.t = t

    def forward(self, t):
        for t_cell in np.arange(self.t, t, self.dt_cell):
            self.y[:] = fgr(self.y, t_cell, self.dt_cell, self.p)
        self.t = t
        return self.y[:]

    def Ta(self, t):
        monitor = mon(t, self.y, p)
        return monitor[Ta_index]


ode_state = ODEState(y, dt_cell, p)


@lru_cache
def get_activation(t: float):
    # Find index modulo 1000
    t_cell_next = t * 1000
    ode_state.forward(t_cell_next)
    return ode_state.Ta(t_cell_next) * 5.0


# Next we will save the displacement of the LV to a file

vtx = dolfinx.io.VTXWriter(
    geometry.mesh.comm, f"{outdir}/displacement.bp", [problem.u], engine="BP4",
)
vtx.write(0.0)

# Here we also use [`io4dolfinx`](https://jsdokken.com/io4dolfinx/README.html) to save the displacement over at different time steps. Currently it is not a straight forward way to save functions and to later load them in a time dependent simulation in FEniCSx. However `io4dolfinx` allows us to save the function to a file and later load it in a time dependent simulation. We will first need to save the mesh to the same file.

filename = Path("function_checkpoint.bp")
io4dolfinx.write_mesh(filename, geometry.mesh)

Ta_history = []
# Next we set up the callback function that will be called at each time step. Here we save the displacement of the LV, the pressure volume loop, and the active tension, and we also plot the pressure volume loop at each time step.


def callback(model, i: int, t: float, save=True):
    Ta_history.append(get_activation(t))
    if save:
        io4dolfinx.write_function(filename, problem.u, time=t, name="displacement")
        vtx.write(t)

        if comm.rank == 0:
            fig = plt.figure(layout="constrained")
            gs = GridSpec(3, 2, figure=fig)
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2, 1])
            ax1.plot(model.history["V_LV"][: i + 1], model.history["p_LV"][: i + 1])
            ax1.set_xlabel("V [mL]")
            ax1.set_ylabel("p [mmHg]")

            ax2.plot(model.history["time"][: i + 1], model.history["p_LV"][: i + 1])
            ax2.set_ylabel("p [mmHg]")
            ax3.plot(model.history["time"][: i + 1], model.history["V_LV"][: i + 1])
            ax3.set_ylabel("V [mL]")
            ax4.plot(model.history["time"][: i + 1], Ta_history[: i + 1])
            ax4.set_ylabel("Ta [kPa]")

            for axi in [ax2, ax3, ax4]:
                axi.set_xlabel("Time [s]")

            logger.debug(f"Saving figure to {outdir / 'pv_loop_incremental.png'}")
            fig.savefig(outdir / "pv_loop_incremental.png")
            plt.close(fig)
            # fig, ax = plt.subplots(4, 1)

            # ax[0].plot(model.history["V_LV"][:i+1], model.history["p_LV"][:i+1])
            # ax[0].set_xlabel("V [mL]")
            # ax[0].set_ylabel("p [mmHg]")

            # ax[1].plot(model.history["time"][:i+1], model.history["p_LV"][:i+1])
            # ax[2].plot(model.history["time"][:i+1], model.history["V_LV"][:i+1])
            # ax[3].plot(model.history["time"][:i+1], Ta_history[:i+1])

            # fig.savefig(outdir / "pv_loop_incremental.png")
            # plt.close(fig)


# ## 5. Coupling Function: 0D $\rightarrow$ 3D
#
# This function defines the interface between the circulation loop and the 3D model.
#
# 1.  **Input**: The circulation model provides the target LV Volume ($V_{LV}$) and current time $t$.
# 2.  **Active State**: We query the pre-computed active tension $T_a(t)$ from the cell model.
# 3.  **Solve 3D**: We update the `Volume` constant in the `StaticProblem` and the activation `Ta`. The solver then finds the displacement field $\mathbf{u}$ that satisfies the volume constraint.
# 4.  **Output**: The Lagrange multiplier associated with the volume constraint is the cavity pressure. We return this pressure (converted to mmHg) to the circulation model.


def p_LV_func(V_LV, t):
    logger.debug("Calculating pressure at time %f", t)
    value = get_activation(t)
    logger.debug("Time %f Activation %f", t, value)
    Ta.assign(value)
    Volume.value = V_LV * 1e-6
    problem.solve()
    pendo = problem.cavity_pressures[0]

    pendo_kPa = pendo.x.array[0] * 1e-3

    return circulation.units.kPa_to_mmHg(pendo_kPa)


# ## 6. Run Coupled Simulation
#
# We initialize the Regazzoni circulation model with our custom pressure function `p_LV=p_LV_func`.
# The model is then integrated in time. At each time step, the circulation model computes a target volume, our 3D model computes the corresponding pressure, and the loop advances.

mL = circulation.units.ureg("mL")
add_units = False
surface_area = geometry.surface_area("ENDO")
initial_volume = (
    geo.mesh.comm.allreduce(geometry.volume("ENDO", u=problem.u), op=MPI.SUM) * 1e6
)
logger.info(f"Initial volume: {initial_volume}")
init_state = {"V_LV": initial_volume * mL}


circulation_model_3D = circulation.regazzoni2020.Regazzoni2020(
    add_units=add_units,
    callback=callback,
    p_LV=p_LV_func,
    verbose=True,
    comm=comm,
    outdir=outdir,
    initial_state=init_state,
)
# Set end time for early stopping if running in CI
end_time = 2 * dt if os.getenv("CI") else None
circulation_model_3D.solve(num_beats=5, initial_state=init_state, dt=dt, T=end_time)
circulation_model_3D.print_info()


# ```{figure} ../../_static/pv_loop_time_dependent_land_circ_lv.png
# ---
# name: pv_loop_time_dependent_land_circ_lv
# ---
# Pressure volume loop for the LV.
# ```
#
# <video controls loop autoplay muted>
#   <source src="../../_static/time_dependent_land_circ_lv.mp4" type="video/mp4">
#   <p>Video showing the motion of the LV.</p>
# </video>
#
# # References
# ```{bibliography}
# :filter: docname in docnames
#
