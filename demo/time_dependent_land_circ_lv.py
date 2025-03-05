# # LV ellipsoid coupled to a 0D circulatory model and a 0D cell model

# In this example we will couple a 3D LV ellipsoid model to a 0D circulatory model and a 0D cell model.
#
# ## The 0D cell model
# The 0D cell model we will use is the one TorORd model {cite}`tomek2019development` which is an electrophysiology model developed for human ventricular myocytes. The model is available at https://github.com/jtmff/torord in CellML format (we use the `endo` cell version of the model). To convert the model to Python code we use the [`gotranx`](https://finsberg.github.io/gotranx/README.html) package. See https://finsberg.github.io/gotranx/docs/cli.html for an example of how to convert a CellML model to Python code. We have also coupled this model to the Land model {cite}`land2017model` which is an excitation-contraction model for cardiac cells. In order to use this model in the 3D simulations we will first run the model to steady state, and then we will use the active tension (`Ta`), which is an intermediate variable in the model, as the active stress in the 3D model. We will also scale the single cell active stress by a factor of 5.0 to get a more pronounced effect in the 3D model.
#
# ## The 0D circulatory model
# The 0D circulatory model we will use is the one from {cite}`regazzoni2022cardiac` which is a lumped parameter model of the circulatory system. We use an existing implementation in the [circulation package](https://computationalphysiology.github.io/circulation/examples/regazzoni.html).

# ## Coupling the 0D cell model to the 3D model
# TBW

# Now we will start by importing the necessary modules
#

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
import adios4dolfinx
import fenicsx_pulse
import cardiac_geometries
import cardiac_geometries.geometry

# Next we set up the logging and the MPI communicator

circulation.log.setup_logging(logging.INFO)
comm = MPI.COMM_WORLD

# Next we create the geometry of the LV ellipsoid

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

geometry = fenicsx_pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <fenicsx_pulse.holzapfelogden.HolzapfelOgden>`

material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

# We use an active stress approach with 30% transverse active stress (see {py:meth}`fenicsx_pulse.active_stress.transversely_active_stress`)

Ta = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta)

# a compressible material model

comp_model = fenicsx_pulse.compressibility.Compressible2()

# and assembles the `CardiacModel`

model = fenicsx_pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# Next we set up the boundary conditions. We use a Robin boundary condition on the epicardium and the base of the LV

alpha_epi = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e8)), "Pa / m",
)
robin_epi = fenicsx_pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
alpha_base = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)), "Pa / m",
)
robin_base = fenicsx_pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])

# For the pressure we use a Lagrange multiplier method to enforce a given volume. The resulting Lagrange multiplier will be the pressure in the cavity.
# To do this we create a `Cavity` object with a given volume, and specify which marker to use for the boundary condition.

initial_volume = geometry.volume("ENDO")
volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(initial_volume))
cavity = fenicsx_pulse.problem.Cavity(marker="ENDO", volume=volume, pressure_name="p_LV", volume_name="V_LV", scale_volume=1e6, scale_pressure=0.0075)
parameters = {"base_bc": fenicsx_pulse.problem.BaseBC.free, "mesh_unit": "m"}

# Next we set up the problem. We can choose between a static and a dynamic problem by setting the `static` variable to `True` or `False`. Currently the dynamic problem is not working (when coupled to a circulation model), so we will use the static problem for now.

static = True

if static:
    outdir = Path("lv_ellipsoid_time_dependent_circulation_static")
    bcs = fenicsx_pulse.BoundaryConditions(robin=(robin_epi, robin_base))
    problem = fenicsx_pulse.problem.StaticProblem(model=model, geometry=geometry, bcs=bcs, cavities=[cavity], parameters=parameters)
else:
    outdir = Path("lv_ellipsoid_time_dependent_circulation_dynamic")
    beta_epi = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m",
    )
    robin_epi_v = fenicsx_pulse.RobinBC(value=beta_epi, marker=geometry.markers["EPI"][0], damping=True)
    beta_base = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m",
    )
    robin_base_v = fenicsx_pulse.RobinBC(value=beta_base, marker=geometry.markers["BASE"][0], damping=True)
    bcs = fenicsx_pulse.BoundaryConditions(robin=(robin_epi, robin_epi_v, robin_base, robin_base_v))

    problem = fenicsx_pulse.problem.DynamicProblem(model=model, geometry=geometry, bcs=bcs, cavities=[cavity], parameters=parameters)


outdir.mkdir(exist_ok=True)

# Now we can solve the problem

log.set_log_level(log.LogLevel.INFO)
problem.solve()

# Next we will load the 0D cell model and run it to steady state. Here we use `gotranx` to load the ode, remove potential singularities and convert it to Python code. We then run the model for 200 beats and save the state of the model at the end of the simulation. We also plot the results. Note also that we extract the index of the active tension (`Ta`) which will be used to drive the 3D model.

ode = gotranx.load_ode("TorOrdLand.ode")
ode = ode.remove_singularities()
code = gotranx.cli.gotran2py.get_code(
    ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen], shape=gotranx.codegen.base.Shape.single,
)
Path("TorOrdLand.py").write_text(code)
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
    for beat in range(nbeats):
        print(f"Solving beat {beat}")
        V_tmp = Vs[beat * len(times) : (beat + 1) * len(times)]
        Cai_tmp = Cais[beat * len(times) : (beat + 1) * len(times)]
        Ta_tmp = Tas[beat * len(times) : (beat + 1) * len(times)]
        solve_beat(times, y, dt_cell, p, V_index, Ca_index, V_tmp, Cai_tmp, Ta_tmp)


    fig, ax = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(10, 10))
    ax[0, 0].plot(all_times, Vs)
    ax[1, 0].plot(all_times, Cais)
    ax[2, 0].plot(all_times, Tas)
    ax[0, 1].plot(times, Vs[-len(times):])
    ax[1, 1].plot(times, Cais[-len(times):])
    ax[2, 1].plot(times, Tas[-len(times):])
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

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, f"{outdir}/displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

# Here we also use [`adios4dolfinx`](https://jsdokken.com/adios4dolfinx/README.html) to save the displacement over at different time steps. Currently it is not a straight forward way to save functions and to later load them in a time dependent simulation in FEniCSx. However `adios4dolfinx` allows us to save the function to a file and later load it in a time dependent simulation. We will first need to save the mesh to the same file.

filename = Path("function_checkpoint.bp")
adios4dolfinx.write_mesh(filename, geometry.mesh)


# Next we set up the callback function that will be called at each time step. Here we save the displacement of the LV, the pressure volume loop, and the active tension, and we also plot the pressure volume loop at each time step.

def callback(model, t: float, save=True):
    model.results["Ta"].append(get_activation(t))
    if save:
        adios4dolfinx.write_function(filename, problem.u, time=t, name="displacement")
        vtx.write(t)

        if comm.rank == 0:
            fig = plt.figure(layout="constrained")
            gs = GridSpec(3, 2, figure=fig)
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2, 1])
            ax1.plot(model.results["V_LV"], model.results["p_LV"])
            ax1.set_xlabel("V [mL]")
            ax1.set_ylabel("p [mmHg]")

            ax2.plot(model.results["time"], model.results["p_LV"])
            ax2.set_ylabel("p [mmHg]")
            ax3.plot(model.results["time"], model.results["V_LV"])
            ax3.set_ylabel("V [mL]")
            ax4.plot(model.results["time"], model.results["Ta"])
            ax4.set_ylabel("Ta [kPa]")

            for axi in [ax2, ax3, ax4]:
                axi.set_xlabel("Time [s]")

            fig.savefig(outdir / "pv_loop_incremental.png")
            plt.close(fig)
            fig, ax = plt.subplots(4, 1)

            ax[0].plot(model.results["V_LV"], model.results["p_LV"])
            ax[0].set_xlabel("V [mL]")
            ax[0].set_ylabel("p [mmHg]")

            ax[1].plot(model.results["time"], model.results["p_LV"])
            ax[2].plot(model.results["time"], model.results["V_LV"])
            ax[3].plot(model.results["time"], model.results["Ta"])

            fig.savefig(outdir / "pv_loop_incremental.png")
            plt.close(fig)


# Now we will set ut the function to calculate the pressure in the LV. In this function we will receive a volume and a time from the 0D circulation model, we will set the active tension from the 0D cell model, and solve the 3D model to get the pressure in the LV.
#

def p_LV_func(V_LV, t):
    print("Calculating pressure at time", t)
    value = get_activation(t)
    print("Time", t, "Activation", value)
    Ta.assign(value)
    Volume.value = V_LV * 1e-6
    problem.solve()

    pendo_kPa = problem.cavity_pressures[0].x.array[0] * 1e-3

    return circulation.units.kPa_to_mmHg(pendo_kPa)

# mL = circulation.units.ureg("mL")
# add_units = False
# init_state = {"V_LV": initial_volume * mL}

HR = 1 * circulation.units.ureg("Hz")
num_beats = 4

if static:
    dt = 0.001
else:
    dt = problem.parameters["dt"].to_base_units()

# circulation_model = circulation.regazzoni2020.Regazzoni2020(
#     add_units=add_units,
#     # callback=callback,
#     # p_LV_func=p_LV_func,
#     verbose=True,
#     comm=comm,
#     outdir=outdir,
#     # initial_state=init_state,
# )

problem.register_circulation_model(circulation.regazzoni2020.Regazzoni2020, verbose=True, outdir=outdir)
# breakpoint()

for beat in range(num_beats):
    for t in problem.circulation_model.times_one_beat(problem.parameters["dt"].to_base_units()):
        print(f"Solving for time {t}")
        problem.solve()
        print(problem.cavity_pressures[0].x.array[0])#, problem.cavities[0].volume.value)
        print(problem.circulation_y.x.array)
        # vtx.write(t)



# circulation_model_3D.solve(num_cycles=5, initial_state=init_state, dt=dt)
# circulation_model_3D.print_info()
