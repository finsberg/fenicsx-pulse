# # Bleeding BiV

# In this example we will take the [BiV example](time_dependent_land_circ_biv.py) from the other tutorial and drain the veins with 2 liter of blood. To model the the effect of bleeding we will use the Zenker model {cite}`zenker2019correction` to find the new heart rate and the Regazzoni model to simulate the circulation.

from pathlib import Path

from mpi4py import MPI
import dolfinx
import logging
import os
from functools import lru_cache

import circulation
from dolfinx import log
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import gotranx
import adios4dolfinx
import cardiac_geometries
import cardiac_geometries.geometry
import fenicsx_pulse

circulation.log.setup_logging(logging.INFO)
logger = logging.getLogger("pulse")
comm = MPI.COMM_WORLD

outdir = Path("bleeding_biv")
outdir.mkdir(exist_ok=True)

geodir = outdir / "geometry"
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.biv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="Quadrature_6",
        comm=comm,
        char_length=1.0,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )


# If the folder already exist, then we just load the geometry
geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)
# Now, let's scale the geometry to be in meters and so that the volume matches the expected volume in the ciculation model
geo.mesh.geometry.x[:] *= 3e-2

geometry = fenicsx_pulse.HeartGeometry.from_cardiac_geometries(
    geo, metadata={"quadrature_degree": 6},
)

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <fenicsx_pulse.holzapfelogden.HolzapfelOgden>`

material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

# We use an active stress approach with 30% transverse active stress (see {py:meth}`fenicsx_pulse.active_stress.transversely_active_stress`)

Ta = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa",
)
active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta)

# We use an incompressible model

comp_model = fenicsx_pulse.compressibility.Compressible2()
viscoeleastic_model = fenicsx_pulse.viscoelasticity.Viscous()

# and assembles the `CardiacModel`

model = fenicsx_pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
    viscoelasticity=viscoeleastic_model,
)

alpha_epi = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e8)),
    "Pa / m",
)
robin_epi = fenicsx_pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
alpha_base = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)),
    "Pa / m",
)
robin_base = fenicsx_pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])


lvv_initial = geometry.volume("ENDO_LV")
lv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(lvv_initial))
lv_cavity = fenicsx_pulse.problem.Cavity(marker="ENDO_LV", volume=lv_volume)

rvv_initial = geometry.volume("ENDO_RV")
rv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(rvv_initial))
rv_cavity = fenicsx_pulse.problem.Cavity(marker="ENDO_RV", volume=rv_volume)

cavities = [lv_cavity, rv_cavity]


parameters = {"base_bc": fenicsx_pulse.problem.BaseBC.free, "mesh_unit": "m"}

outdir = Path("bleeding_biv3")
bcs = fenicsx_pulse.BoundaryConditions(robin=(robin_epi, robin_base))
problem = fenicsx_pulse.problem.StaticProblem(
    model=model, geometry=geometry, bcs=bcs, cavities=cavities, parameters=parameters,
)

outdir.mkdir(exist_ok=True)


# Run first Zenker to get the correct heart rate for normal conditions

zenker_normal = circulation.zenker.Zenker()
zenker_normal.solve(T=100.0, dt=1e-3, dt_eval=0.1)
HR_normal = zenker_normal.results["fHR"][-1]
R_TPR_normal = zenker_normal.results["R_TPR"][-1]
C_PRSW_normal = zenker_normal.results["C_PRSW"][-1]

print(f"HR_normal = {HR_normal}, R_TPR_normal = {R_TPR_normal}, C_PRSW_normal = {C_PRSW_normal}")


# Now we will simulate a bleeding and compute a new heart rate

blood_loss_parameters = {
    "start_withdrawal": 1,
    "end_withdrawal": 2,
    "flow_withdrawal": -2000,
    "flow_infusion": 0,
}
zenker_bleed = circulation.zenker.Zenker(parameters=blood_loss_parameters)
zenker_bleed.solve(T=300.0, dt=1e-3, dt_eval=0.1, initial_state=zenker_normal.state)
HR_bleed = zenker_bleed.results["fHR"][-1]
R_TPR_bleed = zenker_bleed.results["R_TPR"][-1]
C_PRSW_bleed = zenker_bleed.results["C_PRSW"][-1]

print(f"HR_bleed = {HR_bleed}, R_TPR_bleed = {R_TPR_bleed}, C_PRSW_bleed = {C_PRSW_bleed}")

HR_factor = HR_bleed / HR_normal
R_TPR_factor = R_TPR_bleed / R_TPR_normal
C_PRSW_factor = C_PRSW_bleed / C_PRSW_normal

# Create updated parameters for the Regazzoni model yby scaling the heart rate, resistance and compliance parameters

regazzoni_bleed_parmeters = circulation.regazzoni2020.Regazzoni2020.default_parameters()
regazzoni_bleed_parmeters["HR"] = HR_factor
regazzoni_bleed_parmeters["circulation"]["SYS"]["R_AR"] *= R_TPR_factor
regazzoni_bleed_parmeters["circulation"]["SYS"]["R_VEN"] *= R_TPR_factor
for chamber in ["LA", "LV", "RA", "RV"]:
    regazzoni_bleed_parmeters["chambers"][chamber]["EA"] *= C_PRSW_factor
    regazzoni_bleed_parmeters["chambers"][chamber]["EB"] *= C_PRSW_factor
regazzoni_bleed_parmeters["circulation"]["external"] = blood_loss_parameters

# The RR interval will determine the duration of the simulation

RR = 1 / HR_factor

# Now we can solve the problem

log.set_log_level(log.LogLevel.INFO)
problem.solve()

# Now let ut simulate the single cell model to get the activation

ode = gotranx.load_ode("TorOrdLand.ode")
ode = ode.remove_singularities()
code = gotranx.cli.gotran2py.get_code(
    ode,
    scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    shape=gotranx.codegen.base.Shape.single,
)
Path("TorOrdLand.py").write_text(code)
import TorOrdLand

TorOrdLand_model = TorOrdLand.__dict__

Ta_index = TorOrdLand_model["monitor_index"]("Ta")
y = TorOrdLand_model["init_state_values"]()
# Get initial parameter values
p = TorOrdLand_model["init_parameter_values"](i_Stim_Period=RR * 1000)
import numba

fgr = numba.njit(TorOrdLand_model["generalized_rush_larsen"])
mon = numba.njit(TorOrdLand_model["monitor_values"])
V_index = TorOrdLand_model["state_index"]("v")
Ca_index = TorOrdLand_model["state_index"]("cai")

# Time in milliseconds

dt_cell = 0.1

# Let us ensure the 0D model is run to steady state

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
    T = RR * 1000
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
    ax[0, 0].plot(all_times, Vs[-len(all_times) :])
    ax[1, 0].plot(all_times, Cais[-len(all_times) :])
    ax[2, 0].plot(all_times, Tas[-len(all_times) :])
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

# Load the steady state

y = np.load(state_file)

# Create a class to handle the ODE state

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


# Now use the ODEState class to get the activation

ode_state = ODEState(y, dt_cell, p)

dt = 0.01
Tas = {}

for t in np.arange(0, RR + dt, dt):
    t_cell_next = t * 1000
    ode_state.forward(t_cell_next)
    value = ode_state.Ta(t_cell_next) * 3.0
    Tas[f"{t:.3f}"] = value


@lru_cache
def get_activation(t: float):
    # # Find index modulo 1000
    # t_cell_next = t * 1000
    # ode_state.forward(t_cell_next)
    # return ode_state.Ta(t_cell_next) * 5.0
    return Tas[f"{t % RR:.3f}"]


# Let us plot the activation


times = np.linspace(0, 10 * RR, 1000)
activation = [get_activation(t) for t in times]
plt.plot(times, activation)
plt.savefig(outdir / "activation.png")

# Save the displacement for visualization in Paraview

vtx = dolfinx.io.VTXWriter(
    geometry.mesh.comm, f"{outdir}/displacement.bp", [problem.u], engine="BP4",
)
vtx.write(0.0)

# Save checkpoints for later post processing

filename = Path("function_checkpoint.bp")
adios4dolfinx.write_mesh(filename, geometry.mesh)

# The callback function is primarily used to save the results and plot the pressure volume loop as we run the simulation


def callback(model, t: float, save=True):
    model.results["Ta"].append(get_activation(t))
    if save:
        adios4dolfinx.write_function(filename, problem.u, time=t, name="displacement")
        vtx.write(t)

        fig = plt.figure(layout="constrained", figsize=(12, 8))
        gs = GridSpec(3, 4, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[:, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 2])
        ax5 = fig.add_subplot(gs[0, 3])
        ax6 = fig.add_subplot(gs[1, 3])
        ax7 = fig.add_subplot(gs[2, 2:])

        ax1.plot(model.results["V_LV"], model.results["p_LV"])
        ax1.set_xlabel("LVV [mL]")
        ax1.set_ylabel("LVP [mmHg]")

        ax2.plot(model.results["V_RV"], model.results["p_RV"])
        ax2.set_xlabel("RVV [mL]")
        ax2.set_ylabel("RVP [mmHg]")

        ax3.plot(model.results["time"], model.results["p_LV"])
        ax3.set_ylabel("LVP [mmHg]")
        ax4.plot(model.results["time"], model.results["V_LV"])
        ax4.set_ylabel("LVV [mL]")

        ax5.plot(model.results["time"], model.results["p_RV"])
        ax5.set_ylabel("RVP [mmHg]")
        ax6.plot(model.results["time"], model.results["V_RV"])
        ax6.set_ylabel("RVV [mL]")

        ax7.plot(model.results["time"], model.results["Ta"])
        ax7.set_ylabel("Ta [kPa]")

        for axi in [ax3, ax4, ax5, ax6, ax7]:
            axi.set_xlabel("Time [s]")

        fig.savefig(outdir / "pv_loop_incremental.png")
        plt.close(fig)

        pressure_keys = [
            "p_AR_SYS",
            "p_VEN_SYS",
            "p_AR_PUL",
            "p_VEN_PUL",
            "p_LV",
            "p_RV",
            "p_LA",
            "p_RA",
        ]
        flow_keys = [
            "Q_MV",
            "Q_AV",
            "Q_TV",
            "Q_PV",
            "I_ext",
            "Q_AR_SYS",
            "Q_VEN_SYS",
            "Q_AR_PUL",
            "Q_VEN_PUL",
        ]

        fig, ax = plt.subplots(3, 3, sharex=True, figsize=(10, 8))
        for axi, key in zip(ax.flatten(), flow_keys):
            axi.plot(model.results["time"], model.results[key])
            axi.set_title(key)
        fig.suptitle("Flow")
        fig.savefig(outdir / "flow.png")
        plt.close(fig)

        fig, ax = plt.subplots(4, 2, sharex=True, figsize=(10, 8))
        for axi, key in zip(ax.flatten(), pressure_keys):
            axi.plot(model.results["time"], model.results[key])
            axi.set_title(key)
        fig.suptitle("Pressure")
        fig.savefig(outdir / "pressure.png")
        plt.close(fig)

        volumes = circulation.regazzoni2020.Regazzoni2020.compute_volumes(
            model.parameters, model.results,
        )

        fig, ax = plt.subplots(4, 3, sharex=True, figsize=(10, 8))
        for axi, (key, v) in zip(ax.flatten(), volumes.items()):
            axi.plot(model.results["time"], v)
            axi.set_title(key)
        fig.suptitle("Volumes")
        fig.savefig(outdir / "volumes.png")
        plt.close(fig)

# This function will be used to calculate the pressure in the BiV model, it takes the volume from the circulation model and the time and returns the pressure in the left and right ventricle

def p_BiV_func(V_LV, V_RV, t):
    print("Calculating pressure at time", t)
    value = get_activation(t)
    print("Time", t, "Activation", value)

    logger.debug(f"Time{t} with activation: {value}")
    Ta.assign(value)
    lv_volume.value = V_LV * 1e-6
    rv_volume.value = V_RV * 1e-6
    problem.solve()

    lv_pendo_mmHg = circulation.units.kPa_to_mmHg(problem.cavity_pressures[0].x.array[0] * 1e-3)
    rv_pendo_mmHg = circulation.units.kPa_to_mmHg(problem.cavity_pressures[1].x.array[0] * 1e-3)

    return lv_pendo_mmHg, rv_pendo_mmHg


mL = circulation.units.ureg("mL")
add_units = False
surface_area_lv = geometry.surface_area("ENDO_LV")
lvv_init = (
    geo.mesh.comm.allreduce(geometry.volume("ENDO_LV", u=problem.u), op=MPI.SUM) * 1e6 * 1.0
)  # Increase the volume by 5%
surface_area_rv = geometry.surface_area("ENDO_RV")
rvv_init = (
    geo.mesh.comm.allreduce(geometry.volume("ENDO_RV", u=problem.u), op=MPI.SUM) * 1e6 * 1.0
)  # Increase the volume by 5%
logger.info(f"Initial volume (LV): {lvv_init} mL and (RV): {rvv_init} mL")
init_state = {"V_LV": lvv_initial * 1e6 * mL, "V_RV": rvv_initial * 1e6 * mL}


regazzoni_bleed = circulation.regazzoni2020.Regazzoni2020(
    add_units=add_units,
    callback=callback,
    p_BiV_func=p_BiV_func,
    verbose=True,
    comm=comm,
    outdir=outdir,
    initial_state=init_state,
    parameters=regazzoni_bleed_parmeters,
)
# Set end time for early stopping if running in CI
end_time = 2 * dt if os.getenv("CI") else None
regazzoni_bleed.solve(num_cycles=50, initial_state=init_state, dt=dt, T=end_time, checkpoint=RR)

regazzoni_bleed.print_info()

# Below we plot the pressure volume loop, volumes, pressures and flows for 50 beats

# ```{figure} ../_static/bleeding_biv_pv_loop.png
# ---
# name: bleeding_biv_pv_loop
# ---
# Pressure volume loop for the BiV.
# ```
#
# ```{figure} ../_static/bleeding_biv_volumes.png
# ---
# name: bleeding_biv_volumes
# ---
# Volumes
# ```
#
# ```{figure} ../_static/bleeding_biv_pressure.png
# ---
# name: bleeding_biv_pressure
# ---
# Pressures
# ```
#
# ```{figure} ../_static/bleeding_biv_flow.png
# ---
# name: bleeding_biv_flows
# ---
# Volumes
# ```
#

# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
