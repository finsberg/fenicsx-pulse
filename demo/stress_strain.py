# # BiV ellipsoid coupled to a 0D circulatory model and a 0D cell model

# This example is similar to the [LV only example](time_dependent_land_circ_lv.py).


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
import ufl
import scifem
import basix
import ldrb

import adios4dolfinx
import cardiac_geometries
import cardiac_geometries.geometry
import pulse

circulation.log.setup_logging(logging.INFO)
logger = logging.getLogger("pulse")
comm = MPI.COMM_WORLD

geodir = Path("biv_ellipsoid")
if not geodir.exists():
    comm.barrier()
    geo = cardiac_geometries.mesh.biv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="Quadrature_6",
        comm=comm,
        char_length=0.5,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )
    # Create longitudinal fibers
    ldrb_output = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=geo.markers,
        fiber_space="Quadrature_6",
        alpha_epi_lv=-90,
        alpha_endo_lv=-90,
        epi_only=True,
    )

    adios4dolfinx.write_function_on_input_mesh(geodir / "long.bp", ldrb_output.f0, time=0.0, name="long")
    with scifem.xdmf.XDMFFile(geodir / "long.xdmf", [ldrb_output.f0]) as xdmf:
        xdmf.write(0.0)

    # Get markers for the LV, RV and septum need DG_0 space to match the cells
    ldrb_output = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=geo.markers,
        fiber_space="DG_0",
    )
    # LV = 1, RV = 2, SEPTUM = 3
    adios4dolfinx.write_function_on_input_mesh(geodir / "markers.bp", ldrb_output.markers_scalar, time=0.0, name="markers")

    with dolfinx.io.VTXWriter(geo.mesh.comm, geodir / "markers-viz.bp", [ldrb_output.markers_scalar], engine="BP4") as vtx:
        vtx.write(0.0)


# If the folder already exist, then we just load the geometry
geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)
geo.mesh.geometry.x[:] *= 3e-2


# Read markers for the LV, RV and septum
V_markers = dolfinx.fem.functionspace(geo.mesh, ("DG", 0))
markers = dolfinx.fem.Function(V_markers, name="markers")
adios4dolfinx.read_function(geodir / "markers.bp", markers, time=0.0)

# And transform the markers to a meshtag
marker_tags = dolfinx.mesh.meshtags(
    geo.mesh,
    3,
    np.arange(*geo.mesh.topology.index_map(3).local_range),
    markers.x.array.astype(np.int32),
)

# Visualize the markers in paraview to make sure they are correct
with dolfinx.io.XDMFFile(geo.mesh.comm, geodir / "markers.xdmf", "w") as xdmf:
    xdmf.write_mesh(geo.mesh)
    xdmf.write_meshtags(marker_tags, geo.mesh.geometry)

# And create a measure for using the markers dx_markers(1) = LV
# dx_markers(2) = RV, dx_markers(3) = SEPTUM
dx_markers = ufl.Measure("dx", domain=geo.mesh, subdomain_data=marker_tags)


geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <pulse.holzapfelogden.HolzapfelOgden>`

material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
# material_params = pulse.HolzapfelOgden.orthotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

# We use an active stress approach with 30% transverse active stress (see {py:meth}`pulse.active_stress.transversely_active_stress`)

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
active_model = pulse.ActiveStress(geo.f0, activation=Ta)

# We use an incompressible model

comp_model = pulse.compressibility.Compressible2()
# and assembles the `CardiacModel`

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

alpha_epi = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e8)), "Pa / m",
)
robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
alpha_base = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)), "Pa / m",
)
robin_base = pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])


lvv_initial = geo.mesh.comm.allreduce(geometry.volume("ENDO_LV"), op=MPI.SUM)
lv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(lvv_initial))
lv_cavity = pulse.problem.Cavity(marker="ENDO_LV", volume=lv_volume)

rvv_initial = geo.mesh.comm.allreduce(geometry.volume("ENDO_RV"), op=MPI.SUM)
rv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(rvv_initial))
rv_cavity = pulse.problem.Cavity(marker="ENDO_RV", volume=rv_volume)

cavities = [lv_cavity, rv_cavity]


parameters = {"base_bc": pulse.problem.BaseBC.free, "mesh_unit": "m"}


outdir = Path("espen-results")
bcs = pulse.BoundaryConditions(robin=(robin_epi, robin_base))
problem = pulse.problem.StaticProblem(model=model, geometry=geometry, bcs=bcs, cavities=cavities, parameters=parameters)

# Create expressions for the stress and strain
quad_element = basix.ufl.quadrature_element(scheme="default", degree=6, cell=geo.mesh.ufl_cell().cellname())
W = dolfinx.fem.functionspace(geometry.mesh, quad_element)
F = ufl.variable(ufl.grad(problem.u) + ufl.Identity(3))
E = 0.5 * (F.T * F - ufl.Identity(3))
fiber_stress = dolfinx.fem.Function(W, name="fiber_stress")
fiber_stress_expr = dolfinx.fem.Expression(
    ufl.inner(material.sigma(F) * geo.f0, geo.f0),
    W.element.interpolation_points(),
)

fiber_strain = dolfinx.fem.Function(W, name="fiber_strain")
fiber_strain_expr = dolfinx.fem.Expression(
    ufl.inner(E * geo.f0, geo.f0),
    W.element.interpolation_points(),
)

# Read in longitudinal fibers from the file
quad_element_vec = basix.ufl.quadrature_element(scheme="default", degree=6, cell=geo.mesh.ufl_cell().cellname(), value_shape=(3,))
W_vec = dolfinx.fem.functionspace(geometry.mesh, quad_element_vec)
l0 = dolfinx.fem.Function(W_vec, name="long")
adios4dolfinx.read_function(geodir / "long.bp", l0, time=0.0, name="long")
long_strain = dolfinx.fem.Function(W, name="long_strain")
long_strain_expr = dolfinx.fem.Expression(
    ufl.inner(E * l0, l0),
    W.element.interpolation_points(),
)

outdir.mkdir(exist_ok=True)

# Now we can solve the problem

log.set_log_level(log.LogLevel.INFO)
problem.solve()

dt = 0.001
times = np.arange(0.0, 1.0, dt)

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

# Time in milliseconds
dt_cell = 0.1

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
    if comm.rank == 0:
        np.save(state_file, y)

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

num_beats = 5
BCL = 1.0
ts = np.arange(0, num_beats * BCL, dt)
all_Ta = np.zeros_like(ts)
for i, t in enumerate(ts):
    t_cell_next = t * 1000
    ode_state.forward(t_cell_next)
    all_Ta[i] = ode_state.Ta(t_cell_next) * 5.0

@lru_cache
def get_activation(t: float):
    return np.interp(t, ts, all_Ta)

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, f"{outdir}/displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

# Create a file for visualizing the stress and strain in Paraview
# Since these are in a quadrature space, we need to use scifem to visualize them
# as point clouds
xdmf_stress_strain = scifem.xdmf.XDMFFile(outdir / "stress_strain.xdmf", [fiber_stress, fiber_strain, long_strain])
filename = Path(outdir / "function_checkpoint.bp")
adios4dolfinx.write_mesh(filename, geometry.mesh)

Ta_history = []

# Compute volume of the different regions which will be used to compute averages
mesh_volume = comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(dolfinx.fem.Constant(geo.mesh, 1.0) * geometry.dx),
    ), op=MPI.SUM,
)
lv_mesh_volume = comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(dolfinx.fem.Constant(geo.mesh, 1.0) * dx_markers(1)),
    ), op=MPI.SUM,
)
rv_mesh_volume = comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(dolfinx.fem.Constant(geo.mesh, 1.0) * dx_markers(2)),
    ), op=MPI.SUM,
)
septum_mesh_volume = comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(dolfinx.fem.Constant(geo.mesh, 1.0) * dx_markers(3)),
    ), op=MPI.SUM,
)
fiber_stress_average = []
fiber_stress_average_form = dolfinx.fem.form(fiber_stress / mesh_volume * geometry.dx)
fiber_strain_average = []
fiber_strain_average_form = dolfinx.fem.form(fiber_strain / mesh_volume * geometry.dx)
long_strain_average = []
long_strain_average_form = dolfinx.fem.form(long_strain / mesh_volume * geometry.dx)

fiber_stress_lv = []
fiber_stress_rv = []
fiber_stress_septum = []
fiber_stress_lv_form = dolfinx.fem.form(fiber_stress / lv_mesh_volume * dx_markers(1))
fiber_stress_rv_form = dolfinx.fem.form(fiber_stress / rv_mesh_volume * dx_markers(2))
fiber_stress_septum_form = dolfinx.fem.form(fiber_stress / septum_mesh_volume * dx_markers(3))

fiber_strain_lv = []
fiber_strain_rv = []
fiber_strain_septum = []
fiber_strain_lv_form = dolfinx.fem.form(fiber_strain / lv_mesh_volume * dx_markers(1))
fiber_strain_rv_form = dolfinx.fem.form(fiber_strain / rv_mesh_volume * dx_markers(2))
fiber_strain_septum_form = dolfinx.fem.form(fiber_strain / septum_mesh_volume * dx_markers(3))

long_strain_lv = []
long_strain_rv = []
long_strain_septum = []
long_strain_lv_form = dolfinx.fem.form(long_strain / lv_mesh_volume * dx_markers(1))
long_strain_rv_form = dolfinx.fem.form(long_strain / rv_mesh_volume * dx_markers(2))
long_strain_septum_form = dolfinx.fem.form(long_strain / septum_mesh_volume * dx_markers(3))



def callback(model, i: int, t: float, save=True):
    Ta_history.append(get_activation(t))
    if save:
        fiber_strain.interpolate(fiber_strain_expr)
        fiber_stress.interpolate(fiber_stress_expr)
        long_strain.interpolate(long_strain_expr)
        xdmf_stress_strain.write(t)
        adios4dolfinx.write_function(filename, problem.u, time=t, name="displacement")
        vtx.write(t)

        fiber_stress_average.append(comm.allreduce(dolfinx.fem.assemble_scalar(fiber_stress_average_form), op=MPI.SUM))
        fiber_strain_average.append(comm.allreduce(dolfinx.fem.assemble_scalar(fiber_strain_average_form), op=MPI.SUM))
        long_strain_average.append(comm.allreduce(dolfinx.fem.assemble_scalar(long_strain_average_form), op=MPI.SUM))
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        ax[0].plot(ts[:i + 1], fiber_stress_average, label="Fiber stress average")
        ax[0].set_xlabel("Time [s]")
        ax[0].set_title("Fiber stress [Pa]")

        ax[1].plot(ts[:i + 1], fiber_strain_average, label="Fiber strain average")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_title("Fiber strain")

        ax[2].plot(ts[:i + 1], long_strain_average, label="Longitudinal strain average")
        ax[2].set_xlabel("Time [s]")
        ax[2].set_title("Longitudinal strain")

        if comm.rank == 0:
            fig.savefig(outdir / "fiber_stress_strain.png")
        plt.close(fig)

        fiber_stress_lv.append(comm.allreduce(dolfinx.fem.assemble_scalar(fiber_stress_lv_form), op=MPI.SUM))
        fiber_stress_rv.append(comm.allreduce(dolfinx.fem.assemble_scalar(fiber_stress_rv_form), op=MPI.SUM))
        fiber_stress_septum.append(comm.allreduce(dolfinx.fem.assemble_scalar(fiber_stress_septum_form), op=MPI.SUM))
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        ax[0].plot(ts[:i + 1], fiber_stress_lv, label="Fiber stress LV")
        ax[0].set_xlabel("Time [s]")
        ax[0].set_title("Fiber stress (LV) [Pa]")

        ax[1].plot(ts[:i + 1], fiber_stress_rv, label="Fiber stress RV")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_title("Fiber stress (RV) [Pa]")

        ax[2].plot(ts[:i + 1], fiber_stress_septum, label="Fiber stress SEPTUM")
        ax[2].set_xlabel("Time [s]")
        ax[2].set_title("Fiber stress (SEPTUM) [Pa]")

        if comm.rank == 0:
            fig.savefig(outdir / "fiber_stress.png")
        plt.close(fig)

        fiber_strain_lv.append(comm.allreduce(dolfinx.fem.assemble_scalar(fiber_strain_lv_form), op=MPI.SUM))
        fiber_strain_rv.append(comm.allreduce(dolfinx.fem.assemble_scalar(fiber_strain_rv_form), op=MPI.SUM))
        fiber_strain_septum.append(comm.allreduce(dolfinx.fem.assemble_scalar(fiber_strain_septum_form), op=MPI.SUM))
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        ax[0].plot(ts[:i + 1], fiber_strain_lv, label="Fiber strain LV")
        ax[0].set_xlabel("Time [s]")
        ax[0].set_title("Fiber strain (LV)")
        ax[1].plot(ts[:i + 1], fiber_strain_rv, label="Fiber strain RV")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_title("Fiber strain (RV)")
        ax[2].plot(ts[:i + 1], fiber_strain_septum, label="Fiber strain SEPTUM")
        ax[2].set_xlabel("Time [s]")
        ax[2].set_title("Fiber strain (SEPTUM)")

        if comm.rank == 0:
            fig.savefig(outdir / "fiber_strain.png")
        plt.close(fig)

        long_strain_lv.append(comm.allreduce(dolfinx.fem.assemble_scalar(long_strain_lv_form), op=MPI.SUM))
        long_strain_rv.append(comm.allreduce(dolfinx.fem.assemble_scalar(long_strain_rv_form), op=MPI.SUM))
        long_strain_septum.append(comm.allreduce(dolfinx.fem.assemble_scalar(long_strain_septum_form), op=MPI.SUM))
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        ax[0].plot(ts[:i + 1], long_strain_lv, label="Longitudinal strain LV")
        ax[0].set_xlabel("Time [s]")
        ax[0].set_title("Longitudinal strain (LV)")
        ax[1].plot(ts[:i + 1], long_strain_rv, label="Longitudinal strain RV")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_title("Longitudinal strain (RV)")
        ax[1].legend()
        ax[2].plot(ts[:i + 1], long_strain_septum, label="Longitudinal strain SEPTUM")
        ax[2].set_xlabel("Time [s]")
        ax[2].set_title("Longitudinal strain (SEPTUM)")
        ax[2].legend()
        if comm.rank == 0:
            fig.savefig(outdir / "longitudinal_strain.png")
        plt.close(fig)
        import json

        if comm.rank == 0:
            Path(f"{outdir}/stress_strain.json").write_text(
                json.dumps({
                    "time": ts[:i + 1].tolist(),
                    "fiber_stress_average": fiber_stress_average,
                    "fiber_strain_average": fiber_strain_average,
                    "long_strain_average": long_strain_average,
                    "fiber_stress_lv": fiber_stress_lv,
                    "fiber_stress_rv": fiber_stress_rv,
                    "fiber_stress_septum": fiber_stress_septum,
                    "fiber_strain_lv": fiber_strain_lv,
                    "fiber_strain_rv": fiber_strain_rv,
                    "fiber_strain_septum": fiber_strain_septum,
                    "long_strain_lv": long_strain_lv,
                    "long_strain_rv": long_strain_rv,
                    "long_strain_septum": long_strain_septum,
                }),
            )

        fig = plt.figure(layout="constrained", figsize=(12, 8))
        gs = GridSpec(3, 4, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[:, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 2])
        ax5 = fig.add_subplot(gs[0, 3])
        ax6 = fig.add_subplot(gs[1, 3])
        ax7 = fig.add_subplot(gs[2, 2:])

        ax1.plot(model.history["V_LV"][:i+1], model.history["p_LV"][:i+1])
        ax1.set_xlabel("LVV [mL]")
        ax1.set_ylabel("LVP [mmHg]")

        ax2.plot(model.history["V_RV"][:i+1], model.history["p_RV"][:i+1])
        ax2.set_xlabel("RVV [mL]")
        ax2.set_ylabel("RVP [mmHg]")

        ax3.plot(model.history["time"][:i+1], model.history["p_LV"][:i+1])
        ax3.set_ylabel("LVP [mmHg]")
        ax4.plot(model.history["time"][:i+1], model.history["V_LV"][:i+1])
        ax4.set_ylabel("LVV [mL]")

        ax5.plot(model.history["time"][:i+1], model.history["p_RV"][:i+1])
        ax5.set_ylabel("RVP [mmHg]")
        ax6.plot(model.history["time"][:i+1], model.history["V_RV"][:i+1])
        ax6.set_ylabel("RVV [mL]")

        ax7.plot(model.history["time"][:i+1], Ta_history[:i+1])
        ax7.set_ylabel("Ta [kPa]")

        for axi in [ax3, ax4, ax5, ax6, ax7]:
            axi.set_xlabel("Time [s]")

        if comm.rank == 0:
            fig.savefig(outdir / "pv_loop_incremental.png")
        plt.close(fig)


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
lvv_init = geo.mesh.comm.allreduce(geometry.volume("ENDO_LV", u=problem.u), op=MPI.SUM) * 1e6 * 1.0  # Increase the volume by 5%
rvv_init = geo.mesh.comm.allreduce(geometry.volume("ENDO_RV", u=problem.u), op=MPI.SUM) * 1e6 * 1.0  # Increase the volume by 5%
logger.info(f"Initial volume (LV): {lvv_init} mL and (RV): {rvv_init} mL")
init_state = {"V_LV": lvv_initial * 1e6 * mL, "V_RV": rvv_initial * 1e6 * mL}


circulation_model_3D = circulation.regazzoni2020.Regazzoni2020(
    add_units=add_units,
    callback=callback,
    p_BiV=p_BiV_func,
    verbose=True,
    comm=comm,
    outdir=outdir,
    initial_state=init_state,
)
# Set end time for early stopping if running in CI
end_time = 2 * dt if os.getenv("CI") else None
circulation_model_3D.solve(num_beats=num_beats, initial_state=init_state, dt=dt, T=end_time)
circulation_model_3D.print_info()
vtx.close()
xdmf_stress_strain.close()


# ```{figure} ../_static/pv_loop_time_dependent_land_circ_biv.png
# ---
# name: pv_loop_time_dependent_land_circ_biv
# ---
# Pressure volume loop for the BiV.
# ```
#
# <video controls loop autoplay muted>
#   <source src="../_static/time_dependent_land_circ_biv.mp4" type="video/mp4">
#   <p>Video showing the motion of the LV.</p>
# </video>
#
# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
