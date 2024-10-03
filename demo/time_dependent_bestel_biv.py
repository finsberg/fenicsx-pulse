# # LV ellipsoid with time dependent pressure and activation

# In this example we will solve a time dependent mechanics problem for the left ventricle ellipsoid geometry. The pressure and activation will be time dependent.
# We use the Bestel pressure model and the Bestel activation model

from pathlib import Path
import logging
import fenicsx_pulse.problem
import fenicsx_pulse.viscoelasticity
from mpi4py import MPI
import dolfinx
import math
from dolfinx import log
import numpy as np
import circulation.bestel
from scipy.integrate import solve_ivp
import fenicsx_pulse
import cardiac_geometries
import cardiac_geometries.geometry


logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD


geodir = Path("biv-ellipsoid-bestel")
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.biv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="Quadrature_6",
        comm=comm,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )


geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

# Convert mesh to a realistic size in m
geo.mesh.geometry.x[:] *= 3e-2

geometry = fenicsx_pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

print(geometry.volume("ENDO_LV") * 1e6, geometry.volume("ENDO_RV") * 1e6)

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <fenicsx_pulse.holzapfelogden.HolzapfelOgden>`

material_params = fenicsx_pulse.HolzapfelOgden.orthotropic_parameters()
material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore
# material = fenicsx_pulse.material_models.Guccione(f0=geo.f0, s0=geo.s0, n0=geo.n0)

Ta = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
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

traction_lv = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
neumann_lv = fenicsx_pulse.NeumannBC(traction=traction_lv, marker=geometry.markers["ENDO_LV"][0])

traction_rv = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
neumann_rv = fenicsx_pulse.NeumannBC(traction=traction_rv, marker=geometry.markers["ENDO_RV"][0])


alpha_epi = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e8)), "Pa / m",
)
robin_epi_u = fenicsx_pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
beta_epi = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m",
)
robin_epi_v = fenicsx_pulse.RobinBC(value=beta_epi, marker=geometry.markers["EPI"][0], damping=True)

alpha_base = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)), "Pa / m",
)
robin_base_u = fenicsx_pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])
beta_base = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m",
)
robin_base_v = fenicsx_pulse.RobinBC(value=beta_base, marker=geometry.markers["BASE"][0], damping=True)

bcs = fenicsx_pulse.BoundaryConditions(neumann=(neumann_lv, neumann_rv), robin=(robin_epi_u, robin_epi_v, robin_base_u, robin_base_v))


problem = fenicsx_pulse.problem.DynamicProblem(model=model, geometry=geometry, bcs=bcs)


# Now we can solve the problem

log.set_log_level(log.LogLevel.INFO)
problem.solve()

dt = problem.parameters["dt"].to_base_units()
times = np.arange(0.0, 1.0, dt)

lv_pressure_model = circulation.bestel.BestelPressure(
    parameters=dict(
        t_sys_pre=0.17,
        t_dias_pre=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        alpha_pre=5.0,
        alpha_mid=15.0,
        sigma_pre=12000.0,
        sigma_mid=16000.0,
    ),
)
res_lv = solve_ivp(
    lv_pressure_model,
    [0.0, 1.0],
    [0.0],
    t_eval=times,
    method="Radau",
)
lv_pressure = res_lv.y[0]


rv_pressure_model = circulation.bestel.BestelPressure(
    parameters=dict(
        t_sys_pre=0.17,
        t_dias_pre=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        alpha_pre=1.0,
        alpha_mid=10.0,
        sigma_pre=3000.0,
        sigma_mid=4000.0,
    ),
)
res_rv = solve_ivp(
    rv_pressure_model,
    [0.0, 1.0],
    [0.0],
    t_eval=times,
    method="Radau",
)
rv_pressure = res_rv.y[0]

activation_model = circulation.bestel.BestelActivation()
res = solve_ivp(
    activation_model,
    [0.0, 1.0],
    [0.0],
    t_eval=times,
    method="Radau",
)
# Convert the pressure from Pa to kPa
outdir = Path("biv_ellipsoid_time_dependent_bestel")
activation = res.y[0]
vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "displacement_bench.bp", [problem.u], engine="BP4")
vtx.write(0.0)

volume_form = geometry.volume_form(u=problem.u)

import matplotlib.pyplot as plt

lv_volumes = []
rv_volumes = []


for i, (tai, plv, prv, ti) in enumerate(zip(activation, lv_pressure, rv_pressure, times)):
    print(f"Solving for time {ti}, activation {tai}, lv pressure {plv} and rv pressure {prv}")
    traction_lv.assign(plv)
    traction_rv.assign(prv)
    Ta.assign(tai)
    problem.solve()
    vtx.write((i + 1) * dt)

    lv_volumes.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(volume_form * geometry.ds(geometry.markers["ENDO_LV"][0]))))
    rv_volumes.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(volume_form * geometry.ds(geometry.markers["ENDO_RV"][0]))))

    fig,  ax = plt.subplots(4, 1, figsize=(10, 10))
    ax[0].plot(times[:i + 1], lv_pressure[:i + 1], label="LV")
    ax[0].plot(times[:i + 1], rv_pressure[:i + 1], label="RV")
    ax[0].set_title("Pressure")
    ax[1].plot(times[:i + 1], activation[:i + 1])
    ax[1].set_title("Activation")
    ax[2].plot(times[:i + 1], lv_volumes, label="LV")
    ax[2].plot(times[:i + 1], rv_volumes, label="RV")
    ax[2].set_title("Volume")
    ax[3].plot(lv_volumes, lv_pressure[:i + 1], label="LV")
    ax[3].plot(rv_volumes, rv_pressure[:i + 1], label="RV")

    for a in ax:
        a.legend()
    fig.savefig(outdir / "biv_ellipsoid_time_dependent_bestel.png")
    plt.close(fig)
