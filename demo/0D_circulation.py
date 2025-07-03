# # 0D Circulation Model with Regazzoni2020
# This script demonstrates how to set up and solve a 0D circulation model using the Regazzoni2020 model
# from the `circulation` module in the `pulse` package.

from mpi4py import MPI
import dolfinx
import numpy as np
import pulse
import circulation
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logging.getLogger("scifem").setLevel(logging.WARNING)

comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_square(comm, 5, 5)
circulation_model = circulation.regazzoni2020.Regazzoni2020()
y0 = circulation_model.state.copy()
dt = 0.001
theta = 0.5

problem = pulse.problem.StaticProblem(
    model=pulse.CardiacModel(),
    geometry=pulse.Geometry(mesh=domain),
    circulation_model=circulation_model,
    parameters={"0D": True, "dt": dt, "theta": theta},
)

time = np.arange(0, 10, dt)
y = np.zeros((len(y0), len(time)))
y[:, 0] = y0

for i, ti in enumerate(time[1:]):
    if i % 100 == 0:
        print(f"Solving for time {ti:.3f} s")
    problem.solve(ti)
    y[:, i + 1] = problem.circ.x.array[:]

state_names = circulation_model.state_names()
var_names = circulation_model.var_names()
vars = circulation_model.update_static_variables(time, y)

fig, ax = plt.subplots(2, 2, sharex="col", sharey="col", figsize=(12, 8))
ax[0, 0].set_title("Pressures")
ax[0, 0].plot(time, vars[var_names.index("p_LV"), :], label="p_LV")
ax[0, 0].plot(time, vars[var_names.index("p_LA"), :], label="p_LA")
ax[0, 0].plot(time, y[state_names.index("p_AR_SYS"), :], label="p_AR_SYS")
ax[0, 0].plot(time, vars[var_names.index("p_RV"), :], label="p_RV")
ax[0, 0].plot(time, vars[var_names.index("p_RA"), :], label="p_RA")
ax[0, 0].legend()

ax[1, 0].set_title("Volumes")
ax[1, 0].plot(time, y[state_names.index("V_LA"), :], label="V_LA")
ax[1, 0].plot(time, y[state_names.index("V_LV"), :], label="V_LV")
ax[1, 0].plot(time, y[state_names.index("V_RV"), :], label="V_RV")
ax[1, 0].plot(time, y[state_names.index("V_RA"), :], label="V_RA")
ax[1, 0].legend()

ax[0, 1].set_title("LV PV Loop")
ax[0, 1].plot(y[state_names.index("V_LV"), :], vars[var_names.index("p_LV"), :])

ax[1, 1].set_title("RV PV Loop")
ax[1, 1].plot(y[state_names.index("V_RV"), :], vars[var_names.index("p_RV"), :])

for axi in ax.flatten():
    axi.grid()
fig.savefig("circulation_regazzoni2020.png")
