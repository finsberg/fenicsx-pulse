# # LV ellipsoid with time dependent pressure and activation

# In this example we will solve a time dependent mechanics problem for the left ventricle ellipsoid geometry.
#
# The equations of motion can be expressed using the following notation in the reference configuration
#
# $$
# \begin{aligned}
#   \rho \ddot{u} - \nabla\cdot P &= 0, \mbox{ in } \Omega ,\\
#   PN &= pJF^{-T}N, \mbox{ on } \Gamma_{\rm endo}, \\
#   PN\cdot N + \alpha_{\rm epi}u\cdot N + \beta_{\rm epi}\dot{u}\cdot N &= 0, \mbox{ on }  \Gamma_{\rm epi}, \\
#   PN\times N &=0, \mbox{ on }  \Gamma_{\rm epi}, \\
#   PN + \alpha_{\rm top}u + \beta_{\rm top}\dot{u} &= 0, \mbox{ on } \Gamma_{\rm top}.
# \end{aligned}
# $$ (dyn_bc)
#
# If we introduce the notation $a= \ddot{u}, v=\dot{u}$ for the acceleration and velocity, respectively, a weak form of {eq}`dyn_bc` can be written as
#
# $$
# \begin{aligned}
# \int_{\Omega} \rho a \cdot w \, \mathop{}\!\mathrm{d}{}X+ \int_{\Omega} P:Grad(w) \, \mathop{}\!\mathrm{d}{}X-\int_{\Gamma_{\rm endo}} p I JF^{-T}N \cdot w \, \mathop{}\!\mathrm{d}{}S\\
# +\int_{\Gamma_{\rm epi}} \big(\alpha_{\rm epi} u \cdot N + \beta_{\rm epi} v \cdot N \big) w \cdot N \, \mathop{}\!\mathrm{d}{}S\\
# +\int_{\Gamma_{\rm top}} \alpha_{\rm top} u \cdot w + \beta_{\rm top} v \cdot w \, \mathop{}\!\mathrm{d}{}S= 0 \quad \forall w \in H^1(\Omega).
# \end{aligned}
# $$ (weak1)
#
# In order to integrate {eq}`weak1` in time, we need to express $a$ and $v$ in terms of the displacement $u$. For this we use the Generalized $\alpha$-method
#
# ## The generalized $\alpha$ method
#
# The generalized $\alpha$ or G-$\alpha$ methods introduce additional parameters
# $\alpha_f$ and $\alpha_m$ to approximate $v$ and $a$ by evaluating the terms of the
# weak form at times $t_{i+1}-\Delta t\alpha_m$ and  $t_{i+1}-\Delta t\alpha_f$.
# Specifically, the inertia term is evaluated at $t_{i+1}-\Delta t\alpha_m$, and the
# other terms at $t_{i+1}-\Delta t\alpha_f$. The weak form becomes
#
# $$
# \begin{aligned}
# \int_{\Omega} \rho a_{i+1-\alpha_m} \cdot w \, \mathrm{d}X + \int_{\Omega} P_{i+1-\alpha_f}:Grad(w) \, \mathrm{d}X -\int_{\Gamma_{\rm endo}} p I JF_{i+1-\alpha_f}^{-T}N \cdot w \, dS  \\
# +\int_{\Gamma_{\rm epi}} \big(\alpha_{\rm epi} u_{i+1-\alpha_f} \cdot N + \beta_{\rm epi} v_{i+1-\alpha_f} \cdot N \big) w \cdot N \, dS  \\
# +\int_{\Gamma_{\rm top}} \alpha_{\rm top} u_{i+1-\alpha_f} \cdot w + \beta_{\rm top} v_{i+1-\alpha_f} \cdot w \, dS = 0 \quad \forall w \in H^1(\Omega),
# \end{aligned}
# $$ (weak3)
#
# with
#
# $$
# \begin{align*}
#   u_{i+1-\alpha_f} &= (1-\alpha_f)u_{i+1}-\alpha_f u_i, \\
#   v_{i+1-\alpha_f} &= (1-\alpha_f)v_{i+1}-\alpha_f v_i, \\
#   a_{i+1-\alpha_m} &= (1-\alpha_m)a_{i+1}-\alpha_m a_i,
# \end{align*}
# $$
#
# $v_{i+1},a_{i+1}$ given by
#
# $$
# v_{i+1} = v_i + (1-\gamma) \Delta t ~ a_i + \gamma \Delta t ~ a_{i+1}
# $$ (N_v)
#
# $$
# a_{i+1} = \frac{u_{i+1} - (u_i + \Delta t ~ v_i + (0.5 - \beta) \Delta t^2 ~ a_i)}{\beta \Delta t^2}
# $$ (N_a)
#
# and
#
# $$
# \begin{align*}
# F_{i+1-\alpha_f} &= I + \nabla u_{i+1-\alpha_f}, \\
# P_{i+1-\alpha_f} &= P(u_{i+1-\alpha_f}, v_{i+1-\alpha_f}).
# \end{align*}
# $$
#
# Different choices of the four parameters $\alpha_m, \alpha_f, \beta, \gamma$ yield
# methods with different accuracy and stability properties. Tables 1--3 in {cite}`erlicher2002analysis`
# provides an overview of parameter choices for methods in the literature,
# as well as conditions for stability and convergence. We have used the choice $\alpha_m =0.2, \alpha_f=0.4$, and
#
# $$
# \begin{align*}
#   \gamma &= 1/2 + \alpha_f-\alpha_m ,\\
#   \beta &= \frac{(\gamma + 1/2)^2}{4} .
# \end{align*}
# $$
#
# For this choice the solver converges through the time interval of interest, and the convergence is second order.
#
# ## Pressure and activation model
#
# We will use a pressure and activation model from {cite}`bestel2001biomechanical` to drive the simulation. We consider a time-dependent pressure derived from the Bestel model. The solution $p = p(t)$  is characterized as solution to the evolution equation
#
# $$
#         \dot{p}(t) = -|b(t)|p(t) + \sigma_{\mathrm{mid}}|b(t)|_+
#         + \sigma_{\mathrm{pre}}|g_{\mathrm{pre}}(t)|
# $$
#
# with $b(\cdot)$ being the activation function described below:
#
# $$
#         b(t) =& a_{\mathrm{pre}}(t) + \alpha_{\mathrm{pre}}g_{\mathrm{pre}}(t)
#         + \alpha_{\mathrm{mid}} \\
#         a_{\mathrm{pre}}(t) :=& \alpha_{\mathrm{max}} \cdot f_{\mathrm{pre}}(t)
#         + \alpha_{\mathrm{min}} \cdot (1 - f_{\mathrm{pre}}(t)) \\
#         f_{\mathrm{pre}}(t) =& S^+(t - t_{\mathrm{sys}-\mathrm{pre}}) \cdot
#          S^-(t  t_{\mathrm{dias} - \mathrm{pre}}) \\
#         g_{\mathrm{pre}}(t) =& S^-(t - t_{\mathrm{dias} - \mathrm{pre}})
# $$
# with $S^{\pm}$ given by
#
# $$
# S^{\pm}(\Delta t) = \frac{1}{2}(1 \pm \mathrm{tanh}(\frac{\Delta t}{\gamma}))
# $$
#
# Similarly, the active stress is characterized through a time-dependent stress function $\tau$ solution to the evolution equation
#
# $$
#         \dot{\tau}(t) = -|a(t)|\tau(t) + \sigma_0|a(t)|_+
# $$
#
# with $a(\cdot)$ being the activation function and \sigma_0 contractility, where each remaining term is described below:
#
# $$
#         |a(t)|_+ =& \mathrm{max}\{a(t), 0\} \\
#         a(t) :=& \alpha_{\mathrm{max}} \cdot f(t)
#         + \alpha_{\mathrm{min}} \cdot (1 - f(t)) \\
#         f(t) =& S^+(t - t_{\mathrm{sys}}) \cdot S^-(t - t_{\mathrm{dias}}) \\
#         S^{\pm}(\Delta t) =& \frac{1}{2}(1 \pm \mathrm{tanh}(\frac{\Delta t}{\gamma}))
# $$
#
# ## Constitutive model
#
# We will use a nearly incompressible, orthotropic and viscoelastic version of the Holzapfel Ogden model. The material parameters are taken from {cite}`holzapfel2009constitutive`. The anistropic material strain energy function is given by
#
# $$
# \Psi_{\text{aniso}} = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
#         + \frac{a_f}{2 b_f} \mathcal{H}(I_{4\mathbf{f}_0} - 1)
#         \left( e^{ b_f (I_{4\mathbf{f}_0} - 1)_+^2} -1 \right)
#         + \frac{a_s}{2 b_s} \mathcal{H}(I_{4\mathbf{s}_0} - 1)
#         \left( e^{ b_s (I_{4\mathbf{s}_0} - 1)_+^2} -1 \right)
#         + \frac{a_{fs}}{2 b_{fs}} \left( e^{ b_{fs}
#         I_{8 \mathbf{f}_0 \mathbf{s}_0}^2} -1 \right)
# $$
#
# The viscoelastic energy is given by
#
# $$
# \Psi_{\text{visco}} = \frac{\eta}{2} \mathrm{tr} ( \dot{\mathbf{E}}^2 )
# $$
#
# where $\dot{\mathbf{E}}$ is temporal derivative the Green-Lagrange strain tensor and $\eta$ is the viscosity parameter. Here the  temporal derivative is computed by first computing the temporal deformation gradient
#
# $$
# \dot{\mathbf{F}} = \dot{\mathbf{F}} = \dot{\mathbf{I} + \nabla \mathbf{u}} = \nabla \dot{\mathbf{u}} = \nabla \mathbf{v}
# $$
#
# and then
#
# $$
# \mathbf{l} = \dot{\mathbf{F}} \mathbf{F}^{-1} \\
# \mathbf{d} = \frac{1}{2} ( \mathbf{l} + \mathbf{l}^T ) \\
# \dot{\mathbf{E}} = \mathbf{F}^T \mathbf{d} \mathbf{F}
# $$
#
# see {cite}`holzapfel2002nonlinear` Equation 2.139 and 2.163 for more details.
#
# For compressibility we use the following strain energy function
#
# $$
# \Psi_{\text{comp}} = \frac{\kappa}{4} \left( J^2 - 1 - 2 \mathrm{ln}(J) \right)
# $$
#
# where $J$ is the determinant of the deformation gradient.
#
# To solve the problem we first import the necessary packages
#

from pathlib import Path
import logging
import math
import os
from mpi4py import MPI
import dolfinx
from dolfinx import log
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import circulation.bestel
import cardiac_geometries
import cardiac_geometries.geometry
import fenicsx_pulse

# Next we set up the logging and the MPI communicator

logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD

# and create an output directory

outdir = Path("time-dependent-bestel-lv")
outdir.mkdir(parents=True, exist_ok=True)

# Next we create the geometry, which is an ellipsoid with a fiber field varying from -60 to 60 degrees in the endocardium and epicardium and interpolated into a Quadrature_6 function space

geodir = outdir / "geometry"
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

# We load the geometry and convert it to the format used by `fenicsx-pulse`.

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)
geometry = fenicsx_pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# Next we create the material object using the class {py:class}`Holzapfel Ogden model <fenicsx_pulse.holzapfelogden.HolzapfelOgden>`

material_params = fenicsx_pulse.HolzapfelOgden.orthotropic_parameters()
material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore
print(material)

# We set up the active stress model

Ta = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta)
print(active_model)

# and define the compressibility

comp_model = fenicsx_pulse.compressibility.Compressible2()
print(comp_model)

# and viscoelasticity model

viscoelastic_model = fenicsx_pulse.viscoelasticity.Viscous()
print(viscoelastic_model)

# Finally we assembles the `CardiacModel`

model = fenicsx_pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
    viscoelasticity=viscoelastic_model,
)

# Next we define the boundary conditions. First we define a traction on the endocardium

traction = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])

# and Robin boundary conditions on the epicardium and base
#

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

# We assemble the boundary conditions
#

bcs = fenicsx_pulse.BoundaryConditions(robin=(robin_epi_u, robin_epi_v, robin_base_u, robin_base_v), neumann=(neumann,))

# Finally we create a `DynamicProblem`

problem = fenicsx_pulse.problem.DynamicProblem(model=model, geometry=geometry, bcs=bcs, parameters={"base_bc": fenicsx_pulse.problem.BaseBC.free})

# Note that we also specify that the base is free to move, meaning that there will be no Dirichlet boundary conditions on the base. Now we can do an initial solve the problem

log.set_log_level(log.LogLevel.INFO)
problem.solve()

# The next step is to get the activation and pressure as a function of time. For this we use the time step from the problem parameters

dt = problem.parameters["dt"].to_base_units()
times = np.arange(0.0, 1.0, dt)

# We solve the Bestel model for the pressure and activation which is already implemented in the [`circulation` package](https://computationalphysiology.github.io/circulation/examples/bestel.html)

pressure_model = circulation.bestel.BestelPressure()
res = solve_ivp(
    pressure_model,
    [0.0, 1.0],
    [0.0],
    t_eval=times,
    method="Radau",
)
# Convert the pressure from Pa to kPa
pressure = res.y[0]

activation_model = circulation.bestel.BestelActivation()
res = solve_ivp(
    activation_model,
    [0.0, 1.0],
    [0.0],
    t_eval=times,
    method="Radau",
)
# Convert the pressure from Pa to kPa
activation = res.y[0]

# Let us also plot the profiles of the pressure and activation

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
ax[0].plot(times, pressure)
ax[0].set_title("Pressure")
ax[1].plot(times, activation)
ax[1].set_title("Activation")
fig.savefig(outdir / "pressure_activation.png")

# New let us write the displacement to a file using the `VTXWriter` from dolfinx
#

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

# We can also calculate the volume of the left ventricle at each time step, so let us first define the volume form
#

volume_form = dolfinx.fem.form(geometry.volume_form(u=problem.u) * geometry.ds(geometry.markers["ENDO"][0]))
initial_volume = geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(volume_form))
print(f"Initial volume: {initial_volume}")

# and then loop over the time steps and solve the problem for each time step

volumes = []
for i, (tai, pi, ti) in enumerate(zip(activation, pressure, times)):
    print(f"Solving for time {ti}, activation {tai}, pressure {pi}")
    traction.assign(pi)
    Ta.assign(tai)
    problem.solve()
    vtx.write((i + 1) * dt)

    volumes.append(geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(volume_form)))

    if geo.mesh.comm.rank == 0:
        # Plot data as we go
        fig, ax = plt.subplots(4, 1, figsize=(10, 10))
        ax[0].plot(times[:i + 1], pressure[:i + 1])
        ax[0].set_title("Pressure")
        ax[1].plot(times[:i + 1], activation[:i + 1])
        ax[1].set_title("Activation")
        ax[2].plot(times[:i + 1], volumes)
        ax[2].set_title("Volume")
        ax[3].plot(volumes, pressure[:i + 1])
        fig.savefig(outdir / "lv_ellipsoid_time_dependent_bestel.png")
        plt.close(fig)

    if os.getenv("CI") and i > 2:
        # Early stopping for CI
        break


# <video controls loop autoplay muted>
#   <source src="../_static/time_dependent_bestel_lv.mp4" type="video/mp4">
#   <p>Video showing the motion of the LV.</p>
# </video>
#
# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
