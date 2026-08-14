# %% [markdown]
# # Isometric Twitch with the RDQ20-MF Mean-Field Cross-Bridge Model
#
# This demo repeats the [RDQ18 twitch](crossbridge_rdq18.py) with
# [`crossbridge`](https://github.com/ComputationalPhysiology/crossbridge)'s
# `RDQ20MF` model, from Regazzoni, Dedè & Quarteroni
# {cite}`regazzoni2020biophysically`.
#
# ## RDQ18 plus explicit cross-bridge cycling
#
# `RDQ20MF` keeps `RDQ18`'s regulatory-unit (RU) tensor -- the same
# cooperative, nearest-neighbor thin-filament kinetics -- and adds a second,
# explicit layer on top: a mean-field crossbridge (XB) cycle tracking the
# fraction of attached heads in permissive/non-permissive states, with a
# **velocity-dependent detachment rate**. That second layer is what `RDQ18`
# does not have, and it is precisely what gives `RDQ20MF` a non-zero active
# stiffness
#
# $$
# K_a = a_{XB}\,\chi_{SO}(\mathrm{SL})\,[\mu_P^0 + \mu_N^0]
# $$
#
# (Eq. 52 of {cite}`regazzoni2020biophysically`) -- the tissue-level
# stiffness contributed by the population of attached cross-bridges, each
# acting as a linear spring. Where `RDQ18` needed no stabilization because it
# has no strain-rate feedback to destabilize a segregated coupling, `RDQ20MF`
# is the model {class}`pulse.StabilizedActiveStress` was built for: this is
# the regime -- active stiffness exceeding passive stiffness -- the
# stabilization term of {cite}`regazzoni2021oscillation` specifically
# targets.
#
# ## A finer internal time step
#
# `RDQ20MF` resolves the explicit XB cycle, so its internal step (`model.dt`
# $= 2.5\times 10^{-5}$ s, same as `RDQ18`) matters more here: the code below
# still integrates the cell model standalone at that step and only invokes
# the FEM solve every `couple_dt` (1e-2 s), since $\lambda$ does not move
# during an isometric hold and the mechanics only needs $T_a$, $K_a$ at the
# coupling points, not the full-resolution trace.

# %%
from mpi4py import MPI

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from crossbridge import RDQ20MF, calcium_trace

import pulse

# %% [markdown]
# ## 1. The isometric twitch experiment


# %%
def run_isometric_twitch(pre_stretch_mm: float, t_end: float = 0.6, couple_dt: float = 1e-2):
    """
    Runs an isometric twitch on a 10x1x1 mm slab, driven by the RDQ20-MF
    mean-field cross-bridge model.

    Returns the passive fiber stress, and (time, active fiber stress) arrays.
    """
    L = 10.0
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [L, 1.0, 1.0]],
        [10, 2, 2],
    )

    f0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))
    s0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 1.0, 0.0)))

    Ta = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    Ka = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "kPa")

    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    passive_model = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)
    active_model = pulse.StabilizedActiveStress(f0=f0, activation=Ta, active_stiffness=Ka)

    model = pulse.CardiacModel(
        material=passive_model,
        active=active_model,
        compressibility=pulse.Compressible2(),
    )

    boundaries = [
        pulse.Marker(name="X0", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0)),
        pulse.Marker(name="X1", marker=2, dim=2, locator=lambda x: np.isclose(x[0], L)),
    ]
    geo = pulse.Geometry(mesh=mesh, boundaries=boundaries, metadata={"quadrature_degree": 4})

    def dirichlet_bc(V: dolfinx.fem.FunctionSpace) -> list[dolfinx.fem.bcs.DirichletBC]:
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        facets_fixed = geo.facet_tags.find(1)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets_fixed)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0

        facets_stretch = geo.facet_tags.find(2)
        V_x, _ = V.sub(0).collapse()
        dofs_x = dolfinx.fem.locate_dofs_topological((V.sub(0), V_x), 2, facets_stretch)
        u_stretch_x = dolfinx.fem.Function(V_x)
        u_stretch_x.x.array[:] = pre_stretch_mm

        return [
            dolfinx.fem.dirichletbc(u_stretch_x, dofs_x, V.sub(0)),
            dolfinx.fem.dirichletbc(u_fixed, dofs),
        ]

    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))
    parameters = {"mesh_unit": "mm"}
    problem = pulse.StaticProblem(model=model, geometry=geo, bcs=bcs, parameters=parameters)

    Vs = dolfinx.fem.functionspace(mesh, ("DG", 1))
    active_model.lmbda_prev = dolfinx.fem.Function(Vs)
    active_model.lmbda_prev.x.array[:] = 1.0

    # --- Phase 1: passive pre-stretch ---
    problem.solve()

    F = ufl.variable(ufl.grad(problem.u) + ufl.Identity(3))
    f = F * f0
    f_norm = f / ufl.sqrt(ufl.inner(f, f))
    volume = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.det(F) * geo.dx)),
        op=MPI.SUM,
    )
    Tf = dolfinx.fem.form(ufl.inner(model.sigma(F) * f_norm, f_norm) * geo.dx)
    passive_force = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(Tf), op=MPI.SUM) / volume

    # --- Phase 2: standalone cross-bridge twitch at the fixed sarcomere length ---
    cell = RDQ20MF(num_cells=1)
    SL0 = cell.p["SL0"]
    lmbda_pre = 1.0 + pre_stretch_mm / L
    SL_fixed = SL0 * lmbda_pre

    dt_cell = cell.dt
    couple_every = max(1, round(couple_dt / dt_cell))
    n_steps = int(round(t_end / dt_cell))

    times, active_stresses, Ka_trace = [], [], []
    t = 0.0
    for i in range(n_steps):
        Ca = calcium_trace(np.array([t]))[0]
        cell.advance_step(dt_cell, Ca, SL_fixed)
        t += dt_cell

        if i % couple_every == 0:
            ka_val = float(cell.get_active_stiffness()[0])
            Ta.assign(float(cell.get_active_tension()[0]))
            Ka.assign(ka_val)
            problem.solve()
            active_model.update_prev(problem.u)

            total_force = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(Tf), op=MPI.SUM) / volume
            times.append(t)
            active_stresses.append(total_force - passive_force)
            Ka_trace.append(ka_val)

    return passive_force, np.array(times), np.array(active_stresses), np.array(Ka_trace)


# %% [markdown]
# ## 2. Running twitches at increasing pre-stretch

# %%
stretch_amounts = [0.0, 0.5, 1.0, 1.5]  # mm, i.e. 0%, 5%, 10%, 15% strain
strain_pcts = [(s / 10.0) * 100 for s in stretch_amounts]

passive_stresses = []
traces = []

for s, pct in zip(stretch_amounts, strain_pcts):
    p_force, times, active, ka_trace = run_isometric_twitch(s)
    passive_stresses.append(p_force)
    traces.append((times, active))
    print(
        f"stretch {pct:5.1f}%  passive={p_force:8.4f} kPa  "
        f"peak active={active.max():8.4f} kPa  peak Ka={ka_trace.max():10.2f} kPa",
    )

# %% [markdown]
# ## 3. Plotting the results

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for (times, active), pct in zip(traces, strain_pcts):
    ax1.plot(times * 1000, active, linewidth=2, label=f"{pct:.0f}% stretch")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Active fiber stress (kPa)")
ax1.set_title("RDQ20-MF isometric twitch")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

peak_active = [active.max() for _, active in traces]
ax2.plot(
    strain_pcts, peak_active, marker="o", linewidth=2, color="tab:blue", label="Peak active stress",
)
ax2b = ax2.twinx()
ax2b.plot(
    strain_pcts, passive_stresses, marker="^", linewidth=2, color="tab:red", label="Passive stress",
)
ax2.set_xlabel("Stretch (%)")
ax2.set_ylabel("Peak active stress (kPa)", color="tab:blue")
ax2b.set_ylabel("Passive stress (kPa)", color="tab:red")
ax2.set_title("Length-dependent activation (Frank-Starling)")
ax2.grid(True, linestyle="--", alpha=0.6)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Conclusion
#
# `RDQ20MF` reproduces the same Frank-Starling trend as `RDQ18` (they share
# the same RU cooperativity and overlap function), but each twitch above was
# also solved with a large, time-varying active stiffness $K_a$ printed
# alongside the peak active stress -- routinely orders of magnitude larger
# than the tissue's passive stiffness. That the Newton solve converges
# cleanly at every coupling point despite this is exactly
# {class}`pulse.StabilizedActiveStress`'s job: because both faces of this
# slab have prescribed (isometric) displacement, $\lambda$ barely moves
# between coupling points regardless of $T_a$, so the stabilization term
# $K_a(\lambda-\lambda_{prev})$ stays small even while $K_a$ itself is huge
# -- but drop the `update_prev` call after any solve, or let $\lambda$
# actually move (a shortening contraction, not an isometric one), and that
# same $K_a$ is what keeps the coupling from oscillating. See the
# [comparison demo](crossbridge_comparison.py) for how `RDQ20MF`'s twitch
# compares to the other three models tested here.

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
