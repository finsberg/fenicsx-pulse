# %% [markdown]
# # Isometric Twitch with the RDQ18 Regulatory-Unit Model
#
# This demo repeats the [Land (2017) cross-bridge twitch](crossbridge_land2017.py)
# with [`crossbridge`](https://github.com/ComputationalPhysiology/crossbridge)'s
# `RDQ18` model, from Regazzoni, Dedè & Quarteroni {cite}`regazzoni2018active`.
#
# ## A different lineage: regulatory-unit cooperativity, not a fitted cycle
#
# `Land2017`/`Lewalle2024` track a small number of lumped cross-bridge
# populations calibrated against bulk tension measurements. `RDQ18` instead
# derives from a spatially explicit continuous-time Markov chain of
# regulatory units (RUs) along the thin filament, each able to influence its
# neighbors' calcium binding and tropomyosin state -- the mechanism
# generally invoked for cardiac muscle's steep, cooperative
# force-calcium relationship. That $\sim 10^{21}$-state chain is reduced,
# via a closure on nearest-neighbor triplets, to a system of ODEs
# ($\sim$2200 of them) that `crossbridge` integrates directly; length
# dependence enters through an explicit actin-myosin overlap function
# $\chi(\mathrm{SL})$ rather than a fitted gradient.
#
# ## `RDQ18` has no force-velocity effect -- by design
#
# `RDQ18` reports active tension as `Ta_max * compute_permissivity()`: a
# scaling factor times the *fraction of regulatory units in a
# force-permissive state*. Nothing in that quantity, or in the ODEs that
# produce it, depends on shortening velocity $\dot\lambda$ -- which is why
# `RDQ18.get_active_stiffness()` **identically returns zero**. Per its
# docstring, this is a modeling property, not a missing feature: `RDQ18`
# reproduces length-dependent activation but not the Hill force-velocity
# relation, and the corollary is that a segregated (staggered) coupling to
# tissue mechanics is unconditionally stable for this model without
# {class}`pulse.StabilizedActiveStress`'s stabilization term -- the
# instability that class exists to fix ({cite}`regazzoni2021oscillation`) is
# driven entirely by strain-rate feedback that `RDQ18` does not have. We still use
# `StabilizedActiveStress` below, with $K_a \equiv 0$, purely so the same
# coupling code works across all four `crossbridge` demos; it is
# mathematically identical to plain {class}`pulse.ActiveStress` here.
#
# `RDQ18` also differs in how its tension scale is set: `Land2017`,
# `Lewalle2024` and `RDQ20MF` each compute an intrinsic tension from a fixed
# reference parameter (`Tref`/`a_XB`), whereas `RDQ18` exposes `Ta_max`
# directly as a constructor argument -- a free calibration knob rather than
# part of `default_parameters()`.

# %%
from mpi4py import MPI

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from crossbridge import RDQ18, calcium_trace

import pulse

# %% [markdown]
# ## 1. The isometric twitch experiment
#
# `RDQ18` integrates at its own internal time step (`model.dt`, 25
# $\mu$s -- much finer than `Land2017`/`Lewalle2024`'s 1 ms) regardless of
# how often we couple to the FEM solve, so the standalone cell-model loop
# below sub-steps far more often than it reports back to `pulse`.
# `RDQ18` does not define its own reference sarcomere length (`SL0`); we use
# 2.2 $\mu$m, the value used in its own module docstring example, matching
# `RDQ20MF`'s default.


# %%
def run_isometric_twitch(pre_stretch_mm: float, t_end: float = 0.6, couple_dt: float = 1e-2):
    """
    Runs an isometric twitch on a 10x1x1 mm slab, driven by the RDQ18
    regulatory-unit cross-bridge model.

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
    cell = RDQ18(num_cells=1)
    SL0 = cell.p.get("SL0", 2.2)
    lmbda_pre = 1.0 + pre_stretch_mm / L
    SL_fixed = SL0 * lmbda_pre

    dt_cell = cell.dt
    couple_every = max(1, round(couple_dt / dt_cell))
    n_steps = int(round(t_end / dt_cell))

    times, active_stresses = [], []
    t = 0.0
    for i in range(n_steps):
        Ca = calcium_trace(np.array([t]))[0]
        cell.advance_step(dt_cell, Ca, SL_fixed)
        t += dt_cell

        if i % couple_every == 0:
            Ta.assign(float(cell.get_active_tension()[0]))
            Ka.assign(float(cell.get_active_stiffness()[0]))  # always 0.0 for RDQ18
            problem.solve()
            active_model.update_prev(problem.u)

            total_force = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(Tf), op=MPI.SUM) / volume
            times.append(t)
            active_stresses.append(total_force - passive_force)

    return passive_force, np.array(times), np.array(active_stresses)


# %% [markdown]
# ## 2. Running twitches at increasing pre-stretch

# %%
stretch_amounts = [0.0, 0.5, 1.0, 1.5]  # mm, i.e. 0%, 5%, 10%, 15% strain
strain_pcts = [(s / 10.0) * 100 for s in stretch_amounts]

passive_stresses = []
traces = []

for s, pct in zip(stretch_amounts, strain_pcts):
    p_force, times, active = run_isometric_twitch(s)
    passive_stresses.append(p_force)
    traces.append((times, active))
    print(f"stretch {pct:5.1f}%  passive={p_force:8.4f} kPa  peak active={active.max():8.4f} kPa")

# %% [markdown]
# ## 3. Plotting the results

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for (times, active), pct in zip(traces, strain_pcts):
    ax1.plot(times * 1000, active, linewidth=2, label=f"{pct:.0f}% stretch")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Active fiber stress (kPa)")
ax1.set_title("RDQ18 isometric twitch")
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
# `RDQ18` reproduces the Frank-Starling trend through its overlap function
# $\chi(\mathrm{SL})$ and cooperative RU kinetics alone, with $K_a \equiv 0$
# throughout every twitch above -- confirmed by the fact that this demo's
# results would be numerically identical if `active_model` were built as a
# plain {class}`pulse.ActiveStress` instead of
# {class}`pulse.StabilizedActiveStress`. Compare its twitch shape and
# force-calcium steepness against `Land2017`, `Lewalle2024` and `RDQ20MF` in
# the [comparison demo](crossbridge_comparison.py), and reach for `RDQ20MF`
# (below) instead whenever shortening velocity needs to matter.
