---
name: fenicsx-pulse
description: 'Use when writing, reviewing, or debugging Python code that uses fenicsx-pulse (imported as `pulse`), a FEniCSx/dolfinx-based cardiac mechanics solver — building materials, active contraction, compressibility, boundary conditions, geometries, or Static/DynamicProblem simulations, or when the user mentions pulse.CardiacModel, pulse.StaticProblem, HolzapfelOgden, ActiveStress, or cardiac mechanics with dolfinx.'
license: MIT
---

# fenicsx-pulse

`fenicsx-pulse` (import name `pulse`) is a cardiac mechanics solver built on FEniCSx/dolfinx.
It requires dolfinx, mpi4py, petsc4py, ufl, and basix to already be installed (typically via
the `ghcr.io/fenics/dolfinx/dolfinx` container) — never try to `pip install` those.

## Mental model

A simulation is assembled by composing small, independent pieces and handing them to a `Problem`
that builds and solves the nonlinear variational form:

```
CardiacModel(material, active, compressibility, viscoelasticity)
        │
        ▼
StaticProblem / DynamicProblem(model, geometry, bcs, parameters)
```

Each of `material`, `active`, `compressibility`, `viscoelasticity` independently exposes
`strain_energy(C)` and derived `S`/`P` — `CardiacModel` just sums them. New implementations only
need to match that shape (a `Protocol`), not inherit from a base class.

## Minimal end-to-end pattern

```python
import numpy as np
import dolfinx
import cardiac_geometries
import pulse

geo = cardiac_geometries.mesh.lv_ellipsoid(
    outdir="geometry", create_fibers=True, fiber_space="Quadrature_6",
)
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
active_model = pulse.ActiveStress(geo.f0, activation=Ta)

comp_model = pulse.Incompressible()

model = pulse.CardiacModel(material=material, active=active_model, compressibility=comp_model)

traction = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann = pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])

def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
    facets = geo.ffun.find(geo.markers["BASE"][0])
    dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), geo.mesh.topology.dim - 1, facets)
    return [dolfinx.fem.dirichletbc(0.0, dofs, V.sub(0))]

bcs = pulse.BoundaryConditions(neumann=(neumann,), dirichlet=(dirichlet_bc,))

problem = pulse.StaticProblem(model=model, geometry=geometry, bcs=bcs)
problem.solve()

for pressure, activation in zip(np.linspace(0, 5.0, 5), np.linspace(0, 5.0, 5)):
    traction.assign(pressure)   # kPa
    Ta.assign(activation)       # kPa
    problem.solve()
```

For a runnable, more complete version see the repo README and the demos listed below.

## Component cheat sheet

- **Materials** (`pulse.material_models`, all `HyperElasticMaterial`): `HolzapfelOgden` (has
  `transversely_isotropic_parameters()` / `partly_orthotropic_parameters()` /
  `orthotropic_parameters()` convenience constructors), `Guccione`, `NeoHookean`,
  `SaintVenantKirchhoff`, `Usyk`. A material only needs to implement `strain_energy(C)`; `P`, `S`,
  `sigma` are derived automatically via `ufl.diff`. To add a new one, match this shape and export
  it from `material_models/__init__.py` and `pulse/__init__.py`'s `__all__`.
- **Active contraction** (`pulse.active_model` / `pulse.active_stress`): `Passive` (no active
  stress), `ActiveStress`, `FrankStarlingActiveStress`, `StabilizedActiveStress`. All are driven by
  a `Variable` activation parameter (commonly named `Ta`) updated over time with `Ta.assign(...)`.
  - `ActiveStress(f0, activation=Ta, eta=..., formulation=...)`: `formulation` is
    `ActiveStressFormulation.invariant` (default, `Ψ = ½Ta(I4f−1)`) or `.stretch`
    (`Ψ = Ta(λ−1)`). They differ by a factor of λ — switching is a *modelling* change, not a
    refactor. Use `stretch` only when `Ta` comes from a model whose stiffness is defined against λ;
    it requires `eta == 0` (raises otherwise).
  - `StabilizedActiveStress`: adds the Regazzoni & Quarteroni (2021) stabilization term
    `Ψ = Ta·Δλ + ½Ka·Δλ²`, `Δλ = λ − λ_prev`. **Use this whenever `Ta` comes from an external
    force-generation solver** (e.g. a 0D/circuit model) — the plain staggered scheme is not just
    inaccurate but non-convergent once active stiffness exceeds passive stiffness, and refining
    `dt` makes it worse, not better. Callers **must** call `update_prev(u)` after each solve, using
    the same λ that drove the force-generation model. Governed by
    `tests/test_stabilized_active_stress.py`; if you touch this code keep both load-bearing
    properties it checks: the stabilization must vanish at its own fixed point (biases nothing),
    and the assembled Jacobian must match finite differences *and* change with `Ka` (Newton must be
    able to see the term).
- **Compressibility** (`pulse.compressibility`): `Incompressible` (adds a Lagrange-multiplier
  pressure field), `Compressible`, `Compressible2`, `Compressible3` (differing volumetric penalty
  forms). Same `strain_energy`/`S`/`P`/`is_compressible` shape.
- **Viscoelasticity** (`pulse.viscoelasticity`): `NoneViscoElasticity` (default, no-op) or
  `Viscous`, a rate-dependent term contributing when `C_dot`/`F_dot` are supplied — relevant for
  `DynamicProblem`.
- **Geometry** (`pulse.geometry`): `Geometry` wraps a dolfinx mesh plus facet markers/measures
  (`dx`, `ds`), typically built via `Geometry.from_cardiac_geometries(geo, ...)` from a
  `cardiac_geometriesx` object. `HeartGeometry` adds cavity helpers (`volume`, `volume_form`,
  `base_center`) for LV/BiV/cavity problems. Facet/marker lookups raise `MarkerNotFoundError` /
  `MeshTagNotFoundError` (`pulse.exceptions`) rather than returning `None`.
- **Boundary conditions** (`pulse.boundary_conditions`): `BoundaryConditions` is a `NamedTuple` of
  `neumann` / `dirichlet` / `robin` / `body_force` sequences. `NeumannBC(traction, marker)` and
  `RobinBC(value, marker, damping=False, perpendicular=False)` wrap a `Variable`. Dirichlet BCs are
  plain callables `(V: dolfinx.fem.FunctionSpace) -> list[dolfinx.fem.dirichletbc]` — you own the
  DOF-location logic (see `dirichlet_bc` above), pulse does not infer it for you.
- **Units** (`pulse.units`): almost every physical parameter (pressures, activations, stiffnesses,
  traction, Robin values) is a `Variable(value, unit)` (pint units), not a raw float — e.g.
  `Variable(dolfinx.fem.Constant(mesh, 0.0), "kPa")`. Never pass a bare float where a `Variable` is
  expected without checking the surrounding API — it normalizes to SI base units internally via
  `to_base_units()`. Update a `Variable` in place with `.assign(value)`, in the `Variable`'s own
  declared unit (not necessarily SI) — never mutate `.value` directly.
- **Problems** (`pulse.problem`): `StaticProblem(model, geometry, bcs=..., parameters=...)` builds
  the UFL residual and drives a Newton solve (`dolfinx.nls.petsc`); call `.solve()`. `parameters`
  overrides `default_parameters()` (function-space degrees, `base_bc` — `pulse.BaseBC.fixed` or
  `.free` — PETSc SNES options, `mesh_unit`, `base_marker`, ...). `DynamicProblem` extends it with
  time-dependent/inertial terms and adds `dt`, `rho`, `alpha_m`, `alpha_f` parameters (generalized-α
  time integration); after each solve, advance state before the next step.
- **Unloading** (`pulse.unloading`): `PrestressProblem`, `FixedPointUnloader`, `TargetPressure` —
  iterative schemes to back out an unloaded reference geometry from a loaded (e.g. imaged) one.

## Common pitfalls

- Don't hand-roll SI unit conversions — wrap the value in `pulse.Variable(value, unit)` and let
  pint/`to_base_units()` handle it.
- Picking `ActiveStressFormulation.stretch` with nonzero `eta` raises — set `eta=0` or use
  `.invariant`.
- Driving `Ta` from an external force-generation (e.g. circuit/0D) model with plain `ActiveStress`
  in a staggered time loop can silently fail to converge as active stiffness grows; reach for
  `StabilizedActiveStress` and remember to call `update_prev(u)` every step.
- `Incompressible` compressibility changes the function space (adds a pressure field/Lagrange
  multiplier) — solution vectors, `problem.u`, and post-processing code need to account for the
  mixed space this creates.

## Where to look for more

- `demo/` — Jupytext `.py` percent-format notebooks (geometries, boundary_conditions, howto,
  prestress, time_dependent, benchmark), built into the Sphinx docs at
  https://finsberg.github.io/fenicsx-pulse/. Not run by pytest; require the `demo`/`docs` extras.
- `tests/` — one file per module (`test_material_models.py`, `test_active_model.py`,
  `test_stabilized_active_stress.py`, `test_static_problem.py`, ...); the fastest way to see a
  component's expected inputs/outputs in isolation.
- Full architecture notes: `CLAUDE.md` at the repo root.
