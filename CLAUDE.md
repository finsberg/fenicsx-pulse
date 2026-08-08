# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`fenicsx-pulse` (import name `pulse`) is a cardiac mechanics solver built on FEniCSx/dolfinx. It is the successor to the legacy FEniCS-based [`pulse`](https://github.com/finsberg/pulse) package. The distribution name is `fenicsx-pulse`, but Python code imports it as `pulse`.

There is also a `src/fenicsx_pulse/__init__.py` shim that just re-exports `pulse.*` with a `DeprecationWarning` — kept for backwards compatibility with the old import path. Don't add real logic there.

This package requires FEniCSx/dolfinx, which is not pip-installable on its own — it must come from the `ghcr.io/fenics/dolfinx/dolfinx` container image (see Dockerfile / .devcontainer). Assume dolfinx, mpi4py, petsc4py, ufl, and basix are already present in the environment rather than trying to pip install them.

## Common commands

Run from the repository root.

```bash
# Run the full test suite (matches CI)
python3 -m pytest -m "not benchmark"

# Run a single test file / test
python3 -m pytest tests/test_material_models.py
python3 -m pytest tests/test_material_models.py::test_some_case -v

# Lint / format (ruff is the source of truth; line length 100)
ruff check .
ruff format .

# Type check
mypy

# Run all pre-commit hooks (ruff, mypy, cspell, formatting checks) on the whole repo
pre-commit run --all

# Editable install with dev extras
python3 -m pip install -e .[dev]
```

pytest is configured (in `pyproject.toml`) to always compute coverage (`--cov=pulse`) and run everything under `tests/`. Benchmark tests (`pytest-codspeed`, under the `benchmark` extra) are excluded from the normal CI run via `-m "not benchmark"`.

Demos under `demo/` are Jupytext `.py` percent-format notebooks built into the Sphinx docs (`_toc.yml`); they are not part of the pytest suite and require the `demo`/`docs` extras (cardiac-geometriesx, fenicsx-ldrb, circulation, gotranx, etc.).

## Architecture

A simulation is assembled by composing small, mostly-independent pieces, then handing them to a `Problem` that builds and solves the nonlinear variational form.

**Composition graph:**

```
CardiacModel(material, active, compressibility, viscoelasticity)
        │
        ▼
StaticProblem / DynamicProblem(model, geometry, bcs, parameters)
```

- **`material_model.py`** — `Material` / `HyperElasticMaterial` ABCs. A hyperelastic material only needs to implement `strain_energy(C)`; `P`, `S`, `sigma` are derived automatically via `ufl.diff`. Concrete models live in `material_models/` (Holzapfel-Ogden, Guccione, NeoHookean, SaintVenantKirchhoff, Usyk).
- **`active_model.py` / `active_stress.py`** — active contraction models (`Passive`, `ActiveStress`, `FrankStarlingActiveStress`), same `strain_energy`/`S`/`P` interface as materials, driven by a `Variable` activation parameter you update over time (`Ta.assign(...)`).
- **`compressibility.py`** — `Incompressible`, `Compressible`, `Compressible2`, `Compressible3`: pluggable volumetric penalty/incompressibility handling, again exposing `strain_energy`/`S`/`P`/`is_compressible`.
- **`viscoelasticity.py`** — optional rate-dependent term (`Viscous`, `NoneViscoElasticity`) added on top of the elastic response; contributes to `strain_energy`/`S`/`P` when `C_dot`/`F_dot` are supplied.
- **`cardiac_model.py`** — `CardiacModel` is a frozen dataclass that just sums the contributions of the four components above into total `strain_energy`, `S`, `P`, `sigma`. All four components conform to the `Protocol`s defined at the top of this module — new material/active/compressibility/viscoelasticity implementations only need to satisfy that shape, not inherit from anything.
- **`geometry.py`** — `Geometry` wraps a dolfinx mesh plus facet markers/measures (`dx`, `ds`) built from `Marker` locators (or directly from a `cardiac_geometriesx` object via `Geometry.from_cardiac_geometries`). `HeartGeometry` adds cavity-specific helpers (`volume`, `volume_form`, `base_center`) used for LV/BiV/cavity problems.
- **`boundary_conditions.py`** — `BoundaryConditions` is a `NamedTuple` bundling `neumann`/`dirichlet`/`robin`/`body_force` sequences. `NeumannBC`/`RobinBC` wrap a `Variable` traction/stiffness value; Dirichlet BCs are plain callables `(V) -> [dolfinx.fem.dirichletbc, ...]` so callers keep full control over DOF location logic.
- **`units.py`** — `Variable` pairs a raw value (`float`/`dolfinx.fem.Constant`/`dolfinx.fem.Function`) with a `pint` unit and normalizes it to base SI units (`to_base_units()`); `Variable.assign(value)` mutates the underlying dolfinx object in place. Nearly all physical parameters passed into the model (pressures, activations, stiffnesses) are `Variable`s, not raw floats.
- **`problem.py`** — the biggest module. `StaticProblem` builds the UFL residual from a `CardiacModel` + `Geometry` + `BoundaryConditions`, sets up the (possibly mixed, for incompressible) function space, applies BCs, and drives a Newton solve via `dolfinx.nls.petsc`. `DynamicProblem` extends it with time-dependent/inertial terms. `BaseBC` (`fixed`/`free`) selects how the base is constrained by default. Both classes take a `parameters` dict overriding `default_parameters()`.
- **`unloading.py`** — `PrestressProblem`/`FixedPointUnloader`/`TargetPressure`: iterative schemes to back out an unloaded reference geometry from a loaded one.
- **`kinematics.py` / `invariants.py`** — pure UFL helper functions (deformation gradient, `Cdev`, Piola transforms, invariant computations) used throughout the material/active models; no state, safe to unit-test without a mesh.
- **`exceptions.py`** — package-specific exceptions (e.g. `MarkerNotFoundError`, `MeshTagNotFoundError`) raised by `Geometry`/`HeartGeometry` lookups.
- **`cli.py`** — `pulse` console-script entry point (argparse + rich). Subcommands `run`/`validate-config`/`post` are currently stubs (`return NotImplemented`); `version` is implemented.

When adding a new material, active, compressibility, or viscoelasticity model: match the existing `strain_energy(C)` / `S(C, dev)` / `P(F, dev)` protocol shape in `cardiac_model.py` rather than inheriting a base class, and export it from `material_models/__init__.py` (or the relevant module) plus `pulse/__init__.py`'s `__all__` so it's reachable as `pulse.X`.

## Style notes specific to this repo

- ruff handles both linting and formatting (Black-compatible), line length 100, isort groups `mpi4py`/`petsc4py` into their own `mpi` section before third-party imports — follow that ordering in new files.
- mypy and cspell run as pre-commit hooks; cspell checks `src/`, `docs/`, `tests/`, `README.md` against `.cspell_dict.txt` — add domain-specific terms there rather than rewording code/docs to dodge the spellchecker.
- Prefer `dataclass(slots=True)` for small value objects (see `Variable`, `NeumannBC`, `RobinBC`, `Marker`) consistent with the rest of the codebase.
