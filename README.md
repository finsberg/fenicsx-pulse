![_](docs/pulse-logo.png)

[![MIT](https://img.shields.io/github/license/finsberg/fenicsx-pulse)](https://github.com/finsberg/fenicsx-pulse/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/fenicsx-pulse.svg)](https://pypi.org/project/fenicsx_pulse/)
[![Test package](https://github.com/finsberg/fenicsx-pulse/actions/workflows/test_package_coverage.yml/badge.svg)](https://github.com/finsberg/fenicsx-pulse/actions/workflows/test_package_coverage.yml)
[![Pre-commit](https://github.com/finsberg/fenicsx-pulse/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/finsberg/fenicsx-pulse/actions/workflows/pre-commit.yml)
[![Deploy static content to Pages](https://github.com/finsberg/fenicsx-pulse/actions/workflows/build_docs.yml/badge.svg)](https://github.com/finsberg/fenicsx-pulse/actions/workflows/build_docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Create and publish a Docker image](https://github.com/finsberg/fenicsx-pulse/actions/workflows/docker-image.yml/badge.svg)](https://github.com/finsberg/fenicsx-pulse/pkgs/container/fenicsx_pulse)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/finsberg/a7290de789564f03eb6b1ee122fce423/raw/fenicsx-pulse-coverage.json)](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/finsberg/a7290de789564f03eb6b1ee122fce423/raw/fenicsx-pulse-coverage.json)

# fenicsx-pulse

`fenicsx-pulse` is a cardiac mechanics solver based on FEniCSx. It is a successor of [`pulse`](https://github.com/finsberg/pulse) which is a cardiac mechanics solver based on FEniCS.

* Documentation: https://finsberg.github.io/fenicsx-pulse/
* Source code: https://github.com/finsberg/fenicsx-pulse

## Install
You can install the library with `pip`
```bash
python3 -m pip install fenicsx-pulse
```
or with `conda`
```bash
conda install -c conda-forge fenicsx-pulse
```
Note that installing with `pip` requires [FEniCSx already installed](https://fenicsproject.org/download/)

We also provide a pre-built docker image with FEniCSx and `fenicsx_pulse` installed. You pull this image using the command
```bash
docker pull ghcr.io/finsberg/fenicsx-pulse:v0.7.0
```

## Getting started
Here is a minimal example of how to use `fenicsx-pulse` to solve a simple cardiac mechanics problem.

```python
import numpy as np
import dolfinx
import cardiac_geometries
import pulse

# Create a geometry with cardiac-geometries
geo = cardiac_geometries.mesh.lv_ellipsoid(
    outdir="geometry",
    create_fibers=True,
    fiber_space="Quadrature_6",
)
# Convert the geometry to a pulse.Geometry
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# Create a material model
material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)

# Define model for active contraction
Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
active_model = pulse.ActiveStress(geo.f0, activation=Ta)

# Define model for compressibility
comp_model = pulse.Incompressible()

# Assemble into a cardiac model
model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# Define boundary conditions
traction = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa"
)
neumann = pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])


def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
    # Find facets for the BASE marker
    facets = geo.ffun.find(geo.markers["BASE"][0])
    # Locate degrees of freedom for the x-component (sub(0)) on these facets
    dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), geo.mesh.topology.dim - 1, facets)
    # Return the Dirichlet BC object
    return [dolfinx.fem.dirichletbc(0.0, dofs, V.sub(0))]


robin_epi = pulse.RobinBC(
    value=pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e3)),
        "Pa / m",
    ),
    marker=geometry.markers["EPI"][0],
)
robin_base = pulse.RobinBC(
    value=pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e3)),
        "Pa / m",
    ),
    marker=geometry.markers["BASE"][0],
)

bcs = pulse.BoundaryConditions(neumann=(neumann,), dirichlet=(dirichlet_bc,), robin=(robin_base, robin_epi))

# Create a mechanics problem
problem = pulse.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
)
# Perform an initial solve
problem.solve()

# Create a file for storing the solution
vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, "displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

# Assign a pressure and activation and ramp them up in steps
target_pressure = 5.0  # kPa
target_activation = 5.0  # kPa
num_steps = 5
for i, (pressure, activation) in enumerate(
    zip(np.linspace(0, target_pressure, num_steps), np.linspace(0, target_activation, num_steps))
):
    traction.assign(pressure)  # kPa
    Ta.assign(activation)  # kPa
    problem.solve()
    # Save the displacement field
    vtx.write(i + 1)
vtx.close()
```

![_](https://raw.githubusercontent.com/finsberg/fenicsx-pulse/refs/heads/main/_static/readme.png)

A more realistic example is visualized here:



https://github.com/user-attachments/assets/8e2f5d85-3fbf-4e30-9574-22e7f718230c



Check out [the demos](https://finsberg.github.io/fenicsx-pulse/demo/geometries/unit_cube.html) in the documentation for more examples.

## Using with Claude

This repo ships a [Claude Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) at
[`.claude/skills/fenicsx-pulse/SKILL.md`](.claude/skills/fenicsx-pulse/SKILL.md) that teaches
Claude the `fenicsx-pulse` composition model (`CardiacModel`, `StaticProblem`/`DynamicProblem`,
materials, active contraction, boundary conditions, units, ...) so it can write and review code
against this library more accurately.

To use it in your own project with [Claude Code](https://claude.com/product/claude-code), copy the
skill into your project's `.claude/skills/` directory:

```bash
mkdir -p .claude/skills/fenicsx-pulse
curl -o .claude/skills/fenicsx-pulse/SKILL.md \
    https://raw.githubusercontent.com/finsberg/fenicsx-pulse/main/.claude/skills/fenicsx-pulse/SKILL.md
```

Claude will then automatically load it whenever your prompt or code touches `fenicsx-pulse`/`pulse`.
To install it globally for all your projects, copy it to `~/.claude/skills/fenicsx-pulse/` instead.

## Contributing
See https://finsberg.github.io/fenicsx-pulse/CONTRIBUTING.html
