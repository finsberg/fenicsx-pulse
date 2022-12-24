[![MIT](https://img.shields.io/github/license/finsberg/pulsex)](https://github.com/finsberg/pulsex/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/pulsex.svg)](https://pypi.org/project/pulsex/)
[![Test package](https://github.com/finsberg/pulsex/actions/workflows/test_package_coverage.yml/badge.svg)](https://github.com/finsberg/pulsex/actions/workflows/test_package_coverage.yml)
[![Pre-commit](https://github.com/finsberg/pulsex/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/finsberg/pulsex/actions/workflows/pre-commit.yml)
[![Deploy static content to Pages](https://github.com/finsberg/pulsex/actions/workflows/build_docs.yml/badge.svg)](https://github.com/finsberg/pulsex/actions/workflows/build_docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Create and publish a Docker image](https://github.com/finsberg/pulsex/actions/workflows/docker-image.yml/badge.svg)](https://github.com/finsberg/pulsex/pkgs/container/pulsex)

# pulsex

`pulsex` is a cardiac mechanics solver based on FEniCSx. It is a successor of [`pulse`](https://github.com/finsberg/pulse) which is a cardiac mechanics solver based on FEniCS.

---

## Notice

**This repo is a complete rewrite of `pulse` to work with FEniCSx. The package is not yet ready for release.**

If you are using FEniCS please check out [`pulse`](https://github.com/finsberg/pulse) instead

---

* Documentation: https://finsberg.github.io/pulsex/
* Source code: https://github.com/finsberg/pulsex

## Install

To install `pulsex` you need to first [install FEniCSx](https://github.com/FEniCS/dolfinx#installation). Next you can install `pulsex` via pip
```
python3 -m pip install pulsex
```
We also provide a pre-built docker image with FEniCSx and `pulsex` installed. You pull this image using the command
```
docker pull ghcr.io/finsberg/pulsex:v0.1.1
```

## Simple Example

```python
import dolfinx
import numpy as np
import pulsex
from mpi4py import MPI
from petsc4py import PETSc

# Create unit cube mesh
mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)

# Specific list of boundary markers
boundaries = [
    pulsex.Marker(marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0)),
    pulsex.Marker(marker=2, dim=2, locator=lambda x: np.isclose(x[0], 1)),
]
# Create geometry
geo = pulsex.Geometry(
    mesh=mesh,
    boundaries=boundaries,
    metadata={"quadrature_degree": 4},
)

# Create passive material model
material_params = pulsex.HolzapfelOgden.transversely_isotropic_parameters()
f0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((1.0, 0.0, 0.0)))
s0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 1.0, 0.0)))
material = pulsex.HolzapfelOgden(f0=f0, s0=s0, **material_params)

# Create model for active contraction
Ta = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
active_model = pulsex.ActiveStress(f0, activation=Ta)

# Create model for compressibility
comp_model = pulsex.Incompressible()

# Create Cardiac Model
model = pulsex.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# Specific dirichlet boundary conditions on the boundary
def dirichlet_bc(
    state_space: dolfinx.fem.FunctionSpace,
) -> list[dolfinx.fem.bcs.DirichletBCMetaClass]:
    V, _ = state_space.sub(0).collapse()
    facets = geo.facet_tags.find(1)  # Specify the marker used on the boundary
    dofs = dolfinx.fem.locate_dofs_topological((state_space.sub(0), V), 2, facets)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.set(0.0)
    return [dolfinx.fem.dirichletbc(u_fixed, dofs, state_space.sub(0))]

# Use a traction on the opposite boundary
traction = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-1.0))
neumann = pulsex.NeumannBC(traction=traction, marker=2)

# Collect all boundary conditions
bcs = pulsex.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

# Create mechanics problem
problem = pulsex.MechanicsProblem(model=model, geometry=geo, bcs=bcs)

# Set a value for the active stress
Ta.value = 2.0

# Solve the problem
problem.solve()

# Get the solution
u, p = problem.state.split()

# And save to XDMF
xdmf = dolfinx.io.XDMFFile(mesh.comm, "results.xdmf", "w")
xdmf.write_mesh(mesh)
xdmf.write_function(u, 0.0)
xdmf.write_function(p, 0.0)
```


## Contributing

TBW
