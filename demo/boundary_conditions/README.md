# Boundary Conditions in fenicsx-pulse

Defining appropriate boundary conditions is essential for setting up a well-posed
mechanics problem. In `fenicsx-pulse`, boundary conditions are collected in a
`pulse.BoundaryConditions` object and passed to the problem solver.

The library supports several types of boundary conditions, ranging from standard
Dirichlet and Neumann conditions to more complex Robin conditions and global
cavity volume constraints.

```python
import dolfinx
import pulse
import ufl
from mpi4py import MPI
```

```python
# (Mock setup for demonstration purposes)
mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
geo = pulse.Geometry(mesh=mesh)
# Assume we have markers defined in geo.markers, e.g., "ENDO", "EPI", "BASE"
```


## 1. Dirichlet Boundary Conditions

Dirichlet boundary conditions constrain the value of the primary unknown
(displacement) on a specific part of the domain.

In `fenicsx-pulse`, Dirichlet BCs are defined as **functions** (callables).
This is necessary because the underlying function space is created inside the
`Problem` class, so the user cannot create the `dolfinx.fem.dirichletbc` object
directly beforehand. Instead, you provide a function that takes the function
space `V` as input and returns a list of Dirichlet BCs.


```python
def dirichlet_bc_example(V: dolfinx.fem.FunctionSpace) -> list[dolfinx.fem.DirichletBC]:
    """
    Example function to fix the displacement on a specific boundary.
    """
    # 1. Locate degrees of freedom on the boundary.
    #    Assumes 'geo' is available in scope or passed in.
    #    Here we assume marker ID 1 is the boundary to fix.
    marker_id = 1
    facets = geo.facet_tags.find(marker_id)
    dofs = dolfinx.fem.locate_dofs_topological(V, geo.facet_dimension, facets)

    # 2. Define the fixed value (e.g., zero displacement).
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.array[:] = 0.0

    # 3. Create and return the BC object.
    return [dolfinx.fem.dirichletbc(u_fixed, dofs)]
```

## 2. Neumann Boundary Conditions (Pressure)

Neumann boundary conditions in this package primarily represent a
**follower pressure load**. This means the force acts normal to the
**deformed** surface area.

The total force vector $\mathbf{t}$ on a surface element $ds$ is given by:

$$
\mathbf{t} = -p \, J \, \mathbf{F}^{-T} \mathbf{N}
$$

where $p$ is the scalar traction magnitude (pressure), and the term
$J \mathbf{F}^{-T} \mathbf{N}$ represents the area-weighted normal in the current configuration.

```python
# Define the pressure magnitude (can be a Constant or Function).
# It is recommended to wrap it in `pulse.Variable` for unit handling.
pressure = pulse.Variable(dolfinx.fem.Constant(mesh, 1.0), "kPa")
```

```python
# Define the BC on a specific surface marker (e.g., marker 2).
neumann_bc = pulse.NeumannBC(traction=pressure, marker=2)
```


## 3. Robin Boundary Conditions (Springs & Dashpots)

Robin boundary conditions apply a force proportional to the displacement (spring)
or velocity (dashpot/damping). These are often used to model the pericardium
or surrounding tissue support in cardiac mechanics.

The general form for a spring is:

$$
\mathbf{P} \mathbf{N} + k (\mathbf{u} \cdot \mathbf{n}) \mathbf{n} = 0
$$

where $k$ is the stiffness and $\mathbf{n}$ is the normal vector.

Configuration Options:
- `damping`: If True, force is proportional to velocity (viscous damper).
- `perpendicular`: If True, force acts in the tangential plane.
                   If False (default), it acts in the normal direction.
- `normal`: `"current"` acts along the normal $\mathbf{n}$ of the deformed configuration;
            `"reference"` acts along the reference normal $\mathbf{N}$, as in Pfaller et al.
            (2019), which assumes small rotations. Unset (default), springs use `"current"`
            and dampers `"reference"`.

A normal spring on a surface whose normal displacement is already fixed by a Dirichlet
condition (for example a sliding base with $u_x = 0$) barely acts. Use
`perpendicular=True` there to resist the in-plane motion instead. A spring that acts in
all directions is a normal and a perpendicular `RobinBC` with the same stiffness on the
same marker.

```python
# Define stiffness
stiffness = pulse.Variable(dolfinx.fem.Constant(mesh, 1e3), "Pa/m")
```

```python
# Create the Robin BC (e.g., on marker 3)
robin_bc = pulse.RobinBC(value=stiffness, marker=3)
```


## 4. Pericardium (Sliding Springs)

A `RobinBC` on the epicardium measures the gap to the pericardium by projecting
$\mathbf{u}$ onto a normal. On a curved surface, the epicardium sliding along itself then
counts as a normal gap, of second order in the rotation, so stiff springs hold back the
twist of the ventricle.

`PericardiumBC` measures the gap as the change in signed distance $d$ to a surface fixed
in space,

$$
g = d(\mathbf{X} + \mathbf{u}) - d(\mathbf{X}),
$$

and stores the energy $\frac{1}{2} \int k g^2 \, dA$. Its traction
$k g \nabla d(\mathbf{x})$ acts along the normal of the surface, so the epicardium slides
along it without friction, and its Jacobian is symmetric. The springs are at rest in the
reference configuration.

- `surface`: an object with a `signed_distance(x)` method returning a UFL expression,
  positive outside. `pulse.EllipsoidSurface` is an ellipsoid of revolution;
  `EllipsoidSurface.from_cardiac_geometries(geo)` reads the epicardial radii of a
  cardiac-geometriesx `lv_ellipsoid`.
- `damping`: an optional dashpot $c \dot{g} \nabla d$, used by `DynamicProblem`.
- `unilateral`: act only where the epicardium is outside its reference distance ($g > 0$).

```python
# surface = pulse.EllipsoidSurface.from_cardiac_geometries(geo)
surface = pulse.EllipsoidSurface(r_long=0.097, r_short=0.035)  # long axis x, centre 0
pericardium = pulse.PericardiumBC(
    stiffness=pulse.Variable(dolfinx.fem.Constant(mesh, 1e7), "Pa / m"),
    marker=3,
    surface=surface,
)
```

See [](pericardium.py) for a comparison with `RobinBC` on an LV ellipsoid.


## 5. Body Forces

Volumetric forces, such as gravity, can be applied to the entire domain.

```python
gravity = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0, 0, -9.81)))
```


## 6. The BoundaryConditions Container

Finally, all defined conditions are collected into the container.

```python
bcs = pulse.BoundaryConditions(
    dirichlet=(dirichlet_bc_example,),
    neumann=(neumann_bc,),
    robin=(robin_bc,),
    body_force=(gravity,),
    pericardium=(pericardium,),
)
```


## 7. Cavity Volume Constraint

Instead of specifying a known pressure (Neumann BC), you can enforce a specific
**cavity volume**. The solver then treats the cavity pressure as a Lagrange
multiplier (unknown) that adjusts to satisfy the volume constraint.

This is technically a global constraint coupled with the boundary, but it is
handled via the `cavities` argument in the problem solver.

```python
# Target volume
target_vol = dolfinx.fem.Constant(mesh, 150.0)
```

```python
# Define the cavity constraint. 'marker' is the name of the boundary (e.g. "ENDO").
# Note: The geometry object must have this string marker mapped to an ID.
cavity = pulse.problem.Cavity(marker="ENDO", volume=target_vol)
```

Example usage in a problem (commented out):

```python
problem = pulse.StaticProblem(..., bcs=bcs, cavities=[cavity])
```

The solver will compute the pressure required to maintain `target_vol`.
You can access this computed pressure via `problem.cavity_pressures`.



## 8. Helper Parameters

The `StaticProblem` and `DynamicProblem` classes accept a `parameters` dictionary
that can activate predefined boundary behaviors.


### The BaseBC Parameter
For cardiac geometries, handling the basal plane is a common requirement.

- `pulse.BaseBC.fixed`: Automatically applies a Dirichlet BC fixing all
  displacement components (u=0) on the boundary marked as "BASE".
- `pulse.BaseBC.free`: Does not apply any automatic Dirichlet condition to the base. This is for example useful when you want to apply a pure Robin boundary condition.
  You are free to apply your own Dirichlet or Robin conditions manually.


### Rigid Body Constraint
If your problem does not have enough Dirichlet conditions to prevent rigid body
motion (translation/rotation), the solver will fail to converge.

- `rigid_body_constraint=True`: Adds Lagrange multipliers to constrain the
  6 rigid body modes (3 translations, 3 rotations). This is useful for "floating"
  models where you want to study deformation without fixing a specific boundary.

```python
parameters = {"base_bc": pulse.BaseBC.fixed, "rigid_body_constraint": False}
```
