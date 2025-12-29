# Geometries and Basic Setups

This section introduces various geometrical setups ranging from simple unit cubes to idealized ventricular geometries. These demos illustrate how to define domains, boundary markers, fiber architectures, and boundary conditions for different levels of complexity.

## The Unit Cube
**Demo:** [unit_cube.py](unit_cube.py)

This is the "Hello World" of `fenicsx-pulse`. It uses a simple unit cube geometry $\Omega = [0,1]^3$ to demonstrate the core components of the library without the complexity of curvilinear coordinates or complex fiber fields.

* **Geometry**: Built-in Unit Cube.
* **Physics**: Static inflation and active contraction.
* **Key Concepts**: Defining a `CardiacModel`, applying basic Dirichlet and Neumann boundary conditions, and solving a static problem.

## Idealized Left Ventricle (Ellipsoid)
**Demo:** [lv_ellipsoid.py](lv_ellipsoid.py)

This demo simulates a truncated ellipsoid representing an idealized Left Ventricle (LV). It introduces the use of `cardiac-geometries` to generate anatomical shapes and fiber fields.

* **Geometry**: Truncated prolate spheroid (Endocardium, Epicardium, Base).
* **Fibers**: Analytically defined transmural variation (e.g., $+60^\circ$ to $-60^\circ$).
* **Key Concepts**: Handling curvilinear geometries, applying pressure to internal cavities, and modeling passive inflation followed by active contraction.

## Idealized Bi-Ventricle
**Demo:** [biv_ellipsoid.py](biv_ellipsoid.py)

Extending the LV model, this demo simulates a Bi-Ventricular (BiV) geometry containing both the Left and Right Ventricles. This setup allows for the study of interventricular interactions and septal mechanics.

* **Geometry**: Two joined truncated ellipsoids.
* **Fibers**: Generated using the **Laplace-Dirichlet Rule-Based (LDRB)** algorithm via [`fenicsx-ldrb`](https://github.com/finsberg/fenicsx-ldrb).
* **Key Concepts**: Multi-chamber boundary conditions (distinct LV and RV pressures), pericardial constraints (Robin BCs), and complex fiber generation.

## D-shaped Cylinder
**Demo:** [cylinder.py](cylinder.py)

The D-shaped cylinder is a simplified representation of a ventricle that distinguishes between a curved "free wall" and a flat "septum". This geometry is particularly useful for studying regional mechanics and stress differences between septal and free wall regions in a controlled setting.

* **Geometry**: Extruded D-shape with distinct markers for curved and flat inner/outer surfaces.
* **Simulation**: A dynamic, time-dependent simulation coupled with a closed-loop circulation model (Bestel).
* **Key Concepts**: Regional analysis (post-processing specific subdomains), dynamic solvers, and coupling with 0D circulation models.
