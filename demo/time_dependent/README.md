# Time-Dependent Simulations

This section covers the time-dependent cardiac mechanics solvers in `fenicsx-pulse`. Unlike static problems where we solve for equilibrium at a single state, here we integrate the equations of motion over time to simulate a full cardiac cycle.

## Mathematical Formulation

The dynamic simulations solve the balance of linear momentum including inertia and damping effects. The governing equations in the reference configuration are:

$$
\rho \ddot{\mathbf{u}} - \nabla \cdot \mathbf{P} = \mathbf{0} \quad \text{in } \Omega_0
$$

subject to appropriate boundary conditions (Dirichlet, Neumann, or Robin).

* $\mathbf{u}$: Displacement field.
* $\mathbf{P}$: First Piola-Kirchhoff stress tensor.
* $\rho$: Mass density.

### Time Integration
To solve this system numerically, we discretize in time using the **Generalized-$\alpha$ method**. This is an implicit, second-order accurate scheme that allows for control over high-frequency numerical dissipation. It solves for the displacement $\mathbf{u}_{n+1}$, velocity $\mathbf{v}_{n+1}$, and acceleration $\mathbf{a}_{n+1}$ at each time step.

## Benchmark Problems (Bestel Model)

These examples implement the cardiac elastodynamics benchmarks described in {cite}`arostica2025software`. They use a simplified analytical model (the **Bestel model**) to drive the cavity pressure and active tension, focusing on the verification of the mechanical solver and the time integration scheme.

* **[LV Benchmark](time_dependent_bestel_lv.py)**:
    Simulates a beating Left Ventricle (LV) ellipsoid. It verifies the implementation of orthotropic passive material properties, time-dependent active stress, viscoelasticity, and dynamic Robin boundary conditions.

* **[BiV Benchmark](time_dependent_bestel_biv.py)**:
    Extends the benchmark to a Bi-Ventricular (BiV) geometry. This involves applying distinct pressure loads to the LV and RV cavities while handling the complex geometry of the septum and free walls.

## Multiscale Coupling (Circulation & Cell Models)

These examples demonstrate a more physiological setup where the 3D mechanics model is coupled to 0D lumped-parameter models for the circulation and cellular electrophysiology. Note also that these examples
uses a quasi-static formulation (neglecting inertia), so no time integration scheme is required.

**Coupling Strategy:**
* **Electrophysiology**: A 0D cell model (e.g., **TorOrd-Land**) computes the intracellular calcium transient and cross-bridge dynamics to determine the active tension $T_a(t)$.
* **Circulation**: A closed-loop 0D circulation model (e.g., **Regazzoni**) computes the flow and volume changes in the cardiovascular system.
* **Mechanics**: The 3D finite element model replaces the 0D ventricular chamber in the circulation loop. It solves for the cavity pressure required to match the volume computed by the circulation model (or vice-versa).

**Demos:**

* **[LV Multiscale Coupling](time_dependent_land_circ_lv.py)**:
    Couples the LV ellipsoid model with the Regazzoni circulation model and TorOrd-Land cell model. It demonstrates a volume-based coupling strategy where the 3D model acts as a "pressure calculator" for the circulation loop.

* **[BiV Multiscale Coupling](time_dependent_land_circ_biv.py)**:
    Extends the multiscale framework to a Bi-Ventricular geometry. The coupling interface handles two separate cavities (LV and RV), exchanging volumes and pressures for both ventricles simultaneously with the circulation model.
