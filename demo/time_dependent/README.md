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
To solve this system numerically, we discretize in time using the **Generalized-$\alpha$ method** {cite}`erlicher2002analysis`. This is an implicit, second-order accurate scheme that allows for control over high-frequency numerical dissipation. It solves for the displacement $\mathbf{u}_{n+1}$, velocity $\mathbf{v}_{n+1}$, and acceleration $\mathbf{a}_{n+1}$ at each time step.

## Benchmark Problems (Bestel Model)

These examples implement the cardiac elastodynamics benchmarks described in {cite}`arostica2025software`. They use a simplified analytical model (the **Bestel model** {cite}`bestel2001biomechanical`) to drive the cavity pressure and active tension, focusing on the verification of the mechanical solver and the time integration scheme.

* **[LV Benchmark](time_dependent_bestel_lv.py)**:
    Simulates a beating Left Ventricle (LV) ellipsoid. It verifies the implementation of orthotropic passive material properties, time-dependent active stress, viscoelasticity, and dynamic Robin boundary conditions.

* **[BiV Benchmark](time_dependent_bestel_biv.py)**:
    Extends the benchmark to a Bi-Ventricular (BiV) geometry. This involves applying distinct pressure loads to the LV and RV cavities while handling the complex geometry of the septum and free walls.

## Coupling to a circulation

In these examples the 3D mechanics model takes the place of the ventricles in a 0D model of the circulation, and the two exchange cavity volumes and pressures. They differ in how the 0D side is advanced:

| Style | How the 0D side is advanced | Demo | Pick it when |
| --- | --- | --- | --- |
| Five-phase cycle | `pulse.cycle.CycleController`, a Windkessel per cavity | [complete_cycle](complete_cycle.py) | you want the classic phase-driven cycle without a closed loop |
| Split loop | any 0D code, called each step | [land_circulation_biv](land_circulation_biv.py) | the circulation is an external solver |
| Monolithic | `GotranxCirculation`, 0D states inside Newton | [monolithic_3d0d](monolithic_3d0d.py), [monolithic_3d0d_biv](monolithic_3d0d_biv.py) | the circuit is a `.ode` file and you want one Newton system |

The closed loop in the last two is the circulation model of Regazzoni et al. {cite}`regazzoni2022cardiac`. Each demo also chooses its own activation: [complete_cycle](complete_cycle.py) and the monolithic demos use the Bestel model {cite}`bestel2001biomechanical`, and [land_circulation_biv](land_circulation_biv.py) runs the Land crossbridge model {cite}`land2017model` at every quadrature point. [complete_cycle](complete_cycle.py) solves the dynamic problem, [land_circulation_biv](land_circulation_biv.py) the quasi-static one, which needs no time integration scheme, and the monolithic demos can run either.

See also the [Isometric Twitch Experiments & the Frank-Starling Mechanism](../crossbridge/README.md) section, which uses the same quasi-static formulation as [land_circulation_biv](land_circulation_biv.py) but focuses on cellular-scale active tension models rather than a full circulation loop.

## References

```{bibliography}
:filter: docname in docnames
```
