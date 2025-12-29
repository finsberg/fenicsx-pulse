# Pre-stressing (Inverse Mechanics)

In computational cardiac mechanics, we typically generate computational meshes from medical images (such as MRI or CT scans). These images acquire the heart geometry at a specific point in the cardiac cycle, often at **End-Diastole (ED)**.

At this stage, the heart is not stress-free; it is loaded by the end-diastolic pressure and potentially by active tension. However, standard finite element mechanics formulations usually assume that the input mesh corresponds to a **stress-free reference configuration**.

Using a loaded geometry as a stress-free reference leads to incorrect stress predictions and unrealistic deformations. To correct this, we must solve an **Inverse Problem** (often called "pre-stressing" or "unloading") to find the unknown stress-free configuration that, when loaded, matches the patient-specific geometry acquired from imaging.

## The Problem Formulation

Let:
* $\Omega_t$: The **Target (Loaded)** configuration. This is the geometry known from medical images.
* $\Omega_0$: The **Reference (Unloaded)** configuration. This is unknown.
* $\mathbf{t}$: The known traction (pressure) applied to the boundary.

We seek a mapping (or a displacement field) that recovers $\Omega_0$ such that solving the forward mechanics problem on $\Omega_0$ with load $\mathbf{t}$ yields $\Omega_t$.

`fenicsx-pulse` provides two distinct methods to solve this problem, demonstrated in the following examples.

## Method 1: The Inverse Elasticity Problem (IEP)

This method formulates the equilibrium equations directly on the known target configuration $\Omega_t$. We solve for an **inverse displacement field** $\mathbf{u}$ defined on the target mesh such that the reference coordinates $\mathbf{X}$ are given by:

$$
\mathbf{X} = \mathbf{x} + \mathbf{u}(\mathbf{x})
$$

where $\mathbf{x} \in \Omega_t$. The deformation gradient $\mathbf{F}$ is computed as the inverse of the gradient of this mapping. This allows us to solve the problem in a single nonlinear solution procedure (typically using load stepping for stability).

* **Class:** `pulse.unloading.PrestressProblem`
* **Demo:** [Pre-stressing a Bi-Ventricular Geometry](prestress_biv.py)
* **Reference:** The method is described in {cite}`barnafi2024reconstructing`.

## Method 2: Fixed-Point Iteration (Backward Displacement)

This method, often called the **Backward Displacement Method**, uses an iterative approach. It repeatedly solves the standard **forward** mechanics problem to update the guess for the reference geometry.

**Algorithm:**
1.  Initialize the reference guess $\Omega_0^0 = \Omega_t$.
2.  For iteration $k = 0, 1, \dots$:
    a. Solve the forward problem on $\Omega_0^k$ applying the known pressure to get displacement $\mathbf{u}_k$.
    b. Update the reference nodes: $\mathbf{X}^{k+1} = \mathbf{x}_{\text{target}} - \mathbf{u}_k$.
    c. Check convergence: $||\mathbf{X}^{k+1} - \mathbf{X}^k|| < \text{tol}$.

This method is intuitive as it reuses the standard forward solver, but it may require multiple iterations to converge.

* **Class:** `pulse.unloading.FixedPointUnloader`
* **Demo:** [Pre-stressing with Fixed-Point Iteration](prestress_fixedpoint_unloader.py)
* **Reference:** The method is described in {cite}`SELLIER20111461`

## Summary of Differences

| Feature | Inverse Elasticity Problem (IEP) | Fixed-Point Iteration |
| :--- | :--- | :--- |
| **Formulation** | Solve equilibrium on $\Omega_t$ | Iterative forward solves on $\Omega_0^k$ |
| **Mesh** | Mesh remains fixed ($\Omega_t$) | Mesh coordinates update every iteration |
| **Computational Cost** | Generally lower (one system solve) | Higher (multiple forward solves) |
| **Implementation** | Requires specific inverse forms | Wraps standard forward solver |
| **Pulse Class** | `PrestressProblem` | `FixedPointUnloader` |


# References
```{bibliography}
:filter: docname in docnames
