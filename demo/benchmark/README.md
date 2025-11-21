# Cardiac Mechanics Benchmark

This folder contains the implementation of the benchmark problems for cardiac mechanics described in the paper:

> Land, S., Gurev, V., Arens, S. et al.  Verification of cardiac mechanics software: benchmark problems and solutions for testing active and passive material behaviour. > Proc. R. Soc. A 471, 20150641 (2015).DOI: 10.1098/rspa.2015.0641

The benchmark suite consists of three problems of increasing complexity designed to verify the implementation of passive and active cardiac mechanics solvers.

## Problems

### Problem 1: Deformation of a Beam
Deflection of a beam made of anisotropic material under a pressure load. This problem tests the implementation of the transversely isotropic constitutive law (Guccione model) and the handling of Neumann boundary conditions (pressure).

### Problem 2: Inflation of a Ventricle
Inflation of an idealized Left Ventricle (truncated ellipsoid) made of isotropic material. This problem tests the handling of curvilinear geometries and large deformations under pressure loading.

### Problem 3: Inflation and Active Contraction
Inflation and active contraction of the idealized Left Ventricle with a spatially varying fiber architecture. This problem tests the coupling between the passive anisotropic material, the active stress model, and the fiber field definition.
