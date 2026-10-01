# CLI configuration reference

Generated from `pulse.cli.config` by `scripts/gen_cli_reference.py` -- do not edit.
Quantities are strings with units, e.g. `"1 kPa"`.

## `[geometry]`

### FolderGeometry (`folder`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'m'` | Length unit of the mesh coordinates (after `scale`); passed to the problem as mesh_unit |
| `scale` | float | `1.0` | Multiply the mesh coordinates by this factor when loading |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types: cache root, each mesh is cached in its own <hash>/ subfolder |
| `quadrature_degree` | int | `4` | Quadrature degree of the forms |
| `type` | 'folder' | `'folder'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| NoFibers | `FromGeometryFibers(type='from_geometry')` |  |

### BoxGeometry (`box`)

Builtin box; facets tagged X0, X1, Y0, Y1, Z0, Z1 (values 1..6).

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'m'` | Length unit of the mesh coordinates (after `scale`); passed to the problem as mesh_unit |
| `scale` | float | `1.0` | Multiply the mesh coordinates by this factor when loading |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types: cache root, each mesh is cached in its own <hash>/ subfolder |
| `quadrature_degree` | int | `4` | Quadrature degree of the forms |
| `type` | 'box' | `'box'` |  |
| `lx` | float | `1.0` |  |
| `ly` | float | `1.0` |  |
| `lz` | float | `1.0` |  |
| `nx` | int | `3` |  |
| `ny` | int | `3` |  |
| `nz` | int | `3` |  |
| `cell_type` | 'tetrahedron' \| 'hexahedron' | `'tetrahedron'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| NoFibers | `AxisFibers(type='axis', direction='x')` |  |

### LVEllipsoidGeometry (`lv_ellipsoid`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'m'` | Length unit of the mesh coordinates (after `scale`); passed to the problem as mesh_unit |
| `scale` | float | `1.0` | Multiply the mesh coordinates by this factor when loading |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types: cache root, each mesh is cached in its own <hash>/ subfolder |
| `quadrature_degree` | int | `4` | Quadrature degree of the forms |
| `fiber_angle_endo` | float | `60.0` |  |
| `fiber_angle_epi` | float | `-60.0` |  |
| `fiber_space` | str | `'P_1'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| NoFibers | `FromGeometryFibers(type='from_geometry')` |  |
| `type` | 'lv_ellipsoid' | `'lv_ellipsoid'` |  |
| `r_short_endo` | float | `7.0` |  |
| `r_short_epi` | float | `10.0` |  |
| `r_long_endo` | float | `17.0` |  |
| `r_long_epi` | float | `20.0` |  |
| `psize_ref` | float | `3.0` |  |
| `mu_apex_endo` | float | `-3.141592653589793` |  |
| `mu_base_endo` | float | `-1.2722641256100204` |  |
| `mu_apex_epi` | float | `-3.141592653589793` |  |
| `mu_base_epi` | float | `-1.318116071652818` |  |
| `aha` | bool | `False` |  |
| `dmu_factor` | float | `0.25` |  |

### BiVEllipsoidGeometry (`biv_ellipsoid`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'m'` | Length unit of the mesh coordinates (after `scale`); passed to the problem as mesh_unit |
| `scale` | float | `1.0` | Multiply the mesh coordinates by this factor when loading |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types: cache root, each mesh is cached in its own <hash>/ subfolder |
| `quadrature_degree` | int | `4` | Quadrature degree of the forms |
| `fiber_angle_endo` | float | `60.0` |  |
| `fiber_angle_epi` | float | `-60.0` |  |
| `fiber_space` | str | `'P_1'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| NoFibers | `FromGeometryFibers(type='from_geometry')` |  |
| `type` | 'biv_ellipsoid' | `'biv_ellipsoid'` |  |
| `char_length` | float | `0.5` |  |

### CylinderGeometry (`cylinder`)

cardiac_geometries.mesh.cylinder_D_shaped.

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'m'` | Length unit of the mesh coordinates (after `scale`); passed to the problem as mesh_unit |
| `scale` | float | `1.0` | Multiply the mesh coordinates by this factor when loading |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types: cache root, each mesh is cached in its own <hash>/ subfolder |
| `quadrature_degree` | int | `4` | Quadrature degree of the forms |
| `fiber_angle_endo` | float | `60.0` |  |
| `fiber_angle_epi` | float | `-60.0` |  |
| `fiber_space` | str | `'P_1'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| NoFibers | `FromGeometryFibers(type='from_geometry')` |  |
| `type` | 'cylinder' | `'cylinder'` |  |
| `r_inner` | float | `13.0` |  |
| `r_outer` | float | `20.0` |  |
| `height` | float | `40.0` |  |
| `inner_flat_face_distance` | float | `10.0` |  |
| `outer_flat_face_distance` | float | `17.0` |  |
| `char_length` | float | `10.0` |  |

### UKBGeometry (`ukb`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'m'` | Length unit of the mesh coordinates (after `scale`); passed to the problem as mesh_unit |
| `scale` | float | `1.0` | Multiply the mesh coordinates by this factor when loading |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types: cache root, each mesh is cached in its own <hash>/ subfolder |
| `quadrature_degree` | int | `4` | Quadrature degree of the forms |
| `fiber_angle_endo` | float | `60.0` |  |
| `fiber_angle_epi` | float | `-60.0` |  |
| `fiber_space` | str | `'P_1'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| NoFibers | `FromGeometryFibers(type='from_geometry')` |  |
| `type` | 'ukb' | `'ukb'` |  |
| `mode` | int | `-1` |  |
| `std` | float | `1.5` |  |
| `case` | 'ED' \| 'ES' | `'ED'` |  |
| `char_length_max` | float | `5.0` |  |
| `char_length_min` | float | `5.0` |  |
| `clipped` | bool | `False` |  |
| `rotate_base_normal` | list[float] (optional) | – | If set, rotate the mesh so the BASE normal points this way (before caching) |

## `[geometry.fibers]`

### FromGeometryFibers (`from_geometry`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'from_geometry' | `'from_geometry'` |  |

### AxisFibers (`axis`)

Constant fibres along an axis; sheets/normals along the next two axes (cyclically).

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'axis' | `'axis'` |  |
| `direction` | 'x' \| 'y' \| 'z' | `'x'` |  |

### NoFibers (`none`)

No fibre field (isotropic materials only, e.g. neo_hookean or isotropic guccione).

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'none' | `'none'` |  |

## `[material]`

### HolzapfelOgdenMaterial (`holzapfel_ogden`)

| Field | Type | Default | Description |
|---|---|---|---|
| `region` | list[MaterialRegion] | `[]` | Per-cell-marker parameter overrides |
| `type` | 'holzapfel_ogden' | `'holzapfel_ogden'` |  |
| `preset` | 'transversely_isotropic' \| 'partly_orthotropic' \| 'orthotropic' (optional) | `'transversely_isotropic'` | pulse.HolzapfelOgden.<preset>_parameters(); explicit values override it |
| `a` | Quantity (optional) | – |  |
| `b` | float (optional) | – |  |
| `a_f` | Quantity (optional) | – |  |
| `b_f` | float (optional) | – |  |
| `a_s` | Quantity (optional) | – |  |
| `b_s` | float (optional) | – |  |
| `a_fs` | Quantity (optional) | – |  |
| `b_fs` | float (optional) | – |  |

### GuccioneMaterial (`guccione`)

| Field | Type | Default | Description |
|---|---|---|---|
| `region` | list[MaterialRegion] | `[]` | Per-cell-marker parameter overrides |
| `type` | 'guccione' | `'guccione'` |  |
| `C` | Quantity | `'2 kPa'` |  |
| `bf` | float | `8.0` |  |
| `bt` | float | `2.0` |  |
| `bfs` | float | `4.0` |  |

### NeoHookeanMaterial (`neo_hookean`)

| Field | Type | Default | Description |
|---|---|---|---|
| `region` | list[MaterialRegion] | `[]` | Per-cell-marker parameter overrides |
| `type` | 'neo_hookean' | `'neo_hookean'` |  |
| `mu` | Quantity | `'15 kPa'` |  |

### UsykMaterial (`usyk`)

| Field | Type | Default | Description |
|---|---|---|---|
| `region` | list[MaterialRegion] | `[]` | Per-cell-marker parameter overrides |
| `type` | 'usyk' | `'usyk'` |  |
| `C` | Quantity | `'0.88 kPa'` |  |
| `bf` | float | `8.0` |  |
| `bs` | float | `6.0` |  |
| `bn` | float | `3.0` |  |
| `bfs` | float | `12.0` |  |
| `bfn` | float | `3.0` |  |
| `bsn` | float | `3.0` |  |

### SaintVenantKirchhoffMaterial (`saint_venant_kirchhoff`)

| Field | Type | Default | Description |
|---|---|---|---|
| `region` | list[MaterialRegion] | `[]` | Per-cell-marker parameter overrides |
| `type` | 'saint_venant_kirchhoff' | `'saint_venant_kirchhoff'` |  |
| `mu` | Quantity | **required** |  |
| `lmbda` | Quantity | **required** |  |

### MaterialRegion

Per-cell-marker overrides: ``marker`` plus any parameter of the enclosing material.

| Field | Type | Default | Description |
|---|---|---|---|
| `marker` | str | **required** | Cell marker name, or an integer cell tag value (e.g. AHA) |

## `[active]`

### PassiveConfig (`passive`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'passive' | `'passive'` |  |

### ActiveStressConfig (`active_stress`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'active_stress' | `'active_stress'` |  |
| `eta` | float | `0.0` | Transverse fraction of the tension |
| `formulation` | 'invariant' \| 'stretch' | `'invariant'` |  |

## `[compressibility]`

### IncompressibleConfig (`incompressible`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'incompressible' | `'incompressible'` |  |

### CompressibleConfig (`compressible`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'compressible' | `'compressible'` |  |
| `kappa` | Quantity | `'1e6 Pa'` |  |

### Compressible2Config (`compressible2`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'compressible2' | `'compressible2'` |  |
| `kappa` | Quantity | `'1e6 Pa'` |  |

### Compressible3Config (`compressible3`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'compressible3' | `'compressible3'` |  |
| `kappa` | Quantity | `'5e4 Pa'` |  |

## `[viscoelasticity]`

### NoViscoelasticity (`none`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'none' | `'none'` |  |

### ViscousConfig (`viscous`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'viscous' | `'viscous'` |  |
| `eta` | Quantity | `'100 Pa*s'` |  |

## `[bcs]`

### BCsConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `base_bc` | 'fixed' \| 'free' | `'free'` |  |
| `base_marker` | str | `'BASE'` |  |
| `dirichlet` | list[DirichletConfig] | `[]` |  |
| `robin` | list[RobinConfig] | `[]` |  |

### DirichletConfig

Zero displacement on a facet marker (all components, or some: a sliding surface).

| Field | Type | Default | Description |
|---|---|---|---|
| `marker` | str | **required** |  |
| `components` | list['x' \| 'y' \| 'z'] | `['x', 'y', 'z']` |  |

### RobinConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `marker` | str | **required** |  |
| `value` | str | **required** | Stiffness (e.g. '1e3 Pa/m'), or damping ('5e3 Pa*s/m') |
| `damping` | bool | `False` |  |
| `perpendicular` | bool | `False` |  |

## `[[load]]`

### LoadConfig

A prescribed load: a pressure (the Neumann BC on `marker`) or the active tension Ta.

| Field | Type | Default | Description |
|---|---|---|---|
| `target` | 'pressure' \| 'activation' | **required** |  |
| `marker` | str (optional) | – | Facet marker (pressure loads only) |
| `profile` | ConstantProfile \| RampProfile \| TableProfile \| BestelPressureProfile \| BestelActivationProfile | **required** |  |

## `[load.profile]`

### ConstantProfile (`constant`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'constant' | `'constant'` |  |
| `value` | Quantity | **required** |  |

### RampProfile (`ramp`)

`from_value` until `start`, linear until `end`, then `to_value`.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'ramp' | `'ramp'` |  |
| `start` | Quantity | `'0 s'` |  |
| `end` | Quantity | **required** |  |
| `from_value` | Quantity | `'0 kPa'` |  |
| `to_value` | Quantity | **required** |  |

### TableProfile (`table`)

Linear interpolation of a table, held constant outside it; `period` repeats it.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'table' | `'table'` |  |
| `times` | list[float] (optional) | – |  |
| `values` | list[float] (optional) | – |  |
| `file` | path (optional) | – | CSV file (relative to the config) |
| `time_column` | str | `'time'` |  |
| `value_column` | str | `'value'` |  |
| `time_unit` | str | `'s'` |  |
| `value_unit` | str | `'kPa'` |  |
| `period` | Quantity (optional) | – |  |

### BestelPressureProfile (`bestel_pressure`)

| Field | Type | Default | Description |
|---|---|---|---|
| `parameters` | dict[str, str] | `{}` | Overrides of the circulation.bestel defaults, as quantities |
| `type` | 'bestel_pressure' | `'bestel_pressure'` |  |

### BestelActivationProfile (`bestel_activation`)

| Field | Type | Default | Description |
|---|---|---|---|
| `parameters` | dict[str, str] | `{}` | Overrides of the circulation.bestel defaults, as quantities |
| `type` | 'bestel_activation' | `'bestel_activation'` |  |

## `[time]`

### TimeConfig

One time axis: pseudo-time for static problems, physical time for dynamic ones.

| Field | Type | Default | Description |
|---|---|---|---|
| `start_time` | Quantity | `'0 s'` |  |
| `end_time` | Quantity | **required** |  |
| `dt` | Quantity (optional) | – |  |
| `num_steps` | int (optional) | – |  |

## `[problem]`

### ProblemConfig (`static`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'static' \| 'dynamic' | `'static'` |  |
| `u_space` | str | `'P_2'` |  |
| `p_space` | str | `'P_1'` |  |
| `rigid_body_constraint` | bool | `False` |  |
| `rho` | Quantity | `'1000 kg/m**3'` | dynamic only |
| `alpha_m` | float | `0.2` | Generalized-alpha, dynamic only |
| `alpha_f` | float | `0.4` | Generalized-alpha, dynamic only |

## `[solver]`

### SolverConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `max_halvings` | int | `4` | On Newton failure, split the step in two, at most this many times deep |
| `petsc_options` | dict[str, str \| int \| float \| bool] | `{}` | Merged over pulse's defaults |

## `[output]`

### OutputConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `folder` | path | `'output'` |  |
| `save_every` | Quantity (optional) | – | Default: every step |
| `checkpoint_every` | Quantity | `'0 s'` | Restart checkpoint interval; 0 = end only |
| `performance` | bool | `False` | Time Newton solves and runner phases; log every log_every steps and write performance.json |
| `log_every` | int | `10` |  |

## `[postprocess]`

### PostprocessConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `vtx` | bool | `True` |  |
| `fields` | list['fiber_stress' \| 'fiber_strain'] | `[]` |  |
| `points` | dict[str, list[float]] | `{}` | name -> reference coordinates (mesh units, after scale) |
| `vertex_tags` | dict[str, str] | `{}` | name -> vertex marker (e.g. ENDOPT) |
| `plots` | bool | `True` |  |
