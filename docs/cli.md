# Command line interface

`fenicsx-pulse` ships a `pulse` command line tool that runs a static or quasi-static/dynamic
cardiac mechanics simulation (and its postprocessing) purely from a `config.toml` file, so routine
runs — including batch/array jobs on an HPC cluster — don't need a Python script. This page covers
installation, the workflow, the commands and override mechanisms, the output folder layout, and
the built-in templates. See also [Running on a cluster](cli_cluster.md) and the generated
[configuration reference](cli_reference.md).

```{note}
The CLI's v1 scope is static/quasi-static loading and prescribed-activation time stepping
(`[active] type = "active_stress"` driven by a `[[load]] target = "activation"` profile, or
by an injected active model). Circulation coupling, prestress/unloading, gotranx/crossbridge
activation and non-zero Dirichlet values are not expressible via `config.toml` yet — see the
[templates table](#templates) below for the exact deviation from each demo it transcribes.
```

## Installation

The CLI's dependencies (`pydantic`, `pydantic-pint`, `cardiac-geometriesx`, `io4dolfinx`, ...) are
bundled in the `cli` extra:

```bash
pip install "fenicsx-pulse[cli]"
```

Two groups of templates need extra, optional packages, each checked (with an install hint) only
when a config actually needs it:

- A `bestel_pressure`/`bestel_activation` load profile (`bestel_lv`, `bestel_biv`,
  `cylinder_bestel`) needs `pip install circulation scipy`.
- A `biv_ellipsoid`/`ukb` geometry needs `fenicsx-ldrb` for its fibres (`biv_ellipsoid`,
  `ukb_bcs`); `ukb` additionally needs `ukb-atlas` to fetch the atlas mesh itself.

## Quick start

```bash
pulse init case/config.toml --template lv_ellipsoid   # write a starter config
pulse validate-config case/config.toml                 # parse + validate, print the resolved config
pulse geometry case/config.toml                         # generate/cache the mesh, then stop
pulse run case/config.toml                              # run the simulation
pulse post case/config.toml                              # VTX, derived fields, plots, point traces
```

`pulse init --template NAME` copies `NAME`'s `config.toml` (and any companion files) from the
package's built-in templates into place — see the [templates table](#templates) for the full
list, one per row. Without `--template`, it defaults to `lv_ellipsoid`.

`pulse geometry` (also run implicitly by `pulse run` and `pulse post`) generates a mesh once and
reuses it on every later invocation for a generated geometry type (`lv_ellipsoid`,
`biv_ellipsoid`, `cylinder`, `ukb`): it is cached in its own subfolder `geometry.folder/<hash>/`,
keyed on a hash of the `[geometry]` section (everything except `folder`, `unit`, `scale` and
`quadrature_degree`). Changing a geometry parameter therefore creates a *new* subfolder next to
the old one rather than replacing it, and `pulse` never deletes `geometry.folder` itself or
anything in it that it didn't create — `geometry.folder = "."` (the config's own directory) is
safe. Run `pulse geometry` on its own first if you just want to warm that cache (e.g. on a cluster
login node, see [Running on a cluster](cli_cluster.md)); it logs the subfolder it used. The mesh
(and its fibres) is generated in serial on rank 0, whatever the number of ranks, and then read and
distributed by every rank — cardiac-geometriesx's generators are not reliably parallel-safe.

`geometry.type = "folder"` points `folder` at a mesh you (or another `cardiac-geometriesx`
-compatible tool) already generated externally; `pulse` only *loads* it, via
`cardiac_geometries.geometry.Geometry.from_folder`. `geometry.type = "box"` is the other
dependency-free case: a built-in `dolfinx.mesh.create_box`, facets tagged `X0, X1, Y0, Y1, Z0, Z1`,
nothing written to disk.

## Commands

`validate-config`, `geometry`, `run` and `post` all take a config path and accept repeatable
`--set KEY=VALUE` overrides. `init` takes an optional config path (default `config.toml`) but no
`--set` (there's no existing config to override yet). `version` takes neither — it only prints
version numbers. The global `-v/--verbose`, `--log-all-cpus` and `--dry-run` flags are accepted by
every subcommand, either before or after the subcommand name: `pulse -v run config.toml` and
`pulse run config.toml -v` are equivalent.

| Command | Behaviour |
|---|---|
| `pulse init [config.toml] [--template NAME] [--force]` | Write a starter config from a template (default: `lv_ellipsoid`); `--force` overwrites existing files. |
| `pulse validate-config config.toml` | Parse + validate + print the resolved config. Builds nothing (no mesh, no MPI-collective work). |
| `pulse geometry config.toml` | Generate (or load) the geometry and stop. For a generated type the mesh is cached in `geometry.folder/<hash>/` and reused across the other commands and reruns. |
| `pulse run config.toml [--restart] [--overwrite] [--output-folder P] [--petsc-options "..."]` | Run the simulation. |
| `pulse post config.toml [--output-folder P]` | VTX, derived fields, plots and point traces from `results.bp`. |
| `pulse version` | Versions of fenicsx-pulse, dolfinx, mpi4py, petsc4py. |

`pulse --dry-run <command> ...` (e.g. `pulse --dry-run run config.toml --set 'time.dt="1 ms"'`)
logs the raw, argparse-parsed arguments and exits — it does **not** load the TOML file or resolve
`--set`/env overrides, so it can't catch a config or override mistake, only confirm which flags
argparse itself accepted. To actually check that a config (with its overrides applied) parses and
validates, use `pulse validate-config` instead, which does load and resolve everything and prints
the fully resolved result.

(exit-codes)=
### Exit codes

- `0` success.
- `1` a configuration/validation error (`ConfigError`), a command-line usage error (unknown flag,
  missing argument), or a missing `cli` extra (the error names the `pip install
  "fenicsx-pulse[cli]"` fix).
- `2` a runtime failure: a solver failure (Newton did not converge within `solver.max_halvings`),
  or any other unexpected error (mesh generation, I/O, ...). It is logged as a one-line error; run
  with `-v` for the full traceback.

`pulse run` builds the whole simulation — geometry, material, boundary conditions, loads, problem
— before it touches the output folder, so a failure during that setup leaves the output folder
exactly as it was (no `run.json`, and with `--overwrite` the previous results are *not* deleted).
Only once the run has started does a failure set `run.json: status = "failed"` (with the error
message). A marker typo ([[load]], `[bcs]`) can only be checked once the mesh has actually been
built or loaded, since marker names are a property of that mesh — so it's reported early in
`pulse run`, right after the geometry step, but not before it, and before `--overwrite` deletes
anything.

For a **generated** geometry (`lv_ellipsoid`/`biv_ellipsoid`/`cylinder`/`ukb`), building the mesh
itself can be by far the most expensive part of that early setup — run `pulse validate-config`
(parse-time checks only, no mesh) and then `pulse geometry` (builds/caches the mesh once, cheaply
reused by every later command) before submitting a long or queued job, so that if a marker mistake
*is* still there, `pulse run` fails within seconds against the already-cached mesh, rather than
after regenerating it inside the timed job.

(overrides)=
## Overrides

Config values can come from four places, in order of increasing precedence:

1. The TOML file itself.
2. Environment variables: `PULSE_<SECTION>__<KEY>`, e.g. `PULSE_TIME__DT="1 ms"`. The `PULSE_`
   prefix and `__` (double underscore) nesting delimiter are fixed, but the section/key names
   themselves are matched **case-insensitively** against the config schema.
3. `--set dotted.key=value` (repeatable). `value` is parsed as a TOML literal, the same way it
   would appear on the right-hand side of a `key = value` line in the file:
   - `--set 'time.dt="1 ms"'` (a quantity is still a quoted string)
   - `--set material.mu="20 kPa"`
   - `--set 'postprocess.points.apex=[0,0,-0.097]'`
   - List elements are addressed **by index**: `--set 'load.0.profile.to_value="20 kPa"'` sets
     the first `[[load]]` table's `profile.to_value`.
   - Unknown keys are an error (`ConfigError`), never silently dropped.
4. Dedicated flags on `pulse run`/`pulse post`: `--output-folder` and (on `pulse run`)
   `--petsc-options`.

`--output-folder` overrides `output.folder` and, unlike every path *inside* the config file,
resolves against the **current working directory** rather than the config file's directory —
handy for array jobs launched from one shared directory (see
[Running on a cluster](cli_cluster.md)).

`--petsc-options "-ksp_type cg -pc_type hypre"` merges into `solver.petsc_options` (parsed with
`shlex`, each `-key value` pair; a bare `-flag` becomes `True`). Negative numbers are accepted as
option *values*, not mistaken for the next flag, e.g. `--petsc-options "-ksp_rtol -1e-6"`.

Relative paths written *inside* the config file (`geometry.folder` for `type = "folder"`,
`output.folder`, a `[load.profile] file`) resolve against **the config file's own directory**, not
the current working directory — so `pulse run /abs/path/to/config.toml` from anywhere still finds
its companion files, which matters once a cluster job script `cd`s elsewhere before running `srun
pulse run ...`.

Whatever the config resolves to after all four layers, it's written out in full to
`output/config.resolved.toml` at the start of every `pulse run` — the single source of truth for
"what actually ran".

## Config file

Every section is a TOML table; every physical quantity is a string with units (see
[Units](#units)). The full field-by-field listing, generated from the pydantic models, is the
[configuration reference](cli_reference.md); this section shows one short snippet per section, in
the order they're resolved.

### `[geometry]`

```toml
[geometry]
type = "lv_ellipsoid"   # folder | box | lv_ellipsoid | biv_ellipsoid | cylinder | ukb
unit = "mm"              # length unit of the mesh coordinates
quadrature_degree = 4
fiber_space = "P_2"
[geometry.fibers]
type = "from_geometry"   # from_geometry | axis | none
```

### `[material]`

```toml
[material]
type = "holzapfel_ogden"   # holzapfel_ogden | guccione | neo_hookean | usyk | saint_venant_kirchhoff
preset = "transversely_isotropic"
```

`[[material.region]]` overrides one or more parameters on a cell marker (or an integer cell tag):

```toml
[[material.region]]
marker = "10"
a = "22.8 kPa"
```

### `[active]`

```toml
[active]                 # omitted ⇒ passive
type = "active_stress"   # passive | active_stress
eta = 0.3
```

### `[compressibility]`

```toml
[compressibility]
type = "incompressible"   # incompressible | compressible | compressible2 | compressible3
```

### `[viscoelasticity]`

```toml
[viscoelasticity]
type = "none"   # none | viscous
```

`viscous` (like a Robin condition with `damping = true`) only acts on velocities, so it is
rejected unless `problem.type = "dynamic"`.

### `[bcs]`

```toml
[bcs]
base_bc = "fixed"      # fixed | free
base_marker = "BASE"
[[bcs.robin]]
marker = "EPI"
value = "1e3 Pa/m"
```

### `[[load]]`

A pressure load is the Neumann boundary condition on `marker`; an activation load drives `Ta` in
an `active_stress` model. Both are a `profile` evaluated at the current time:

```toml
[[load]]
target = "pressure"   # pressure | activation
marker = "ENDO"        # pressure only
[load.profile]
type = "ramp"           # constant | ramp | table | bestel_pressure | bestel_activation
start = "0 s"
end = "1 s"
to_value = "15 kPa"
```

### `[time]`

```toml
[time]
start_time = "0 s"
end_time = "2 s"
num_steps = 20   # exactly one of dt / num_steps
```

There is one time axis for every run: pseudo-time for a static problem (each step is an
equilibrium solve, not physical time), physical time for a dynamic one. Phases (e.g. "ramp the
pressure, then ramp the activation") are expressed as separate loads whose `ramp`/`table` windows
occupy different parts of that one axis — see the `lv_ellipsoid` template above. `num_steps` and
`dt` both describe the same axis; whichever is given, the other is derived
(`dt = (end_time - start_time) / num_steps`). With `dt` given, `end_time - start_time` must be an
integer multiple of it, and `output.save_every`/`checkpoint_every` (when non-zero) must be integer
multiples of the *effective* `dt` (all to a 1e-9 relative tolerance) — otherwise the config is
rejected instead of silently rounding the run length or the output grid.

### `[problem]`

```toml
[problem]
type = "static"   # static | dynamic
u_space = "P_2"
p_space = "P_1"
```

### `[solver]`

```toml
[solver]
max_halvings = 4
```

### `[output]`

```toml
[output]
folder = "output"
save_every = "10 ms"
checkpoint_every = "0 s"
performance = false
```

### `[postprocess]`

```toml
[postprocess]
vtx = true
fields = ["fiber_stress", "fiber_strain"]
points = { apex = [0.0, 0.0, -0.097] }
```

## Outputs

`pulse run` writes into `output.folder` (default `output`, relative to the config file):

```text
output/
  config.resolved.toml   # the fully resolved configuration of the (latest) run
  run.json                # versions, n_ranks, start/end wall time, status: running/finished/failed
  output.log              # log file (output_all_cpus.log too when running on >1 rank)
  results.bp               # io4dolfinx: u (+ p if incompressible), every output.save_every
  loads.csv                 # t [s], every load [Pa], volume_<marker> [m^3] for cavity markers
  restart.bp                 # io4dolfinx: the mechanics_* state functions
  restart.json                # the latest complete checkpoint's time/step and a physics hash
  performance.json             # timing summary (only with output.performance = true)
  post/                         # written by `pulse post`, see below
```

`loads.csv` has one row per saved time: `t` in seconds, every `[[load]]`'s value in pascal
(`pressure_<marker>` or `activation`), and `volume_<marker>` in cubic metres for every cavity
marker (`ENDO`, `LV`, `RV`) that carries a pressure load — summed over MPI ranks.

`pulse run` never writes VTX itself — only the io4dolfinx files above. `pulse post config.toml`
reads `results.bp` (which can happen later, on any number of ranks, independent of how many ranks
the run itself used) and writes into `post/`. The config given to `pulse post` must describe the
same physics as the run that wrote `results.bp` — the same check as for `--restart` (below),
against the hash in `restart.json`, or, if the run stopped before writing its first checkpoint,
against `config.resolved.toml`. Only `[output]`, `[postprocess]` and the run length may differ;
anything else (e.g. an edited `geometry.nx`) is refused with a `ConfigError` naming
`config.resolved.toml` to compare with, rather than crashing or silently producing wrong results.

```text
output/post/
  displacement.bp   # results.bp converted to VTX (ParaView), if postprocess.vtx (default true)
  fields.bp           # derived DG1 fields (fiber_stress, fiber_strain), if postprocess.fields
  points.csv            # u at postprocess.points / vertex_tags, every saved time
  loads.png              # loads [kPa] and cavity volumes [mL] vs t, if matplotlib is installed
  pv_loop_<marker>.png    # pressure-volume loop per cavity marker with both a pressure and a volume
```

On more than one rank, the plots are produced on rank 0 only, contained so a plotting failure
never deadlocks the other ranks (a warning is logged and the remaining postprocessing steps still
run); the VTX files are always complete.

### `--overwrite` and `--restart`

Re-running into a non-empty output folder (e.g. an array-job index collision, or simply rerunning
by hand) is refused by default — `pulse` never silently deletes anything:

- `--overwrite` deletes *only the artifacts `pulse` itself wrote* (everything listed above, plus
  `post/`) and starts fresh. It does so only *after* the new config has been fully validated (the
  simulation is built first), so a mistake in the config never costs the previous results.
  Anything else in that folder — notably your own `config.toml`, if the output folder happens to
  be the config's own directory — is left untouched.
- `--restart` continues from `restart.json`/`restart.bp` instead. It refuses if the run's
  **physics** has changed since the checkpoint was written: the check is a hash of the whole
  resolved config *excluding* the run length (`time.end_time`/`num_steps` — the effective `dt` and
  `start_time` are hashed instead, so changing `num_steps` without changing `dt` is still caught)
  and excluding `[output]`, `[postprocess]` and `[solver]` entirely (`max_halvings` and
  `petsc_options` only change *how* a step is solved, not the physics). `geometry.folder` only matters for
  `geometry.type = "folder"` (where it *is* the mesh being simulated); for every generated
  geometry type it's just a cache location and is excluded like any other non-physics path.
  Restarting on a **different number of MPI ranks** than the original run is allowed. If the run
  stopped before its first checkpoint there is no `restart.json` yet, and both `--restart` and a
  plain rerun are refused with a message saying so — use `--overwrite` to start over.
- A restart never rewrites a `results.bp` timestamp that's already there: io4dolfinx *appends* a
  duplicate write at an existing timestamp, and its reader returns the *first* match, so
  re-writing the same time would be silently ignored on read anyway — the runner simply skips it.

## Newton failures

Each step calls `problem.solve()`. If Newton fails to converge, the step is halved and retried
(`_advance(t, dt/2, level+1)` twice), recursively, up to `solver.max_halvings` levels deep (default
4). If the deepest halving still fails, `pulse run` raises `SolverFailure` (exit code 2) and the
problem — every state function, old-state function, and for a dynamic problem `v_old`/`a_old` and
the `dt` `Constant` — is restored to exactly where it was before the failed `step()` call, even
when some of the halves had already converged. The last checkpoint on disk is therefore always
intact and consistent. Since `[solver]` is not part of the physics hash, you may continue with
`--restart` and a larger `solver.max_halvings` (e.g. `--set solver.max_halvings=8`) or different
`--petsc-options`. A smaller `time.dt` or a changed load (e.g. a gentler ramp) changes the physics:
the restart is refused, so rerun with `--overwrite` (or into a new output folder) instead.

(performance)=
## Performance

Set `[output] performance = true` to have `pulse run` build a
`pulse.PerformanceMonitor(log_frequency=output.log_every)` and hand it to the simulation. Every
`log_every` steps it logs a line with the Newton and KSP iteration counts of the latest step, the
number of halvings so far, and the accumulated wall-clock time (rank 0's) spent in `step`,
`newton_solve`, `update_fields`, `save`, `checkpoint`, `volumes` and `loads`. At the end of the run it logs a
summary table and writes `performance.json` (an `--overwrite` artifact) with the same totals.

From Python, pass the same monitor to `build_simulation`/`StaticProblem`/`DynamicProblem`
directly:

```python
from pulse import PerformanceMonitor
from pulse.cli.overrides import load_config
from pulse.cli.runner import build_simulation

conf = load_config("config.toml")
monitor = PerformanceMonitor(log_frequency=10)
sim = build_simulation(conf, monitor=monitor)
```

## Using pulse from Python

`pulse run`/`pulse post` are thin wrappers around a step API any Python script (or a coupled
driver, e.g. simcardemsx) can call directly:

```python
from pulse.cli.overrides import load_config
from pulse.cli.runner import build_simulation

conf = load_config("config.toml")
sim = build_simulation(conf)  # geometry=..., active_model=... may be injected
sim.start()  # creates the output folder and writes the loads.csv header
dt = conf.time.dt_s()
for _ in range(conf.time.n_steps()):
    sim.step(dt)
    sim.save()
sim.checkpoint()
```

`build_simulation` accepts `geometry=` and `active_model=` to reuse an already-built geometry or
swap `[active]` for an externally driven active model (an injected active model and a
`[[load]] target = "activation"` are mutually exclusive — only one may drive `Ta`). `geometry=`
takes a `pulse.cli.geometry.CLIGeometry`, as returned by
`pulse.cli.geometry.build_geometry(conf.geometry)` (the mesh plus fibres and cell/vertex tags), not
a bare `pulse.HeartGeometry`; the simulation keeps it as `sim.geo`, with `sim.geo.geometry` the
`pulse.HeartGeometry` and `sim.geo.mesh` the mesh. The config's markers are checked against it
like against a built geometry.

(units)=
## Units

Every physical quantity in the config is a pint string, `"<value> <unit>"` (e.g.
`dt = "1 ms"`, `mu = "15 kPa"`) — a bare number is rejected with a validation error naming the
field. Internally, time is always SI seconds and every load/pressure is SI pascal; `loads.csv`
and `config.resolved.toml` are written in those units too (`geometry.unit`/`scale` only affect the
mesh coordinates, via `mesh_unit`, not the quantities above).

(templates)=
## Templates

Each template under `src/pulse/cli/templates/<name>/` is a runnable `config.toml` reproducing one
of the `demo/` scripts as closely as the CLI schema allows; `pulse init --template NAME` copies it
into place. Where a demo does something the schema can't express yet, the template's header
comment documents the deviation — summarized here:

| Template | Demo | Needs beyond `.[cli,test]` | Known deviation |
|---|---|---|---|
| `unit_cube` | `demo/geometries/unit_cube.py` | — | Pressure and activation applied together in one pseudo-time step (matches the demo's single combined solve). |
| `benchmark1` | `demo/benchmark/problem1.py` | — | Pressure steps are a 5-row table over pseudo-time 1..5 s (the same five values the demo loops over). |
| `benchmark2` | `demo/benchmark/problem2.py` | — | A 10-step ramp to 10 kPa with adaptive halving replaces the demo's own continuation scheme. |
| `benchmark3` | `demo/benchmark/problem3.py` | — | 19 pseudo-time steps ramp pressure and Ta together instead of the demo's 20 linspace points. |
| `lv_ellipsoid` | `demo/geometries/lv_ellipsoid.py` | — | The demo's single pressure step, then single activation step, become two one-step ramps over pseudo-time `[0, 1]` s and `[1, 2]` s. |
| `lv_sliding_base` | `demo/boundary_conditions/lv_ellipsoid_fixed_x.py` | — | Pressure then Ta are ramps over pseudo-time `[0, 1]` s and `[1, 2]` s instead of two single steps. |
| `spatial_material` | `demo/howto/spatial_material.py` | — | The 10x-stiffer AHA segment 10 is a `[[material.region]]` directly on the cell tag, instead of the demo's hand-built 0/1 region function. |
| `biv_ellipsoid` | `demo/geometries/biv_ellipsoid.py` | `fenicsx-ldrb` | Fibres come from `cardiac_geometries`' `create_fibers` (60/-60, `P_2`) instead of the demo's explicit `ldrb` call with zero sheet angles. |
| `ukb_bcs` | `demo/boundary_conditions/ukb_bcs.py` | `ukb-atlas`, `fenicsx-ldrb` | Pressures and Ta are tables instead of the demo's explicit Python loop; needs network access on first run (atlas download, cached afterwards). |
| `cylinder_bestel` | `demo/geometries/cylinder.py` | `circulation`, `scipy` | 100 quasi-static steps of 10 ms to `t = 1.0 s`; the demo's regional stress/strain plots are replaced by `pulse post`'s derived fields. |
| `bestel_lv` | `demo/time_dependent/time_dependent_bestel_lv.py` | `circulation`, `scipy` | 1000 steps of 1 ms to `t = 1.0 s`, saved every 10 ms; the load trajectory leads the demo's by about 1-2 steps (the CLI evaluates and labels loads at each step's end time, not its start). |
| `bestel_biv` | `demo/time_dependent/time_dependent_bestel_biv.py` | `circulation`, `scipy`, `fenicsx-ldrb` | Same 1000-step/1 ms schedule and load-timing offset as `bestel_lv`; cavity volumes in `loads.csv` are allreduced over ranks (the demo's are rank-local). |

## See also

- [Configuration reference](cli_reference.md) — every section/field, generated from the pydantic
  models, so it can't drift from the code.
- [Running on a cluster](cli_cluster.md) — SLURM array jobs, wall-time/`--restart` patterns, and
  solver advice for large meshes.
