# Running on a cluster

The `pulse` CLI is designed so one `config.toml` can drive an array-job parameter sweep, and a
long run can survive a wall-time kill and be continued (`--restart`), possibly on a different node
allocation. This page assumes SLURM but the same ideas (override flags, exit codes, restart) apply
to any scheduler.

(mesh-once-run-many)=
## Mesh once, run many

Generating a mesh (especially a realistic `lv_ellipsoid`/`biv_ellipsoid`/`ukb` geometry) can take
much longer than a short SLURM job's queue-wait budget wants, and every array-job task sharing one
`config.toml` would otherwise regenerate (or race to regenerate) it independently. Build it once,
on a login node or a short single-task job, before submitting the sweep:

```bash
pulse validate-config config.toml   # parse-time checks only: no mesh, no MPI-collective work
pulse geometry config.toml          # build/cache the mesh once
```

`pulse geometry` is exactly the same step `pulse run` would do lazily on first use, cached in
`geometry.folder/<hash>/`, a subfolder keyed by a hash of the `[geometry]` section — so running it
up front is purely an optimization, not a separate code path; every array-job task's own `pulse
run` reuses the cache rather than rebuilding it. A sweep over a *geometry* parameter (e.g.
`--set geometry.char_length=...` per task) is safe too: each distinct geometry gets its own
subfolder, so tasks never overwrite or delete each other's mesh. Tasks that need the same, not yet
cached mesh may each generate it, but every job generates into a private temporary folder that is
atomically renamed into place; whichever finishes first wins, and the others reuse its mesh and
discard their own copy. Warming the cache with `pulse geometry` first (once per distinct geometry)
avoids that duplicated work. It's also the cheapest way to surface a marker-name typo
(`[[load]]`, `[bcs]`) ahead of the timed sweep: that check needs the actual mesh, so it only runs
once `pulse run` has built or loaded the geometry (see {ref}`Exit codes <exit-codes>`) — with the
mesh already cached, that happens within seconds rather than after a from-scratch mesh generation
inside the job.

## A SLURM array job for a parameter sweep

`--output-folder` and `--set` let one base `config.toml` be varied per array-job task without
copying files:

```bash
#!/bin/bash
#SBATCH --job-name=pulse-sweep
#SBATCH --array=0-9
#SBATCH --ntasks=64
#SBATCH --time=04:00:00

DT=(0.5 1 2 5 10 0.5 1 2 5 10)
srun pulse run config.toml \
    --output-folder "runs/${SLURM_ARRAY_TASK_ID}" \
    --set "time.dt=\"${DT[$SLURM_ARRAY_TASK_ID]} ms\"" \
    --overwrite
```

`--output-folder` resolves against the **current working directory** the job runs in (unlike every
path *inside* the config file, which resolves against the config file's own directory — see
{ref}`Overrides <overrides>`), so `runs/${SLURM_ARRAY_TASK_ID}` above lands next to wherever the
job script itself runs from. `--overwrite` makes the task safe to resubmit: if that array index's
output folder already has results in it (e.g. a resubmit after a scheduler-level failure, or
resubmitting the whole array because task 3 failed), `pulse` would otherwise refuse to touch it
rather than silently deleting a previous task's output from under a differently-indexed rerun.

## Surviving a wall-time kill: `--restart`

For a run whose simulated time exceeds what a single job's wall-time allows, set a checkpoint
interval and let the job resubmit itself onto the same output folder:

```bash
#!/bin/bash
#SBATCH --job-name=pulse-longrun
#SBATCH --time=24:00:00
#SBATCH --ntasks=64

# output.checkpoint_every = "0.1 s" in config.toml
if [ -f output/restart.json ]; then FLAG=--restart; else FLAG=--overwrite; fi
srun pulse run config.toml $FLAG
```

Resubmit the same script (e.g. from a scheduler dependency chain, `sbatch --dependency=afterany`,
or cron) until `run.json: status == "finished"`. Each resubmission picks up `--restart`
automatically once `output/restart.json` exists (written after the first successful checkpoint).
Before that — the first submission, or a job killed before its first checkpoint, which leaves a
`results.bp` but no `restart.json` — there is nothing to continue from, so the script starts over
with `--overwrite` (without it, `pulse` refuses to touch the existing `results.bp`, saying that no
restart checkpoint exists yet). `--overwrite` is safe here: it only deletes `pulse`'s own
artifacts, and only after the config has been validated.

**What may change across a restart, and what may not:** `pulse` refuses `--restart` if the run's
*physics* has changed since the last checkpoint, comparing a hash of the whole resolved config
except the run length (`time.end_time`/`num_steps`) and everything under
`[output]`/`[postprocess]`/`[solver]`. So between restarts you may freely:

- Extend `time.end_time`, or change `num_steps`, to run longer than originally configured — as
  long as the *effective* `dt` (`(end_time - start_time) / num_steps` when `num_steps` is given)
  stays the same; changing `num_steps` without a matching `end_time` change is refused rather than
  silently mixing time steps.
- Change anything under `[output]` (`save_every`, `checkpoint_every`, `performance`, `log_every`)
  or `[postprocess]` (`vtx`, `fields`, `points`, `vertex_tags`, `plots`).
- Change anything under `[solver]` (`max_halvings`, `petsc_options`, or `--petsc-options`): they
  change how a step is solved, not the physics. This is the way to continue after a solver failure
  (exit code 2) — e.g. `--restart --set solver.max_halvings=8`.
- Run on a **different number of MPI ranks** than the original job used (the checkpoint is read
  and redistributed across however many ranks the new job has).

But not, without `pulse` refusing with an error naming the mismatch:

- `time.start_time`, `time.dt` (or the effective `dt` derived from `num_steps`), `[geometry]`
  (except `geometry.folder` for a *generated* geometry type, which is just a cache location
  there, not the physics itself — it does count for `geometry.type = "folder"`, where it's the
  actual mesh being simulated), `[material]`, `[active]`, `[compressibility]`,
  `[viscoelasticity]`, `[bcs]`, `[[load]]` (including a `[load.profile] file` table's *contents*,
  not just its path — editing that CSV also counts as a physics change), `[problem]`. So a
  solver failure that needs a smaller `time.dt` or a gentler load cannot be continued: rerun it
  with `--overwrite` (or into a new output folder).

## Exit codes for job-script branching

`0` success, `1` a configuration error (`ConfigError`, a command-line usage error, or a missing
`cli` extra), `2` a runtime failure (Newton failing to converge within `solver.max_halvings`, or
any other unexpected error, e.g. from mesh generation or I/O). A parse-time mistake (bad TOML,
wrong units, an unknown key) is always caught before any mesh is built or loaded; a marker-name
mistake (which needs the actual mesh to check) is caught right after that — still well before the
collective solve loop, but only cheap in wall-time if the geometry was already built/cached ahead
of time (see [Mesh once, run many](#mesh-once-run-many)). Either way, a job script can branch on
the exit code directly:

```bash
srun pulse run config.toml $FLAG
case $? in
  0) echo "done" ;;
  1) echo "config or usage error, not retrying" >&2; exit 1 ;;
  2) echo "runtime failure (solver, I/O, mesh generation): check output/run.json and the log" >&2; exit 1 ;;
esac
```

`output/run.json` (`status: "running"|"finished"|"failed"`, plus `n_ranks`, wall-clock start/end,
and — on a failure — the error) is the same information in a form a later step or a monitoring
script can read back out of the output folder itself. It is only written once the run has started:
a failure while setting up the simulation (config, geometry, markers, boundary conditions) leaves
the output folder untouched, so check the exit code and the job's own log for those.

## Env var overrides with scheduler-provided variables

`PULSE_<SECTION>__<KEY>` env vars (matched case-insensitively against the schema, e.g.
`PULSE_TIME__DT`) sit below `--set` and above the TOML file in precedence — convenient for piping
a scheduler's own environment straight through without constructing a `--set` string in the job
script:

```bash
export PULSE_OUTPUT__FOLDER="runs/${SLURM_ARRAY_TASK_ID}"
export PULSE_TIME__DT="${DT_MS} ms"
srun pulse run config.toml
```

(Note `PULSE_OUTPUT__FOLDER` here still resolves against the config file's directory like any
other in-file path, since it's not the dedicated `--output-folder` flag — use that flag instead if
you need current-working-directory-relative resolution, as in the array-job example above.)

## Solver advice for large meshes

`solver.petsc_options` (and the matching `--petsc-options` flag) are merged over `pulse`'s problem
defaults, so a large mesh (a fine realistic ventricular geometry, or a sweep run at production
resolution) that needs a different linear solver can set it without editing Python:

```toml
[solver]
petsc_options = { ksp_type = "cg", pc_type = "hypre", pc_hypre_type = "boomeramg" }
```

Algebraic multigrid scales much better across MPI ranks and mesh sizes than the direct (MUMPS)
factorization `pulse` otherwise defaults to for the small-to-moderate meshes in the templates.
Fine-tune further with `--petsc-options`/`solver.petsc_options` if the default iterative
tolerances need adjusting for a particular mesh.
