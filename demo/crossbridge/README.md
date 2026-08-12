# Isometric Twitch Experiments & the Frank-Starling Mechanism

These examples all run the same experiment -- stretch a slab of myocardial
tissue to a fixed length, lock the boundaries (isometric condition), then
activate it -- to study how active tension depends on pre-stretch, using
progressively more physiologically detailed models of active tension
generation. They use a quasi-static formulation (no inertia), so no time
integration scheme is required.

* **[Frank-Starling Twitch (ad hoc)](frank_starling_twitch.py)**:
    Reproduces the Frank-Starling mechanism with a simple, curve-fitted
    stretch-dependent multiplier ({class}`pulse.FrankStarlingActiveStress`)
    applied on top of a constant activation level. Cheap and dependency-free;
    a good default when only the qualitative length-tension trend matters.

## Cross-bridge cycling models ([`crossbridge`](https://github.com/ComputationalPhysiology/crossbridge))

These replace the fitted multiplier with an actual sub-cellular
force-generation model, driven by a realistic calcium transient and coupled
through {class}`pulse.StabilizedActiveStress` (using the model's own active
tension $T_a$ *and* active stiffness $K_a$, per {cite}`regazzoni2021oscillation`).
Length-dependent activation and, for two of the four models, a genuine
force-velocity relationship, fall out of the model's own kinetics rather than
being imposed.

* **[Land (2017)](crossbridge_land2017.py)**: three-state cross-bridge
  cycle with curve-fitted length-dependent-activation gradients.
* **[Lewalle (2024)](crossbridge_lewalle2024.py)**: the same cross-bridge
  cycle, with length dependence replaced by a mechanistic myosin
  OFF-state force-feedback loop.
* **[RDQ18](crossbridge_rdq18.py)**: cooperative regulatory-unit thin-filament
  kinetics with explicit filament overlap; no force-velocity effect.
* **[RDQ20-MF](crossbridge_rdq20mf.py)**: RDQ18's regulatory-unit kinetics plus
  an explicit, velocity-dependent cross-bridge cycle.
* **[Model Comparison](crossbridge_comparison.py)**: runs all four side by
  side under matched conditions and compares twitch shape,
  Frank-Starling steepness, and computational cost.
