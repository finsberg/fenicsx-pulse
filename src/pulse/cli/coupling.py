"""Build a `pulse.coupling` coupling from ``[circulation]``."""

from typing import Any

from ..circulation import ChamberCoupling, GotranxCirculation, GotranxNumpyCirculation
from ..coupling import (
    CycleCoupling,
    MonolithicCoupling,
    NoCoupling,
    Phase,
    SplitCoupling,
)
from ..cycle import CycleParams, PrescribedInflow, Windkessel
from .config import Config, ConfigError, CycleCavityConfig, si

_GOTRANX_HINT = 'circulation.type = {kind!r} needs the "gotranx" package: pip install gotranx'


def cycle_params(cavity: CycleCavityConfig) -> CycleParams:
    """`pulse.cycle.CycleParams` (SI) of one ``[[circulation.cavity]]``."""
    wk = cavity.windkessel
    return CycleParams(
        t_zero=si(cavity.t_zero),
        preload_pressure=si(cavity.preload_pressure),
        t_end_diastole=si(cavity.t_end_diastole),
        p_end_diastole=si(cavity.p_end_diastole),
        p_fill=si(cavity.p_fill),
        period=si(cavity.period),
        windkessel=Windkessel(
            p_init=si(wk.p_init),
            compliance=si(wk.compliance),
            resistance=si(wk.resistance),
            characteristic_impedance=si(wk.characteristic_impedance),
        ),
        filling=PrescribedInflow(rate=si(cavity.filling_rate)),
        min_ejection_duration=si(cavity.min_ejection_duration),
    )


def _generate(cls: type, circulation: Any) -> Any:
    try:
        return cls(
            ode_file=circulation.ode_file,
            parameters=dict(circulation.parameters),
            drop_components=tuple(circulation.drop_components),
        )
    except Exception as e:  # gotranx raises many kinds on a bad file or component name
        raise ConfigError(
            f"circulation.ode_file {circulation.ode_file}: cannot generate the model "
            f"(drop_components = {circulation.drop_components}): {e!r}",
        ) from e


def check_ode_names(circulation: Any, model: GotranxNumpyCirculation) -> None:
    """Every name ``[circulation]`` uses must exist in the generated model, and every missing
    variable must be fed by a chamber or an input. Raises one ConfigError listing them all."""
    states, missing = set(model.state_names), set(model.missing_names)
    errors = []
    if model.ignored_parameters:
        errors.append(f"unknown parameters {list(model.ignored_parameters)}")
    unknown = sorted(set(circulation.initial_state) - states)
    if unknown:
        errors.append(f"unknown initial_state names {unknown} (states: {sorted(states)})")
    for c in circulation.chamber:
        if c.volume_state not in states:
            errors.append(f"chamber {c.marker}: {c.volume_state!r} is not a state")
        if c.pressure_missing not in missing:
            errors.append(f"chamber {c.marker}: {c.pressure_missing!r} is not a missing variable")
    unknown = sorted(set(circulation.inputs) - missing)
    if unknown:
        errors.append(f"inputs {unknown} are not missing variables")
    fed = {c.pressure_missing for c in circulation.chamber} | set(circulation.inputs)
    unfed = sorted(missing - fed)
    if unfed:
        errors.append(
            f"missing variables {unfed} are neither a chamber pressure nor an input "
            "(add them to [circulation.inputs], or keep the component that computes them)",
        )
    unknown = sorted(set(circulation.record) - set(model.monitor_names))
    if unknown:
        errors.append(f"unknown record names {unknown} (monitored: {sorted(model.monitor_names)})")
    if errors:
        raise ConfigError(f"circulation ({circulation.ode_file.name}): " + "; ".join(errors))


def build_coupling(conf: Config) -> Any:
    """The `pulse.coupling` coupling ``[circulation]`` describes (`NoCoupling` for none)."""
    circulation = conf.circulation
    if circulation.type == "none":
        return NoCoupling()
    if circulation.type == "cycle":
        return CycleCoupling(
            {c.marker: cycle_params(c) for c in circulation.cavity},
            preconditioner_lag=conf.solver.preconditioner_lag,
        )
    try:
        import gotranx  # noqa: F401
    except ImportError as e:
        raise ConfigError(f"{_GOTRANX_HINT.format(kind=circulation.type)} ({e})") from e
    numpy_model = _generate(GotranxNumpyCirculation, circulation)
    check_ode_names(circulation, numpy_model)
    chambers = [
        ChamberCoupling(
            marker=c.marker,
            volume_state=c.volume_state,
            pressure_missing=c.pressure_missing,
        )
        for c in circulation.chamber
    ]
    inputs = {name: Phase(si(i.period)) for name, i in circulation.inputs.items()}
    if circulation.type == "split":
        return SplitCoupling(
            numpy_model,
            chambers,
            inputs=inputs,
            initial_state=dict(circulation.initial_state),
            record=tuple(circulation.record),
        )
    return MonolithicCoupling(
        _generate(GotranxCirculation, circulation),
        chambers,
        inputs=inputs,
        initial_state=dict(circulation.initial_state),
        scheme=circulation.scheme,
        monitor_model=numpy_model,
        record=tuple(circulation.record),
    )
