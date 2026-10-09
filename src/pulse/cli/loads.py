"""Prescribed loads: pressure (Neumann) and activation (Ta) profiles as functions of time.

Every profile is a pure function of ``t`` (seconds) returning SI base units (Pa), so a restart
needs no load state. Bestel profiles are integrated once at build time (one period from 0 when
`period` is set; `peak` normalisation needs it), interval by interval on the ``[time]`` grid, so
their values up to ``t`` never depend on ``end_time``.
"""

import csv
import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import dolfinx
import numpy as np

import pulse

from ..units import ureg
from .config import (
    BestelActivationProfile,
    BestelPressureProfile,
    ConfigError,
    ConstantProfile,
    LoadConfig,
    ProfileConfig,
    RampProfile,
    TableProfile,
    TimeConfig,
    si,
)

logger = logging.getLogger(__name__)

_BESTEL_HINT = (
    'Bestel profiles need the "circulation" and "scipy" packages: pip install circulation scipy'
)


def _uses_bestel(loads: Sequence[LoadConfig]) -> bool:
    return any(
        isinstance(load.profile, (BestelPressureProfile, BestelActivationProfile)) for load in loads
    )


def require_optional_packages(loads: Sequence[LoadConfig]) -> None:
    if not _uses_bestel(loads):
        return
    try:
        import circulation.bestel  # noqa: F401
        import scipy.integrate  # noqa: F401
    except ImportError as e:
        raise ConfigError(f"{_BESTEL_HINT} ({e})") from e


def _read_table(profile: TableProfile) -> tuple[np.ndarray, np.ndarray]:
    if profile.file is None:
        assert profile.times is not None and profile.values is not None
        times = np.asarray(profile.times, dtype=float)
        values = np.asarray(profile.values, dtype=float)
    else:
        try:
            with open(profile.file, newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError as e:
            raise ConfigError(f"table file {profile.file}: {e}") from e
        for column in (profile.time_column, profile.value_column):
            if not rows or column not in rows[0]:
                raise ConfigError(f"table file {profile.file}: no column {column!r}")
        try:
            times = np.array([float(r[profile.time_column]) for r in rows])
            values = np.array([float(r[profile.value_column]) for r in rows])
        except ValueError as e:
            raise ConfigError(f"table file {profile.file}: {e}") from e
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ConfigError(
                f"table file {profile.file}: need >= 2 rows with strictly increasing times",
            )
    t_factor = ureg.Quantity(1, profile.time_unit).to_base_units().magnitude
    v_factor = ureg.Quantity(1, profile.value_unit).to_base_units().magnitude
    return times * t_factor, values * v_factor


def _bestel(profile, t_start: float, t_end: float, dt: float) -> Callable[[float], float]:
    import circulation.bestel
    from scipy.integrate import solve_ivp

    if isinstance(profile, BestelPressureProfile):
        model = circulation.bestel.BestelPressure(parameters=profile.si_parameters())
    else:
        model = circulation.bestel.BestelActivation(parameters=profile.si_parameters())
    period = si(profile.period) if profile.period is not None else None
    t0, t1 = (0.0, period) if period is not None else (t_start, t_end)
    n = max(1, round((t1 - t0) / dt))
    times = t0 + dt * np.arange(n + 1)
    values = np.zeros(n + 1)
    # One solve_ivp per interval: the value at times[k] depends only on times[:k + 1].
    for k in range(n):
        res = solve_ivp(
            model,
            [times[k], times[k + 1]],
            [values[k]],
            method="Radau",
            rtol=1e-8,
            atol=1e-6,
        )
        if not res.success:
            raise ConfigError(f"Bestel profile integration failed: {res.message}")
        values[k + 1] = res.y[0, -1]
    if profile.peak is not None:
        top = float(np.max(np.abs(values)))
        if top == 0.0:
            raise ConfigError("Bestel profile: cannot normalise a trace that is zero everywhere")
        values = values / top * si(profile.peak)
    if period is not None:
        return lambda t: float(np.interp(t % period, times, values))
    return lambda t: float(np.interp(t, times, values))


def build_profile(
    profile: ProfileConfig,
    t_start: float,
    t_end: float,
    dt: float,
) -> Callable[[float], float]:
    """A function of time in seconds returning the load in SI base units (Pa)."""
    if isinstance(profile, ConstantProfile):
        value = si(profile.value)
        return lambda t: value
    if isinstance(profile, RampProfile):
        t0, t1 = si(profile.start), si(profile.end)
        a, b = si(profile.from_value), si(profile.to_value)

        def ramp(t: float) -> float:
            if t <= t0:
                return a
            if t >= t1:
                return b
            return a + (b - a) * (t - t0) / (t1 - t0)

        return ramp
    if isinstance(profile, TableProfile):
        times, values = _read_table(profile)
        period = si(profile.period) if profile.period is not None else None

        def table(t: float) -> float:
            if period is not None:
                t = t % period
            return float(np.interp(t, times, values))

        return table
    if isinstance(profile, (BestelPressureProfile, BestelActivationProfile)):
        return _bestel(profile, t_start, t_end, dt)
    raise ConfigError(f"Unsupported profile {profile!r}")  # pragma: no cover


def make_load_variables(
    loads: Sequence[LoadConfig],
    mesh: dolfinx.mesh.Mesh,
) -> dict[str, pulse.Variable]:
    """One Constant-backed ``pulse.Variable`` in Pa per load, keyed by ``LoadConfig.name``."""
    return {
        load.name: pulse.Variable(
            dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
            "Pa",
        )
        for load in loads
    }


@dataclass
class Load:
    name: str
    variable: pulse.Variable
    fn: Callable[[float], float]


@dataclass
class LoadSet:
    loads: list[Load]

    def update(self, t: float) -> None:
        for load in self.loads:
            load.variable.assign(load.fn(t))

    def values(self, t: float) -> dict[str, float]:
        return {load.name: load.fn(t) for load in self.loads}

    @property
    def names(self) -> list[str]:
        return [load.name for load in self.loads]

    @property
    def pressure_markers(self) -> list[str]:
        prefix = "pressure_"
        return [n[len(prefix) :] for n in self.names if n.startswith(prefix)]


def build_loads(
    loads: Sequence[LoadConfig],
    variables: dict[str, pulse.Variable],
    time: TimeConfig,
) -> LoadSet:
    require_optional_packages(loads)
    t_start = time.start_s()
    t_end = time.end_s()
    dt = time.dt_s()
    return LoadSet(
        [
            Load(load.name, variables[load.name], build_profile(load.profile, t_start, t_end, dt))
            for load in loads
        ],
    )
