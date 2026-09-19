"""Embedding a 0D circulation model in the mechanics problem.

The usual way to drive a mechanics problem from a lumped-parameter circuit is
to alternate between them: solve the circuit for a volume, solve the mechanics
for the pressure that volume implies, hand it back, repeat until the two agree.
That works, but the two solvers only ever see each other's output, never each
other's derivatives.

This module supports the other arrangement, where the circuit's states become
unknowns of the same Newton system as the displacement. The coupling terms then
appear in the Jacobian and are differentiated along with everything else.

Nothing here knows about any particular circuit. A model is anything satisfying
:class:`CirculationModel`, which is a small enough interface that a
hand-written circuit satisfies it as readily as a generated one. The intended
source is a `.ode` file translated by `gotranx`; :class:`GotranxCirculation`
does that, and is the only part of this module that imports it.

Time stepping belongs to the problem, not the model. A model supplies
:math:`f` in :math:`dy/dt = f(t, y, m)` and nothing else, so the same circuit
can be advanced by different schemes without touching it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CirculationModel",
    "ChamberCoupling",
    "GotranxCirculation",
    "mL",
    "mmHg",
]

#: One milliliter in cubic meters. Circuit models are conventionally written in
#: milliliters and millimeters of mercury, while everything in `pulse` is in
#: base SI units, so the two coupling rows need converting. These are the only
#: places that conversion happens.
mL = 1e-6

#: One millimeter of mercury in pascals.
mmHg = 133.322387415


@runtime_checkable
class CirculationModel(Protocol):
    """A lumped-parameter circuit whose states can join a Newton system.

    Implementations supply the right-hand side of :math:`dy/dt = f(t, y, m)` as
    UFL expressions, where :math:`m` are quantities the circuit needs but does
    not compute, such as the pressure in a chamber that has been replaced by a
    3D model.
    """

    @property
    def state_names(self) -> Sequence[str]:
        """Names of the circuit states, in the order `rhs` returns them."""
        ...

    @property
    def missing_names(self) -> Sequence[str]:
        """Names of the values the circuit expects to be supplied."""
        ...

    @property
    def initial_states(self) -> np.ndarray:
        """Initial value of each state, ordered as :attr:`state_names`."""
        ...

    def rhs(self, t: Any, states: Sequence[Any], missing: Sequence[Any]) -> Sequence[Any]:
        """Return :math:`f(t, y, m)`, one scalar expression per state.

        Arguments may be UFL expressions rather than numbers, so the
        implementation must not branch on their values or coerce them with
        `float`. Conditionals have to be expressed with `ufl.conditional`.
        """
        ...


@dataclass(slots=True)
class ChamberCoupling:
    """Ties one cavity of the geometry to one chamber of the circuit.

    The chamber's volume stops being something the circuit predicts and becomes
    the volume of the deformed cavity; its pressure stops being something the
    circuit computes and becomes the cavity pressure the mechanics problem
    already carries as a Lagrange multiplier.

    Parameters
    ----------
    marker : str
        Name of the surface bounding the cavity, e.g. ``"ENDO"``.
    volume_state : str
        Circuit state holding that chamber's volume, e.g. ``"V_LV"``. It stays
        an unknown; what changes is that a constraint now ties it to the
        deformed cavity volume.
    pressure_missing : str
        Value the circuit expects to be supplied for that chamber's pressure,
        e.g. ``"p_LV"``.
    """

    marker: str
    volume_state: str
    pressure_missing: str


@dataclass
class GotranxCirculation:
    """A :class:`CirculationModel` generated from a `.ode` file by `gotranx`.

    The file is translated to UFL in memory rather than written to disk, so
    there is no generated module to go stale and nothing for concurrent
    processes to race over. Translation costs a `sympy` pass, which for a
    circuit of a dozen states is not worth caching.

    Parameters
    ----------
    ode_file : Path | str
        The `.ode` file.
    parameters : dict[str, float] | None
        Parameter values, by name, overriding the file's defaults.
    drop_components : Sequence[str]
        Components to subtract before generating code. Whatever the remainder
        still uses but no longer defines becomes a missing variable, which is
        how a chamber closure is replaced by an external model. Dropping a
        component that computes an activation phase with ``Mod`` is also what
        makes the remainder translatable at all, since UFL has no ``Mod``.
    """

    ode_file: Path | str
    parameters: dict[str, float] | None = None
    drop_components: Sequence[str] = ()
    _model: dict = field(init=False, repr=False, default_factory=dict)
    _parameter_values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import gotranx
        import gotranx.cli.gotran2ufl
        from gotranx.codegen.python import Format

        ode = gotranx.load_ode(Path(self.ode_file))
        for name in self.drop_components:
            ode = ode - ode.get_component(name)
        logger.debug(
            f"Generating UFL for {Path(self.ode_file).name} "
            f"({len(ode.states)} states, missing {sorted(ode.missing_variables)})",
        )

        code = gotranx.cli.gotran2ufl.get_code(ode, format=Format.none)
        exec(code, self._model)

        # Dropping a component drops its parameters too, so a caller passing the
        # whole parameter set of the unsplit model will be handing over names
        # this model no longer has. Take what applies and say what was left.
        known, unknown = {}, []
        for name, value in (self.parameters or {}).items():
            try:
                self._model["parameter_index"](name)
            except KeyError:
                unknown.append(name)
            else:
                known[name] = value
        if unknown:
            logger.debug(
                f"Ignoring {len(unknown)} parameter(s) not in the generated model "
                f"(expected after dropping {list(self.drop_components)}): {sorted(unknown)}",
            )
        self._parameter_values = self._model["init_parameter_values"](**known)

        # gotranx orders states and missing variables alphabetically, which is
        # rarely the declaration order. Everything below goes through the
        # generated index functions so that ordering never has to be guessed.
        self._state_names = tuple(
            sorted((s.name for s in ode.states), key=self._model["state_index"]),
        )
        self._missing_names = tuple(
            sorted(ode.missing_variables, key=self._model["missing_index"]),
        )

    @property
    def state_names(self) -> Sequence[str]:
        return self._state_names

    @property
    def missing_names(self) -> Sequence[str]:
        return self._missing_names

    @property
    def initial_states(self) -> np.ndarray:
        return np.asarray(self._model["init_state_values"]())

    def state_index(self, name: str) -> int:
        """Position of a state in :attr:`state_names`."""
        return int(self._model["state_index"](name))

    def missing_index(self, name: str) -> int:
        """Position of a missing value in :attr:`missing_names`."""
        return int(self._model["missing_index"](name))

    def rhs(self, t: Any, states: Sequence[Any], missing: Sequence[Any]) -> Sequence[Any]:
        return self._model["rhs"](t, states, self._parameter_values, missing)
