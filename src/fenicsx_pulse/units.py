from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import dolfinx
import numpy as np
import pint

ureg = pint.UnitRegistry()
T = TypeVar("T", bound=float | dolfinx.fem.Function | dolfinx.fem.Constant)


def mesh_factor(mesh_unit: str) -> float:
    return ureg(mesh_unit).to_base_units().magnitude


def assign(variable: "Variable", value) -> None:
    if isinstance(variable.value, (float, int)) or np.isscalar(variable.value):
        variable.value = value
    elif isinstance(variable.value, dolfinx.fem.Function):
        if isinstance(value, (float, int, np.ndarray)) or np.isscalar(value):
            variable.value.x.array[:] = value
        elif isinstance(value, dolfinx.fem.Function):
            variable.value.x.array[:] = value.x.array
        else:
            raise ValueError(f"Cannot assign value of type {type(value)}")
    elif isinstance(variable.value, dolfinx.fem.Constant):
        variable.value.value = value
    else:
        raise ValueError(f"Cannot assign value of type {type(variable.value)}")


@dataclass(slots=True)
class Variable(Generic[T]):
    value: T
    unit: pint.Unit | str | None

    original_unit: pint.Unit = field(init=False, repr=False)
    original_value: T = field(init=False, repr=False)
    factor: float = field(init=False, repr=False)

    def __post_init__(self):
        # Check unit
        if isinstance(self.unit, str):
            self.original_unit = ureg(self.unit).units
        elif self.unit is None:
            self.original_unit = ureg.dimensionless
        else:
            self.original_unit = self.unit

        quantity = (1 * self.original_unit).to_base_units()
        self.factor = quantity.magnitude
        self.unit = quantity.units

    def to_base_units(self) -> float | Any:
        return self.value * self.factor

    def __str__(self) -> str:
        return f"{self.to_base_units()} {self.unit} ({self.value} {self.original_unit})"

    @classmethod
    def from_quantity(cls, quantity: pint.Quantity) -> "Variable":
        return cls(quantity.magnitude, quantity.units)

    def assign(self, value) -> None:
        assign(self, value)
