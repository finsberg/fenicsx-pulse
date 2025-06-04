import operator
from dataclasses import dataclass

from mpi4py import MPI

import dolfinx
import numpy as np


def check_value_greater_than(
    f: float | dolfinx.fem.Function | dolfinx.fem.Constant | np.ndarray,
    bound: float,
    inclusive: bool = False,
) -> bool:
    """Check that the value of f is greater than the given bound

    Parameters
    ----------
    f : float | dolfinx.fem.Function | dolfinx.fem.Constant | np.ndarray
        The variable to be checked
    bound : float
        The lower bound
    inclusive: bool
        Whether to include the bound in the check or not, by default False

    Returns
    -------
    bool
        True if value is greater than the lower bound,
        otherwise false
    """
    op = operator.ge if inclusive else operator.gt
    if np.isscalar(f):
        return op(f, bound)
    elif isinstance(f, dolfinx.fem.Constant):
        return op(f.value.max(), bound)
    elif isinstance(f, dolfinx.fem.Function):
        return op(
            f.function_space.mesh.comm.allreduce(f.x.array.max(), op=MPI.MAX),
            bound,
        )

    raise PulseException(  # pragma: no cover
        f"Invalid type for f: {type(f)}. Expected 'float', "
        "'dolfinx.fem.Constant', 'numpy array' or 'dolfinx.fem.Function'",
    )


def check_value_lower_than(
    f: float | dolfinx.fem.Function | dolfinx.fem.Constant,
    bound: float,
    inclusive: bool = False,
) -> bool:
    """Check that the value of f is lower than the given bound

    Parameters
    ----------
    f : float | dolfinx.fem.Function | dolfinx.fem.Constant | np.ndarray
        The variable to be checked
    bound : float
        The upper bound
    inclusive: bool
        Whether to include the bound in the check or not, by default False

    Returns
    -------
    bool
        True if value is greater than the lower bound,
        otherwise false
    """
    op = operator.le if inclusive else operator.lt
    if np.isscalar(f):
        return op(f, bound)
    elif isinstance(f, dolfinx.fem.Constant):
        return op(f.value.min(), bound)
    elif isinstance(f, dolfinx.fem.Function):
        return op(
            f.function_space.mesh.comm.allreduce(f.x.array.min(), op=MPI.MIN),
            bound,
        )

    raise PulseException(  # pragma: no cover
        f"Invalid type for f: {type(f)}. Expected 'float', "
        "'dolfinx.fem.Constant', 'numpy array' or 'dolfinx.fem.Function'",
    )


def check_value_between(
    f: float | dolfinx.fem.Function | dolfinx.fem.Constant,
    lower_bound: float,
    upper_bound: float,
    inclusive: bool = False,
) -> bool:
    """Check if value of `f` is between lower and upper bound

    Parameters
    ----------
    f : float | dolfinx.fem.Function | dolfinx.fem.Constant
        The variable to check
    lower_bound : float
        The lower bound
    upper_bound : float
        The upper bound
    inclusive: bool
        Whether to include the bound in the check or not, by default False

    Returns
    -------
    bool
        Return True if the value is between lower_bound and upper_bound,
        otherwise return False
    """
    return check_value_greater_than(
        f,
        lower_bound,
        inclusive=inclusive,
    ) and check_value_lower_than(f, upper_bound, inclusive=inclusive)


class PulseException(Exception):
    pass


@dataclass
class InvalidRangeError(ValueError, PulseException):
    name: str
    expected_range: tuple[float, float]

    def __str__(self) -> str:
        return (
            f"Invalid range for variable {self.name}. "
            f"Expected variable to be in the range: {self.expected_range}"
        )


@dataclass
class MissingModelAttribute(AttributeError, PulseException):
    attr: str
    model: str

    def __str__(self) -> str:
        return f"Missing required attributed {self.attr!r} for model {self.model!r}"


class MeshTagNotFoundError(PulseException):
    def __str__(self) -> str:
        return "No mesh tags found"


@dataclass
class MarkerNotFoundError(PulseException):
    marker: str

    def __str__(self) -> str:
        return f"Marker {self.marker} not found in geometry"
