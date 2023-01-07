"""Top-level package for fenicsx_pulse."""
from importlib.metadata import metadata

meta = metadata("fenicsx_pulse")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

from . import kinematics
from . import invariants
from . import material_model
from .material_model import HyperElasticMaterial
from . import compressibility
from .compressibility import Compressible, Incompressible
from . import exceptions
from . import cardiac_model
from .cardiac_model import CardiacModel
from . import active_model
from . import active_stress
from .active_stress import ActiveStress
from . import geometry
from .geometry import Geometry, Marker
from . import boundary_conditions
from .boundary_conditions import NeumannBC, RobinBC, BoundaryConditions
from . import mechanicsproblem
from .mechanicsproblem import MechanicsProblem
from .linear_elastic import LinearElastic
from .holzapfelogden import HolzapfelOgden

__all__ = [
    "kinematics",
    "invariants",
    "material_model",
    "Material",
    "HyperElasticMaterial",
    "LinearElastic",
    "exceptions",
    "HolzapfelOgden",
    "compressibility",
    "Compressible",
    "Incompressible",
    "cardiac_model",
    "CardiacModel",
    "active_model",
    "active_stress",
    "ActiveStress",
    "geometry",
    "Geometry",
    "NeumannBC",
    "RobinBC",
    "boundary_conditions",
    "mechanicsproblem",
    "MechanicsProblem",
    "BoundaryConditions",
    "Marker",
]
