"""Top-level package for pulsex."""
from importlib.metadata import metadata

meta = metadata("pulsex")
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
]
