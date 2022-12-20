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
from .material_model import HyperElasticMaterialModel

from .linear_elastic import LinearElastic

__all__ = [
    "kinematics",
    "invariants",
    "material_model",
    "Material",
    "HyperElasticMaterialModel",
    "LinearElastic",
]
