"""Top-level package for fenicsx_pulse."""

from importlib.metadata import metadata

meta = metadata("fenicsx_pulse")
__version__ = meta["Version"]
__author__ = meta["Author-email"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

from . import (
    active_model,
    active_stress,
    boundary_conditions,
    cardiac_model,
    compressibility,
    exceptions,
    geometry,
    invariants,
    kinematics,
    material_model,
    material_models,
    problem,
    units,
    utils,
    viscoelasticity,
)
from .active_stress import ActiveStress
from .boundary_conditions import BoundaryConditions, NeumannBC, RobinBC
from .cardiac_model import CardiacModel
from .compressibility import Compressibility, Compressible, Incompressible
from .geometry import Geometry, HeartGeometry, Marker
from .material_model import HyperElasticMaterial, Material
from .material_models import (
    Guccione,
    HolzapfelOgden,
    LinearElastic,
    NeoHookean,
    SaintVenantKirchhoff,
)
from .problem import BaseBC, DynamicProblem, StaticProblem
from .units import Variable, ureg
from .viscoelasticity import NoneViscoElasticity, ViscoElasticity, Viscous

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
    "BoundaryConditions",
    "Marker",
    "material_models",
    "NeoHookean",
    "SaintVenantKirchhoff",
    "Compressibility",
    "Guccione",
    "units",
    "ureg",
    "utils",
    "Variable",
    "HeartGeometry",
    "problem",
    "StaticProblem",
    "DynamicProblem",
    "BaseBC",
    "viscoelasticity",
    "NoneViscoElasticity",
    "ViscoElasticity",
    "Viscous",
]
