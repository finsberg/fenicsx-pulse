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
    cli,
    compressibility,
    exceptions,
    geometry,
    invariants,
    kinematics,
    material_model,
    material_models,
    prestress,
    problem,
    units,
    utils,
    viscoelasticity,
)
from .active_stress import ActiveStress
from .boundary_conditions import BoundaryConditions, NeumannBC, RobinBC
from .cardiac_model import CardiacModel
from .compressibility import (
    Compressibility,
    Compressible,
    Compressible2,
    Compressible3,
    Incompressible,
)
from .geometry import Geometry, HeartGeometry, Marker
from .material_model import HyperElasticMaterial, Material
from .material_models import (
    Guccione,
    HolzapfelOgden,
    NeoHookean,
    SaintVenantKirchhoff,
    Usyk,
)
from .prestress import PrestressProblem
from .problem import BaseBC, DynamicProblem, StaticProblem
from .units import Variable, ureg
from .viscoelasticity import NoneViscoElasticity, ViscoElasticity, Viscous

__all__ = [
    "kinematics",
    "invariants",
    "material_model",
    "problem",
    "cli",
    "Material",
    "HyperElasticMaterial",
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
    "__version__",
    "prestress",
    "PrestressProblem",
    "Usyk",
    "Compressible2",
    "Compressible3",
]
