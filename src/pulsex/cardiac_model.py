from dataclasses import dataclass

import dolfinx

from . import kinematics
from .active_model import ActiveModel
from .compressibility import Compressibility
from .material_model import HyperElasticMaterial


@dataclass(frozen=True, slots=True)
class CardiacModel:
    material: HyperElasticMaterial
    active: ActiveModel
    compressibility: Compressibility

    def strain_energy(self, F, p: dolfinx.fem.Function | None = None):
        self.compressibility.register(p)
        # If active strain we would need to get the elastic
        # part of the deformation gradient
        Fe = self.active.F(F)
        J = kinematics.Jacobian(Fe)
        return (
            self.material.strain_energy(Fe)
            + self.active.strain_energy(F)
            + self.compressibility.strain_energy(J)
        )
