from dataclasses import dataclass
from typing import Protocol

import dolfinx

from . import kinematics


class ActiveModel(Protocol):
    def strain_energy(self, F) -> dolfinx.fem.Form: ...
    def Fe(self, F) -> dolfinx.fem.Form: ...


class Compressibility(Protocol):
    def strain_energy(self, J) -> dolfinx.fem.Form: ...
    def register(self, p: dolfinx.fem.Function | None) -> None: ...


class HyperElasticMaterial(Protocol):
    def strain_energy(self, F) -> dolfinx.fem.Form: ...


@dataclass(frozen=True, slots=True)
class CardiacModel:
    material: HyperElasticMaterial
    active: ActiveModel
    compressibility: Compressibility

    def strain_energy(self, F, p: dolfinx.fem.Function | None = None):
        self.compressibility.register(p)
        # If active strain we would need to get the elastic
        # part of the deformation gradient
        Fe = self.active.Fe(F)
        J = kinematics.Jacobian(Fe)
        Jm13 = J ** (-1 / 3)
        return (
            self.material.strain_energy(Jm13 * Fe)
            + self.active.strain_energy(Jm13 * F)
            + self.compressibility.strain_energy(J)
        )
