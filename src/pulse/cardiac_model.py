"""This module defines the cardiac model.

The cardiac model is a combination of a material model,
an active model, and a compressibility model.
"""

from dataclasses import dataclass, field
from typing import Protocol

import dolfinx

from . import kinematics
from .viscoelasticity import NoneViscoElasticity


class ActiveModel(Protocol):
    def strain_energy(self, F) -> dolfinx.fem.Form: ...

    def Fe(self, F) -> dolfinx.fem.Form: ...


class Compressibility(Protocol):
    def strain_energy(self, J) -> dolfinx.fem.Form: ...

    def is_compressible(self) -> bool: ...

    def register(self, p: dolfinx.fem.Function | None) -> None: ...


class HyperElasticMaterial(Protocol):
    def strain_energy(self, F) -> dolfinx.fem.Form: ...


class ViscoElasticity(Protocol):
    def strain_energy(self, E_dot) -> dolfinx.fem.Form: ...


@dataclass(frozen=True, slots=True)
class CardiacModel:
    material: HyperElasticMaterial
    active: ActiveModel
    compressibility: Compressibility
    viscoelasticity: ViscoElasticity = field(default_factory=NoneViscoElasticity)
    decouple_deviatoric_volumetric: bool = False

    def strain_energy(self, F, p: dolfinx.fem.Function | None = None):
        self.compressibility.register(p)
        # If active strain we would need to get the elastic
        # part of the deformation gradient
        Fe = self.active.Fe(F)
        J = kinematics.Jacobian(Fe)

        if self.decouple_deviatoric_volumetric:
            Jm13 = J ** (-1 / 3)
        else:
            Jm13 = 1.0

        return (
            self.material.strain_energy(Jm13 * Fe)
            + self.active.strain_energy(Jm13 * F)
            + self.compressibility.strain_energy(J)
        )

    def viscoelastic_strain_energy(self, E_dot):
        return self.viscoelasticity.strain_energy(E_dot)
