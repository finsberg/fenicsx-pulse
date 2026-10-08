"""Pydantic models for the ``pulse`` CLI configuration file.

This module must not import dolfinx itself: it is used by ``pulse validate-config`` and by the
fast unit tests. Every physical quantity is a pint quantity (``"<value> <unit>"`` in TOML); bare
numbers are rejected. Section models never assume they are the root, so simcardemsx can compose
them into its own config.
"""

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Union, cast

import pint
from pint import Quantity
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_pint import PydanticPintQuantity, set_registry

from ..units import ureg

# Share pulse's registry so config quantities combine with pulse.Variable units.
set_registry(ureg)

Time = Annotated[Quantity, PydanticPintQuantity("s")]
Pressure = Annotated[Quantity, PydanticPintQuantity("kPa")]
Density = Annotated[Quantity, PydanticPintQuantity("kg/m**3")]
Viscosity = Annotated[Quantity, PydanticPintQuantity("Pa*s")]
Compliance = Annotated[Quantity, PydanticPintQuantity("m**3/Pa")]
Resistance = Annotated[Quantity, PydanticPintQuantity("Pa*s/m**3")]
FlowRate = Annotated[Quantity, PydanticPintQuantity("m**3/s")]


class ConfigError(ValueError):
    """Invalid configuration (exit code 1)."""


def si(q: Quantity) -> float:
    """Magnitude of ``q`` in SI base units."""
    return float(q.to_base_units().magnitude)


def _q(default: str) -> Quantity:
    """Type-only cast for a quantity field's string default (see beat.cli.config._q).

    pydantic-pint parses the string at validation time because every model sets
    ``validate_default=True``; always use it as ``default_factory=lambda: _q(...)``.
    """
    return default  # type: ignore[return-value]


def _parse(v: Any, what: str) -> Quantity:
    if not isinstance(v, str):
        raise ValueError(f"{what} must be a quantity string like '1 kPa', got {v!r}")
    try:
        q = ureg.Quantity(v)
    except (pint.errors.PintError, ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"{what} must be a valid quantity, got {v!r}: {e}") from e
    return q


def _check_unit(v: Any, unit: str, what: str) -> str:
    """Check that the quantity string ``v`` has the dimensionality of ``unit``."""
    q = _parse(v, what)
    if q.dimensionality != ureg.Quantity(1, unit).dimensionality:
        raise ValueError(f"{what} must have units like {unit!r}, got {v!r}")
    return v


def _is_multiple(value: float, step: float) -> bool:
    """Whether ``value`` is an integer multiple of ``step`` (1e-9 relative tolerance)."""
    n = value / step
    return abs(n - round(n)) <= 1e-9 * max(1.0, abs(n))


def _check_unit_name(v: str, unit: str, what: str) -> str:
    try:
        dim = ureg.Quantity(1, v).dimensionality
    except (pint.errors.PintError, ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"{what} must be a valid unit, got {v!r}: {e}") from e
    if dim != ureg.Quantity(1, unit).dimensionality:
        raise ValueError(f"{what} must be a unit like {unit!r}, got {v!r}")
    return v


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


# --- geometry ---------------------------------------------------------------------------


class FromGeometryFibers(_Base):
    type: Literal["from_geometry"] = "from_geometry"


class AxisFibers(_Base):
    """Constant fibres along an axis; sheets/normals along the next two axes (cyclically)."""

    type: Literal["axis"] = "axis"
    direction: Literal["x", "y", "z"] = "x"


class NoFibers(_Base):
    """No fibre field (isotropic materials only); "isotropic" (beat's name) is an alias."""

    type: Literal["none", "isotropic"] = "none"

    @field_validator("type")
    @classmethod
    def _alias(cls, v: str) -> str:
        return "none"


FibersConfig = Annotated[
    Union[FromGeometryFibers, AxisFibers, NoFibers],
    Field(discriminator="type"),
]


class _GeometryBase(_Base):
    unit: str = Field(
        default="m",
        description="Length unit of the mesh coordinates (after `scale`); passed to the problem "
        "as mesh_unit",
    )
    scale: float = Field(
        default=1.0,
        gt=0,
        description="Multiply the mesh coordinates by this factor when loading",
    )
    folder: Path = Field(
        default=Path("geometry"),
        description="type=folder: folder to read. Generated types: cache root, each mesh is "
        "cached in its own <hash>/ subfolder",
    )
    quadrature_degree: int = Field(default=4, ge=1, description="Quadrature degree of the forms")

    @field_validator("unit")
    @classmethod
    def _is_length(cls, v: str) -> str:
        return _check_unit_name(v, "m", "geometry.unit")

    def generator_kwargs(self) -> dict[str, Any]:
        exclude = {"type", "unit", "folder", "fibers", "scale", "quadrature_degree"}
        exclude |= {"rotate_base_normal", "ldrb"}
        return self.model_dump(exclude=exclude)


class FolderGeometry(_GeometryBase):
    type: Literal["folder"] = "folder"
    fibers: FibersConfig = Field(default_factory=FromGeometryFibers)


class BoxGeometry(_GeometryBase):
    """Builtin box; facets tagged X0, X1, Y0, Y1, Z0, Z1 (values 1..6)."""

    type: Literal["box"] = "box"
    lx: float = Field(default=1.0, gt=0)
    ly: float = Field(default=1.0, gt=0)
    lz: float = Field(default=1.0, gt=0)
    nx: int = Field(default=3, ge=1)
    ny: int = Field(default=3, ge=1)
    nz: int = Field(default=3, ge=1)
    cell_type: Literal["tetrahedron", "hexahedron"] = "tetrahedron"
    fibers: FibersConfig = Field(default_factory=AxisFibers)


class _FiberAngles(_GeometryBase):
    fiber_angle_endo: float = 60.0
    fiber_angle_epi: float = -60.0
    fiber_space: str = "P_1"
    fibers: FibersConfig = Field(default_factory=FromGeometryFibers)


class LVEllipsoidGeometry(_FiberAngles):
    type: Literal["lv_ellipsoid"] = "lv_ellipsoid"
    r_short_endo: float = 7.0
    r_short_epi: float = 10.0
    r_long_endo: float = 17.0
    r_long_epi: float = 20.0
    psize_ref: float = 3.0
    mu_apex_endo: float = -3.141592653589793
    mu_base_endo: float = -1.2722641256100204
    mu_apex_epi: float = -3.141592653589793
    mu_base_epi: float = -1.318116071652818
    aha: bool = False
    dmu_factor: float = 0.25


class BiVEllipsoidGeometry(_FiberAngles):
    type: Literal["biv_ellipsoid"] = "biv_ellipsoid"
    char_length: float = 0.5


class CylinderGeometry(_FiberAngles):
    """cardiac_geometries.mesh.cylinder_D_shaped."""

    type: Literal["cylinder"] = "cylinder"
    r_inner: float = 13.0
    r_outer: float = 20.0
    height: float = 40.0
    inner_flat_face_distance: float = 10.0
    outer_flat_face_distance: float = 17.0
    char_length: float = 10.0


class LDRBAngles(_Base):
    """Per-ventricle LDRB fibre angles in degrees (ldrb.dolfinx_ldrb's keywords).

    The defaults are those of demo/time_dependent/complete_cycle.py (after Doste et al. 2019).
    """

    alpha_endo_lv: float = 60.0
    alpha_epi_lv: float = -60.0
    alpha_endo_rv: float = 90.0
    alpha_epi_rv: float = -25.0
    beta_endo_lv: float = -20.0
    beta_epi_lv: float = 20.0
    beta_endo_rv: float = 0.0
    beta_epi_rv: float = 20.0


class UKBGeometry(_FiberAngles):
    type: Literal["ukb"] = "ukb"
    mode: int = -1
    std: float = 1.5
    case: Literal["ED", "ES"] = "ED"
    char_length_max: float = 5.0
    char_length_min: float = 5.0
    clipped: bool = False
    rotate_base_normal: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="If set, rotate the mesh so the BASE normal points this way (before caching)",
    )
    ldrb: LDRBAngles | None = Field(
        default=None,
        description="Per-ventricle LDRB angles; replaces fiber_angle_endo/fiber_angle_epi, "
        "which are then ignored (computed after rotate_base_normal, before caching)",
    )


GENERATED_GEOMETRY_TYPES = ("lv_ellipsoid", "biv_ellipsoid", "cylinder", "ukb")
# config type -> cardiac_geometries.mesh function name
GENERATORS = {
    "lv_ellipsoid": "lv_ellipsoid",
    "biv_ellipsoid": "biv_ellipsoid",
    "cylinder": "cylinder_D_shaped",
    "ukb": "ukb",
}
# Facet markers whose enclosed volume is reported in loads.csv (when they carry a pressure load).
CAVITY_MARKERS = ("ENDO", "LV", "RV")

GeometryConfig = Annotated[
    Union[
        FolderGeometry,
        BoxGeometry,
        LVEllipsoidGeometry,
        BiVEllipsoidGeometry,
        CylinderGeometry,
        UKBGeometry,
    ],
    Field(discriminator="type"),
]

# --- material ---------------------------------------------------------------------------


class MaterialRegion(BaseModel):
    """Per-cell-marker overrides: ``marker`` plus any parameter of the enclosing material."""

    model_config = ConfigDict(extra="allow")

    marker: str = Field(description="Cell marker name, or an integer cell tag value (e.g. AHA)")

    def values(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class _MaterialBase(_Base):
    # Parameters in kPa (quantity strings); every other parameter is a dimensionless float.
    # ClassVar and no leading underscore: pydantic turns `_name` attributes into private
    # attributes, so every subclass re-annotates these as ClassVar too.
    PRESSURE_FIELDS: ClassVar[tuple[str, ...]] = ()
    FLOAT_FIELDS: ClassVar[tuple[str, ...]] = ()

    region: list[MaterialRegion] = Field(
        default_factory=list,
        description="Per-cell-marker parameter overrides",
    )

    @model_validator(mode="after")
    def _check_regions(self) -> "_MaterialBase":
        allowed = set(self.PRESSURE_FIELDS) | set(self.FLOAT_FIELDS)
        for region in self.region:
            for key, value in region.values().items():
                if key not in allowed:
                    raise ValueError(
                        f"material.region: unknown parameter {key!r}; allowed: {sorted(allowed)}",
                    )
                if key in self.PRESSURE_FIELDS:
                    _check_unit(value, "kPa", f"material.region.{key} (a pressure)")
                elif isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"material.region.{key} must be a number, got {value!r}")
        return self


class HolzapfelOgdenMaterial(_MaterialBase):
    PRESSURE_FIELDS: ClassVar[tuple[str, ...]] = ("a", "a_f", "a_s", "a_fs")
    FLOAT_FIELDS: ClassVar[tuple[str, ...]] = ("b", "b_f", "b_s", "b_fs")

    type: Literal["holzapfel_ogden"] = "holzapfel_ogden"
    preset: Literal["transversely_isotropic", "partly_orthotropic", "orthotropic"] | None = Field(
        default="transversely_isotropic",
        description="pulse.HolzapfelOgden.<preset>_parameters(); explicit values override it",
    )
    a: Pressure | None = None
    b: float | None = None
    a_f: Pressure | None = None
    b_f: float | None = None
    a_s: Pressure | None = None
    b_s: float | None = None
    a_fs: Pressure | None = None
    b_fs: float | None = None


class GuccioneMaterial(_MaterialBase):
    PRESSURE_FIELDS: ClassVar[tuple[str, ...]] = ("C",)
    FLOAT_FIELDS: ClassVar[tuple[str, ...]] = ("bf", "bt", "bfs")

    type: Literal["guccione"] = "guccione"
    C: Pressure = Field(default_factory=lambda: _q("2 kPa"))
    bf: float = 8.0
    bt: float = 2.0
    bfs: float = 4.0


class NeoHookeanMaterial(_MaterialBase):
    PRESSURE_FIELDS: ClassVar[tuple[str, ...]] = ("mu",)

    type: Literal["neo_hookean"] = "neo_hookean"
    mu: Pressure = Field(default_factory=lambda: _q("15 kPa"))


class UsykMaterial(_MaterialBase):
    PRESSURE_FIELDS: ClassVar[tuple[str, ...]] = ("C",)
    FLOAT_FIELDS: ClassVar[tuple[str, ...]] = ("bf", "bs", "bn", "bfs", "bfn", "bsn")

    type: Literal["usyk"] = "usyk"
    C: Pressure = Field(default_factory=lambda: _q("0.88 kPa"))
    bf: float = 8.0
    bs: float = 6.0
    bn: float = 3.0
    bfs: float = 12.0
    bfn: float = 3.0
    bsn: float = 3.0


class SaintVenantKirchhoffMaterial(_MaterialBase):
    PRESSURE_FIELDS: ClassVar[tuple[str, ...]] = ("mu", "lmbda")

    type: Literal["saint_venant_kirchhoff"] = "saint_venant_kirchhoff"
    mu: Pressure
    lmbda: Pressure


MaterialConfig = Annotated[
    Union[
        HolzapfelOgdenMaterial,
        GuccioneMaterial,
        NeoHookeanMaterial,
        UsykMaterial,
        SaintVenantKirchhoffMaterial,
    ],
    Field(discriminator="type"),
]

# --- active / compressibility / viscoelasticity ------------------------------------------


class PassiveConfig(_Base):
    type: Literal["passive"] = "passive"


class ActiveStressConfig(_Base):
    type: Literal["active_stress"] = "active_stress"
    eta: float = Field(default=0.0, ge=0, le=1, description="Transverse fraction of the tension")
    formulation: Literal["invariant", "stretch"] = "invariant"

    @model_validator(mode="after")
    def _stretch_needs_eta_zero(self) -> "ActiveStressConfig":
        if self.formulation == "stretch" and self.eta != 0:
            raise ValueError("active.formulation = 'stretch' requires eta = 0")
        return self


ActiveConfig = Annotated[Union[PassiveConfig, ActiveStressConfig], Field(discriminator="type")]


class IncompressibleConfig(_Base):
    type: Literal["incompressible"] = "incompressible"


class CompressibleConfig(_Base):
    type: Literal["compressible"] = "compressible"
    kappa: Pressure = Field(default_factory=lambda: _q("1e6 Pa"))


class Compressible2Config(_Base):
    type: Literal["compressible2"] = "compressible2"
    kappa: Pressure = Field(default_factory=lambda: _q("1e6 Pa"))


class Compressible3Config(_Base):
    type: Literal["compressible3"] = "compressible3"
    kappa: Pressure = Field(default_factory=lambda: _q("5e4 Pa"))


CompressibilityConfig = Annotated[
    Union[IncompressibleConfig, CompressibleConfig, Compressible2Config, Compressible3Config],
    Field(discriminator="type"),
]


class NoViscoelasticity(_Base):
    type: Literal["none"] = "none"


class ViscousConfig(_Base):
    type: Literal["viscous"] = "viscous"
    eta: Viscosity = Field(default_factory=lambda: _q("100 Pa*s"))


ViscoelasticityConfig = Annotated[
    Union[NoViscoelasticity, ViscousConfig],
    Field(discriminator="type"),
]

# --- boundary conditions ----------------------------------------------------------------


class DirichletConfig(_Base):
    """Zero displacement on a facet marker (all components, or some: a sliding surface)."""

    marker: str
    components: list[Literal["x", "y", "z"]] = Field(
        default_factory=lambda: cast(list[Literal["x", "y", "z"]], ["x", "y", "z"]),
        min_length=1,
    )

    @field_validator("components")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("bcs.dirichlet.components must be unique")
        return v


class RobinConfig(_Base):
    marker: str
    value: str = Field(description="Stiffness (e.g. '1e3 Pa/m'), or damping ('5e3 Pa*s/m')")
    damping: bool = False
    perpendicular: bool = False

    @model_validator(mode="after")
    def _units(self) -> "RobinConfig":
        unit = "Pa*s/m" if self.damping else "Pa/m"
        _check_unit(self.value, unit, f"bcs.robin.value (damping={self.damping})")
        return self


class BCsConfig(_Base):
    base_bc: Literal["fixed", "free"] = "free"
    base_marker: str = "BASE"
    dirichlet: list[DirichletConfig] = Field(default_factory=list)
    robin: list[RobinConfig] = Field(default_factory=list)


# --- loads ------------------------------------------------------------------------------

# Units of every circulation.bestel parameter (all SI in that package).
BESTEL_PRESSURE_UNITS = {
    "t_sys_pre": "s",
    "t_dias_pre": "s",
    "gamma": "s",
    "a_max": "1/s",
    "a_min": "1/s",
    "alpha_pre": "1/s",
    "alpha_mid": "1/s",
    "sigma_pre": "Pa",
    "sigma_mid": "Pa",
}
BESTEL_ACTIVATION_UNITS = {
    "t_sys": "s",
    "t_dias": "s",
    "gamma": "s",
    "a_max": "1/s",
    "a_min": "1/s",
    "sigma_0": "Pa",
}


class ConstantProfile(_Base):
    type: Literal["constant"] = "constant"
    value: Pressure


class RampProfile(_Base):
    """`from_value` until `start`, linear until `end`, then `to_value`."""

    type: Literal["ramp"] = "ramp"
    start: Time = Field(default_factory=lambda: _q("0 s"))
    end: Time
    from_value: Pressure = Field(default_factory=lambda: _q("0 kPa"))
    to_value: Pressure

    @model_validator(mode="after")
    def _order(self) -> "RampProfile":
        if self.end <= self.start:
            raise ValueError("ramp: end must be after start")
        return self


class TableProfile(_Base):
    """Linear interpolation of a table, held constant outside it; `period` repeats it."""

    type: Literal["table"] = "table"
    times: list[float] | None = None
    values: list[float] | None = None
    file: Path | None = Field(default=None, description="CSV file (relative to the config)")
    time_column: str = "time"
    value_column: str = "value"
    time_unit: str = "s"
    value_unit: str = "kPa"
    period: Time | None = None

    @field_validator("time_unit")
    @classmethod
    def _time_unit(cls, v: str) -> str:
        return _check_unit_name(v, "s", "time_unit")

    @field_validator("value_unit")
    @classmethod
    def _value_unit(cls, v: str) -> str:
        return _check_unit_name(v, "kPa", "value_unit")

    @model_validator(mode="after")
    def _source(self) -> "TableProfile":
        inline = self.times is not None or self.values is not None
        if inline == (self.file is not None):
            raise ValueError("table: give either `times` and `values`, or `file`")
        if inline:
            if self.times is None or self.values is None:
                raise ValueError("table: give both `times` and `values`")
            if len(self.times) != len(self.values) or len(self.times) < 2:
                raise ValueError("table: `times` and `values` need the same length (>= 2)")
            if any(b <= a for a, b in zip(self.times, self.times[1:])):
                raise ValueError("table: `times` must be strictly increasing")
        if self.period is not None and self.period.magnitude <= 0:
            raise ValueError("table: period must be positive")
        return self


class _BestelBase(_Base):
    UNITS: ClassVar[dict[str, str]] = {}
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Overrides of the circulation.bestel defaults, as quantities",
    )
    period: Time | None = Field(
        default=None,
        description="Integrate one period from t = 0 on the [time] grid, then repeat it",
    )
    peak: Pressure | None = Field(
        default=None,
        description="Divide the trace by its largest value, then scale it to this (needs period)",
    )

    @model_validator(mode="after")
    def _positive(self) -> "_BestelBase":
        if self.period is not None and self.period.magnitude <= 0:
            raise ValueError("Bestel profile: period must be positive")
        if self.peak is not None and self.peak.magnitude <= 0:
            raise ValueError("Bestel profile: peak must be positive")
        if self.peak is not None and self.period is None:
            raise ValueError(
                "Bestel profile: peak needs period (normalising a non-periodic trace would "
                "depend on time.end_time)",
            )
        return self

    @field_validator("parameters")
    @classmethod
    def _check(cls, v: dict[str, str]) -> dict[str, str]:
        for key, value in v.items():
            if key not in cls.UNITS:
                raise ValueError(f"unknown Bestel parameter {key!r}; allowed: {sorted(cls.UNITS)}")
            _check_unit(value, cls.UNITS[key], f"Bestel parameter {key}")
        return v

    def si_parameters(self) -> dict[str, float]:
        return {k: si(ureg.Quantity(v)) for k, v in self.parameters.items()}


class BestelPressureProfile(_BestelBase):
    UNITS: ClassVar[dict[str, str]] = BESTEL_PRESSURE_UNITS
    type: Literal["bestel_pressure"] = "bestel_pressure"


class BestelActivationProfile(_BestelBase):
    UNITS: ClassVar[dict[str, str]] = BESTEL_ACTIVATION_UNITS
    type: Literal["bestel_activation"] = "bestel_activation"


ProfileConfig = Annotated[
    Union[
        ConstantProfile,
        RampProfile,
        TableProfile,
        BestelPressureProfile,
        BestelActivationProfile,
    ],
    Field(discriminator="type"),
]


class LoadConfig(_Base):
    """A prescribed load: a pressure (the Neumann BC on `marker`) or the active tension Ta."""

    target: Literal["pressure", "activation"]
    marker: str | None = Field(default=None, description="Facet marker (pressure loads only)")
    profile: ProfileConfig

    @model_validator(mode="after")
    def _marker(self) -> "LoadConfig":
        if self.target == "pressure" and not self.marker:
            raise ValueError("a pressure load needs a marker")
        if self.target == "activation" and self.marker is not None:
            raise ValueError("an activation load takes no marker")
        return self

    @property
    def name(self) -> str:
        return "activation" if self.target == "activation" else f"pressure_{self.marker}"


# --- time / problem / solver -----------------------------------------------------------


class TimeConfig(_Base):
    """One time axis: pseudo-time for static problems, physical time for dynamic ones."""

    start_time: Time = Field(default_factory=lambda: _q("0 s"))
    end_time: Time
    dt: Time | None = None
    num_steps: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "TimeConfig":
        if (self.dt is None) == (self.num_steps is None):
            raise ValueError("time: give exactly one of `dt` or `num_steps`")
        if self.end_time <= self.start_time:
            raise ValueError("time: end_time must be after start_time")
        if self.dt is not None and self.dt.magnitude <= 0:
            raise ValueError("time: dt must be positive")
        if self.dt is not None and not _is_multiple(self.end_s() - self.start_s(), self.dt_s()):
            raise ValueError(
                "time: end_time - start_time must be an integer multiple of dt "
                f"({self.end_s() - self.start_s():g} s vs dt = {self.dt_s():g} s)",
            )
        return self

    def start_s(self) -> float:
        return si(self.start_time)

    def end_s(self) -> float:
        return si(self.end_time)

    def dt_s(self) -> float:
        if self.dt is not None:
            return si(self.dt)
        assert self.num_steps is not None
        return (self.end_s() - self.start_s()) / self.num_steps

    def n_steps(self) -> int:
        if self.num_steps is not None:
            return self.num_steps
        return max(1, round((self.end_s() - self.start_s()) / self.dt_s()))


class ProblemConfig(_Base):
    type: Literal["static", "dynamic"] = "static"
    u_space: str = "P_2"
    p_space: str = "P_1"
    rigid_body_constraint: bool = False
    rho: Density = Field(default_factory=lambda: _q("1000 kg/m**3"), description="dynamic only")
    alpha_m: float = Field(default=0.2, description="Generalized-alpha, dynamic only")
    alpha_f: float = Field(default=0.4, description="Generalized-alpha, dynamic only")


class SolverConfig(_Base):
    max_halvings: int = Field(
        default=4,
        ge=0,
        description="On Newton failure, split the step in two, at most this many times deep",
    )
    petsc_options: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Merged over pulse's defaults",
    )
    preconditioner_lag: int | None = Field(
        default=None,
        ge=1,
        description="circulation.type = 'cycle' only: the steady-state snes_lag_preconditioner "
        "(refreshed after every phase change)",
    )


# --- circulation ------------------------------------------------------------------------


class WindkesselConfig(_Base):
    """A three-element Windkessel (pulse.cycle.Windkessel)."""

    p_init: Pressure
    compliance: Compliance
    resistance: Resistance
    characteristic_impedance: Resistance = Field(default_factory=lambda: _q("0 Pa*s/m**3"))


class CycleCavityConfig(_Base):
    """One cavity of the five-phase cycle (pulse.cycle.CycleParams)."""

    marker: str
    period: Time
    t_zero: Time
    t_end_diastole: Time
    preload_pressure: Pressure
    p_end_diastole: Pressure = Field(description="PRELOAD ends here; also the prestress target")
    p_fill: Pressure
    filling_rate: FlowRate
    min_ejection_duration: Time = Field(default_factory=lambda: _q("10 ms"))
    windkessel: WindkesselConfig

    @model_validator(mode="after")
    def _timing(self) -> "CycleCavityConfig":
        if not 0 < si(self.t_zero) <= si(self.t_end_diastole) < si(self.period):
            raise ValueError("circulation.cavity: need 0 < t_zero <= t_end_diastole < period")
        return self


class NoCirculation(_Base):
    type: Literal["none"] = "none"


class CycleCirculation(_Base):
    """CycleController: one CavityControl and Windkessel per cavity."""

    type: Literal["cycle"] = "cycle"
    cavity: list[CycleCavityConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> "CycleCirculation":
        markers = [c.marker for c in self.cavity]
        if len(set(markers)) != len(markers):
            raise ValueError(f"circulation.cavity markers must be unique, got {markers}")
        return self


class ChamberConfig(_Base):
    """Ties a cavity (facet marker) to a chamber of the .ode circuit."""

    marker: str
    volume_state: str = Field(description="The circuit state holding the chamber volume (mL)")
    pressure_missing: str = Field(description="The missing variable the chamber pressure feeds")


class PhaseInputConfig(_Base):
    """A time-derived missing variable: t mod period (e.g. Regazzoni's beat_phase)."""

    type: Literal["phase"] = "phase"
    period: Time

    @model_validator(mode="after")
    def _positive(self) -> "PhaseInputConfig":
        if self.period.magnitude <= 0:
            raise ValueError("circulation.inputs: period must be positive")
        return self


class _OdeCirculation(_Base):
    ode_file: Path = Field(
        description='gotranx .ode file: a path relative to the config, or "<package>:<file>" '
        'for a file inside an installed package (e.g. "circulation:regazzoni2020.ode"); the '
        "physics hash covers its contents",
    )
    drop_components: list[str] = Field(default_factory=list)
    parameters: dict[str, float] = Field(
        default_factory=dict,
        description="Parameter overrides by name, in the .ode file's own units (plain numbers)",
    )
    initial_state: dict[str, float] = Field(
        default_factory=dict,
        description="Initial values by state name, in the .ode file's own units; coupled "
        "chamber volumes always come from the mesh",
    )
    record: list[str] = Field(
        default_factory=list,
        description="Monitored expressions of the .ode file to add to loads.csv",
    )
    chamber: list[ChamberConfig] = Field(min_length=1)
    inputs: dict[str, PhaseInputConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _names(self) -> "_OdeCirculation":
        for what in ("marker", "volume_state", "pressure_missing"):
            values = [getattr(c, what) for c in self.chamber]
            if len(set(values)) != len(values):
                raise ValueError(f"circulation.chamber {what}s must be unique, got {values}")
        volumes = sorted(set(self.initial_state) & {c.volume_state for c in self.chamber})
        if volumes:
            raise ValueError(
                f"circulation.initial_state must not set coupled chamber volumes {volumes}: "
                "they come from the mesh",
            )
        clash = sorted(set(self.inputs) & {c.pressure_missing for c in self.chamber})
        if clash:
            raise ValueError(f"circulation.inputs {clash} are chamber pressures already")
        if len(set(self.record)) != len(self.record):
            raise ValueError("circulation.record names must be unique")
        return self


class SplitCirculation(_OdeCirculation):
    """The .ode circuit stepped by forward Euler, one mechanics solve per step."""

    type: Literal["split"] = "split"


class MonolithicCirculation(_OdeCirculation):
    """The .ode circuit's states solved in the mechanics' own Newton system."""

    type: Literal["monolithic"] = "monolithic"
    scheme: Literal["backward_euler", "bdf2"] = "backward_euler"


CirculationConfig = Annotated[
    Union[NoCirculation, CycleCirculation, SplitCirculation, MonolithicCirculation],
    Field(discriminator="type"),
]


def coupled_markers(circulation: Any) -> list[str]:
    """Facet markers whose cavity the circulation couples."""
    if circulation.type == "cycle":
        return [c.marker for c in circulation.cavity]
    if circulation.type in ("split", "monolithic"):
        return [c.marker for c in circulation.chamber]
    return []


# --- prestress ----------------------------------------------------------------------------


class PrestressTarget(_Base):
    marker: str
    pressure: Pressure


class PrestressConfig(_Base):
    """Recover the unloaded reference configuration before the run (PrestressProblem)."""

    ramp_steps: int = Field(default=20, ge=1)
    cache_folder: Path = Field(
        default=Path("prestress"),
        description="Cache root (relative to the config); each result in its own <hash>/ "
        "subfolder; never deleted by --overwrite",
    )
    inflate_steps: int = Field(
        default=0,
        ge=0,
        description="> 0: ramp the chamber volumes back to the imaged ones in this many static "
        "steps before the run (split/monolithic only)",
    )
    target: list[PrestressTarget] = Field(
        default_factory=list,
        description="Cavity pressures of the imaged mesh; not allowed with circulation.type = "
        "'cycle', whose targets are p_end_diastole",
    )

    @model_validator(mode="after")
    def _unique(self) -> "PrestressConfig":
        markers = [t.marker for t in self.target]
        if len(set(markers)) != len(markers):
            raise ValueError(f"prestress.target markers must be unique, got {markers}")
        return self


# --- output / postprocess ---------------------------------------------------------------


class OutputConfig(_Base):
    folder: Path = Path("output")
    save_every: Time | None = Field(default=None, description="Default: every step")
    checkpoint_every: Time = Field(
        default_factory=lambda: _q("0 s"),
        description="Restart checkpoint interval; 0 = end only",
    )
    performance: bool = Field(
        default=False,
        description="Time Newton solves and runner phases; log every log_every steps and write "
        "performance.json",
    )
    log_every: int = Field(default=10, ge=1)

    def save_stride(self, dt: float) -> int:
        if self.save_every is None:
            return 1
        return max(1, round(si(self.save_every) / dt))

    def checkpoint_stride(self, dt: float) -> int:
        every = si(self.checkpoint_every)
        return round(every / dt) if every > 0 else 0


class PostprocessConfig(_Base):
    vtx: bool = True
    fields: list[Literal["fiber_stress", "fiber_strain"]] = Field(default_factory=list)
    points: dict[str, list[float]] = Field(
        default_factory=dict,
        description="name -> reference coordinates (mesh units, after scale)",
    )
    vertex_tags: dict[str, str] = Field(
        default_factory=dict,
        description="name -> vertex marker (e.g. ENDOPT)",
    )
    plots: bool = True


class Config(_Base):
    geometry: GeometryConfig
    material: MaterialConfig = Field(default_factory=HolzapfelOgdenMaterial)
    active: ActiveConfig = Field(default_factory=PassiveConfig)
    compressibility: CompressibilityConfig = Field(default_factory=IncompressibleConfig)
    viscoelasticity: ViscoelasticityConfig = Field(default_factory=NoViscoelasticity)
    bcs: BCsConfig = Field(default_factory=BCsConfig)
    load: list[LoadConfig] = Field(default_factory=list)
    circulation: CirculationConfig = Field(default_factory=NoCirculation)
    prestress: PrestressConfig | None = None
    time: TimeConfig
    problem: ProblemConfig = Field(default_factory=ProblemConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)

    @model_validator(mode="after")
    def _cross_checks(self) -> "Config":
        dt = self.time.dt_s()
        tol = 1e-12 * dt
        if self.output.save_every is not None:
            save_every = si(self.output.save_every)
            if save_every < dt - tol:
                raise ValueError("output.save_every must be >= the time step")
            if not _is_multiple(save_every, dt):
                raise ValueError(
                    f"output.save_every must be an integer multiple of the time step ({dt:g} s)",
                )
        every = si(self.output.checkpoint_every)
        if 0 < every < dt - tol:
            raise ValueError("output.checkpoint_every must be 0 or >= the time step")
        if every > 0 and not _is_multiple(every, dt):
            raise ValueError(
                f"output.checkpoint_every must be 0 or an integer multiple of the time step "
                f"({dt:g} s)",
            )
        if self.problem.type != "dynamic":
            if self.viscoelasticity.type == "viscous":
                raise ValueError(
                    "viscoelasticity.type = 'viscous' only has an effect for "
                    "problem.type = 'dynamic'",
                )
            damped = [r.marker for r in self.bcs.robin if r.damping]
            if damped:
                raise ValueError(
                    f"bcs.robin damping = true (markers {damped}) only has an effect for "
                    "problem.type = 'dynamic'",
                )
        names = [load.name for load in self.load]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"load: duplicate loads {duplicates} (one per target and marker)")
        for load in self.load:
            profile = load.profile
            period = getattr(profile, "period", None)
            if isinstance(profile, _BestelBase) and period is not None:
                if not _is_multiple(si(period), dt):
                    raise ValueError(
                        f"load {load.name}: the Bestel period ({si(period):g} s) must be an "
                        f"integer multiple of the time step ({dt:g} s)",
                    )
        circulation = self.circulation
        coupled = coupled_markers(circulation)
        if coupled and self.geometry.unit != "m":
            raise ValueError(
                f"circulation.type = {circulation.type!r} needs geometry.unit = 'm' (cavity "
                f"volumes and pressures are SI), got {self.geometry.unit!r}",
            )
        if circulation.type == "cycle" and self.time.start_s() != 0.0:
            raise ValueError("circulation.type = 'cycle' needs time.start_time = 0 s")
        if circulation.type == "split" and self.problem.type != "static":
            raise ValueError(
                "circulation.type = 'split' needs problem.type = 'static' (its initial solve "
                "at t0 would otherwise be a spurious dynamic step)",
            )
        if self.solver.preconditioner_lag is not None and circulation.type != "cycle":
            raise ValueError("solver.preconditioner_lag only applies to circulation.type = 'cycle'")
        loaded = sorted({load.marker for load in self.load if load.marker in coupled})
        if loaded:
            raise ValueError(
                f"load: pressure loads on coupled cavity markers {loaded}; the wall load there "
                "comes from the cavity's pressure unknown",
            )
        prestress = self.prestress
        if prestress is not None:
            targets = {t.marker for t in prestress.target}
            if circulation.type == "cycle" and targets:
                raise ValueError(
                    "prestress.target is not allowed with circulation.type = 'cycle': the "
                    "targets are each cavity's p_end_diastole",
                )
            if circulation.type != "cycle" and not targets:
                raise ValueError("prestress needs at least one [[prestress.target]]")
            if prestress.inflate_steps > 0:
                if circulation.type not in ("split", "monolithic"):
                    raise ValueError(
                        "prestress.inflate_steps > 0 needs circulation.type = 'split' or "
                        "'monolithic'",
                    )
                missing = sorted(set(coupled) - targets)
                if missing:
                    raise ValueError(
                        f"prestress.inflate_steps > 0: chamber markers {missing} need a "
                        "[[prestress.target]] (re-inflation goes back to their imaged volumes)",
                    )
            if self.problem.rigid_body_constraint:
                raise ValueError(
                    "prestress cannot be combined with problem.rigid_body_constraint = true",
                )
            if self.geometry.type == "box":
                raise ValueError("prestress needs a cardiac geometry, not geometry.type = 'box'")
        return self


ALL_MODELS: tuple[type[BaseModel], ...] = tuple(
    obj
    for obj in list(globals().values())
    if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
)
