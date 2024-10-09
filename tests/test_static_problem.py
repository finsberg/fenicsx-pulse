import math

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import cardiac_geometries
import fenicsx_pulse
import fenicsx_pulse.problem
from fenicsx_pulse.problem import BaseBC, StaticProblem


def get_geo_biv(geodir):
    comm = MPI.COMM_WORLD

    if not (geodir / "mesh.xdmf").exists():
        comm.barrier()
        cardiac_geometries.mesh.cardiac_geometries.mesh.biv_ellipsoid(
            outdir=geodir,
            create_fibers=True,
            fiber_space="Quadrature_6",
            comm=comm,
            fiber_angle_epi=-60,
            fiber_angle_endo=60,
        )
    return cardiac_geometries.geometry.Geometry.from_folder(
        comm=comm,
        folder=geodir,
    )


def get_geo_flat_base_mm(geodir):
    comm = MPI.COMM_WORLD

    if not (geodir / "mesh.xdmf").exists():
        comm.barrier()
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=geodir,
            create_fibers=True,
            fiber_space="Quadrature_6",
            comm=comm,
            psize_ref=10.0,
            fiber_angle_epi=-60,
            fiber_angle_endo=60,
        )
    return cardiac_geometries.geometry.Geometry.from_folder(
        comm=comm,
        folder=geodir,
    )


def get_geo_no_flat_base_m(geodir):
    comm = MPI.COMM_WORLD

    if not (geodir / "mesh.xdmf").exists():
        comm.barrier()
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=geodir,
            create_fibers=True,
            fiber_space="Quadrature_6",
            r_short_endo=0.025,
            r_short_epi=0.035,
            r_long_endo=0.09,
            r_long_epi=0.097,
            psize_ref=0.03,
            mu_apex_endo=-math.pi,
            mu_base_endo=-math.acos(5 / 17),
            mu_apex_epi=-math.pi,
            mu_base_epi=-math.acos(5 / 20),
            comm=comm,
            fiber_angle_epi=-60,
            fiber_angle_endo=60,
        )
    return cardiac_geometries.geometry.Geometry.from_folder(
        comm=comm,
        folder=geodir,
    )


def get_geo(geo_str, tmp_path_factory):
    geodir = tmp_path_factory.mktemp(geo_str)
    if geo_str == "geo_flat_base_mm":
        return get_geo_flat_base_mm(geodir)
    elif geo_str == "geo_no_flat_base_m":
        return get_geo_no_flat_base_m(geodir)
    else:
        raise ValueError(f"Unknown geometry {geo_str}")


def cardiac_model(geo, comp_model_cls):
    material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore
    Ta = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0)),
        "kPa",
    )
    active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta)
    comp_model = comp_model_cls()

    return fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )


def handle_base_bc(base_bc, geometry):
    if base_bc == BaseBC.fixed:
        robin = ()
    elif base_bc == BaseBC.free:
        alpha_epi = fenicsx_pulse.Variable(
            dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e8)),
            "Pa / m",
        )
        robin_epi = fenicsx_pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])

        alpha_base = fenicsx_pulse.Variable(
            dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)),
            "Pa / m",
        )
        robin_base = fenicsx_pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])
        robin = (robin_epi, robin_base)

    else:
        raise ValueError(f"Unknown base_bc {base_bc}")

    return robin


def handle_control_lv(
    control,
    geometry,
    target_lvp,
    target_lvv,
    robin,
    initial_volume,
    initial_lvp,
):
    comm = geometry.mesh.comm
    if control == "pressure":
        traction = fenicsx_pulse.Variable(
            dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(initial_lvp)),
            "Pa",
        )
        neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])
        bcs = fenicsx_pulse.BoundaryConditions(neumann=(neumann,), robin=robin)
        cavities = []

        def gen(problem):
            dvlp = (target_lvp - initial_lvp) / 3.0

            for lvp in np.arange(0.0, target_lvp + 1e-9, dvlp):
                traction.assign(lvp)
                problem.solve()
                lvv = comm.allreduce(geometry.volume("ENDO", u=problem.u), op=MPI.SUM)
                yield lvp, lvv

    elif control == "volume":
        volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(initial_volume))
        cavity = fenicsx_pulse.problem.Cavity(marker="ENDO", volume=volume)
        cavities = [cavity]
        bcs = fenicsx_pulse.BoundaryConditions(robin=robin)

        def gen(problem):
            dlv = (target_lvv - initial_volume) / 3.0

            for lvv in np.arange(initial_volume, target_lvv + 1e-9, dlv):
                volume.value = lvv
                problem.solve()
                new_lvv = comm.allreduce(geometry.volume("ENDO", u=problem.u), op=MPI.SUM)
                lvp = comm.allgather(problem.cavity_pressures[0].x.array)[0]
                assert np.isclose(lvv, new_lvv, rtol=0.01)
                yield lvp, lvv

    else:
        raise ValueError(f"Unknown control {control}")

    return bcs, cavities, gen


@pytest.mark.parametrize(
    "rigid, geo_str, base_bc",
    [
        (False, "geo_no_flat_base_m", BaseBC.fixed),
        (True, "geo_no_flat_base_m", BaseBC.fixed),
        (False, "geo_flat_base_mm", BaseBC.fixed),
        (True, "geo_flat_base_mm", BaseBC.fixed),
        (False, "geo_no_flat_base_m", BaseBC.free),
        (True, "geo_no_flat_base_m", BaseBC.free),
        (False, "geo_flat_base_mm", BaseBC.free),
        (True, "geo_flat_base_mm", BaseBC.free),
    ],
)
@pytest.mark.parametrize("control", ["pressure", "volume"])
@pytest.mark.parametrize(
    "comp_model_cls",
    [
        fenicsx_pulse.compressibility.Incompressible,
        fenicsx_pulse.compressibility.Compressible2,
    ],
)
def test_static_problem_lv(
    rigid,
    geo_str,
    base_bc,
    control,
    comp_model_cls,
    tmp_path_factory,
):
    geo = get_geo(geo_str, tmp_path_factory)
    geometry = fenicsx_pulse.HeartGeometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": 6},
    )
    model = cardiac_model(geo, comp_model_cls)

    robin = handle_base_bc(geometry=geometry, base_bc=base_bc)

    initial_volume = geo.mesh.comm.allreduce(geometry.volume("ENDO"), op=MPI.SUM)
    initial_lvp = 0.0

    target_lvp = 300.0  # Pa
    target_lvv = initial_volume * 1.05  # 5% increase in volume

    bcs, cavities, gen = handle_control_lv(
        control=control,
        geometry=geometry,
        target_lvp=target_lvp,
        target_lvv=target_lvv,
        robin=robin,
        initial_volume=initial_volume,
        initial_lvp=initial_lvp,
    )

    problem = StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters={
            "mesh_unit": geo_str.split("_")[-1],
            "base_bc": base_bc,
            "rigid_body_constraint": rigid,
        },
        cavities=cavities,
    )
    problem.solve()
    for lvp, lvv in gen(problem):
        print(f"LVP: {lvp}, LVV: {lvv}")

    lvv_inflated = lvv
    lvp_inflated = lvp

    if control == "pressure":
        assert np.isclose(lvp_inflated, target_lvp, rtol=0.01)
        assert lvv_inflated > initial_volume  # Volume will increase
    else:
        assert np.isclose(lvv_inflated, target_lvv, rtol=0.01)
        assert lvp_inflated > initial_lvp  # Pressure will increase

    # Now add some active stress
    for tai in [0.1, 0.5, 0.7]:
        model.active.activation.assign(tai)
        problem.solve()
        lvv = geo.mesh.comm.allreduce(geometry.volume("ENDO", u=problem.u), op=MPI.SUM)
        print(f"Ta: {tai}, LVV: {lvv}")

    # When pressure is the control, the pressure will remain constant and the volume
    # will change in response to change in Ta.
    # When volume is the control, the volume will remain constant and the
    # pressure will change.

    if control == "pressure":
        # Pressure remains the same to volume should decrease
        assert lvv < lvv_inflated
    else:
        # Volume remains the same so pressure should increase
        lvp = geo.mesh.comm.allgather(problem.cavity_pressures[0].x.array)[0]

        assert lvp > lvp_inflated
