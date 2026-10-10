# # Pericardium: sliding springs versus normal springs
#
# The pericardium is a stiff sac around the heart, separated from it by a thin film of
# fluid. It holds the epicardium in place in the normal direction but lets it slide along
# the sac without friction {cite}`pfaller2019importance`. A common model is a spring on
# the epicardium, a `pulse.RobinBC`, with stiffnesses up to $5 \cdot 10^7$ Pa/m
# {cite}`strocchi2020simulating`.
#
# A `RobinBC` measures the gap to the pericardium by projecting the displacement onto a
# normal, $k (\mathbf{u} \cdot \mathbf{n}) \mathbf{n}$. On a curved surface, the
# epicardium sliding along itself then counts as a normal gap. A rotation by
# $\theta$ about the long axis gives a gap of about $r (1 - \cos \theta)$, so stiff springs
# hold back the twist of the ventricle.
#
# `pulse.PericardiumBC` measures the gap as the change in signed distance $d$ to a surface
# fixed in space, $g = d(\mathbf{X} + \mathbf{u}) - d(\mathbf{X})$, and stores the energy
# $\frac{1}{2} \int k g^2 \, dA$. A point that slides along the surface keeps its
# distance to it, and feels no force.
#
# This demo contracts an LV ellipsoid held by either epicardial condition, at the same
# stiffness, and compares the twist, the motion of the base and apex, and the volumes.
#
# ---

import math
from pathlib import Path

from mpi4py import MPI

import cardiac_geometries
import cardiac_geometries.geometry
import dolfinx
import numpy as np
import ufl

import pulse

# ## Geometry
#
# An LV ellipsoid in metres, with its long axis along $x$, the apex at
# $x = -97$ mm and the base at $x \approx 24$ mm.

outdir = Path("lv_ellipsoid_pericardium")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"
comm = MPI.COMM_WORLD

if not (geodir / "mesh.xdmf").exists():
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="Quadrature_6",
        r_short_endo=0.025,
        r_short_epi=0.035,
        r_long_endo=0.09,
        r_long_epi=0.097,
        psize_ref=0.008,
        mu_apex_endo=-math.pi,
        mu_base_endo=-math.acos(5 / 17),
        mu_apex_epi=-math.pi,
        mu_base_epi=-math.acos(5 / 20),
        comm=comm,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )

geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo)
markers = {name: geometry.markers[name][0] for name in ("ENDO", "EPI", "BASE")}

# The pericardium is the epicardial ellipsoid itself, read from the parameters the mesh
# was made with. In general it can be any object with a `signed_distance(x)` method that
# returns the signed distance to the surface (positive outside) as a UFL expression.

surface = pulse.EllipsoidSurface.from_cardiac_geometries(geo)
print(surface)

# ## Model
#
# A transversely isotropic Holzapfel-Ogden material, contracted by an active stress
# along the fibres.


def constant(value):
    return dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(value))


def create_model():
    material = pulse.HolzapfelOgden(
        f0=geo.f0,
        s0=geo.s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    Ta = pulse.Variable(constant(0.0), "kPa")
    model = pulse.CardiacModel(
        material=material,
        active=pulse.ActiveStress(geo.f0, activation=Ta),
        compressibility=pulse.Compressible(),
    )
    return model, Ta


# ## Boundary conditions
#
# The base hangs on springs. They resist its in-plane motion firmly, and its motion along
# the long axis only weakly, so that the base can descend towards the apex as the
# ventricle contracts. On the epicardium we put either a `RobinBC` along the current
# normal (pulse's default for springs) or a `PericardiumBC`, both with
# $k = 10^7$ Pa/m.

k_epi = 1e7  # Pa/m


def create_bcs(epicardium: str):
    pressure = pulse.Variable(constant(0.0), "kPa")
    neumann = [pulse.NeumannBC(traction=pressure, marker=markers["ENDO"])]
    robin = [
        pulse.RobinBC(value=pulse.Variable(constant(1e5), "Pa / m"), marker=markers["BASE"]),
        pulse.RobinBC(
            value=pulse.Variable(constant(1e6), "Pa / m"),
            marker=markers["BASE"],
            perpendicular=True,
        ),
    ]
    pericardium = []
    k = pulse.Variable(constant(k_epi), "Pa / m")
    if epicardium == "RobinBC":
        robin.append(pulse.RobinBC(value=k, marker=markers["EPI"]))
    elif epicardium == "PericardiumBC":
        pericardium.append(pulse.PericardiumBC(stiffness=k, marker=markers["EPI"], surface=surface))
    else:
        raise ValueError(epicardium)
    bcs = pulse.BoundaryConditions(neumann=neumann, robin=robin, pericardium=pericardium)
    return bcs, pressure


# ## Measurements
#
# We measure the cavity volume, the mean displacement of the base and of the apex along
# the long axis, and the mean rotation about the long axis of the epicardium in slices
# along it. The twist is the rotation of an apical slice minus that of a basal one. The
# gap is how far the epicardium has moved away from (or into) the pericardium, as a root
# mean square of $d(\mathbf{X} + \mathbf{u}) - d(\mathbf{X})$. The smallest Jacobian
# determinant $J$ tells us whether any element is about to invert.


def measurements(problem):
    u = problem.u
    V = problem.u_space
    X = V.tabulate_dof_coordinates()
    U = u.x.array.reshape(-1, 3)
    n_owned = V.dofmap.index_map.size_local

    def dofs(marker):
        found = dolfinx.fem.locate_dofs_topological(V, 2, geometry.facet_tags.find(marker))
        return found[found < n_owned]

    epi, base = dofs(markers["EPI"]), dofs(markers["BASE"])

    def mean(values):
        total = comm.allreduce(np.sum(values, axis=0), op=MPI.SUM)
        return total / comm.allreduce(len(values), op=MPI.SUM)

    def rotation(lo, hi):
        sel = epi[(X[epi, 0] > lo) & (X[epi, 0] < hi)]
        r0 = X[sel, 1:]
        r1 = r0 + U[sel, 1:]
        angle = np.arctan2(r0[:, 0] * r1[:, 1] - r0[:, 1] * r1[:, 0], (r0 * r1).sum(axis=1))
        return float(np.degrees(mean(angle)))

    # The cavity is open at the base, so measure its volume from a point on the base
    x_base = geometry.base_center("BASE", u)
    F = ufl.Identity(3) + ufl.grad(u)
    x = geometry.X + u - ufl.as_vector(x_base)
    volume_form = (-1 / 3) * ufl.det(F) * ufl.dot(x, ufl.inv(F).T * geometry.facet_normal)
    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(volume_form * geometry.ds(markers["ENDO"]))),
        op=MPI.SUM,
    )

    ds_epi = geometry.ds(markers["EPI"])
    gap = surface.signed_distance(geometry.X + u) - surface.signed_distance(geometry.X)
    gap2, area = (
        comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(f * ds_epi)), op=MPI.SUM)
        for f in (gap**2, constant(1.0))
    )

    Q = dolfinx.fem.functionspace(geometry.mesh, ("DG", 0))
    J = dolfinx.fem.Function(Q)
    J.interpolate(dolfinx.fem.Expression(ufl.det(F), Q.element.interpolation_points))
    apex = epi[X[epi, 0] < -0.095]

    slices = np.linspace(-0.09, 0.02, 12)
    return {
        "volume [mL]": volume * 1e6,
        "base [mm]": 1e3 * mean(U[base, 0]),
        "apex [mm]": 1e3 * mean(U[apex, 0]),
        "basal rotation [deg]": rotation(0.0, 0.02),
        "apical rotation [deg]": rotation(-0.085, -0.065),
        "gap [mm]": 1e3 * math.sqrt(gap2 / area),
        "min J": comm.allreduce(J.x.array.min(), op=MPI.MIN),
        "profile": (
            slices[:-1] + np.diff(slices) / 2,
            [rotation(a, b) for a, b in zip(slices[:-1], slices[1:])],
        ),
    }


# ## Simulation
#
# We inflate the ventricle to an end-diastolic pressure of 1.5 kPa, then raise the active
# tension to 40 kPa at that pressure. `solve` returns `False` if Newton does not
# converge, so we check every step.


def run(epicardium: str):
    model, Ta = create_model()
    bcs, pressure = create_bcs(epicardium)
    problem = pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters={"base_bc": pulse.BaseBC.free, "mesh_unit": "m"},
    )

    for value in np.linspace(0.0, 1.5, 4)[1:]:
        pressure.assign(value)
        if not problem.solve():
            raise RuntimeError(f"{epicardium}: Newton did not converge at {value} kPa")
    end_diastole = measurements(problem)

    for value in np.linspace(0.0, 40.0, 9)[1:]:
        Ta.assign(value)
        if not problem.solve():
            raise RuntimeError(f"{epicardium}: Newton did not converge at Ta = {value} kPa")
    end_systole = measurements(problem)
    return end_diastole, end_systole


results = {name: run(name) for name in ("RobinBC", "PericardiumBC")}

# ## Results

if comm.rank == 0:
    rows = [
        ("EDV [mL]", lambda ed, es: ed["volume [mL]"]),
        ("ESV [mL]", lambda ed, es: es["volume [mL]"]),
        ("EF [%]", lambda ed, es: 100 * (1 - es["volume [mL]"] / ed["volume [mL]"])),
        ("base, systole [mm]", lambda ed, es: es["base [mm]"] - ed["base [mm]"]),
        ("apex, systole [mm]", lambda ed, es: es["apex [mm]"] - ed["apex [mm]"]),
        ("basal rotation [deg]", lambda ed, es: es["basal rotation [deg]"]),
        ("apical rotation [deg]", lambda ed, es: es["apical rotation [deg]"]),
        (
            "twist [deg]",
            lambda ed, es: es["apical rotation [deg]"] - es["basal rotation [deg]"],
        ),
        ("gap, systole [mm]", lambda ed, es: es["gap [mm]"]),
        ("min J", lambda ed, es: es["min J"]),
    ]
    print(f"{'':24}" + "".join(f"{name:>16}" for name in results))
    for label, f in rows:
        print(f"{label:24}" + "".join(f"{f(*results[name]):16.2f}" for name in results))

# The displacements of the base and the apex are along the long axis, from end diastole to
# end systole; negative is towards the apex. Rotations are positive counterclockwise when
# seen from the apex.

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed")
else:
    if comm.rank == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, (_, end_systole) in results.items():
            x, rotation = end_systole["profile"]
            ax.plot(1e3 * x, rotation, marker="o", label=name)
        ax.set_xlabel("Position along the long axis [mm] (apex left, base right)")
        ax.set_ylabel("Rotation at end systole [deg]")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "rotation.png")

# ![rotation](lv_ellipsoid_pericardium/rotation.png)
#
# ## Discussion
#
# Both conditions hold the apex in place and let the base descend towards it, as the
# pericardium does in vivo, and they fill and eject about the same volumes. On this mesh
# the `RobinBC` resists the rotation of the epicardium, so the ventricle twists less, and
# yet it lets the epicardium move further from the pericardium. How much a `RobinBC` along
# the current normal resists the twist depends on the mesh: on finer meshes it resists it
# less, and lets the epicardium move further still. Along the reference normal it resists
# the twist far more, on any mesh. The `PericardiumBC` lets the epicardium slide, keeps it
# at its distance from the pericardium on any mesh, and lets the ventricle twist at least
# as much as one without epicardial springs.
#
# Both conditions keep the epicardium near where it is in the reference configuration.
# Here that is the unloaded ventricle, so stiff springs also resist filling. Softer
# springs ($k \lesssim 10^6$ Pa/m) resist it less, and there the two conditions differ
# less too.
#
# `PericardiumBC` takes a `damping` coefficient for a dashpot in a `DynamicProblem`, and
# `unilateral=True` makes it act only where the epicardium moves outward. In systole the
# ventricle moves inward, so a unilateral pericardium does nothing then.
