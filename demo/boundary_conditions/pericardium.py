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
# stiffness, and compares the twist, the motion of the base and apex, and the volumes. It
# ends with a movie of the two side by side. The pericardium is the epicardial ellipsoid,
# first as a formula and then, for comparison, as a surface made from the mesh.
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

# A geometry from an image has no such formula. For that, `pulse.MeshSurface` makes a
# smooth surface through the facets with a given marker, as they are now; see
# [the next demo](pericardium_ukb.py). We run it here too, to compare it with the
# ellipsoid that these facets approximate.

mesh_surface = pulse.MeshSurface.from_geometry(geometry, "EPI")
print(mesh_surface)

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
# normal (pulse's default for springs) or a `PericardiumBC`, with the ellipsoid or with
# the surface from the mesh, all with $k = 10^7$ Pa/m.

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
    elif epicardium == "MeshSurface":
        pericardium.append(
            pulse.PericardiumBC(stiffness=k, marker=markers["EPI"], surface=mesh_surface),
        )
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
# converge, so we check every step. After every step we keep the displacement,
# interpolated onto linear elements, for the movie at the end.

V1 = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (3,)))
frames: dict[str, list[tuple[str, np.ndarray]]] = {}


def run(epicardium: str):
    model, Ta = create_model()
    bcs, pressure = create_bcs(epicardium)
    problem = pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters={"base_bc": pulse.BaseBC.free, "mesh_unit": "m"},
    )
    u1 = dolfinx.fem.Function(V1)
    frames[epicardium] = []

    def keep(p, ta):
        u1.interpolate(problem.u)
        label = f"p = {p:.1f} kPa, Ta = {ta:.0f} kPa"
        frames[epicardium].append((label, u1.x.array.reshape(-1, 3).copy()))

    keep(0.0, 0.0)
    for step, value in enumerate(np.linspace(0.0, 1.5, 7)[1:]):
        pressure.assign(value)
        if not problem.solve():
            raise RuntimeError(f"{epicardium}: Newton did not converge at {value} kPa")
        if step % 2 == 1:
            keep(value, 0.0)
    end_diastole = measurements(problem)

    for value in np.linspace(0.0, 40.0, 9)[1:]:
        Ta.assign(value)
        if not problem.solve():
            raise RuntimeError(f"{epicardium}: Newton did not converge at Ta = {value} kPa")
        keep(1.5, value)
    end_systole = measurements(problem)
    return end_diastole, end_systole


results = {name: run(name) for name in ("RobinBC", "PericardiumBC", "MeshSurface")}

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
# seen from the base. The last column is the `PericardiumBC` with the surface from the mesh;
# its gap is measured to the ellipsoid, like the others.

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed")
else:
    if comm.rank == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, (_, end_systole) in results.items():
            x, rotation = end_systole["profile"]
            style = {"linestyle": "--", "marker": "x"} if name == "MeshSurface" else {"marker": "o"}
            ax.plot(1e3 * x, rotation, label=name, **style)
        ax.set_xlabel("Position along the long axis [mm] (apex left, base right)")
        ax.set_ylabel("Rotation at end systole [deg]")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "rotation.png")

# ![rotation](lv_ellipsoid_pericardium/rotation.png)
#
# ## Movie
#
# To see the difference, we render the first two runs side by side, from the side (top row)
# and from the apex (bottom row). The stripes follow the tissue. They run along the long axis
# in the reference configuration, and wind into helices as the ventricle twists. The grey
# shape is the reference configuration, and its outer surface is the pericardium.


def animate(path, names=("RobinBC", "PericardiumBC"), substeps=6, hold=10, fps=20):
    cells, cell_types, points = dolfinx.plot.vtk_mesh(V1)
    grid = pyvista.UnstructuredGrid(cells, cell_types, points)
    for name in names:
        for i, (_, u) in enumerate(frames[name]):
            grid[f"{name} {i}"] = u
    # Subdivide the surface for sharper stripes. The displacements are interpolated
    # linearly onto the new points, so the motion is still the computed one.
    surface = grid.extract_surface().subdivide(2, subfilter="linear")
    X = surface.points.copy()
    stripes = np.sin(6 * np.arctan2(X[:, 2], X[:, 1]))
    labels = [label for label, _ in frames[names[0]]]

    # From the side with the base up, and from the apex: (direction, up, zoom)
    views = [((0.0, 1.0, 0.3), (1.0, 0.0, 0.0), 1.1), ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0)]
    plotter = pyvista.Plotter(
        shape=(len(views), len(names)),
        row_weights=[1.5, 1.0],
        window_size=(720, 688),
        border=False,
        off_screen=True,
    )
    plotter.set_background("white")
    meshes: dict[str, list[pyvista.PolyData]] = {name: [] for name in names}
    for row, (view, viewup, zoom) in enumerate(views):
        for col, name in enumerate(names):
            plotter.subplot(row, col)
            if row == 0:
                plotter.add_text(name, font_size=12, color="black", position="upper_edge")
            plotter.add_mesh(pyvista.PolyData(X, surface.faces), color="grey", opacity=0.15)
            mesh = pyvista.PolyData(X.copy(), surface.faces)
            mesh["stripes"] = stripes
            plotter.add_mesh(
                mesh,
                cmap=["#8b1a2b", "#f2c9cf"],
                show_scalar_bar=False,
                smooth_shading=True,
            )
            meshes[name].append(mesh)
            plotter.view_vector(view, viewup=viewup)
            plotter.camera.zoom(zoom)

    def draw(t):
        """The picture at load step `t`, interpolated linearly between the steps around it"""
        i = min(int(t), len(labels) - 2)
        for name, copies in meshes.items():
            u = (i + 1 - t) * surface[f"{name} {i}"] + (t - i) * surface[f"{name} {i + 1}"]
            for mesh in copies:
                mesh.points = X + u
        plotter.subplot(len(views) - 1, 0)
        plotter.add_text(
            labels[round(t)],
            font_size=9,
            color="black",
            position="lower_left",
            name="label",
        )
        return plotter.screenshot(return_img=True)

    # Forwards, then back so that the movie loops
    forwards = np.linspace(0, len(labels) - 1, substeps * (len(labels) - 1) + 1)
    times = np.concatenate([forwards, np.full(hold, forwards[-1]), forwards[::-1], np.zeros(hold)])
    with imageio.get_writer(path, fps=fps, macro_block_size=8, quality=6) as writer:
        for t in times:
            writer.append_data(draw(t))
    plotter.close()


try:
    import imageio.v2 as imageio
    import imageio_ffmpeg  # noqa: F401
    import pyvista
except ImportError:
    print("pyvista, imageio and imageio-ffmpeg are needed for the movie")
else:
    # In parallel each process has only its own part of the mesh. The page shows a copy
    # of this file, `_static/pericardium.mp4`.
    if comm.size == 1:
        animate(outdir / "pericardium.mp4")

# <video width="720" controls loop autoplay muted>
#   <source src="../../_static/pericardium.mp4" type="video/mp4">
#   <p>Video of the ventricle contracting with a RobinBC and with a PericardiumBC.</p>
# </video>
#
# With the `PericardiumBC` the stripes wind further and the base comes further down. With
# the `RobinBC` the epicardium comes away from the grey shape towards the apex, where the
# `PericardiumBC` keeps it on the pericardium.
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
# The surface made from the mesh gives the same as the ellipsoid: the twist differs by
# 0.03 degrees, the volumes by 0.2 mL and the descent of the base by 0.2 mm. The surface
# only approximates the ellipsoid, and its integral is a sum over the nodes of the
# epicardium rather than over the quadrature points; both contribute.
#
# Both conditions keep the epicardium near where it is in the reference configuration.
# Here that is the unloaded ventricle, so stiff springs also resist filling. Softer
# springs ($k \lesssim 10^6$ Pa/m) resist it less, and there the two conditions differ
# less too.
#
# `PericardiumBC` takes a `damping` coefficient for a dashpot in a `DynamicProblem`, and
# `unilateral=True` makes it act only where the epicardium moves outward. In systole the
# ventricle moves inward, so a unilateral pericardium does nothing then.
