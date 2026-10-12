# # Pericardium around a biventricular geometry
#
# In [the previous demo](pericardium.py) the pericardium was an ellipsoid, because the
# ventricle was one. A heart from an image is no ellipsoid. For such a mesh,
# `pulse.MeshSurface` makes the pericardium from the mesh itself: a smooth surface through
# the facets of the epicardium, where they are when the surface is made.
#
# This demo contracts a biventricular geometry from the UK Biobank atlas inside such a
# pericardium, with a `pulse.PericardiumBC`, and compares it with the usual epicardial
# springs, a `pulse.RobinBC`, at the same stiffness.
#
# ---

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
# The mean end-diastolic shape of the
# [UK Biobank atlas](https://github.com/ComputationalPhysiology/ukb-atlas), cut off at the
# base, in millimetres. We rotate it so that the long axis is along $x$, with the base on
# top, in the plane $x = 26.5$ mm, and the apex at $x \approx -57$ mm. `rotate` aligns the
# normal of the base with $x$, but that normal may point into the heart or out of it. If
# the base ends up below, we turn the mesh and the fibres half a turn about $z$. The mesh
# is made and rotated on one process.

outdir = Path("ukb_pericardium")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"
comm = MPI.COMM_WORLD

if comm.rank == 0 and not (geodir / "mesh.xdmf").exists():
    atlas = cardiac_geometries.mesh.ukb(
        outdir=outdir / "atlas",
        comm=MPI.COMM_SELF,
        mode=-1,
        std=0,
        case="ED",
        create_fibers=True,
        char_length_max=7.0,
        char_length_min=7.0,
        clipped=True,
        fiber_angle_endo=60,
        fiber_angle_epi=-60,
        fiber_space="Quadrature_6",
    )
    atlas = atlas.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")
    base = dolfinx.mesh.compute_midpoints(
        atlas.mesh,
        2,
        atlas.ffun.find(atlas.markers["BASE"][0]),
    )
    if base[:, 0].mean() < atlas.mesh.geometry.x[:, 0].mean():
        atlas.mesh.geometry.x[:, :2] *= -1
        for field in (atlas.f0, atlas.s0, atlas.n0):
            field.x.array.reshape(-1, 3)[:, :2] *= -1
    atlas.save_folder(folder=geodir)
comm.barrier()

geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo)
markers = {name: geometry.markers[name][0] for name in ("LV", "RV", "EPI", "BASE")}

# ## The pericardium
#
# The pericardium is the epicardium as it is now, in the reference configuration.
# `MeshSurface.from_geometry` reads the facets with the given marker and makes a surface
# through them that has no edges between the facets: each facet stands for a plane, and the
# signed distance $d$ to the surface is a weighted mean of the distances to the planes of
# the facets within `radius` of the point. Features smaller than the radius are smoothed
# out, and the surface reaches no further than the radius from the facets, so it has to be
# larger than the gap we expect. Unless we give one, it is two facet sizes or a tenth of
# the size of the surface, whichever is larger: 14.7 mm here. At the base, where the
# epicardium ends, the surface goes on in the planes of the last facets, so that tissue that
# slides past the edge is still on a pericardium.

surface = pulse.MeshSurface.from_geometry(geometry, "EPI")
print(surface)

# The surface gives the distance to it at any points, as numbers, with its gradient and
# Hessian. Two millimetres out from the middle of the first three facets:

distance, gradient, hessian = surface.evaluate(surface.points[:3] + 2.0 * surface.normals[:3])
print(distance)

# A form cannot evaluate this where the epicardium has moved to. So the problem keeps the
# distance, gradient and Hessian at the current position of each node of the epicardium,
# and updates them before every evaluation of the residual and the Jacobian, also within
# a line search. Newton's method then sees the surface itself. We only have to pass the
# surface to the `PericardiumBC`.

# ## Model
#
# As in the previous demo: a transversely isotropic Holzapfel-Ogden material, contracted
# by an active stress along the fibres.


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
# The base hangs on springs, firm in its plane and weak along the long axis, so that it can
# descend towards the apex. On the epicardium we put either a `RobinBC` along the current
# normal or the `PericardiumBC`, both with $k = 10^6$ Pa/m.

k_epi = 1e6  # Pa/m


def create_bcs(epicardium: str):
    p_lv = pulse.Variable(constant(0.0), "kPa")
    p_rv = pulse.Variable(constant(0.0), "kPa")
    neumann = [
        pulse.NeumannBC(traction=p_lv, marker=markers["LV"]),
        pulse.NeumannBC(traction=p_rv, marker=markers["RV"]),
    ]
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
    return bcs, p_lv, p_rv


# ## Measurements
#
# We measure the two cavity volumes, the mean displacement of the base and of the apex
# along the long axis, and the rotation of the left ventricular endocardium about the long
# axis of the left ventricle, in a slice near the apex and one near the base. The gap is
# how far each node of the epicardium is from where it started, measured by the distance
# to the pericardium: $d(\mathbf{X} + \mathbf{u}) - d(\mathbf{X})$.


def gap_at(X, U):
    return surface.evaluate(X + U, hessian=False)[0] - surface.evaluate(X, hessian=False)[0]


def measurements(problem):
    u = problem.u
    V = problem.u_space
    X = V.tabulate_dof_coordinates()
    U = u.x.array.reshape(-1, 3)
    n_owned = V.dofmap.index_map.size_local

    def dofs(marker):
        found = dolfinx.fem.locate_dofs_topological(V, 2, geometry.facet_tags.find(marker))
        return found[found < n_owned]

    epi, base, lv = dofs(markers["EPI"]), dofs(markers["BASE"]), dofs(markers["LV"])

    def mean(values):
        total = comm.allreduce(np.sum(values, axis=0), op=MPI.SUM)
        return total / comm.allreduce(len(values), op=MPI.SUM)

    # The long axis of the left ventricle is along x, through the middle of its endocardium
    centre = mean(X[lv])
    bottom = comm.allreduce(X[lv, 0].min(initial=np.inf), op=MPI.MIN)
    top = comm.allreduce(X[lv, 0].max(initial=-np.inf), op=MPI.MAX)

    def rotation(lo, hi):
        height = (X[lv, 0] - bottom) / (top - bottom)
        sel = lv[(height > lo) & (height < hi)]
        r0 = X[sel, 1:] - centre[1:]
        r1 = r0 + U[sel, 1:]
        angle = np.arctan2(r0[:, 0] * r1[:, 1] - r0[:, 1] * r1[:, 0], (r0 * r1).sum(axis=1))
        return float(np.degrees(mean(angle)))

    # The cavities are open at the base, so measure their volumes from a point on the base
    x_base = geometry.base_center("BASE", u)
    F = ufl.Identity(3) + ufl.grad(u)
    x = geometry.X + u - ufl.as_vector(x_base)
    volume_form = (-1 / 3) * ufl.det(F) * ufl.dot(x, ufl.inv(F).T * geometry.facet_normal)

    def volume(marker):
        form = dolfinx.fem.form(volume_form * geometry.ds(markers[marker]))
        return comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)

    gap = gap_at(X[epi], U[epi])
    Q = dolfinx.fem.functionspace(geometry.mesh, ("DG", 0))
    J = dolfinx.fem.Function(Q)
    J.interpolate(dolfinx.fem.Expression(ufl.det(F), Q.element.interpolation_points))
    apex = lv[X[lv, 0] < bottom + 0.05 * (top - bottom)]
    return {
        "LV [mL]": volume("LV") * 1e-3,
        "RV [mL]": volume("RV") * 1e-3,
        "base [mm]": mean(U[base, 0]),
        "apex [mm]": mean(U[apex, 0]),
        "basal rotation [deg]": rotation(0.7, 0.9),
        "apical rotation [deg]": rotation(0.1, 0.3),
        "gap [mm]": np.sqrt(mean(gap**2)),
        "largest gap [mm]": comm.allreduce(np.abs(gap).max(initial=0.0), op=MPI.MAX),
        "min J": comm.allreduce(J.x.array.min(), op=MPI.MIN),
    }


# ## Simulation
#
# We inflate the left ventricle to 1.5 kPa and the right to a third of that, then raise
# the active tension to 40 kPa at those pressures. `solve` returns `False` if Newton does
# not converge, so we check every step. After every step we keep the displacement,
# interpolated onto linear elements, for the movie at the end.

V1 = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (3,)))
frames: dict[str, list[tuple[str, np.ndarray]]] = {}


def run(epicardium: str):
    model, Ta = create_model()
    bcs, p_lv, p_rv = create_bcs(epicardium)
    problem = pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters={"base_bc": pulse.BaseBC.free, "mesh_unit": "mm"},
    )
    u1 = dolfinx.fem.Function(V1)
    frames[epicardium] = []

    def keep(p, ta):
        u1.interpolate(problem.u)
        label = f"p = {p:.1f} kPa, Ta = {ta:.0f} kPa"
        frames[epicardium].append((label, u1.x.array.reshape(-1, 3).copy()))

    keep(0.0, 0.0)
    for step, value in enumerate(np.linspace(0.0, 1.5, 7)[1:]):
        p_lv.assign(value)
        p_rv.assign(value / 3)
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


results = {name: run(name) for name in ("RobinBC", "PericardiumBC")}

# ## Results

if comm.rank == 0:
    rows = [
        ("LV EDV [mL]", lambda ed, es: ed["LV [mL]"]),
        ("LV ESV [mL]", lambda ed, es: es["LV [mL]"]),
        ("RV EDV [mL]", lambda ed, es: ed["RV [mL]"]),
        ("RV ESV [mL]", lambda ed, es: es["RV [mL]"]),
        ("base, systole [mm]", lambda ed, es: es["base [mm]"] - ed["base [mm]"]),
        ("apex, systole [mm]", lambda ed, es: es["apex [mm]"] - ed["apex [mm]"]),
        ("basal rotation [deg]", lambda ed, es: es["basal rotation [deg]"]),
        ("apical rotation [deg]", lambda ed, es: es["apical rotation [deg]"]),
        (
            "twist [deg]",
            lambda ed, es: es["apical rotation [deg]"] - es["basal rotation [deg]"],
        ),
        ("gap, diastole [mm]", lambda ed, es: ed["gap [mm]"]),
        ("gap, systole [mm]", lambda ed, es: es["gap [mm]"]),
        ("largest gap, systole [mm]", lambda ed, es: es["largest gap [mm]"]),
        ("min J", lambda ed, es: es["min J"]),
    ]
    print(f"{'':28}" + "".join(f"{name:>16}" for name in results))
    for label, f in rows:
        print(f"{label:28}" + "".join(f"{f(*results[name]):16.2f}" for name in results))

# The displacements of the base and the apex are along the long axis, from end diastole to
# end systole; negative is towards the apex. Rotations are positive counterclockwise when
# seen from the base. The gap is a root mean square over the nodes of the epicardium, and
# the largest gap the largest among them.
#
# ## Movie
#
# The two runs side by side, from the side (top row) and from the apex (bottom row). The
# grey shell is the pericardium: the epicardium in the reference configuration. The colour
# is the gap, blue where the epicardium has moved inwards from the pericardium and red
# where it has moved outwards.


def animate(path, names=("RobinBC", "PericardiumBC"), substeps=6, hold=10, fps=20):
    cells, cell_types, points = dolfinx.plot.vtk_mesh(V1)
    grid = pyvista.UnstructuredGrid(cells, cell_types, points)
    for name in names:
        for i, (_, u) in enumerate(frames[name]):
            grid[f"{name} {i}"] = u
    # Only the epicardium has a gap
    on_epicardium = np.zeros(len(points))
    facets = geometry.facet_tags.find(markers["EPI"])
    on_epicardium[dolfinx.fem.locate_dofs_topological(V1, 2, facets)] = 1.0
    grid["epicardium"] = on_epicardium
    skin = grid.extract_surface(algorithm="dataset_surface")
    X = skin.points.copy()
    epicardium = skin["epicardium"] > 0.5
    labels = [label for label, _ in frames[names[0]]]

    # From the side with the base up, and from the apex: (direction, up, zoom)
    views = [((-0.25, 1.0, 0.0), (1.0, 0.0, 0.0), 1.25), ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.25)]
    plotter = pyvista.Plotter(
        shape=(len(views), len(names)),
        window_size=(720, 720),
        border=False,
        off_screen=True,
    )
    plotter.set_background("white")
    bar = {
        "title": "",
        "color": "black",
        "n_labels": 5,
        "fmt": "%.0f",
        "width": 0.45,
        "height": 0.08,
        "position_x": 0.5,
        "position_y": 0.04,
        "label_font_size": 14,
    }
    meshes: dict[str, list[pyvista.PolyData]] = {name: [] for name in names}
    for row, (view, viewup, zoom) in enumerate(views):
        for col, name in enumerate(names):
            plotter.subplot(row, col)
            if row == 0:
                plotter.add_text(name, font_size=12, color="black", position="upper_edge")
            # The far half of the pericardium, behind the heart
            plotter.add_mesh(
                pyvista.PolyData(X, skin.faces),
                color="grey",
                opacity=0.35,
                culling="front",
            )
            mesh = pyvista.PolyData(X.copy(), skin.faces)
            mesh["gap"] = np.zeros(len(X))
            plotter.add_mesh(
                mesh,
                scalars="gap",
                cmap="coolwarm",
                clim=(-4.0, 4.0),
                nan_color="lightgrey",
                smooth_shading=True,
                show_scalar_bar=(row, col) == (len(views) - 1, len(names) - 1),
                scalar_bar_args=bar,
            )
            meshes[name].append(mesh)
            plotter.view_vector(view, viewup=viewup)
            plotter.camera.zoom(zoom)
    plotter.add_text("gap [mm]", font_size=9, color="black", position=(72, 14))

    def draw(t):
        """The picture at load step `t`, interpolated linearly between the steps around it"""
        i = min(int(t), len(labels) - 2)
        for name, copies in meshes.items():
            u = (i + 1 - t) * skin[f"{name} {i}"] + (t - i) * skin[f"{name} {i + 1}"]
            gap = np.full(len(X), np.nan)
            gap[epicardium] = gap_at(X[epicardium], u[epicardium])
            for mesh in copies:
                mesh.points = X + u
                mesh["gap"] = gap
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
    # of this file, `_static/pericardium_ukb.mp4`.
    if comm.size == 1:
        animate(outdir / "pericardium_ukb.mp4")

# <video width="720" controls loop autoplay muted>
#   <source src="../../_static/pericardium_ukb.mp4" type="video/mp4">
#   <p>Video of the two ventricles contracting with a RobinBC and with a PericardiumBC.</p>
# </video>
#
# ## Discussion
#
# The pericardium made from the mesh holds this heart as the ellipsoid held the ventricle
# of the previous demo: the apex stays where it is and the base comes down towards it, by
# 11 mm. Newton's method needs as many iterations as with the `RobinBC`.
#
# Here the two conditions give nearly the same motion and the same volumes. They differ in
# how they treat sliding, and there is less of it than in the previous demo: the left
# ventricle twists by 14 degrees where the ellipsoid twisted by more than 30, and the gap
# that a `RobinBC` sees in a rotation grows with the square of the angle. What is left is
# of the size of the gap itself, about a millimetre.
#
# The right ventricle ejects little, with either condition, because it fills little. The
# reference configuration is the end-diastolic shape of the atlas, and the springs hold the
# epicardium to it. So they resist filling beyond it, most of all of the right ventricle
# with its thin wall. Softer springs let it fill more. A simulation of a heartbeat would
# start from an unloaded configuration (see [prestressing](../prestress/README.md)) and
# take its pressures from a circulation model.
#
# The largest gap, 5 mm, is well within the radius of the surface. With softer springs or
# a finer mesh the epicardium may move further than two facet sizes from the pericardium.
# The surface then warns that points are out of its reach, and needs a larger `radius`.
