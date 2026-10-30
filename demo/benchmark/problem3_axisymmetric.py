# # Problem 3 on an axisymmetric geometry
#
# This example solves Problem 3 of the cardiac mechanics benchmark {cite}`land2015verification`
# (see [Problem 3](problem3.py)) the way one of its participants did: on the meridian of the
# ventricle, with curved (isoparametric) elements. That entry, by Simone Pezzuto, was written in
# legacy FEniCS and is available at https://bitbucket.org/peppu/mechbench.
#
# The truncated ellipsoid is a solid of revolution, and so is the loading. The fiber field is not
# symmetric under a reflection through a meridian plane, so the ventricle twists, but nothing
# depends on the angle about the axis. The problem can then be solved on one meridian, a 2D
# domain with coordinates $(Z, R)$, for a displacement with three components: the axial and
# radial displacement $u_Z$ and $u_R$, and the twist $\phi$, the angle a point turns by about
# the axis.
#
# The deformation is $z = Z + u_Z$, $r = R + u_R$, $\theta = \Theta + \phi$ and its gradient,
# in the orthonormal bases of cylindrical coordinates, is
#
# $$
# \mathbf{F} = \begin{pmatrix}
#     z_{,Z} & z_{,R} & 0 \\
#     r_{,Z} & r_{,R} & 0 \\
#     r \phi_{,Z} & r \phi_{,R} & r / R
# \end{pmatrix} .
# $$
#
# A point of the meridian stands for the circle it sweeps, so every integral is weighted by
# $2 \pi R$. {py:class}`pulse.AxisymmetricHeartGeometry` does both, and the rest of the problem
# is set up as in 3D.
#
# ---

import math
import typing
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np

import cardiac_geometries
import pulse

comm = MPI.COMM_WORLD

# ## 1. Geometry and fibers
#
# `cardiac_geometries.mesh.lv_ellipsoid_2D` makes the meridian of the ellipsoid of Problems 2
# and 3, with the long axis along $x$ and the radial direction along $y$. The apex is at
# $x = -17$ mm (endocardium) and $x = -20$ mm (epicardium), and the base at $x = 5$ mm. The
# segment of the axis between the two apices is marked `APEX`: the points there must stay on
# the axis, which {py:class}`pulse.AxisymmetricHeartGeometry` imposes with $u_R = 0$.
#
# As in Pezzuto's entry, the elements are of order 2, so the endocardium and the epicardium are
# curved, and the size of the elements is 3 mm at the base and 1.5 mm at the apex, divided by
# `ndiv`.
#
# The fibers have three components, in the axial, radial and circumferential direction. The
# fiber angle goes from $+90^\circ$ at the endocardium to $-90^\circ$ at the epicardium,
# linearly in the transmural coordinate $t$ the benchmark defines through the ellipsoid itself:
# a point is on the ellipsoid with radii $r_s(t) = 7 + 3t$ and $r_l(t) = 17 + 3t$. They are
# evaluated at the quadrature points.

quadrature_degree = 6


def create_geometry(ndiv: int) -> tuple[pulse.AxisymmetricHeartGeometry, typing.Any]:
    geodir = Path(f"lv_ellipsoid_2D-problem3-ndiv{ndiv}")
    if not geodir.exists():
        comm.barrier()
        cardiac_geometries.mesh.lv_ellipsoid_2D(
            outdir=geodir,
            r_short_endo=7.0,
            r_short_epi=10.0,
            r_long_endo=17.0,
            r_long_epi=20.0,
            mu_apex_endo=math.pi,
            mu_base_endo=math.acos(5 / 17),
            mu_apex_epi=math.pi,
            mu_base_epi=math.acos(5 / 20),
            psize_ref=3.0 / ndiv,
            create_fibers=True,
            fiber_angle_endo=90,
            fiber_angle_epi=-90,
            fiber_space=f"Quadrature_{quadrature_degree}",
            order=2,
            comm=comm,
        )
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
    geometry = pulse.AxisymmetricHeartGeometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": quadrature_degree},
        axis_marker="APEX",
    )
    return geometry, geo


# ## 2. Model, boundary conditions and solution
#
# The model is the one of [Problem 3](problem3.py): transversely isotropic Guccione material
# ($C = 2$ kPa, $b_f = 8$, $b_t = 2$, $b_{fs} = 4$), incompressible, with an active stress
# $T_a$ along the fibers. The base is fixed, in all three components, and the pressure acts on
# the endocardium. The pressure goes to 15 kPa and $T_a$ to 60 kPa, together.


def solve(ndiv: int) -> dict[str, float]:
    geometry, geo = create_geometry(ndiv)

    def constant(value: float) -> dolfinx.fem.Constant:
        return dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(value))

    material = pulse.Guccione(
        f0=geo.f0,
        s0=geo.s0,
        n0=geo.n0,
        C=constant(2.0),
        bf=constant(8.0),
        bt=constant(2.0),
        bfs=constant(4.0),
    )
    Ta = constant(0.0)
    model = pulse.CardiacModel(
        material=material,
        active=pulse.ActiveStress(geo.f0, activation=pulse.Variable(Ta, "kPa")),
        compressibility=pulse.Incompressible(),
    )

    pressure = constant(0.0)
    neumann = pulse.NeumannBC(
        traction=pulse.Variable(pressure, "kPa"),
        marker=geo.markers["ENDO"][0],
    )
    problem = pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=pulse.BoundaryConditions(neumann=(neumann,)),
        parameters={"base_bc": pulse.BaseBC.fixed},
    )

    steps = 20
    for p, ta in zip(np.linspace(0, 15.0, steps + 1)[1:], np.linspace(0, 60.0, steps + 1)[1:]):
        pressure.value = p
        Ta.value = ta
        if not problem.solve():
            raise RuntimeError(f"Newton did not converge at p = {p:.2f} kPa, Ta = {ta:.2f} kPa")

    # The position of the apices along the axis
    X = dolfinx.fem.Function(problem.u.function_space)
    X.interpolate(lambda x: (x[0], x[1], np.zeros_like(x[0])))
    result = {"ndiv": ndiv}
    for name, marker in [("endo", "ENDOPT"), ("epi", "EPIPT")]:
        tag = geo.markers[marker][0]
        x = pulse.utils.evaluate_at_vertex_tag(X, geo.vfun, tag)
        u = pulse.utils.evaluate_at_vertex_tag(problem.u, geo.vfun, tag)
        result[name] = pulse.utils.gather_broadcast_array(comm, x + u)[0, 0]
    result["volume"] = comm.allreduce(geometry.volume("ENDO", problem.u), op=MPI.SUM)
    result["dofs"] = sum(
        V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        for V in (problem.u_space, problem.p_space)
    )
    result["u"] = problem.u
    return result


results = [solve(ndiv) for ndiv in (1, 2, 4)]

# ## 3. Results
#
# Pezzuto's axisymmetric results with elements of order 2 were, for the same `ndiv`
# (positions of the apices in mm, cavity volume in mm$^3$):
#
# | ndiv | dofs | volume | epicardial apex | endocardial apex |
# |---:|---:|---:|---:|---:|
# | 1 | 404 | 1794.49 | -15.427 | -11.932 |
# | 2 | 1272 | 1798.55 | -15.463 | -12.285 |
# | 4 | 4906 | 1796.17 | -15.466 | -12.342 |
# | 64 | 1116576 | 1795.89 | -15.452 | -12.347 |
#
# where `ndiv = 64` is the finest mesh he solved on. The meshes are not the same, as gmsh
# meshes the meridian anew, so the numbers do not have to agree at a given `ndiv`, but they
# should converge to the same values.

if comm.rank == 0:
    print(f"{'ndiv':>4} {'dofs':>8} {'volume':>10} {'epi':>9} {'endo':>9}")
    for r in results:
        print(
            f"{r['ndiv']:>4d} {r['dofs']:>8d} {r['volume']:>10.2f} "
            f"{r['epi']:>9.3f} {r['endo']:>9.3f}",
        )

# ## 4. Visualization
#
# The meridian before and after, on the finest mesh.

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed")
else:
    u = results[-1]["u"]
    mesh = u.function_space.mesh
    V1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    u1 = dolfinx.fem.Function(V1)
    u1.interpolate(u)
    # The triangles and coordinates of the dofs of V1, in the order of u1's values
    topology, _, x = dolfinx.plot.vtk_mesh(V1)
    cells = topology.reshape(-1, 4)[:, 1:]
    X, R = x[:, 0], x[:, 1]
    U = u1.x.array.reshape(-1, 3)

    fig, ax = plt.subplots()
    ax.triplot(X, R, cells, color="0.7", linewidth=0.5, label="Reference")
    ax.triplot(X + U[:, 0], R + U[:, 1], cells, color="C0", linewidth=0.5, label="Deformed")
    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("R (mm)")
    ax.set_aspect("equal")
    ax.legend()
    fig.savefig("problem3_axisymmetric.png", dpi=200)

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
