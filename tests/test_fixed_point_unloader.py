"""FixedPointUnloader: unloading a loaded geometry and loading it again gives the loaded
geometry back, on any number of processes.

The unloader moves the mesh itself, node by node. It used to evaluate the displacement at each
process's own nodes with collectives per node, so two processes holding different numbers of
nodes hung.
"""

from mpi4py import MPI

import dolfinx
import numpy as np

import pulse

PRESSURE = 1e3  # Pa


def _geometry():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    boundaries = [
        pulse.Marker(name="X0", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0.0)),
        pulse.Marker(name="Z1", marker=3, dim=2, locator=lambda x: np.isclose(x[2], 1.0)),
    ]
    return pulse.Geometry(mesh=mesh, boundaries=boundaries)


def _model():
    return pulse.CardiacModel(
        material=pulse.NeoHookean(mu=pulse.Variable(10.0, "kPa")),
        active=pulse.Passive(),
        compressibility=pulse.Compressible(),
    )


def _bcs(geometry, traction):
    """Clamped at x = 0, the traction on z = 1 bends the cube."""

    def clamp(V):
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, geometry.facet_tags.find(1))
        return [dolfinx.fem.dirichletbc(dolfinx.fem.Function(V), dofs)]

    return pulse.BoundaryConditions(
        neumann=(pulse.NeumannBC(traction=traction, marker=3),),
        dirichlet=(clamp,),
    )


def _traction(mesh):
    return pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "Pa")


def test_loading_the_unloaded_geometry_gives_the_loaded_one():
    geometry = _geometry()
    mesh = geometry.mesh
    loaded = mesh.geometry.x.copy()
    traction = _traction(mesh)
    unloader = pulse.FixedPointUnloader(
        geometry=geometry,
        model=_model(),
        bcs=_bcs(geometry, traction),
        targets=[pulse.unloading.TargetPressure(traction=traction, target=PRESSURE)],
        unload_parameters={"max_iter": 20, "tol": 1e-8, "ramp_steps": 3},
    )
    u_pre = unloader.unload()
    # The unloader puts the mesh back where it found it
    np.testing.assert_array_equal(mesh.geometry.x, loaded)

    geometry.deform(u_pre)
    traction = _traction(mesh)
    problem = pulse.StaticProblem(model=_model(), geometry=geometry, bcs=_bcs(geometry, traction))
    for p in np.linspace(0.0, PRESSURE, 4)[1:]:
        traction.assign(p)
        assert problem.solve()

    reloaded = mesh.geometry.x + pulse.geometry.nodal_values(problem.u)
    comm = mesh.comm
    error = comm.allreduce(np.abs(reloaded - loaded).max(), op=MPI.MAX)
    displacement = comm.allreduce(np.abs(u_pre.x.array).max(), op=MPI.MAX)
    assert displacement > 1e-2
    assert error < 1e-6 * displacement
