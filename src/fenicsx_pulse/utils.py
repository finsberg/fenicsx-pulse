import dolfinx
import numpy as np
import numpy.typing as npt


def vertex_to_dofmap(V: dolfinx.fem.FunctionSpace) -> npt.NDArray[np.int32]:
    """Compute a mapping from vertices to dofs in a function space

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        The function space

    Returns
    -------
    npt.NDArray[np.int32]
        The mapping from vertices to dofs
    """
    mesh = V.mesh
    num_vertices_per_cell = dolfinx.cpp.mesh.cell_num_entities(mesh.topology.cell_type, 0)

    dof_layout = np.empty((num_vertices_per_cell,), dtype=np.int32)
    for i in range(num_vertices_per_cell):
        var = V.dofmap.dof_layout.entity_dofs(0, i)
        assert len(var) == 1
        dof_layout[i] = var[0]

    num_vertices = mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts

    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    assert (
        c_to_v.offsets[1:] - c_to_v.offsets[:-1] == c_to_v.offsets[1]
    ).all(), "Single cell type supported"

    vertex_to_dof_map = np.empty(num_vertices, dtype=np.int32)
    vertex_to_dof_map[c_to_v.array] = V.dofmap.list[:, dof_layout].reshape(-1)
    return vertex_to_dof_map


def unroll_dofmap(dofs: npt.NDArray[np.int32], bs: int) -> npt.NDArray[np.int32]:
    """
    Given a two-dimensional dofmap of size `(num_cells, num_dofs_per_cell)`
    Expand the dofmap by its block size such that the resulting array
    is of size `(num_cells, bs*num_dofs_per_cell)`
    """
    num_cells, num_dofs_per_cell = dofs.shape
    unrolled_dofmap = np.repeat(dofs, bs).reshape(num_cells, num_dofs_per_cell * bs) * bs
    unrolled_dofmap += np.tile(np.arange(bs), num_dofs_per_cell)
    return unrolled_dofmap


def evaluate_at_vertex_tag(
    u: dolfinx.fem.Function,
    vt: dolfinx.mesh.MeshTags,
    tag: int,
) -> npt.NDArray[np.int32]:
    """
    Given a function `u` and a vertex tag `vt` return the values of `u` at the vertices
    tagged with `tag`

    Parameters
    ----------
    u: dolfinx.fem.Function
        The function to evaluate
    vt: dolfinx.mesh.MeshTags
        The vertex tags
    tag: int
        The tag to evaluate at

    Returns
    -------
    npt.NDArray[np.int32]
        The values of `u` at the vertices tagged with `tag`
    """
    v2d = vertex_to_dofmap(u.function_space)
    block_index = v2d[vt.find(tag)]
    dofs = unroll_dofmap(block_index.reshape(-1, 1), u.function_space.dofmap.bs)
    return u.x.array[dofs]


def gather_broadcast_array(comm, local_array):
    """
    Collects local arrays from all processes on the root process
    and distributes the global array to all processes.
    Assumes that the local arrays are either the same of empty / None on all processes.

    Parameters
    ----------
    comm: MPI.Comm
        The MPI communicator
    local_array: np.ndarray
        The local array

    Returns
    -------
    np.ndarray
        The global array on the root process
    """
    all_arrays = comm.gather(local_array, root=0)
    if comm.rank == 0:
        nonempty_arrays = [array for array in all_arrays if array is not None and len(array) > 0]
        # Check that all nonempty arrays are the same
        if len(nonempty_arrays) == 0:
            global_array = np.array([])
        elif len(nonempty_arrays) == 1:
            global_array = nonempty_arrays[0]
        else:
            all_same = True
            for array in nonempty_arrays[1:]:
                all_same = all_same and np.allclose(array, nonempty_arrays[0])
            if not all_same:
                raise ValueError("Local arrays are not the same", nonempty_arrays)
            global_array = nonempty_arrays[0]

    else:
        global_array = np.array([])
    return comm.bcast(global_array, root=0)
