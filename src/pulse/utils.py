import dolfinx
import numpy as np
import numpy.typing as npt
import scifem
import ufl


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
    v2d = scifem.vertex_to_dofmap(u.function_space)
    block_index = v2d[vt.find(tag)]
    dofs = scifem.utils.unroll_dofmap(block_index.reshape(-1, 1), u.function_space.dofmap.bs)
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


def matrix_is_zero(A: ufl.core.expr.Expr) -> bool:
    n = ufl.domain.find_geometric_dimension(A)
    for i in range(n):
        for j in range(n):
            value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(A[i, j] * ufl.dx))
            print(i, j, value)
            is_zero = np.isclose(value, 0)
            if not is_zero:
                return False
    return True


def float2object(
    f: float,
    obj_str: str,
    mesh: dolfinx.mesh.Mesh,
    V: dolfinx.fem.FunctionSpace,
):
    if obj_str == "float":
        return f
    if obj_str == "Constant":
        return dolfinx.fem.Constant(mesh, f)
    if obj_str == "Function":
        v = dolfinx.fem.Function(V)
        v.x.array[:] = f
        return v
    raise ValueError(f"Invalid object string {obj_str!r}")
