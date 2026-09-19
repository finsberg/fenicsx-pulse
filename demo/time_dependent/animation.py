"""Record the moving geometry and turn it into a video.

A sibling module of the demos in this directory, not a demo itself and not part
of the documentation build -- the same arrangement as `calibration.py`.

Two halves, kept apart:

:class:`FrameRecorder`
    Runs inside a simulation. Interpolates the displacement onto a linear space
    once per saved step and stores the nodal values with the mesh topology in
    the form VTK wants. Needs dolfinx.

:func:`render`
    Runs afterwards on the saved file. Needs pyvista and matplotlib but not
    dolfinx, so regenerating a video does not mean re-running a beat.

The split is what lets the documentation ship finished assets: the demos run
two steps under CI, while the videos and figures in `_static/` come from full
runs done by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["FrameRecorder", "base_normal", "render", "save_pv_figure"]

CHAMBER_COLOURS = {"LV": "#c81e3c", "RV": "#2a6fb0"}


class FrameRecorder:
    """Keep the deformed geometry every `every` steps, for later rendering.

    The displacement is interpolated onto a linear space before storing. The
    simulation may run on quadratic elements, but a surface rendered from the
    corner values of the same cells looks the same at a quarter of the size on
    disk.
    """

    def __init__(self, mesh, every: int = 5, enabled: bool = True, up=None):
        self.every = max(int(every), 1)
        self.enabled = enabled
        self.up = None if up is None else np.asarray(up, dtype=float)
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []
        self._mesh = mesh
        self._u1 = None
        self._topology = None

    def _setup(self):
        import dolfinx

        space = dolfinx.fem.functionspace(self._mesh, ("Lagrange", 1, (3,)))
        self._u1 = dolfinx.fem.Function(space)
        cells, cell_types, points = dolfinx.plot.vtk_mesh(space)
        self._topology = {
            "cells": np.asarray(cells),
            "cell_types": np.asarray(cell_types),
            "points": np.asarray(points),
        }

    def record(self, u, t: float, step: int) -> None:
        if not self.enabled or step % self.every:
            return
        if self._u1 is None:
            self._setup()
        self._u1.interpolate(u)
        self.frames.append(self._u1.x.array.copy().reshape(-1, 3))
        self.times.append(float(t))

    def save(self, path: Path) -> Path | None:
        """Write the frames, or do nothing if there are none."""
        if not self.frames:
            return None
        assert self._topology is not None
        extra = {} if self.up is None else {"up": self.up}
        np.savez_compressed(
            path,
            u=np.asarray(self.frames),
            time=np.asarray(self.times),
            **self._topology,
            **extra,
        )
        return Path(path)


def base_normal(geometry, marker: str = "BASE") -> np.ndarray:
    """Area-averaged outward normal of a surface, as a unit vector.

    Used to stand the ventricle upright in the video. Taking it from the
    geometry is safer than assuming an axis, since a mesh may have been rotated
    or written and read back without the rotation surviving.
    """
    from mpi4py import MPI

    import dolfinx
    import ufl

    comm = geometry.mesh.comm
    n = ufl.FacetNormal(geometry.mesh)
    ds = geometry.ds(geometry.markers[marker][0])
    vector = np.array(
        [
            comm.allreduce(
                dolfinx.fem.assemble_scalar(dolfinx.fem.form(n[i] * ds)),
                op=MPI.SUM,
            )
            for i in range(3)
        ],
    )
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError(f"surface {marker!r} has no net normal; is it closed?")
    return vector / norm


def _grid(frames: dict):
    import pyvista as pv

    grid = pv.UnstructuredGrid(
        frames["cells"],
        frames["cell_types"],
        frames["points"].astype(float),
    )
    # Extract the surface once and remember where each of its points came
    # from, so every later frame is a lookup rather than another extraction.
    surface = grid.extract_surface(pass_pointid=True)
    return surface, np.asarray(surface["vtkOriginalPointIds"], dtype=int)


def _orientation(points: np.ndarray):
    """Guess which way is up from the shape itself.

    A fallback for frame files written without a base normal. The long axis is
    the leading principal direction of the point cloud, and the base is
    whichever end of it spreads out more, a flat cut through the valve plane
    being wider than an apex.
    """
    centred = points - points.mean(axis=0)
    _, _, directions = np.linalg.svd(centred, full_matrices=False)
    long_axis = directions[0]

    along = centred @ long_axis
    radial = np.linalg.norm(centred - np.outer(along, long_axis), axis=1)
    high = radial[along > np.quantile(along, 0.85)].mean()
    low = radial[along < np.quantile(along, 0.15)].mean()
    if low > high:  # the wider end is the base, and the base goes up
        long_axis = -long_axis

    # Look at it side-on, from a direction perpendicular to the long axis and
    # tipped slightly so the picture is not a flat silhouette.
    side = directions[1]
    view = side + 0.35 * np.cross(long_axis, side)
    return long_axis / np.linalg.norm(long_axis), view / np.linalg.norm(view)


def _orientation_about(points: np.ndarray, up: np.ndarray):
    """A side-on view direction, given which way is up.

    The widest direction across the base is where the two ventricles sit side by
    side, so looking along it would hide one behind the other. The camera goes
    at right angles to it instead, tipped towards the base for some depth.
    """
    up = up / np.linalg.norm(up)
    centred = points - points.mean(axis=0)
    in_plane = centred - np.outer(centred @ up, up)
    _, _, directions = np.linalg.svd(in_plane, full_matrices=False)
    widest = directions[0]
    view = np.cross(up, widest) + 0.3 * up
    return up, view / np.linalg.norm(view)


def _subdivide(surface, levels: int):
    """A smoother copy of the surface, for looks only.

    The demos run on a few hundred vertices, which is plenty for the mechanics
    but reads as a faceted lump on screen. Subdividing changes no result and is
    applied identically to every frame, so the motion shown is the computed one.
    """
    if levels <= 0:
        return surface.copy()
    return surface.subdivide(levels, subfilter="loop")


def _loop_panel(traces, chambers, width_px, height_px, dpi=100):
    """A figure showing the loops, plus the artists that advance over it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    artists = {}
    for chamber in chambers:
        V, p = traces[f"V_{chamber}"], traces[f"p_{chamber}"]
        colour = CHAMBER_COLOURS.get(chamber, "#444444")
        # The whole loop faintly, so its shape is visible from the first
        # frame, with the part already reached drawn on top.
        ax.plot(V, p, color=colour, alpha=0.22, linewidth=1.0)
        (trace,) = ax.plot([], [], color=colour, linewidth=1.8, label=chamber)
        (head,) = ax.plot([], [], "o", color=colour, markersize=6)
        artists[chamber] = (trace, head)

    ax.set_xlabel("volume [mL]")
    ax.set_ylabel("pressure [mmHg]")
    if len(chambers) > 1:
        ax.legend(loc="upper right", fontsize="small", frameon=False)
    ax.set_title("Pressure-volume loop", fontsize="medium")
    ax.spines[["top", "right"]].set_visible(False)
    clock = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize="small")
    fig.tight_layout()
    return fig, ax, artists, clock


def _panel_image(fig):
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return image


def render(
    frames_path: Path | str,
    traces_path: Path | str,
    out_path: Path | str,
    chambers: Sequence[str] = ("LV",),
    fps: int = 25,
    size: tuple[int, int] = (560, 520),
    zoom: float = 1.0,
    title: str | None = None,
    subdivide: int = 2,
    view_vector: tuple[float, float, float] | None = None,
    quality: int = 6,
) -> Path:
    """Render the recorded motion beside the loop it traces.

    The camera and the colour range are fixed once across all frames; letting
    either follow the current frame would rescale the motion from frame to
    frame.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True

    frames = dict(np.load(frames_path))
    traces = dict(np.load(traces_path))
    times, displacements = frames["time"], frames["u"]

    surface, original = _grid(frames)
    rest = np.asarray(frames["points"], dtype=float)[original]

    magnitude = np.linalg.norm(displacements, axis=-1)
    clim = (0.0, float(magnitude.max()))

    def shape_at(index: int):
        surface.points = rest + displacements[index][original]
        surface["displacement [mm]"] = magnitude[index][original] * 1e3
        return _subdivide(surface, subdivide)

    width, height = size
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background("white")
    display = shape_at(0)
    plotter.add_mesh(
        display,
        scalars="displacement [mm]",
        clim=(clim[0] * 1e3, clim[1] * 1e3),
        cmap="viridis",
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "|u| [mm]",
            "color": "black",
            # Two labels, not four: VTK centres the title over the bar, where
            # a middle label would sit on top of it.
            "n_labels": 2,
            "vertical": False,
            "position_x": 0.28,
            "position_y": 0.015,
            "width": 0.44,
            "height": 0.05,
            "title_font_size": 13,
            "label_font_size": 11,
        },
    )
    # Set the camera on the widest frame, then leave it alone.
    points = np.asarray(frames["points"], dtype=float)
    if "up" in frames:
        # The recorder stored the base normal, which is exact.
        viewup, derived_view = _orientation_about(points, np.asarray(frames["up"], dtype=float))
    else:
        # Guess it from the shape, for files written before it was recorded.
        viewup, derived_view = _orientation(points)

    widest = int(np.argmax([np.ptp(rest + u[original], axis=0).max() for u in displacements]))
    plotter.add_mesh(shape_at(widest), opacity=0.0, show_scalar_bar=False)
    plotter.view_vector(view_vector if view_vector is not None else derived_view, viewup=viewup)
    plotter.camera.zoom(zoom)
    if title:
        plotter.add_text(title, font_size=10, color="black")

    fig, _, artists, clock = _loop_panel(traces, chambers, width, height)

    # The traces are sampled every step, the frames every `every` steps. Line
    # them up by time rather than index, which also survives a run that stopped
    # early.
    trace_time = traces["time"]
    indices = np.searchsorted(trace_time, times).clip(0, trace_time.size - 1)

    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=8, quality=quality)
    try:
        for frame, (u, t, upto) in enumerate(zip(displacements, times, indices)):
            current = shape_at(frame)
            # Same topology every frame, so update the displayed mesh in place
            # rather than rebuilding and re-adding it to the scene.
            display.points = current.points
            display["displacement [mm]"] = current["displacement [mm]"]
            plotter.render()
            left = np.asarray(plotter.screenshot(return_img=True))

            for chamber in chambers:
                trace, head = artists[chamber]
                V, p = traces[f"V_{chamber}"], traces[f"p_{chamber}"]
                trace.set_data(V[: upto + 1], p[: upto + 1])
                head.set_data([V[upto]], [p[upto]])
            clock.set_text(f"t = {t:.2f} s")
            right = _panel_image(fig)

            rows = min(left.shape[0], right.shape[0])
            writer.append_data(np.hstack([left[:rows], right[:rows]]))
    finally:
        writer.close()
        plotter.close()
        import matplotlib.pyplot as plt

        plt.close(fig)

    return out_path


def save_pv_figure(
    traces_path: Path | str,
    out_path: Path | str,
    chambers: Sequence[str] = ("LV",),
    title: str | None = None,
) -> Path:
    """The finished loop as a still, to show beside the video."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = dict(np.load(traces_path))
    fig, (loop, trace) = plt.subplots(
        1,
        2,
        figsize=(11, 4.2),
        layout="constrained",
        width_ratios=(1.0, 1.3),
    )
    for chamber in chambers:
        colour = CHAMBER_COLOURS.get(chamber, "#444444")
        loop.plot(
            traces[f"V_{chamber}"], traces[f"p_{chamber}"], color=colour,
            linewidth=1.3, label=chamber,
        )
        trace.plot(
            traces["time"], traces[f"p_{chamber}"], color=colour,
            linewidth=1.3, label=f"p {chamber}",
        )
    loop.set_xlabel("volume [mL]")
    loop.set_ylabel("pressure [mmHg]")
    loop.set_title(title or "Pressure-volume loop")
    trace.set_xlabel("time [s]")
    trace.set_ylabel("pressure [mmHg]")
    trace.set_title("Pressure over the beat")
    for axis in (loop, trace):
        axis.spines[["top", "right"]].set_visible(False)
        if len(chambers) > 1:
            axis.legend(frameon=False, fontsize="small")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
