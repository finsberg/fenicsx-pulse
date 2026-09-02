# # UK Biobank BiV Mesh Coupled to a Closed-Loop Circulation Circuit
#
# This demo builds a single illustrative figure that merges a PyVista render of the
# clipped UK Biobank (UKB) atlas bi-ventricular mesh with a schematic drawing of the
# closed-loop lumped-parameter circulation model of Regazzoni et al. {cite}`regazzoni2022cardiac`
# (as implemented in `circulation.regazzoni2020.Regazzoni2020`), whose systemic and
# pulmonary circuits are each represented by a windkessel-type RC network.
#
# The circuit is drawn so that its LV/RV chambers coincide with the actual 3D
# ventricular cavities of the rendered mesh: the aortic/mitral valves sit directly on
# top of the LV cavity and the pulmonic/tricuspid valves directly on top of the RV
# cavity, with the systemic loop (LV -> aorta -> ... -> RA -> RV) arching wide above
# the mesh and the pulmonary loop (RV -> pulmonary artery -> ... -> LA -> LV) nested
# inside it.
#
# ---

# ## Imports

import json
from pathlib import Path

from mpi4py import MPI

import dolfinx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyvista
from PIL import Image

import cardiac_geometries

# ## 1. Mesh generation
#
# We generate the **clipped** UKB atlas BiV mesh (base cut off at the valve plane).
# Fibers are not needed for this figure, so we skip `create_fibers` to keep mesh
# generation fast; this also means the only facet markers available are `LV`, `RV`,
# `EPI` and `BASE` (individual valve markers require `create_fibers=True`).

OUTDIR = Path("ukb_circulation_figure")


def generate_mesh(outdir: Path) -> Path:
    geodir = outdir / "geometry"
    if not (geodir / "geometry.bp").exists():
        cardiac_geometries.mesh.ukb(
            outdir=geodir,
            comm=MPI.COMM_WORLD,
            mode=-1,
            std=0,
            case="ED",
            create_fibers=False,
            char_length_max=3.0,
            char_length_min=3.0,
            clipped=True,
        )
    return geodir


# ## 2. Rendering the mesh with PyVista
#
# We render a "base-on" view: the camera looks down into the open (clipped) base
# plane so that the LV and RV cavities are both clearly visible side by side, with
# the epicardium forming a shell below them. The viewing direction is derived
# directly from the geometry (long axis apex->base, and the LV/RV separation axis)
# rather than hard-coded, so it is robust to changes in mesh orientation.

LV_COLOR = "#a3213e"
RV_COLOR = "#2a5a8c"
EPI_COLOR = "#e2a08e"
BASE_COLOR = "#e0b13e"
PHI = 48.0  # pitch (deg) about the lateral (RV->LV) axis; tilts toward a base-on view
WINDOW = (1600, 2000)
ZOOM = 1.15
DIST_FACTOR = 1.6


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _rotate_about_axis(v: np.ndarray, axis: np.ndarray, theta_deg: float) -> np.ndarray:
    theta = np.radians(theta_deg)
    axis = axis / np.linalg.norm(axis)
    return (
        v * np.cos(theta)
        + np.cross(axis, v) * np.sin(theta)
        + axis * np.dot(axis, v) * (1 - np.cos(theta))
    )


def _facet_grid(geo, fdim, marker_name):
    tag = geo.markers[marker_name][0]
    facets = geo.ffun.find(tag)
    vtk = dolfinx.plot.vtk_mesh(geo.mesh, fdim, facets)
    return pyvista.UnstructuredGrid(*vtk)


def _compute_camera(geo, lv, rv, epi, base):
    lv_c = np.array(lv.cell_centers().points).mean(axis=0)
    rv_c = np.array(rv.cell_centers().points).mean(axis=0)
    base_c = np.array(base.cell_centers().points).mean(axis=0)
    epi_pts = np.array(epi.cell_centers().points)
    apex = epi_pts[np.argmax(np.linalg.norm(epi_pts - base_c, axis=1))]

    long_axis = base_c - apex
    long_axis /= np.linalg.norm(long_axis)

    lat = lv_c - rv_c
    lat = lat - np.dot(lat, long_axis) * long_axis
    lat /= np.linalg.norm(lat)  # RV -> LV, stays horizontal in screen space

    view_dir0 = np.cross(long_axis, lat)
    view_dir0 /= np.linalg.norm(view_dir0)
    view_dir = _rotate_about_axis(view_dir0, lat, PHI)

    xall = np.array(geo.mesh.geometry.x)
    center = (xall.min(axis=0) + xall.max(axis=0)) / 2
    diag = np.linalg.norm(xall.max(axis=0) - xall.min(axis=0))
    return dict(center=center, diag=diag, view_dir=view_dir, long_axis=long_axis)


def _setup_camera(p, cam):
    p.camera.position = tuple(cam["center"] + cam["view_dir"] * cam["diag"] * DIST_FACTOR)
    p.camera.focal_point = tuple(cam["center"])
    p.camera.up = tuple(cam["long_axis"])
    p.reset_camera()
    p.camera.zoom(ZOOM)


def _render(lv, rv, epi, base, cam, lit: bool):
    lighting = "three lights" if lit else None
    p = pyvista.Plotter(off_screen=True, window_size=WINDOW, lighting=lighting)
    p.set_background("white")
    for grid, color in ((epi, EPI_COLOR), (lv, LV_COLOR), (rv, RV_COLOR), (base, BASE_COLOR)):
        p.add_mesh(
            grid, color=color, lighting=lit,
            smooth_shading=lit, specular=0.3 if lit else 0, ambient=0.35 if lit else 0,
        )
    _setup_camera(p, cam)
    img = p.screenshot(transparent_background=True, return_img=True)
    p.close()
    return img


def _color_mask(img, hexcolor, tol=10):
    target = np.array(_hex_to_rgb(hexcolor))
    rgb = img[:, :, :3].astype(int)
    dist = np.linalg.norm(rgb - target[None, None, :], axis=2)
    return (dist < tol) & (img[:, :, 3] > 0)


def render_mesh(geodir: Path, outdir: Path) -> dict:
    """Render the BiV mesh and locate the LV/RV cavity openings in image space."""
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=MPI.COMM_WORLD, folder=geodir)
    fdim = geo.mesh.topology.dim - 1

    lv = _facet_grid(geo, fdim, "LV")
    rv = _facet_grid(geo, fdim, "RV")
    epi = _facet_grid(geo, fdim, "EPI")
    base = _facet_grid(geo, fdim, "BASE")

    cam = _compute_camera(geo, lv, rv, epi, base)
    flat_img = _render(lv, rv, epi, base, cam, lit=False)  # unlit -> exact colors for masking
    lit_img = _render(lv, rv, epi, base, cam, lit=True)  # lit -> the actual figure artwork

    alpha = flat_img[:, :, 3]
    rows, cols = np.where(np.any(alpha > 0, axis=1))[0], np.where(np.any(alpha > 0, axis=0))[0]
    margin = 15
    rmin, rmax = max(0, rows[0] - margin), min(flat_img.shape[0] - 1, rows[-1] + margin)
    cmin, cmax = max(0, cols[0] - margin), min(flat_img.shape[1] - 1, cols[-1] + margin)

    flat_c = flat_img[rmin : rmax + 1, cmin : cmax + 1, :]
    lit_c = lit_img[rmin : rmax + 1, cmin : cmax + 1, :]

    lv_mask = _color_mask(flat_c, LV_COLOR)
    rv_mask = _color_mask(flat_c, RV_COLOR)

    def centroid(mask):
        ys, xs = np.where(mask)
        return float(xs.mean()), float(ys.mean())

    def top_row(mask):
        return float(np.where(mask)[0].min())

    h, w = lit_c.shape[:2]
    info = dict(
        width=w, height=h,
        lv_xy=centroid(lv_mask), rv_xy=centroid(rv_mask),
        lv_top=top_row(lv_mask), rv_top=top_row(rv_mask),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lit_c).save(outdir / "mesh_render.png")
    with open(outdir / "mesh_render_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


# ## 3. Drawing the circulation circuit
#
# The circuit is drawn with plain matplotlib primitives (zig-zag resistors, parallel-
# plate capacitors, diode-style one-way valves) directly on top of the mesh image, in
# a shared data-coordinate system. Pixel coordinates from the render are mapped into
# that space via `px_to_data` so the valve stubs land exactly on the LV/RV openings.
#
# Blood is colored by oxygenation (red = oxygenated, blue = deoxygenated), matching
# common physiology diagrams: `AR,SYS`/`VEN,PUL` are red, `VEN,SYS`/`AR,PUL` are blue.
# The systemic loop (larger, higher pressure) arcs high and wide; the pulmonary loop
# (smaller, lower pressure) nests inside it -- both attach straight into the LV/RV
# cavities of the 3D mesh, per the parameter names used in
# `circulation.regazzoni2020.Regazzoni2020`.

RED = "#9c2c3e"
BLUE = "#2d5c8c"
INK = "#33302c"
LW = 2.0


def _resistor(ax, x, y0, y1, color=INK, n=6, width=1.0, lw=LW, zorder=8):
    ylo, yhi = min(y0, y1), max(y0, y1)
    body = 0.72 * (yhi - ylo)
    lead = 0.5 * ((yhi - ylo) - body)
    ys = np.linspace(ylo + lead, yhi - lead, 2 * n + 1)
    xs = np.array([x + (width / 2 if i % 2 else -width / 2) for i in range(len(ys))])
    xs[0] = xs[-1] = x
    ax.plot([x, x], [ylo, ylo + lead], color=color, lw=lw, zorder=zorder, solid_capstyle="round")
    ax.plot([x, x], [yhi - lead, yhi], color=color, lw=lw, zorder=zorder, solid_capstyle="round")
    ax.plot(
        xs, ys, color=color, lw=lw, zorder=zorder, solid_capstyle="round", solid_joinstyle="round",
    )


def _capacitor(ax, x, y0, y1, color=INK, plate_halfwidth=1.5, gap=1.0, lw=LW, zorder=8):
    ylo, yhi = min(y0, y1), max(y0, y1)
    ymid = 0.5 * (ylo + yhi)
    yp1, yp2 = ymid - gap / 2, ymid + gap / 2
    ax.plot([x, x], [ylo, yp1], color=color, lw=lw, zorder=zorder, solid_capstyle="round")
    ax.plot([x, x], [yp2, yhi], color=color, lw=lw, zorder=zorder, solid_capstyle="round")
    for yp in (yp1, yp2):
        ax.plot(
            [x - plate_halfwidth, x + plate_halfwidth], [yp, yp], color=color, lw=lw * 1.5,
            zorder=zorder, solid_capstyle="round",
        )


def _valve(ax, x, y0, y1, flow_up=True, color=INK, lw=LW, zorder=8, halfwidth=1.5):
    ylo, yhi = min(y0, y1), max(y0, y1)
    ymid = 0.5 * (ylo + yhi)
    h = 0.42 * (yhi - ylo)
    if flow_up:
        tri = [(x - halfwidth, ymid - h / 2), (x + halfwidth, ymid - h / 2), (x, ymid + h / 2)]
        bar_y = ymid + h / 2
    else:
        tri = [(x - halfwidth, ymid + h / 2), (x + halfwidth, ymid + h / 2), (x, ymid - h / 2)]
        bar_y = ymid - h / 2
    ax.plot([x, x], [ylo, yhi], color=color, lw=lw, zorder=zorder - 1, solid_capstyle="round")
    poly = mpatches.Polygon(
        tri, closed=True, facecolor="#fbf8f3", edgecolor=color, lw=lw, zorder=zorder,
    )
    ax.add_patch(poly)
    ax.plot(
        [x - halfwidth, x + halfwidth], [bar_y, bar_y],
        color=color, lw=lw * 1.5, zorder=zorder, solid_capstyle="round",
    )


def _atrium(ax, x, y, r, txt, facecolor, zorder=8):
    circ = mpatches.Circle((x, y), r, facecolor=facecolor, edgecolor=INK, lw=LW, zorder=zorder)
    ax.add_patch(circ)
    ax.text(
        x, y, txt, ha="center", va="center", zorder=zorder + 1,
        fontsize=13, fontweight="bold", color="white",
    )


def _vline(ax, x, y0, y1, color=INK, lw=LW, zorder=7):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=zorder, solid_capstyle="round")


def _hline_split(ax, x0, x1, y, c0, c1, lw=LW, zorder=7):
    xm = 0.5 * (x0 + x1)
    ax.plot([x0, xm], [y, y], color=c0, lw=lw, zorder=zorder, solid_capstyle="round")
    ax.plot([xm, x1], [y, y], color=c1, lw=lw, zorder=zorder, solid_capstyle="round")


def _flow_arrow(ax, x, y, dx, dy, color=INK, zorder=9, scale=1.25):
    style = f"-|>,head_width={0.3 * scale},head_length={0.5 * scale}"
    ax.annotate(
        "", xy=(x + dx, y + dy), xytext=(x, y), zorder=zorder,
        arrowprops=dict(arrowstyle=style, color=color, lw=0),
    )


def _elem_label(ax, x, y, text, color=INK, fontsize=8.2, ha="left", zorder=10, dx=1.3):
    ax.text(
        x + (dx if ha == "left" else -dx), y, text, ha=ha, va="center",
        fontsize=fontsize, color=color, family="monospace", zorder=zorder,
    )


def _section_label(ax, x, y, text, color, fontsize=13, ha="center"):
    ax.text(
        x, y, text, ha=ha, va="center", fontsize=fontsize, color=color,
        fontweight="bold", style="italic", zorder=10,
    )


def draw_circuit(ax, info: dict, mesh_extent: tuple[float, float, float, float]) -> None:
    mesh_x0, mesh_x1, mesh_y0, mesh_y1 = mesh_extent
    img_w, img_h = info["width"], info["height"]

    def px_to_data(px, py):
        x = mesh_x0 + (px / img_w) * (mesh_x1 - mesh_x0)
        y = mesh_y1 - (py / img_h) * (mesh_y1 - mesh_y0)
        return x, y

    lv_x, lv_y = px_to_data(*info["lv_xy"])
    rv_x, rv_y = px_to_data(*info["rv_xy"])
    _, lv_top = px_to_data(0, info["lv_top"])
    _, rv_top = px_to_data(0, info["rv_top"])

    DELTA = 4.3
    AV_x, MV_x, PV_x, TV_x = lv_x - DELTA, lv_x + DELTA, rv_x - DELTA, rv_x + DELTA
    STUB = 2.6

    chamber_kw = dict(
        ha="center", va="center", fontsize=13, fontweight="bold", color="white", zorder=10,
    )
    ax.text(lv_x, lv_y, "LV", **chamber_kw)
    ax.text(rv_x, rv_y, "RV", **chamber_kw)

    VALVE_H_OUT, R_H_OUT, C_H_OUT = 4.6, 8.0, 5.2
    VALVE_H_IN, R_H_IN, C_H_IN = 4.0, 6.0, 4.0
    BUS_SYS_Y, BUS_PUL_Y = 87.0, 66.0
    RA_R, LA_R = 4.4, 4.0

    # --- systemic: rising (arterial, red) LV -> AV -> R_AR,SYS -> C_AR,SYS -> bus ---
    y = lv_top
    _vline(ax, AV_x, y, y + STUB, color=RED); y += STUB
    _valve(ax, AV_x, y, y + VALVE_H_OUT, flow_up=True, color=RED)
    _elem_label(ax, AV_x, y + VALVE_H_OUT / 2, "AV", color=RED, ha="right", dx=2.0)
    y += VALVE_H_OUT
    _resistor(ax, AV_x, y, y + R_H_OUT, color=RED)
    _elem_label(ax, AV_x, y + R_H_OUT / 2, r"$R_{AR,SYS}$", color=RED, ha="right", dx=2.0)
    y += R_H_OUT
    _capacitor(ax, AV_x, y, y + C_H_OUT, color=RED)
    _elem_label(ax, AV_x, y + C_H_OUT / 2, r"$C_{AR,SYS}$", color=RED, ha="right", dx=2.0)
    y += C_H_OUT
    _vline(ax, AV_x, y, BUS_SYS_Y, color=RED)  # ascending aorta (flexible length)

    # --- systemic: descending (venous, blue) bus -> C_VEN,SYS -> R_VEN,SYS -> RA -> TV -> RV ---
    y = BUS_SYS_Y
    _capacitor(ax, TV_x, y - C_H_OUT, y, color=BLUE)
    _elem_label(ax, TV_x, y - C_H_OUT / 2, r"$C_{VEN,SYS}$", color=BLUE, dx=2.0)
    y -= C_H_OUT
    _resistor(ax, TV_x, y - R_H_OUT, y, color=BLUE)
    _elem_label(ax, TV_x, y - R_H_OUT / 2, r"$R_{VEN,SYS}$", color=BLUE, dx=2.0)
    y -= R_H_OUT
    ra_y = y - 2.0 - RA_R
    _vline(ax, TV_x, ra_y + RA_R, y, color=BLUE)
    _atrium(ax, TV_x, ra_y, RA_R, "RA", RV_COLOR)
    _vline(ax, TV_x, rv_top + STUB, ra_y - RA_R, color=BLUE)  # vena cava (flexible length)
    _valve(ax, TV_x, rv_top + STUB, rv_top + STUB + VALVE_H_OUT, flow_up=False, color=BLUE)
    _elem_label(ax, TV_x, rv_top + STUB + VALVE_H_OUT / 2, "TV", color=BLUE, dx=2.0)
    _vline(ax, TV_x, rv_top, rv_top + STUB, color=BLUE)

    _hline_split(ax, AV_x, TV_x, BUS_SYS_Y, RED, BLUE)
    _flow_arrow(ax, AV_x + 0.35 * (TV_x - AV_x), BUS_SYS_Y, 3.4, 0, color=RED)
    _flow_arrow(ax, AV_x + 0.72 * (TV_x - AV_x), BUS_SYS_Y, 3.4, 0, color=BLUE)
    _flow_arrow(ax, AV_x, lv_top + STUB + 1.8, 0, 1.6, color=RED)
    _flow_arrow(ax, TV_x, ra_y - RA_R - 3.2, 0, -1.6, color=BLUE)
    _section_label(ax, (AV_x + TV_x) / 2, BUS_SYS_Y + 3.4, "Systemic circulation", RED)

    # --- pulmonary: rising (venous->lungs, blue) RV -> PV -> R_AR,PUL -> C_AR,PUL -> bus ---
    y = rv_top
    _vline(ax, PV_x, y, y + STUB, color=BLUE); y += STUB
    _valve(ax, PV_x, y, y + VALVE_H_IN, flow_up=True, color=BLUE)
    _elem_label(ax, PV_x, y + VALVE_H_IN / 2, "PV", color=BLUE, ha="right", dx=2.0)
    y += VALVE_H_IN
    _resistor(ax, PV_x, y, y + R_H_IN, color=BLUE, width=0.85)
    _elem_label(ax, PV_x, y + R_H_IN / 2, r"$R_{AR,PUL}$", color=BLUE, ha="right", dx=2.0)
    y += R_H_IN
    _capacitor(ax, PV_x, y, y + C_H_IN, color=BLUE, plate_halfwidth=1.3)
    _elem_label(ax, PV_x, y + C_H_IN / 2, r"$C_{AR,PUL}$", color=BLUE, ha="right", dx=2.0)
    y += C_H_IN
    _vline(ax, PV_x, y, BUS_PUL_Y, color=BLUE)

    # --- pulmonary: descending (oxygenated, red): bus -> R_VEN,PUL -> C_VEN,PUL -> LA -> MV -> LV
    y = BUS_PUL_Y
    _resistor(ax, MV_x, y - R_H_IN, y, color=RED, width=0.85)
    _elem_label(ax, MV_x, y - R_H_IN / 2, r"$R_{VEN,PUL}$", color=RED, dx=2.0)
    y -= R_H_IN
    _capacitor(ax, MV_x, y - C_H_IN, y, color=RED, plate_halfwidth=1.3)
    _elem_label(ax, MV_x, y - C_H_IN / 2, r"$C_{VEN,PUL}$", color=RED, dx=2.0)
    y -= C_H_IN
    la_y = y - 1.6 - LA_R
    _vline(ax, MV_x, la_y + LA_R, y, color=RED)
    _atrium(ax, MV_x, la_y, LA_R, "LA", LV_COLOR)
    _vline(ax, MV_x, lv_top + STUB, la_y - LA_R, color=RED)
    _valve(ax, MV_x, lv_top + STUB, lv_top + STUB + VALVE_H_IN, flow_up=False, color=RED)
    _elem_label(ax, MV_x, lv_top + STUB + VALVE_H_IN / 2, "MV", color=RED, dx=2.0)
    _vline(ax, MV_x, lv_top, lv_top + STUB, color=RED)

    _hline_split(ax, PV_x, MV_x, BUS_PUL_Y, BLUE, RED)
    _flow_arrow(ax, PV_x + 0.35 * (MV_x - PV_x), BUS_PUL_Y, -3.2, 0, color=BLUE)
    _flow_arrow(ax, PV_x + 0.72 * (MV_x - PV_x), BUS_PUL_Y, -3.2, 0, color=RED)
    _flow_arrow(ax, PV_x, rv_top + STUB + 1.6, 0, 1.6, color=BLUE)
    _flow_arrow(ax, MV_x, la_y - LA_R - 2.6, 0, -1.6, color=RED)
    _section_label(
        ax, (PV_x + MV_x) / 2 - 2.0, BUS_PUL_Y + 3.4, "Pulmonary circulation", BLUE, fontsize=12,
    )

    # --- legend ---
    lx, ly = 10.0, 34.0
    _resistor(ax, lx, ly, ly + 4.5, color=INK, width=0.85, lw=1.6)
    ax.text(lx + 2.6, ly + 2.25, "resistance $R$", ha="left", va="center", fontsize=9, color=INK)
    ly -= 7.0
    _capacitor(ax, lx, ly, ly + 4.0, color=INK, plate_halfwidth=1.2, lw=1.6)
    ax.text(lx + 2.6, ly + 2.0, "compliance $C$", ha="left", va="center", fontsize=9, color=INK)
    ly -= 7.5
    _valve(ax, lx, ly, ly + 3.6, flow_up=True, color=INK, lw=1.6, halfwidth=1.2)
    ax.text(lx + 2.6, ly + 1.8, "one-way valve", ha="left", va="center", fontsize=9, color=INK)
    ly -= 6.0
    ax.plot([lx - 1.2, lx + 1.2], [ly, ly], color=RED, lw=2.2, solid_capstyle="round")
    ax.text(lx + 2.6, ly, "oxygenated blood", ha="left", va="center", fontsize=9, color=INK)
    ly -= 3.6
    ax.plot([lx - 1.2, lx + 1.2], [ly, ly], color=BLUE, lw=2.2, solid_capstyle="round")
    ax.text(lx + 2.6, ly, "deoxygenated blood", ha="left", va="center", fontsize=9, color=INK)


# ## 4. Putting it all together


def make_figure(outdir: Path) -> Path:
    geodir = generate_mesh(outdir)
    info = render_mesh(geodir, outdir)
    img = np.array(Image.open(outdir / "mesh_render.png"))

    fig, ax = plt.subplots(figsize=(12, 13.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 101)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#fbf8f3")
    ax.set_facecolor("#fbf8f3")

    mesh_x0, mesh_x1, mesh_y0 = 23.0, 77.0, 0.0
    mesh_y1 = mesh_y0 + (mesh_x1 - mesh_x0) * info["height"] / info["width"]
    ax.imshow(img, extent=(mesh_x0, mesh_x1, mesh_y0, mesh_y1), zorder=5)

    draw_circuit(ax, info, (mesh_x0, mesh_x1, mesh_y0, mesh_y1))

    title = "Closed-loop circulation model coupled to a 3D bi-ventricular geometry"
    subtitle = (
        "UK Biobank atlas mesh (clipped)   •   "
        "lumped-parameter circuit after Regazzoni et al. (2022)"
    )
    ax.text(50, 99.5, title, ha="center", va="top", fontsize=15.5, fontweight="bold", color=INK)
    ax.text(
        50, 97.3, subtitle, ha="center", va="top", fontsize=10.5, color="#6b6660", style="italic",
    )

    outpath = outdir / "ukb_circulation_figure.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    path = make_figure(OUTDIR)
    print(f"Saved figure to {path}")
