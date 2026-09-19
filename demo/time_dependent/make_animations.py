"""Turn finished runs into the videos and figures the documentation shows.

Run by hand after a full simulation, never as part of the documentation build:

    cd demo/time_dependent
    PULSE_DYNAMIC=1 python3 monolithic_3d0d.py        # writes frames + traces
    PULSE_DYNAMIC=0 python3 monolithic_3d0d_biv.py
    python3 make_animations.py

Each demo records the moving geometry every few steps as it runs, alongside the
pressure-volume traces. This reads both back and writes an `.mp4` and a `.png`
into `_static/` for the pages to embed. The demos take two steps under CI, so
nothing here is on the critical path of a documentation build and the pages
still show a whole beat.

`--check` reports what is available without rendering anything.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import animation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("make_animations")

HERE = Path(__file__).parent
STATIC = HERE.parent.parent / "_static"

#: Where each run left its files, and what to call the result.
ASSETS = {
    "monolithic_3d0d_lv": {
        "results": HERE / "results_monolithic_3d0d",
        "frames": "frames-dynamic-stretch.npz",
        "traces": "traces_monolithic-dynamic-stretch.npz",
        "chambers": ("LV",),
        "title": "LV ellipsoid, monolithic 3D-0D (dynamic)",
        "zoom": 1.0,
    },
    "monolithic_3d0d_biv": {
        "results": HERE / "results_monolithic_3d0d_biv",
        "frames": "frames-quasistatic.npz",
        "traces": "traces_biv-quasistatic.npz",
        "chambers": ("LV", "RV"),
        "title": "UKB biventricular mesh, monolithic 3D-0D",
        "zoom": 1.1,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=None, help="assets to build")
    parser.add_argument("--check", action="store_true", help="report inputs, render nothing")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    names = args.names or list(ASSETS)
    STATIC.mkdir(parents=True, exist_ok=True)

    for name in names:
        spec = ASSETS[name]
        frames = spec["results"] / spec["frames"]
        traces = spec["results"] / spec["traces"]

        missing = [p for p in (frames, traces) if not p.exists()]
        if missing:
            logger.warning(
                f"{name}: missing {', '.join(p.name for p in missing)} -- "
                f"run the demo for a full beat first (see this module's docstring)",
            )
            continue
        if args.check:
            logger.info(f"{name}: ready ({frames.name}, {traces.name})")
            continue

        figure = animation.save_pv_figure(
            traces,
            STATIC / f"pv_loop_{name}.png",
            chambers=spec["chambers"],
            title=spec["title"],
        )
        logger.info(f"{name}: wrote {figure.relative_to(STATIC.parent)}")

        video = animation.render(
            frames,
            traces,
            STATIC / f"{name}.mp4",
            chambers=spec["chambers"],
            fps=args.fps,
            zoom=spec["zoom"],
            title=spec["title"],
        )
        logger.info(f"{name}: wrote {video.relative_to(STATIC.parent)}")


if __name__ == "__main__":
    main()
