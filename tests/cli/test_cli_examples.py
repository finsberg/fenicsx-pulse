import importlib.util
import shutil

from mpi4py import MPI

import pytest

from pulse.cli import TEMPLATES_DIR, _available_templates, main
from pulse.cli.overrides import load_config

EXPECTED = {
    "unit_cube",
    "benchmark1",
    "benchmark2",
    "benchmark3",
    "lv_ellipsoid",
    "lv_sliding_base",
    "spatial_material",
    "biv_ellipsoid",
    "ukb_bcs",
    "cylinder_bestel",
    "bestel_lv",
    "bestel_biv",
    "complete_cycle",
    "split_biv",
    "monolithic_lv",
    "monolithic_biv",
}
# optional packages each template needs to *run* (not to validate)
NEEDS = {
    "biv_ellipsoid": ["ldrb"],
    "ukb_bcs": ["ukb", "ldrb"],
    "cylinder_bestel": ["circulation", "scipy"],
    "bestel_lv": ["circulation", "scipy"],
    "bestel_biv": ["circulation", "scipy", "ldrb"],
    "complete_cycle": ["ukb", "ldrb", "circulation", "scipy"],
    "split_biv": ["ldrb", "gotranx", "circulation", "scipy"],
    "monolithic_lv": ["gotranx", "circulation", "scipy"],
    "monolithic_biv": ["ukb", "ldrb", "gotranx", "circulation", "scipy"],
}
# cheap overrides for a smoke run: first step(s) only
SMOKE = {
    "benchmark2": ["time.num_steps=1", 'time.end_time="0.1 s"'],
    "benchmark3": ["time.num_steps=1", 'time.end_time="0.0526315789 s"'],
    "cylinder_bestel": ['time.end_time="0.02 s"'],
    "bestel_lv": ['time.end_time="0.002 s"', 'output.save_every="1 ms"'],
    "bestel_biv": ['time.end_time="0.002 s"', 'output.save_every="1 ms"'],
    "split_biv": ['time.end_time="2 ms"', 'output.save_every="1 ms"'],
    "monolithic_lv": [
        'time.end_time="4 ms"',
        'output.save_every="2 ms"',
        "prestress.ramp_steps=10",
        "prestress.inflate_steps=8",
    ],
}
# Prestressed UKB meshes take minutes: these templates are validated, not smoke-run.
VALIDATE_ONLY = {"complete_cycle", "monolithic_biv"}


def test_every_expected_template_exists():
    assert set(_available_templates()) == EXPECTED


# optional packages a template needs to *validate*: its .ode file lives in that package
VALIDATE_NEEDS = {
    "split_biv": ["circulation"],
    "monolithic_lv": ["circulation"],
    "monolithic_biv": ["circulation"],
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_template_validates(name, tmp_path):
    missing = [m for m in VALIDATE_NEEDS.get(name, []) if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"needs {missing}")
    tmp = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    assert main(["init", str(tmp / "config.toml"), "--template", name]) == 0
    conf = load_config(tmp / "config.toml", environ={})
    header = (TEMPLATES_DIR / name / "config.toml").read_text().splitlines()[0]
    assert header.startswith("# Transcribes demo/")
    assert conf.output.folder == (tmp / "output").resolve()


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(EXPECTED - {"unit_cube"} - VALIDATE_ONLY))
def test_template_smoke_run(name, tmp_path):
    missing = [m for m in NEEDS.get(name, []) if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"needs {missing}")
    tmp = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    cfg = tmp / "config.toml"
    assert main(["init", str(cfg), "--template", name]) == 0
    sets = []
    for item in SMOKE.get(name, []):
        sets += ["--set", item]
    assert main(["run", str(cfg), *sets]) == 0
    loads_csv = (tmp / "output" / "loads.csv").read_text().splitlines()
    assert len(loads_csv) >= 3  # header + at least 2 data rows
    assert main(["post", str(cfg), *sets, "--set", "postprocess.plots=false"]) == 0
    if MPI.COMM_WORLD.rank == 0:
        shutil.rmtree(tmp / "geometry", ignore_errors=True)
        shutil.rmtree(tmp / "prestress", ignore_errors=True)
