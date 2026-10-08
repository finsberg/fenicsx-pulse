from pathlib import Path

from pulse.cli import config as c
from pulse.cli.config_reference import SECTIONS, render_reference

# Models deliberately left out of the reference: the private bases (their fields are listed
# under each subclass) and the top-level `Config` (its fields are the sections themselves).
UNDOCUMENTED = {
    c._Base,
    c._GeometryBase,
    c._FiberAngles,
    c._MaterialBase,
    c._BestelBase,
    c._OdeCirculation,
    c.Config,
}


def test_reference_mentions_every_section_and_is_up_to_date():
    text = render_reference()
    for section in (
        "[geometry]",
        "[geometry.fibers]",
        "[geometry.ldrb]",
        "[material]",
        "[active]",
        "[compressibility]",
        "[viscoelasticity]",
        "[bcs]",
        "[[load]]",
        "[load.profile]",
        "[circulation]",
        "[prestress]",
        "[time]",
        "[problem]",
        "[solver]",
        "[output]",
        "[postprocess]",
    ):
        assert section in text
    doc = Path(__file__).parents[2] / "docs" / "cli_reference.md"
    assert doc.read_text() == text, "Run `python scripts/gen_cli_reference.py`"


def test_reference_documents_every_config_model():
    documented = {model for _, models in SECTIONS for model in models}
    missing = [m.__name__ for m in c.ALL_MODELS if m not in documented | UNDOCUMENTED]
    assert missing == [], "Add these to pulse.cli.config_reference.SECTIONS"
