"""Regenerate docs/cli_reference.md from the pulse.cli.config pydantic models.

Run after any change to src/pulse/cli/config.py:

    python scripts/gen_cli_reference.py

tests/cli/test_cli_reference.py fails if docs/cli_reference.md is out of date.
"""

from pathlib import Path

from pulse.cli.config_reference import render_reference

out = Path(__file__).parents[1] / "docs" / "cli_reference.md"
out.write_text(render_reference())
print(f"Wrote {out}")
