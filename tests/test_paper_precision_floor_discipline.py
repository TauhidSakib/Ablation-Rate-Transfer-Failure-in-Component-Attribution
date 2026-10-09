"""The K=100-class NAF values must never appear in the draft without their floor.

These numbers sit at (seed 1: 2.0x) or below (seed 2: 0.97x) the bf16 precision
floor. Quoted alone they read as a measured null; they are not one. This test
enforces the rule mechanically so an edit cannot quietly reintroduce a bare value.

Guarded means: the word "floor" appears within four lines, or the line carries the
dagger footnote marker used in tables.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
PAPER = ROOT / "PAPER.md"

# every value at or beneath the measured floor (0.00025 seed 1, 0.00027 seed 2)
SUB_FLOOR = re.compile(r"0\.0005\b|0\.00051|0\.00061|0\.00028|0\.00026\b|0\.00016|0\.00002\b")
WINDOW = 4


@pytest.mark.skipif(not PAPER.exists(), reason="PAPER.md not present")
def test_sub_floor_values_always_carry_their_floor():
    lines = PAPER.read_text(encoding="utf-8").split("\n")
    offenders = []
    for i, line in enumerate(lines):
        if not SUB_FLOOR.search(line):
            continue
        if "\u2020" in line:  # table cell carrying the footnote marker
            continue
        context = "\n".join(lines[max(0, i - WINDOW): i + WINDOW + 1]).lower()
        if "floor" not in context:
            offenders.append(f"  L{i + 1}: {line.strip()[:100]}")
    assert not offenders, (
        "Sub-precision-floor NAF values quoted without their floor:\n"
        + "\n".join(offenders)
        + "\n\nAdd the floor within 4 lines, or mark the table cell with \u2020."
    )


@pytest.mark.skipif(not PAPER.exists(), reason="PAPER.md not present")
def test_no_pending_markers_in_draft():
    """A submission draft must not contain pending-number markers."""
    text = PAPER.read_text(encoding="utf-8")
    assert "\u29d7" not in text, "PAPER.md still contains \u29d7 pending markers"


def test_precision_floor_is_a_variance_share():
    """A floor is noise/residual. Any run reporting one outside [0,1) is a bug.

    This is the dimensional check that caught the wrong-checkpoint bug; see
    the precision-floor checkpoint guard.
    """
    import json

    checked = 0
    for pf in (ROOT / "runs").glob("*/precision_floor.json"):
        j = json.loads(pf.read_text(encoding="utf-8"))
        floor = j["floor_fraction_of_residual"]
        assert 0.0 <= floor < 1.0, (
            f"{pf}: floor {floor} is not a variance share. The run almost certainly "
            f"compared against a different checkpoint -- re-run with the matching --config."
        )
        checked += 1
    if checked == 0:
        pytest.skip("no precision_floor.json files present")
