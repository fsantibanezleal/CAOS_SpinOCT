"""No unreviewed magic constant in solver code.

WHY
---
This gate was written for a sibling package after four defects of one shape were found in a single
afternoon: a sigmoid bounding predictions to (0, 1), a blow-up bound of 2.0 in raw units, a sparse
regression threshold carrying units of one over the driver, and a noise-scale prior capped at 0.5 in
raw target units. Every one was correct for that product's first observable, a fraction on the unit
interval, hard-coded into a method, and never revisited as the matrix grew. Two sat in BASELINE
rungs, so their failure read as a satisfying result rather than a bug, and two were published as
scientific null results before being withdrawn.

This package carries the same risk more sharply, for a reason specific to its field. The optimal
control literature for magnetization switching writes the same physical quantity four different
ways: exchange as ``+sum J S.S``, ``-sum J S.S``, with and without the one-half double-counting
factor, and over unit vectors rather than spin vectors; anisotropy as meV per atom, meV per formula
unit, joules per cubic metre, or an effective field in tesla. A number that silently assumes one of
those produces a confident wrong answer in another, and the plot still looks right.

On top of that, the central quantity of this package is dimensionally treacherous on its own:
``Phi = int |b|^2 dt`` is in T^2 s, not joules, and becomes an energy only through a circuit
constant. ``spinoct.units.CircuitModel`` makes that assumption a required, described object rather
than a literal, and this gate keeps it that way.

WHAT THIS DOES
--------------
Walks the AST of every solver module and collects each float literal that could plausibly be
compared against a measured quantity. Each must appear in ``scripts/unit_constants_allow.json``
with a written justification, which forces the author to answer one question at the moment they
write the number: what unit does this assume?

Ratios, probabilities, tolerances relative to a computed scale, and iteration counts are all
legitimate. They just have to be declared as such.

Run:  python scripts/check_unit_constants.py
      python scripts/check_unit_constants.py --update   (re-baseline, then review the diff)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOW = REPO / "scripts" / "unit_constants_allow.json"

# spinoct.units is the declared home of physical constants: every value there is named with its
# unit, carries a primary source, and is asserted by self_check(). Those are reviewed as data,
# against CODATA, rather than as solver logic. Everything else must justify its numbers.
SKIP_PARTS = ("/units/",)

ROOTS = (REPO / "src" / "spinoct",)

# Values that cannot carry a unit assumption in any code path: identities, halves, and simple
# ratios used for arithmetic rather than for comparison against a measured quantity.
UNIVERSAL = {0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 6.0, 100.0, 10.0, 1000.0}


def collect() -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if any(part in "/" + rel for part in SKIP_PARTS):
                continue
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            lines = text.split("\n")
            hits = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, float):
                    continue
                value = float(node.value)
                if abs(value) in UNIVERSAL or value == 0:
                    continue
                hits.append(
                    {
                        "value": value,
                        "line": node.lineno,
                        "source": lines[node.lineno - 1].strip()[:110],
                    }
                )
            if hits:
                found[rel] = hits
    return found


def key(rel: str, hit: dict) -> str:
    return f"{rel}::{hit['value']!r}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="re-baseline; review the diff")
    args = parser.parse_args()

    found = collect()
    allow = json.loads(ALLOW.read_text(encoding="utf-8")) if ALLOW.exists() else {}

    if args.update:
        merged = dict(allow)
        added = 0
        for rel, hits in found.items():
            for hit in hits:
                marker = key(rel, hit)
                if marker not in merged:
                    merged[marker] = {
                        "reason": "TODO: state what unit this number assumes",
                        "source": hit["source"],
                    }
                    added += 1
                else:
                    merged[marker]["source"] = hit["source"]
        for marker in list(merged):
            rel = marker.split("::")[0]
            if rel in found and not any(key(rel, h) == marker for h in found[rel]):
                del merged[marker]
            elif rel not in found:
                del merged[marker]
        ALLOW.write_text(
            json.dumps(dict(sorted(merged.items())), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"baseline written: {len(merged)} constants ({added} new, marked TODO)")
        return 0

    problems: list[str] = []
    for rel, hits in found.items():
        for hit in hits:
            marker = key(rel, hit)
            entry = allow.get(marker)
            if entry is None:
                problems.append(
                    f"{rel}:{hit['line']}: unreviewed constant {hit['value']!r}\n"
                    f"      {hit['source']}\n"
                    "      What unit does this number assume? Add it to "
                    "scripts/unit_constants_allow.json with a justification, or derive it "
                    "from the inputs."
                )
            elif str(entry.get("reason", "")).startswith("TODO"):
                problems.append(
                    f"{rel}:{hit['line']}: constant {hit['value']!r} is baselined but not justified\n"
                    f"      {hit['source']}"
                )

    if problems:
        print(f"UNIT-CONSTANT GATE FAILED: {len(problems)} problem(s)\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    total = sum(len(v) for v in found.values())
    print(f"UNIT-CONSTANT GATE OK: {total} constants across {len(found)} modules, all justified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
