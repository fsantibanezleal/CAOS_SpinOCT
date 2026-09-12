"""Guards on the documentation and the source references.

These keep the repo honest as it grows: every theory doc must cite a real DOI, and the display
version must agree across the manifest, the VERSION file, the package and the changelog.
"""

from __future__ import annotations

import re
from pathlib import Path

import spinoct

REPO = Path(__file__).resolve().parents[1]


def test_display_version_is_consistent_everywhere() -> None:
    version_file = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    assert version_file == spinoct.__display_version__
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"[{version_file}]" in changelog
    # The manifest carries the semver form with the padding dropped.
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    semver = ".".join(str(int(part)) for part in version_file.split("."))
    assert f'version = "{semver}"' in pyproject


def test_every_theory_doc_cites_a_real_doi() -> None:
    theory = REPO / "docs" / "theory"
    docs = sorted(theory.glob("*.md"))
    assert docs, "no theory docs found"
    doi = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        assert doi.search(text), f"{doc.name} cites no DOI"


def test_no_em_dash_or_emoji_in_repo_text() -> None:
    """ADR-0067: no em-dash, no emoji in repo content."""
    banned = re.compile(r"[\u2014\u2013\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
    for pattern in ("*.md", "src/**/*.py", "tests/**/*.py"):
        for path in REPO.glob(pattern):
            if ".venv" in path.parts or "build" in path.parts:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                match = banned.search(line)
                assert match is None, f"{path.name}:{number} has a banned character {match.group()!r}"
