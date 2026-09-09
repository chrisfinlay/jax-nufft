"""Refuse to publish unless the release tag and the built artifacts agree.

Issue #59 shipped a package whose reported version named a release that did
not exist, because two files each held a copy of the number and only one was
updated. ``src/jax_nufft/_version.py`` is the single source now, and this is
the gate that keeps the git tag honest against it.

It reads the version out of the built ``dist/`` filenames rather than the
source tree, so what is checked is the artifact that is about to be uploaded.
PyPI will not let a filename be re-uploaded, so a wrong version here is not
recoverable -- the only fix is to burn the number and cut another.

usage: check_release_version.py <tag>
"""

from __future__ import annotations

import sys
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

DIST = Path("dist")
EXPECTED_NAME = canonicalize_name("jax-nufft")


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def versions_in_dist() -> set[str]:
    """The version segment of every sdist and wheel built."""
    found = set()
    for path in sorted(DIST.iterdir()):
        if path.suffix == ".whl":
            name, version, *_ = path.stem.split("-")
        elif path.name.endswith(".tar.gz"):
            name, _, version = path.name[: -len(".tar.gz")].rpartition("-")
        else:
            continue
        if canonicalize_name(name) != EXPECTED_NAME:
            fail(f"{path.name} is not a jax-nufft artifact")
        found.add(version)
    return found


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        fail("usage: check_release_version.py <tag>")
    tag = argv[1].removeprefix("v")

    if not DIST.is_dir() or not (built := versions_in_dist()):
        fail("no built artifacts under dist/ to check")
    if len(built) > 1:
        fail(f"dist/ holds more than one version: {sorted(built)}")
    packaged = built.pop()

    try:
        tag_version = Version(tag)
        packaged_version = Version(packaged)
    except InvalidVersion as exc:
        fail(f"cannot parse a version: {exc}")

    # Compare parsed versions, not strings: 0.2.0 and 0.2 are the same release
    # and a tag written either way should pass.
    if tag_version != packaged_version:
        fail(
            f"release tag {argv[1]!r} means version {tag_version}, but the built "
            f"artifacts are {packaged_version}. Update "
            f"src/jax_nufft/_version.py, or retag."
        )
    if packaged_version.is_devrelease or packaged_version.is_prerelease:
        fail(
            f"refusing to publish {packaged_version} to PyPI: it is a "
            "development or pre-release version. Drop the suffix in "
            "src/jax_nufft/_version.py before releasing."
        )

    print(f"tag {argv[1]} matches built version {packaged_version}")


if __name__ == "__main__":
    main(sys.argv)
