"""Keep ``tests/jax_floor_probe.py`` honest, and the three floor declarations agreed.

The probe itself runs in its own CI job against ``jax`` pinned to the declared
floor (see ``.github/workflows/test.yml``); this module runs the same two checks
against whatever ``jax`` the pixi environment resolved. That is not redundant.
The probe's real failure mode is not "a symbol is missing" but "the scanner
returned nothing and the job passed vacuously" -- a derived list that derives
zero symbols is green at every floor, forever. So the tests below pin what the
scan must contain, and pin the alias forms it has to understand, in the
environment that actually gets run on every push.

Every cell here runs the *one* jax the pixi environment resolved, so on its own
this module can only ever watch the probe succeed -- and a check that has never
been seen to fail is not known to be a check at all. The cells named
``..._reports_...`` / ``..._rejects_...`` are the negative controls that close
that, and two of them are the real thing rather than a fixture: deleting
``jax.typeof`` off the installed module reproduces jax 0.5.0's
``AttributeError: module 'jax' has no attribute 'typeof'`` exactly, in process,
so both halves of issue #21's bug are exercised as *failures* here without a
second jax anywhere.

``test_pixi_features_declare_the_same_floor`` covers the other half of issue
#21's original bug: the floor is declared in *three* places (``pyproject.toml``
and both pixi CPU/GPU features) and only the ``pyproject.toml`` one is what the
probe job installs. Two of the three agreeing is the same silence the probe
exists to break. ``test_floor_job_python_matches_requires_python`` is its
sibling on the interpreter axis, which the workflow has to write out longhand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import FrameType

import jax
import pytest
import tomllib

from tests import jax_floor_probe
from tests.jax_floor_probe import (
    REPO_ROOT,
    ProbeError,
    check_installed_is_floor,
    check_primitive_pattern,
    check_symbols,
    close,
    declared_floor,
    jax_return_value_methods,
    jax_symbols,
    main,
    resolve_symbol,
)

# Anchors for the derived set: names that ``src/jax_nufft/wgridder.py`` cannot
# stop using without the primitive pattern itself changing. They are **not** the
# authority on what gets probed -- the AST scan is, and it currently finds many
# more -- and the count is not meaningful: it is not a transcription of the 18
# symbols issue #53 records the #21 review as having checked by hand, since that
# list is recorded nowhere in the repository, and any resemblance in size is
# coincidence. They exist for one purpose: a scan that silently stops finding
# things must fail here rather than pass everywhere. Written out by hand
# deliberately, because a second derived list would fail in the same way as the
# first.
ANCHOR_SYMBOLS = frozenset(
    {
        "jax.Array",
        "jax.core.ShapedArray",
        "jax.disable_jit",
        "jax.dtypes.canonicalize_dtype",
        "jax.extend.core.Primitive",
        "jax.grad",
        "jax.interpreters.ad.Zero",
        "jax.interpreters.ad.is_undefined_primal",
        "jax.interpreters.ad.primitive_jvps",
        "jax.interpreters.ad.primitive_transposes",
        "jax.interpreters.batching.not_mapped",
        "jax.interpreters.batching.primitive_batchers",
        "jax.interpreters.mlir.lower_fun",
        "jax.interpreters.mlir.register_lowering",
        "jax.jit",
        "jax.jvp",
        "jax.lax.dynamic_slice",
        "jax.lax.scan",
        "jax.linear_transpose",
        "jax.tree_util.register_pytree_node",
        "jax.typeof",
    }
)


@pytest.fixture
def jax_without_typeof(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete ``jax.typeof`` from the installed jax, restoring it afterwards.

    This is issue #21's bug, reproduced on the jax the pixi environment
    resolved: jax 0.5.0 shipped no ``jax.typeof`` and ``wgridder.py`` called it,
    so every operator under :func:`jax.disable_jit` raised ``AttributeError``.
    Deleting the attribute produces the identical ``AttributeError: module 'jax'
    has no attribute 'typeof'`` from the identical lines, which is what lets the
    cells below watch both checks *fail* -- the one thing 0.5.0 would have shown
    and the pinned environment never can. ``jax.typeof`` is a plain module
    attribute with no lazy ``__getattr__`` behind it, so the deletion is real
    and ``monkeypatch`` puts it back.
    """
    monkeypatch.delattr(jax, "typeof")


def test_declared_floor_is_a_version_string() -> None:
    floor = declared_floor()
    assert floor[0].isdigit(), floor
    assert floor.count(".") >= 1, floor


def test_declared_floor_requires_a_lower_bound(tmp_path: Path) -> None:
    """A ``jax`` requirement with no ``>=`` is a floor the probe cannot exercise."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["jax", "numpy>=1.24"]\n')
    with pytest.raises(ProbeError, match="no '>=' lower bound"):
        declared_floor(pyproject)


def test_declared_floor_is_not_confused_by_jax_finufft(tmp_path: Path) -> None:
    """``jax-finufft>=1.3.0`` also starts with "jax" and must not be read as the floor.

    The fabricated floor is deliberately a version nothing in this repository
    declares. Reading it back as ``0.6.0`` would look like a pass with the
    repository's real floor substituted in, so a ``declared_floor`` that had
    quietly become a hard-coded constant -- the one thing the function's own
    docstring forbids -- would sail through a fixture that used ``0.6.0``.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["jax-finufft>=1.3.0", "jax>=9.9.9"]\n')
    assert declared_floor(pyproject) == "9.9.9"


@pytest.mark.parametrize("requirement", ["jax~=0.6.0", "jax==0.6.0", "jax==0.6.*"])
def test_declared_floor_rejects_non_ge_pins_loudly(tmp_path: Path, requirement: str) -> None:
    """``~=`` and ``==`` do imply a minimum, and are still refused, on purpose.

    The job installs ``jax==<floor>`` and then asserts the installed version
    *is* that floor -- a statement about a ``>=`` bound, not a compatible-release
    or pinned one. Rather than guess, the parser refuses and says so. Pinned
    here because "stricter than PEP 508" is a decision that must not decay into
    a silent one: the failure has to stay a named requirement and a non-zero
    exit, never a bogus floor.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\ndependencies = ["{requirement}"]\n')
    with pytest.raises(ProbeError, match="no '>=' lower bound"):
        declared_floor(pyproject)


def test_declared_floor_rejects_an_environment_marker(tmp_path: Path) -> None:
    """A marker is refused rather than ignored.

    The probe is stdlib-only so CI can read the floor before installing
    anything, which means no ``packaging.markers``. Dropping the marker instead
    would let a requirement that does not apply to the job's platform set the
    floor the job then claims to have exercised -- and with two
    marker-partitioned entries, whichever came first would win.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\ndependencies = [\n"
        '  "jax>=0.6.0; sys_platform == \\"win32\\"",\n'
        '  "jax>=0.9.0; sys_platform != \\"win32\\"",\n]\n'
    )
    with pytest.raises(ProbeError, match="environment marker"):
        declared_floor(pyproject)


def test_pixi_features_declare_the_same_floor() -> None:
    """``pyproject.toml`` and both pixi features must agree on the jax floor.

    Issue #21 had to raise all three. Only the ``pyproject.toml`` one is what
    the probe job installs, so a pixi feature left behind would be a floor no
    CI job exercises -- the exact hole this issue closes.
    """
    floor = declared_floor()
    with (REPO_ROOT / "pixi.toml").open("rb") as handle:
        pixi = tomllib.load(handle)
    declared = {
        f"feature.{name}": pixi["feature"][name]["dependencies"]["jax"] for name in ("cpu", "gpu")
    }
    assert declared == {"feature.cpu": f">={floor}", "feature.gpu": f">={floor}"}


def _floor_job_steps() -> list[str]:
    """The lines of the ``jax-floor`` job in ``.github/workflows/test.yml``.

    A four-line indentation scan rather than PyYAML, because the pixi ``test``
    environment does not ship PyYAML and adding a dependency to read four lines
    of a file this repository writes by hand is a poor trade. It is enough for
    what is asserted: the job's block starts at ``  jax-floor:`` at two-space
    indent and ends at the next line indented that far, so a missing job is an
    empty list and fails loudly rather than vacuously.
    """
    lines = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line == "  jax-floor:"]
    assert len(starts) == 1, "expected exactly one 'jax-floor:' job in test.yml"
    body: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line.strip() and not line.startswith("   "):
            break
        body.append(line)
    return body


def test_floor_job_exists_and_runs_the_probe() -> None:
    """The workflow still has a ``jax-floor`` job, and it still runs the probe.

    Nothing else would notice the job being deleted: the probe would keep
    passing in this module against the pixi jax, and the declared floor would
    go back to being exercised by nothing at all -- the exact state issue #53
    exists to end.
    """
    body = "\n".join(_floor_job_steps())
    assert "jax_floor_probe.py --print-floor" in body
    assert 'python -m pip install "jax==${{ steps.floor.outputs.version }}"' in body
    assert "run: python tests/jax_floor_probe.py" in body


def test_floor_job_python_matches_requires_python() -> None:
    """The floor job's interpreter must be ``requires-python``'s lower bound.

    The sibling of ``test_pixi_features_declare_the_same_floor`` on the other
    axis. The jax floor is *read* from ``pyproject.toml``; the python floor
    cannot be, because ``actions/setup-python`` takes no expression there, so
    the workflow writes ``python-version: "3.11"`` by hand and claims in a
    comment that it is ``requires-python``. Raise ``requires-python`` to
    ``>=3.12`` and, without this cell, the job would go on probing an
    interpreter the project no longer supports and stay green -- hard-coding a
    floor nothing checks, which is what #53 calls out.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        requires_python = tomllib.load(handle)["project"]["requires-python"]
    bound = re.fullmatch(r">=\s*(?P<version>\d+\.\d+)", requires_python.strip())
    assert bound is not None, f"requires-python {requires_python!r} is not a bare '>=' bound"
    declared = [
        line.split(":", 1)[1].strip().strip('"')
        for line in _floor_job_steps()
        if line.strip().startswith("python-version:")
    ]
    assert declared == [bound.group("version")], (
        f"the jax-floor job runs python {declared} but pyproject.toml declares "
        f"requires-python = {requires_python!r}"
    )


def test_symbol_scan_still_finds_the_anchors() -> None:
    """The derived set must be a strict superset of the anchors.

    This is the anti-vacuity gate. A scanner that returns an empty set -- an
    alias form it stops understanding, a directory that moves -- makes the CI
    probe green at every jax version ever released, and nothing else in the
    repository would notice.
    """
    derived = jax_symbols()
    assert ANCHOR_SYMBOLS <= derived, sorted(ANCHOR_SYMBOLS - derived)
    assert len(derived) > len(ANCHOR_SYMBOLS)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import jax\njax.typeof(x)\n", "jax.typeof"),
        ("import jax.numpy as jnp\njnp.exp(x)\n", "jax.numpy.exp"),
        ("import jax.extend as jex\njex.core.Primitive('p')\n", "jax.extend.core.Primitive"),
        ("from jax.interpreters import ad\nad.Zero(a)\n", "jax.interpreters.ad.Zero"),
        ("from jax import grad\ngrad(f)\n", "jax.grad"),
        ("import jax.numpy\njax.numpy.fft.fftn(x)\n", "jax.numpy.fft.fftn"),
        # A ``Call`` ends the chain, so an instance attribute of the result is
        # not mistaken for a module attribute of ``jax.numpy``.
        ("import jax.numpy as jnp\njnp.array(x).shape\n", "jax.numpy.array"),
    ],
)
def test_scanner_understands_each_alias_form(tmp_path: Path, source: str, expected: str) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text(source)
    assert expected in jax_symbols(tmp_path, scan_dirs=("src",))


def test_scanner_ignores_non_jax_roots(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text(
        "import jax\nimport numpy as np\nnp.exp(plan.uvw_m)\njax.jit(f)\n"
    )
    assert jax_symbols(tmp_path, scan_dirs=("src",)) == {"jax", "jax.jit"}


def test_every_derived_jax_symbol_resolves() -> None:
    """The installed jax has every symbol ``src/`` and ``tests/`` touch.

    Trivially true here (the pixi env resolves a current jax); the point is that
    the same function is what the floor job runs, so it cannot rot unnoticed.
    """
    assert len(check_symbols()) >= len(ANCHOR_SYMBOLS)


def test_primitive_pattern_behaves() -> None:
    """The miniature of the wgridder's three linear primitives, on this jax."""
    check_primitive_pattern()


# --------------------------------------------------------------------------
# Negative controls. Everything above watches the probe succeed; a check never
# seen to fail is not known to be a check. These are the cells that watch each
# of its mechanisms fail, and they are why a version of this module that has
# quietly stopped detecting anything cannot stay green.
# --------------------------------------------------------------------------


def test_resolve_symbol_reports_an_absent_attribute() -> None:
    """The ``getattr`` walk is the detector for 83 of the 84 derived symbols.

    ``resolve_symbol`` imports the longest importable prefix and walks the rest
    with ``getattr``; if that walk stops happening, every symbol resolves, and
    the floor job passes at every jax version ever released.
    """
    with pytest.raises(AttributeError, match="a_symbol_that_never_existed"):
        resolve_symbol("jax.a_symbol_that_never_existed")


def test_resolve_symbol_reports_an_unimportable_root() -> None:
    """Nothing under ``not_a_package_xyz`` is importable, so no prefix resolves."""
    with pytest.raises(AttributeError, match="cannot import any prefix"):
        resolve_symbol("not_a_package_xyz.thing")


def test_check_symbols_reports_a_missing_symbol(tmp_path: Path) -> None:
    """A scanned tree naming a symbol jax lacks must raise, listing it.

    ``check_symbols`` takes a ``root``, so this needs no second jax: the tree is
    fabricated, the installed jax is whatever pixi resolved, and the symbol is
    one no jax will ever have.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "m.py").write_text("import jax\njax.not_a_real_symbol_xyz\n")
    with pytest.raises(ProbeError, match="not_a_real_symbol_xyz"):
        check_symbols(tmp_path)


def test_check_symbols_lists_every_missing_symbol_not_just_the_first(tmp_path: Path) -> None:
    """One CI round trip per missing symbol would be the wrong shape at a floor bump."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "m.py").write_text(
        "import jax\njax.absent_one_xyz\njax.absent_two_xyz\njax.jit\n"
    )
    with pytest.raises(ProbeError) as caught:
        check_symbols(tmp_path)
    message = str(caught.value)
    assert "absent_one_xyz" in message and "absent_two_xyz" in message
    # 4 scanned: the two absent names, `jax.jit`, and bare `jax` from the import.
    assert message.startswith("2 of 4 jax symbols")


def test_check_symbols_catches_the_issue_21_symbol_going_missing(
    jax_without_typeof: None,
) -> None:
    """Issue #21's bug, on the real repository tree, seen as a failure.

    Not a fixture tree: this is ``src/`` and ``tests/`` as they are, with the
    one symbol jax 0.5.0 lacked removed from the installed module. The message
    must name it, because that message is the entire product of the CI job.
    """
    with pytest.raises(ProbeError, match=r"jax\.typeof") as caught:
        check_symbols()
    assert "1 of " in str(caught.value)


def test_primitive_pattern_catches_the_issue_21_symbol_going_missing(
    jax_without_typeof: None,
) -> None:
    """The other half of #21: the miniature has to fail too, independently.

    The two checks are meant to be independent -- symbol availability says the
    name resolves, the pattern says it behaves -- and at jax 0.5.0 both fired.

    This pins only that *some* line of the miniature still spells
    ``jax.typeof``; there are two that must, and which they are is
    ``test_primitive_pattern_spells_jax_typeof_in_both_rules``'s job.
    """
    with pytest.raises(AttributeError, match="no attribute 'typeof'"):
        check_primitive_pattern()


def test_primitive_pattern_spells_jax_typeof_in_both_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``jax.typeof`` must be reached from the eager impl *and* from the jvp rule.

    ``wgridder.py`` spells ``jax.typeof`` in exactly two places: the eager
    ``def_impl``, which is the line jax 0.5.0 raised on, and the ``ad.Zero``
    branch of the jvp rule. A miniature that reached an aval some other way in
    either of them would pass a floor the library fails, and no assertion on
    *results* can see the difference -- the numbers come out the same. So this
    watches the call itself, by the name of the probe function it came from.
    Nothing weaker distinguishes the two: deleting ``jax.typeof`` outright still
    fails the miniature as long as *one* of the two sites survives.

    The ``ad.Zero`` half is the reason ``battery`` invokes the jvp rule by hand.
    No transform above it reaches that branch -- JAX eliminates the primitive
    before a symbolically-zero tangent arrives -- so left alone the line runs
    zero times, and pinning it here is what keeps it running.
    """
    probe_file = jax_floor_probe.__file__
    callers: set[str] = set()
    real_typeof = jax.typeof

    def spy(value: object) -> object:
        # The *innermost* probe function, not every probe frame on the stack:
        # the eager impl runs underneath the jvp rule during a jvp trace, so
        # crediting the whole chain would report ``_jvp`` for a call the eager
        # impl made and the ad.Zero branch could go dead unnoticed. Synthetic
        # frames are skipped because the eager call sits in a genexpr.
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename == probe_file and not frame.f_code.co_name.startswith("<"):
                callers.add(frame.f_code.co_name)
                break
            frame = frame.f_back
        return real_typeof(value)

    monkeypatch.setattr(jax, "typeof", spy)
    check_primitive_pattern()
    assert {"_eager", "_jvp"} <= callers, sorted(callers)


def test_miniature_exercises_every_method_surface() -> None:
    """Every method the sources call on a jax return value must actually run here.

    ``check_symbols`` cannot see these -- a ``Call`` ends an attribute chain, and
    no static scan knows what a call returns -- so the miniature calling them is
    the only coverage they get, and a hand-kept list of which ones would rot
    exactly like a hand-kept symbol list. The set is derived instead, and this
    asserts on *executed* lines rather than on the source text: a surface whose
    line is present but unreachable is the state ``ad.Zero`` was in, probed by
    nothing while looking probed.
    """
    surfaces = jax_return_value_methods()
    assert surfaces, "the method-surface scan found nothing; it has stopped working"
    probe_file = jax_floor_probe.__file__
    lines = Path(probe_file).read_text().splitlines()
    executed: set[int] = set()

    def tracer(frame: FrameType, event: str, arg: object) -> object:
        if frame.f_code.co_filename != probe_file:
            return None
        if event == "line":
            executed.add(frame.f_lineno)
        return tracer

    # Restore whatever was there rather than clearing: this repository runs no
    # coverage plugin today, and a session that added one should not silently
    # lose its tracer from here on.
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        check_primitive_pattern()
    finally:
        sys.settrace(previous)

    ran = "\n".join(lines[number - 1] for number in sorted(executed))
    unprobed = {name: sorted(sites) for name, sites in surfaces.items() if f".{name}" not in ran}
    assert not unprobed, (
        "src/ and tests/ call these methods on jax return values, and no line the "
        f"miniature actually executed calls them: {unprobed}"
    )


def test_probe_comparator_rejects_a_mismatch() -> None:
    """Every numeric assertion in the miniature funnels through ``close``.

    A comparator that stopped comparing would leave the whole battery running
    and checking nothing.
    """
    close([1.0, 2.0], [1.0, 2.0], "identical")
    with pytest.raises(ProbeError, match="deliberate mismatch"):
        close([1.0, 2.0], [1.0, 2.5], "deliberate mismatch")


def test_check_installed_is_floor_rejects_a_different_version() -> None:
    """pip is happy to resolve something other than the pin, and then the job proves nothing."""
    assert check_installed_is_floor(jax.__version__) == jax.__version__
    with pytest.raises(ProbeError, match="must run at the declared floor"):
        check_installed_is_floor("0.0.1")


# --------------------------------------------------------------------------
# ``main`` -- the entry point CI actually invokes, and the only place the two
# checks are wired to an exit code.
# --------------------------------------------------------------------------


def test_main_print_floor_prints_the_floor_and_exits_zero(capsys: pytest.CaptureFixture) -> None:
    """``--print-floor`` is what the workflow interpolates into ``pip install``.

    It must print the floor and nothing else: the CI step assigns the whole
    stdout to a shell variable.
    """
    assert main(["--print-floor"]) == 0
    assert capsys.readouterr().out == f"{declared_floor()}\n"


def test_main_runs_both_checks_and_reports_each(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The success path end to end, with only the version assertion relaxed.

    ``check_installed_is_floor`` is the one thing that cannot hold here -- the
    pixi environments deliberately track a current jax, not the floor -- so it
    is the one thing stubbed. Both checks then run for real, and each status
    line has to be earned.
    """
    monkeypatch.setattr("tests.jax_floor_probe.check_installed_is_floor", lambda floor: floor)
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "symbol availability:" in out
    assert "primitive pattern:" in out
    assert out.rstrip().endswith(f"jax floor probe: OK at jax {declared_floor()}")


@pytest.mark.parametrize("failing", ["check_symbols", "check_primitive_pattern"])
def test_main_exits_one_when_either_check_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, failing: str
) -> None:
    """Either check failing must reach CI as exit 1 and a ``FAIL:`` line.

    Parametrized over both because ``main`` calling only one of them is a
    single-line edit that nothing else here would notice -- and a probe that
    skips the primitive pattern still prints the pattern's success line.
    """

    def boom(*args: object, **kwargs: object) -> None:
        raise ProbeError(f"deliberate failure in {failing}")

    monkeypatch.setattr("tests.jax_floor_probe.check_installed_is_floor", lambda floor: floor)
    monkeypatch.setattr(f"tests.jax_floor_probe.{failing}", boom)
    assert main([]) == 1
    captured = capsys.readouterr()
    assert f"FAIL: deliberate failure in {failing}" in captured.err
    assert f"deliberate failure in {failing}" not in captured.out


def test_main_exits_one_and_shows_frames_for_a_non_probe_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A jax-side failure is the outcome this job exists to diagnose.

    At jax 0.5.0 the primitive pattern raised a bare ``AttributeError``, and the
    whole CI output was one line naming neither file nor line. The traceback is
    the diagnosis; the summary line stays so the log's last words still read.
    """
    monkeypatch.setattr("tests.jax_floor_probe.check_installed_is_floor", lambda floor: floor)
    monkeypatch.setattr(
        "tests.jax_floor_probe.check_symbols",
        lambda *a, **k: (_ for _ in ()).throw(AttributeError("no attribute 'typeof'")),
    )
    assert main([]) == 1
    err = capsys.readouterr().err
    assert "FAIL: AttributeError: no attribute 'typeof'" in err
    assert "Traceback (most recent call last)" in err
    assert "jax_floor_probe.py" in err


def test_main_exits_one_when_the_floor_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An unreadable floor must fail, not pin an empty version.

    ``declared_floor`` runs before the ``--print-floor`` branch, so this is what
    stops the workflow's ``pip install "jax=="`` from ever being written: exit 1
    and an empty stdout, which under ``set -euo pipefail`` fails the step.
    """

    def boom(*args: object, **kwargs: object) -> None:
        raise ProbeError("deliberate: no 'jax' entry")

    monkeypatch.setattr("tests.jax_floor_probe.declared_floor", boom)
    assert main(["--print-floor"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "FAIL: deliberate: no 'jax' entry" in captured.err
