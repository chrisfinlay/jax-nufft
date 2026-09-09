"""Exercise the ``jax`` floor that ``pyproject.toml`` declares (issue #53).

Nothing else in the repository runs the declared minimum. Every test, on both
precision legs, on CPU and on a GH200, across the whole Python matrix, runs the
single ``jax`` the pixi lockfile resolves. So ``pyproject.toml`` saying
``jax>=0.5.0`` while ``wgridder.py`` called ``jax.typeof`` -- exported in 0.6.0,
so every operator under :func:`jax.disable_jit` raised ``AttributeError`` on the
declared minimum -- was invisible until someone read both files side by side
(found on issue #21, fixed there by raising the floor). This module is what
reads both files, mechanically, on every push.

It is deliberately standalone: **stdlib plus ``jax`` only**, no ``jax_nufft``,
no ``jax_finufft``, no ``pytest``, no ``numpy`` beyond what ``jax`` itself
pulls in. That is what lets CI ``pip install jax==<floor>`` and be done in a few
seconds. Running the *whole suite* at the floor instead would need a conda-forge
``jax-finufft`` that solves against a jax that old on ``linux-64`` /
``linux-aarch64`` / ``osx-arm64``; issue #53 puts that out of scope, and it has
not been attempted -- hence a probe rather than an environment.

Two checks, in order:

``check_symbols``
    Every module-level ``jax.*`` attribute chain ``src/`` and ``tests/`` touch
    is imported and ``getattr``-ed. The list is **derived** by walking the AST
    of every module -- it is never maintained by hand, because a
    hand-maintained list is the same failure mode as a hand-maintained version
    floor.

    "Module-level attribute chain" is the exact scope, and it is narrower than
    "every ``jax`` symbol the repository uses": a ``Call`` terminates a chain,
    so methods of a *returned* object -- ``jax.typeof(...).to_tangent_aval()``,
    ``jax.jit(...).lower()``, ``jnp.zeros(...).at[...]``, ``.astype``,
    ``.real``, ``.reshape`` -- are not in the derived set and cannot be. A
    static scan does not know the type a call returns, and resolving those
    names would mean calling arbitrary repository code inside the probe. They
    are covered by ``check_primitive_pattern`` instead, which calls them for
    real; see the list there.

``check_primitive_pattern``
    A miniature of ``wgridder.py``'s three linear primitives on a 3x3 matmul:
    a ``batch_shape`` param, ``mlir.lower_fun`` lowering, ``ad.Zero(jax.typeof
    (out).to_tangent_aval())``, two mutually-transposing rules, a batching rule
    that pushes the mapped axis into the params, and the eager ``def_impl`` --
    driven through ``jit``, ``grad``, ``jvp``, ``linear_transpose``, nested
    ``vmap`` inside ``grad``, ``grad(grad(...))`` and ``disable_jit``. Symbol
    availability says the name resolves; this says the pattern behaves.

    It is also where the method surfaces above get exercised, since the scan
    cannot see them: ``.to_tangent_aval()`` (the jvp rule is invoked directly
    with an ``ad.Zero`` tangent, because no transform below reaches that branch
    on its own), ``jax.jit(...).lower()``, ``.at[...].set()``, ``.astype()``,
    ``.real`` and ``.reshape()`` -- the six method names ``src/`` and
    ``tests/`` call on jax return values, across nine distinct call surfaces.

``jax.core.get_aval`` is deprecated in recent jax and warns, which under this
repository's ``filterwarnings = ["error"]`` is a failure; ``jax.typeof(...)
.to_tangent_aval()`` is the replacement and is what ``wgridder.py`` uses, so it
is what the miniature uses too. Keeping the two on the same idiom is the point:
a probe on a different idiom would pass a floor the library fails.

Run it directly (this is what CI does)::

    python -m pip install "jax==$(python tests/jax_floor_probe.py --print-floor)"
    python tests/jax_floor_probe.py

``tests/test_jax_floor.py`` runs the same two checks against whatever jax the
pixi environment resolved, so the probe itself cannot rot between releases.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import re
import sys
import tomllib
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directories scanned for ``jax`` usage. ``docs/`` is prose and ``tools/``
#: does not exist; these two are the shipped package and its tests.
SCAN_DIRS = ("src", "tests")


class ProbeError(AssertionError):
    """A floor check failed. Carries a message CI can read without context."""


# --------------------------------------------------------------------------
# 1. The declared floor, read out of pyproject.toml
# --------------------------------------------------------------------------

# PEP 508: the name runs until the first extras bracket, comparison operator,
# marker semicolon or whitespace. Anchored and negative-lookahead'd so that
# ``jax-finufft>=1.3.0`` -- the other requirement starting with "jax" -- does
# not match. ``spec`` stops at the marker semicolon and ``marker`` takes the
# rest, so a marker is *seen* rather than silently swallowed (see below).
_JAX_REQUIREMENT = re.compile(
    r"^jax(?![\w.-])\s*(?:\[[^\]]*\])?\s*(?P<spec>[^;]*)(?P<marker>;.*)?$"
)
# One clause of a comma-separated version specifier. ``===`` and the two-char
# operators come first so the alternation cannot bite off just ``=`` or ``<``.
_CLAUSE = re.compile(r"^(?P<op>===|==|!=|~=|<=|>=|<|>)\s*(?P<version>\S+)$")
# Release segments only. Anything with an epoch, a wildcard, a pre/post/dev
# suffix or a local version fails to match, and the caller then refuses to
# reason about the constraint rather than guessing at PEP 440 ordering.
_NUMERIC_RELEASE = re.compile(r"[0-9]+(?:\.[0-9]+)*")


def _release(version: str) -> tuple[int, ...] | None:
    """``(0, 6, 0)`` for a plain dotted release, else ``None`` for "cannot compare".

    Deliberately not a PEP 440 implementation. ``packaging`` is not importable
    here -- the module is stdlib-plus-jax so CI can read the floor before
    installing anything -- and a half-right ordering that silently mis-ranks
    ``0.6.0rc1`` would be worse than no ordering at all. ``None`` makes the
    caller fail closed.
    """
    if _NUMERIC_RELEASE.fullmatch(version) is None:
        return None
    return tuple(int(part) for part in version.split("."))


def _satisfies(floor: str, op: str, version: str) -> bool | None:
    """Does ``floor`` satisfy the clause ``op version``? ``None`` if uncomparable.

    Zero-padded release comparison, which is what PEP 440 specifies for the
    release segment: ``0.6`` and ``0.6.0`` are the same version, so ``!=0.6``
    does exclude a floor written ``0.6.0``.
    """
    left, right = _release(floor), _release(version)
    if left is None or right is None:
        return None
    width = max(len(left), len(right))
    left += (0,) * (width - len(left))
    right += (0,) * (width - len(right))
    return {
        ">=": left >= right,
        ">": left > right,
        "<=": left <= right,
        "<": left < right,
        "!=": left != right,
    }[op]


def _floor_from_spec(path: Path, requirement: str, spec: str) -> str:
    """The single ``>=`` bound of ``spec``, or a ``ProbeError`` saying why not.

    The rule is *fail closed*: the probe returns a floor only when it has
    understood every clause and checked that the floor it is about to hand to
    ``pip install jax==<floor>`` actually satisfies all of them. The job
    installs that pin **alone**, with none of the project's other requirements
    to correct it, so a floor read out of half a specifier is a green tick on a
    version the project itself excludes -- ``jax>=0.6.0,!=0.6.0`` being the
    sharp case, and ``jax>=0.6.0,>=0.9.0`` (the real floor is 0.9.0) the easy
    one to write by accident when raising a bound.
    """
    clauses: list[tuple[str, str]] = []
    for raw in spec.split(","):
        clause = raw.strip()
        if not clause:
            continue
        parsed = _CLAUSE.match(clause)
        if parsed is None:
            raise ProbeError(
                f"{path} declares {requirement!r}, whose clause {clause!r} is not a "
                "comparison this probe can parse. It will not guess at a floor from a "
                "specifier it has not understood. Write 'jax>=X.Y.Z' or teach "
                "declared_floor()."
            )
        clauses.append((parsed.group("op"), parsed.group("version")))

    lower = [version for op, version in clauses if op == ">="]
    if not lower:
        raise ProbeError(
            f"{path} declares {requirement!r} with no '>=' lower bound; "
            "there is no floor for this probe to exercise. Only '>=' is read: "
            "the job installs 'jax==<floor>' and asserts that is what ran, which "
            "'~=' and '==' do not state. Write 'jax>=X.Y.Z' or teach declared_floor()."
        )
    if len(lower) > 1:
        raise ProbeError(
            f"{path} declares {requirement!r} with {len(lower)} '>=' lower bounds "
            f"({', '.join(lower)}). The effective floor is the highest of them, and a "
            "probe that took the first would install and then claim to have exercised a "
            "version the project does not even allow. Declare one '>=' bound."
        )
    floor = lower[0]

    # A single bare '>=' is the whole specifier: nothing to intersect, and the
    # version string goes through untouched (a pre-release floor stays legal).
    if len(clauses) == 1:
        return floor

    for op, version in clauses:
        if op in {"==", "~=", "==="}:
            raise ProbeError(
                f"{path} declares {requirement!r}, combining '>=' with {op}{version}. "
                f"A '>=' bound says the job may install {floor}; {op}{version} says it may "
                "not, and this probe will not pick a winner -- that is how a pin the "
                "project excludes gets a green tick. Declare a single 'jax>=X.Y.Z'."
            )
        ok = _satisfies(floor, op, version)
        if ok is None:
            raise ProbeError(
                f"{path} declares {requirement!r}, and this probe cannot order "
                f"{floor!r} against {version!r} without 'packaging'. It is stdlib-only "
                "by design, so it refuses the requirement rather than shipping a floor "
                "it has not checked. Declare a single 'jax>=X.Y.Z'."
            )
        if not ok:
            raise ProbeError(
                f"{path} declares {requirement!r}, whose '>=' bound {floor} does not "
                f"satisfy its own {op}{version} clause. The job installs 'jax=={floor}' "
                "alone -- none of the project's other requirements are there to correct "
                "it -- so it would pass on a version this requirement excludes."
            )
    return floor


def declared_floor(pyproject: Path | None = None) -> str:
    """Return the ``>=`` lower bound the project declares for ``jax``.

    Read rather than hard-coded, so this probe follows the declaration
    automatically. A probe carrying its own copy of the version could go stale
    against ``pyproject.toml``, which is precisely the bug being fixed.

    Only ``>=`` is accepted, and everything else raises. ``jax~=0.6.0`` and
    ``jax==0.6.0`` do each imply a minimum, so rejecting them is stricter than
    PEP 508 requires -- deliberately. This probe installs ``jax==<floor>`` and
    asserts the installed version *is* that floor, which is a statement about a
    ``>=`` bound and not about a compatible-release or pinned one; guessing a
    floor out of ``~=`` would make the job's claim about what it exercised
    quietly wrong. The failure is loud (non-zero exit, named requirement), so
    it cannot hide a bad floor; a maintainer who wants ``~=`` has to teach this
    function what the job should then install.

    An environment marker is rejected for the same reason. Evaluating one needs
    ``packaging.markers``, and this module is stdlib-plus-jax by design so that
    CI can run it before installing anything; silently ignoring the marker
    would let a ``jax>=…; sys_platform == "win32"`` entry set the floor for a
    Linux job that the requirement does not even apply to.

    Everything past the ``>=`` is refused rather than ignored, for the same
    reason and by the same rule -- see :func:`_floor_from_spec`. A second
    ``jax`` entry is refused too: two requirements intersect exactly like two
    clauses of one, and taking the first is the same silent guess.
    """
    path = pyproject or (REPO_ROOT / "pyproject.toml")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    requirements = data.get("project", {}).get("dependencies", [])
    matched: list[tuple[str, re.Match[str]]] = []
    for requirement in requirements:
        match = _JAX_REQUIREMENT.match(requirement.strip())
        if match is None:
            continue
        if match.group("marker"):
            raise ProbeError(
                f"{path} declares {requirement!r} with an environment marker. This probe "
                "is stdlib-only and cannot evaluate markers, so it will not guess whether "
                "the requirement applies to the job's platform. Declare 'jax' unconditionally."
            )
        matched.append((requirement, match))
    if not matched:
        raise ProbeError(f"{path} has no 'jax' entry in [project] dependencies.")
    if len(matched) > 1:
        listed = ", ".join(repr(requirement) for requirement, _ in matched)
        raise ProbeError(
            f"{path} declares 'jax' {len(matched)} times ({listed}). The effective "
            "requirement is their intersection; reading the first would let the job "
            "install a version another entry forbids. Declare 'jax' once."
        )
    requirement, match = matched[0]
    return _floor_from_spec(path, requirement, match.group("spec"))


def check_installed_is_floor(floor: str) -> str:
    """Fail unless the installed ``jax`` is exactly the declared floor.

    The job is worthless if the environment quietly resolved something newer,
    and pip is perfectly happy to do that. Only the standalone run asserts
    this; ``tests/test_jax_floor.py`` calls the two checks directly, since the
    pixi environments deliberately track a current jax.
    """
    import jax

    installed = jax.__version__
    if installed != floor:
        raise ProbeError(
            f"probe must run at the declared floor: pyproject.toml says jax>={floor} "
            f"but jax {installed} is installed. Install 'jax=={floor}' exactly."
        )
    return installed


# --------------------------------------------------------------------------
# 2. Symbol availability, from a derived list
# --------------------------------------------------------------------------


def _python_files(root: Path) -> list[Path]:
    """Every ``.py`` under ``root``, sorted so failure output is deterministic."""
    return sorted(root.rglob("*.py"))


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a pure Name/Attribute chain, else ``None``.

    Returning ``None`` for anything else is what keeps ``jnp.array(x).shape``
    out of the results: the intervening ``Call`` breaks the chain, so the
    recorded symbol is ``jax.numpy.array`` and the instance attribute is not
    mistaken for a module one.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _jax_aliases(tree: ast.Module) -> dict[str, str]:
    """Map each local name bound to something under ``jax`` to its dotted path.

    Covers the four binding forms the codebase actually uses: ``import jax``,
    ``import jax.numpy as jnp``, ``import jax.extend as jex`` and
    ``from jax.interpreters import ad, batching, mlir``.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "jax" and not alias.name.startswith("jax."):
                    continue
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    # ``import jax.numpy`` binds the top-level name ``jax``.
                    aliases["jax"] = "jax"
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if node.level or not module or (module != "jax" and not module.startswith("jax.")):
                continue
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
    return aliases


def jax_symbols(root: Path | None = None, scan_dirs: tuple[str, ...] = SCAN_DIRS) -> set[str]:
    """Every ``jax.*`` symbol reachable from the scanned sources, by AST.

    AST rather than regex because the answer depends on the *binding* of the
    root name -- ``jnp`` means ``jax.numpy`` in one file and would mean nothing
    in another, ``ad`` is ``jax.interpreters.ad`` only because of a
    ``from``-import several hundred lines earlier, and a regex for ``jax\\.``
    misses every one of those. It also has to know that a ``Call`` ends an
    attribute chain, which is a parse, not a pattern.

    Two sources of symbols:

    * every name a ``from jax... import`` binds, which is a symbol outright;
    * every maximal attribute chain rooted at a jax-bound name.
    """
    base = root or REPO_ROOT
    symbols: set[str] = set()
    for scan_dir in scan_dirs:
        for path in _python_files(base / scan_dir):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            aliases = _jax_aliases(tree)
            if not aliases:
                continue
            symbols.update(aliases.values())
            # A chain is maximal when it is not itself the ``.value`` of an
            # enclosing Attribute; recording inner nodes too would only add
            # prefixes of symbols already recorded.
            inner = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute) or id(node) in inner:
                    continue
                dotted = _dotted(node)
                if dotted is None:
                    continue
                head, _, tail = dotted.partition(".")
                if head in aliases:
                    symbols.add(f"{aliases[head]}.{tail}" if tail else aliases[head])
    return {symbol for symbol in symbols if symbol == "jax" or symbol.startswith("jax.")}


def jax_return_value_methods(
    root: Path | None = None, scan_dirs: tuple[str, ...] = SCAN_DIRS
) -> dict[str, set[str]]:
    """Methods the sources call on the *return value* of a ``jax`` call.

    The complement of :func:`jax_symbols`, and the reason that function's
    docstring is careful to say "module attribute". A ``Call`` terminates an
    attribute chain, so ``jax.typeof(x).to_tangent_aval()`` records
    ``jax.typeof`` and drops ``to_tangent_aval`` -- correctly, because the
    scanner has no way to know what type the call returned, and resolving the
    method would mean executing repository code inside the probe.

    Those methods are still jax API, they still move between releases (``.at``,
    ``.lower()`` and ``.to_tangent_aval()`` especially), and something has to
    watch them. They cannot be watched by ``getattr``, so they are watched by
    :func:`check_primitive_pattern` calling them for real -- and this function
    is what says *which*, so that list is derived rather than hand-kept and a
    newly used method surface cannot slip in unprobed. See
    ``tests/test_jax_floor.py::test_miniature_exercises_every_method_surface``.

    Returns ``{method name: {"jax.typeof(...).to_tangent_aval", ...}}`` -- the
    name to probe, and the call surfaces that motivate it.

    One level only: in ``jax.jit(f).lower(x).compile()`` the ``.lower`` is
    recorded and ``.compile`` is not, because the receiver of ``.compile`` is
    itself a call whose type is just as unknowable. The miniature runs that
    whole chain anyway; it is only the *enumeration* that stops at one.
    """
    base = root or REPO_ROOT
    surfaces: dict[str, set[str]] = {}
    for scan_dir in scan_dirs:
        for path in _python_files(base / scan_dir):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            aliases = _jax_aliases(tree)
            if not aliases:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Call):
                    continue
                dotted = _dotted(node.value.func)
                if dotted is None:
                    continue
                head, _, tail = dotted.partition(".")
                if head not in aliases:
                    continue
                called = f"{aliases[head]}.{tail}" if tail else aliases[head]
                if called == "jax" or called.startswith("jax."):
                    surfaces.setdefault(node.attr, set()).add(f"{called}(...).{node.attr}")
    return surfaces


def resolve_symbol(symbol: str) -> object:
    """Import-and-``getattr`` one dotted symbol, raising ``AttributeError`` if absent.

    Submodules are not always attributes of their parent until imported, so the
    longest importable prefix is imported first and the remainder walked with
    ``getattr``.
    """
    parts = symbol.split(".")
    obj = None
    imported = 0
    for stop in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:stop]))
        except ImportError:
            continue
        imported = stop
        break
    if obj is None:
        raise AttributeError(f"cannot import any prefix of {symbol!r}")
    for part in parts[imported:]:
        obj = getattr(obj, part)
    return obj


def check_symbols(root: Path | None = None) -> list[str]:
    """Resolve every derived symbol; raise listing all failures, not just the first.

    All of them, because the useful output when a floor is too low is the whole
    set of names that floor lacks -- one per run would mean one CI round trip
    per missing symbol.
    """
    symbols = sorted(jax_symbols(root))
    missing = []
    for symbol in symbols:
        try:
            resolve_symbol(symbol)
        except (AttributeError, ImportError) as exc:
            missing.append(f"  {symbol}: {type(exc).__name__}: {exc}")
    if missing:
        import jax

        raise ProbeError(
            f"{len(missing)} of {len(symbols)} jax symbols used by src/ and tests/ are "
            f"missing from jax {jax.__version__}:\n" + "\n".join(missing)
        )
    return symbols


# --------------------------------------------------------------------------
# 3. A miniature of the wgridder's primitive pattern
# --------------------------------------------------------------------------


def close(got: object, want: object, what: str) -> None:
    """Raise :class:`ProbeError` unless ``got`` matches ``want`` elementwise.

    Module level rather than nested inside :func:`check_primitive_pattern` for
    one reason: it is the single comparison every numeric assertion in the
    miniature funnels through, so a comparator that silently stops comparing
    kills the whole check while leaving it green. As a nested closure no test
    could reach its failure path; here one can, and does
    (``test_probe_comparator_rejects_a_mismatch``).
    """
    import jax.numpy as jnp

    # float32-safe: the standalone probe runs without the suite's x64 switch.
    if not bool(jnp.allclose(jnp.asarray(got), jnp.asarray(want), rtol=1e-5, atol=1e-5)):
        raise ProbeError(f"primitive pattern: {what} gave {got!r}, expected {want!r}")


def check_primitive_pattern() -> None:
    """Build and drive a 3x3 matmul copy of ``wgridder.py``'s primitive pattern.

    Deliberately built inside the function rather than at import time: a
    ``Primitive`` registers itself globally by name, so constructing these at
    module scope would make importing the probe twice (script and test module)
    a duplicate-registration hazard, and would run the pattern's construction
    even for ``--print-floor``.
    """
    from functools import partial

    import jax
    import jax.extend as jex
    import jax.numpy as jnp
    from jax.interpreters import ad, batching, mlir

    matrix = jnp.asarray([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0], [1.0, 0.25, -2.0]])
    fwd_p = jex.core.Primitive("jax_floor_probe_fwd")  # x -> M x
    bwd_p = jex.core.Primitive("jax_floor_probe_bwd")  # y -> M^T y

    def _batched(core, x, batch_shape):  # the wgridder's ``_apply_batched``
        if not batch_shape:
            return core(x)
        n_batch = 1
        for dim in batch_shape:
            n_batch *= dim
        out = jax.vmap(core)(x.reshape((n_batch, *x.shape[len(batch_shape) :])))
        return out.reshape((*batch_shape, *out.shape[1:]))

    def _fwd_lowering(x, *, batch_shape):
        return _batched(lambda v: matrix @ v, x, batch_shape)

    def _bwd_lowering(y, *, batch_shape):
        return _batched(lambda v: matrix.T @ v, y, batch_shape)

    def _abstract(x, *, batch_shape):
        return jax.core.ShapedArray((*batch_shape, 3), x.dtype)

    def _eager(lowering, *operands, **params):  # the wgridder's ``def_impl``
        # The ``jax.typeof`` here is the exact call that made jax 0.5.0 fail:
        # every bind outside a trace lands on this line.
        _abstract(*(jax.typeof(operand) for operand in operands), **params)
        return jax.jit(lambda *a: lowering(*a, **params))(*operands)

    def _jvp(prim, primals, tangents, **params):
        # The ``ad.Zero`` branch is defensive here exactly as it is in
        # ``wgridder.py``: no transform below reaches it (JAX dead-code-
        # eliminates the primitive first), but it is part of the rule contract
        # and it is the second place the library spells ``jax.typeof``.
        (x,), (tangent,) = primals, tangents
        out = prim.bind(x, **params)
        if isinstance(tangent, ad.Zero):
            return out, ad.Zero(jax.typeof(out).to_tangent_aval())
        return out, prim.bind(tangent, **params)

    def _transpose(other, ct, x, **params):  # fwd^T is bwd, bwd^T is fwd
        assert ad.is_undefined_primal(x)
        if isinstance(ct, ad.Zero):
            return [None]
        return [other.bind(ct, **params).astype(x.aval.dtype)]

    def _batcher(prim, args, dims, **params):
        (x,), (batch_dim,) = args, dims
        xs = jnp.moveaxis(x, batch_dim, 0)
        params = {**params, "batch_shape": (xs.shape[0], *params["batch_shape"])}
        return prim.bind(xs, **params), 0

    for prim, lowering, other in ((fwd_p, _fwd_lowering, bwd_p), (bwd_p, _bwd_lowering, fwd_p)):
        prim.def_abstract_eval(_abstract)
        prim.def_impl(partial(_eager, lowering))
        mlir.register_lowering(prim, mlir.lower_fun(lowering, multiple_results=False))
        ad.primitive_jvps[prim] = partial(_jvp, prim)
        ad.primitive_transposes[prim] = partial(_transpose, other)
        batching.primitive_batchers[prim] = partial(_batcher, prim)

    fwd = partial(fwd_p.bind, batch_shape=())
    x0 = jnp.asarray([1.0, -2.0, 3.0])
    ct = jnp.asarray([0.5, 1.5, -1.0])
    want_fwd = matrix @ x0
    want_bwd = matrix.T @ ct

    def battery(label):
        close(jax.jit(fwd)(x0), want_fwd, f"{label}: jit")
        close(fwd(x0), want_fwd, f"{label}: eager bind (def_impl)")
        close(bwd_p.bind(ct, batch_shape=()), want_bwd, f"{label}: eager bind, transpose primitive")
        close(jax.grad(lambda x: jnp.vdot(fwd(x), ct))(x0), want_bwd, f"{label}: grad")
        close(jax.jvp(fwd, (x0,), (x0,))[1], want_fwd, f"{label}: jvp")
        close(jax.linear_transpose(fwd, x0)(ct)[0], want_bwd, f"{label}: linear_transpose")
        # vmap nested inside grad: the batcher pushes both axes into the param.
        stack = jnp.stack([jnp.stack([x0, ct]), jnp.stack([ct, x0])])  # (2, 2, 3)
        nested = jax.grad(lambda xs: jnp.sum(jax.vmap(jax.vmap(fwd))(xs)))(stack)
        close(nested, jnp.broadcast_to(matrix.sum(axis=0), stack.shape), f"{label}: vmap in grad")

        # grad(grad(...)): the second reverse pass transposes the transpose.
        def scalar(t):  # t^2 * ||M x0||^2, so the second derivative is constant
            return jnp.sum(fwd(t * x0) ** 2)

        close(jax.grad(jax.grad(scalar))(1.0), 2.0 * jnp.sum(want_fwd**2), f"{label}: grad(grad)")

        # The ``ad.Zero`` branch of the jvp rule, driven by hand. No transform
        # above reaches it -- JAX dead-code-eliminates the primitive before a
        # symbolically-zero tangent can arrive -- so left to the battery alone
        # the branch runs zero times, and with it the only *executed* use of
        # ``jax.typeof(...).to_tangent_aval()``, which the AST scan cannot see
        # either (a ``Call`` terminates a chain). Invoking the registered rule
        # directly is what makes both true statements: the line runs, and the
        # aval it builds is the one the contract requires.
        zero_in = ad.Zero(jax.typeof(x0).to_tangent_aval())
        out, tangent_out = ad.primitive_jvps[fwd_p]((x0,), (zero_in,), batch_shape=())
        close(out, want_fwd, f"{label}: jvp rule, ad.Zero tangent (primal)")
        if not isinstance(tangent_out, ad.Zero):
            raise ProbeError(
                f"primitive pattern: {label}: jvp rule given an ad.Zero tangent returned "
                f"{type(tangent_out).__name__}, expected ad.Zero -- the branch that spells "
                "jax.typeof(out).to_tangent_aval() did not run."
            )
        close(tangent_out.aval.shape, (3,), f"{label}: ad.Zero tangent aval shape")

        # Methods called on jax return values, which check_symbols cannot
        # reach: these are every such surface src/ and tests/ use.
        close(jnp.zeros(3).at[1].set(2.0), jnp.asarray([0.0, 2.0, 0.0]), f"{label}: .at[].set()")
        close(
            jnp.zeros_like(x0).at[0].add(1.0).astype(jnp.float32).real.reshape((3,)),
            jnp.asarray([1.0, 0.0, 0.0]),
            f"{label}: .at[].add() / .astype() / .real / .reshape()",
        )

    battery("jit enabled")
    with jax.disable_jit():
        battery("disable_jit")

    # The AOT chain tests/test_chunked_strategy.py:369 uses, in full:
    # jax.jit(fn).lower(*args).compile().memory_analysis(). Every link of it is
    # a method on a return value, so none of it is in the derived set. Outside
    # the battery because jax itself refuses the AOT path under disable_jit
    # ("Disable jit is not supported in the AOT path"), so there is only one
    # leg to run it on. ``memory_analysis`` returns None on backends that do
    # not expose it, which the caller there also tolerates.
    compiled = jax.jit(fwd).lower(x0).compile()
    close(compiled(x0), want_fwd, "AOT: jit(...).lower().compile()")
    compiled.memory_analysis()


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print-floor",
        action="store_true",
        help="print the jax version pyproject.toml declares and exit (used by CI to pin pip)",
    )
    args = parser.parse_args(argv)

    def say(message: str) -> None:
        # Flushed, so the ordered narrative survives being interleaved with the
        # stderr failure line in a CI log.
        print(message, flush=True)

    try:
        # Inside the try, including for --print-floor: an unreadable floor is a
        # ProbeError naming the requirement, and that message is more use to
        # the reader of a CI log than the traceback it would otherwise become.
        # Either way stdout stays empty and the exit code is 1, so the
        # workflow's `version="$(...)"` under `set -euo pipefail` fails the step
        # rather than writing an empty pin.
        floor = declared_floor()
        if args.print_floor:
            print(floor)
            return 0
        installed = check_installed_is_floor(floor)
        say(f"jax {installed} == declared floor (pyproject.toml: jax>={floor})")
        symbols = check_symbols()
        say(f"symbol availability: {len(symbols)} derived jax symbols from src/ + tests/, present")
        check_primitive_pattern()
        say(
            "primitive pattern: jit / grad / jvp / linear_transpose / vmap-in-grad / "
            "grad(grad) / disable_jit / ad.Zero jvp / AOT lower+compile all agree "
            f"with the dense 3x3, and {len(jax_return_value_methods())} method surfaces "
            "the symbol scan cannot see ran"
        )
    except ProbeError as exc:
        # Self-describing by construction -- every ProbeError message names the
        # file, the requirement or the symbol -- so a traceback would only bury
        # it.
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        # Anything else is a *jax* failure at the floor, which is the outcome
        # this job exists to diagnose, and the useful part of it is the frame:
        # jax 0.5.0's `AttributeError: module 'jax' has no attribute 'typeof'`
        # says nothing about which of the two spellings raised until you can
        # see the line. One-line summary first so the log's last words are
        # still readable, then the frames.
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if not (isinstance(exc, ModuleNotFoundError) and exc.name == "jax"):
            # jax not being installed at all needs no frame -- the pip step
            # above it has already gone red and the message says everything.
            # Any other failure is a version incompatibility, and then where it
            # bit is the whole answer.
            traceback.print_exc()
        return 1
    say(f"jax floor probe: OK at jax {floor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
