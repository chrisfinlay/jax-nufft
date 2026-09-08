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
    Every ``jax`` symbol ``src/`` and ``tests/`` touch is imported and
    ``getattr``-ed. The list is **derived** by walking the AST of every module
    -- it is never maintained by hand, because a hand-maintained list is the
    same failure mode as a hand-maintained version floor.

``check_primitive_pattern``
    A miniature of ``wgridder.py``'s three linear primitives on a 3x3 matmul:
    a ``batch_shape`` param, ``mlir.lower_fun`` lowering, ``ad.Zero(jax.typeof
    (out).to_tangent_aval())``, two mutually-transposing rules, a batching rule
    that pushes the mapped axis into the params, and the eager ``def_impl`` --
    driven through ``jit``, ``grad``, ``jvp``, ``linear_transpose``, nested
    ``vmap`` inside ``grad``, ``grad(grad(...))`` and ``disable_jit``. Symbol
    availability says the name resolves; this says the pattern behaves.

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
from pathlib import Path

import tomllib

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
# not match.
_JAX_REQUIREMENT = re.compile(r"^jax(?![\w.-])\s*(?:\[[^\]]*\])?\s*(?P<spec>[^;]*)")
_LOWER_BOUND = re.compile(r">=\s*(?P<version>[0-9][^,\s]*)")


def declared_floor(pyproject: Path | None = None) -> str:
    """Return the ``>=`` lower bound the project declares for ``jax``.

    Read rather than hard-coded, so this probe follows the declaration
    automatically. A probe carrying its own copy of the version could go stale
    against ``pyproject.toml``, which is precisely the bug being fixed.
    """
    path = pyproject or (REPO_ROOT / "pyproject.toml")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    requirements = data.get("project", {}).get("dependencies", [])
    for requirement in requirements:
        match = _JAX_REQUIREMENT.match(requirement.strip())
        if match is None:
            continue
        bound = _LOWER_BOUND.search(match.group("spec"))
        if bound is None:
            raise ProbeError(
                f"{path} declares {requirement!r} with no '>=' lower bound; "
                "there is no floor for this probe to exercise."
            )
        return bound.group("version")
    raise ProbeError(f"{path} has no 'jax' entry in [project] dependencies.")


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
    rtol = 1e-5  # float32-safe: this runs without the suite's x64 switch

    def close(got, want, what):
        if not bool(jnp.allclose(jnp.asarray(got), jnp.asarray(want), rtol=rtol, atol=1e-5)):
            raise ProbeError(f"primitive pattern: {what} gave {got!r}, expected {want!r}")

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

    battery("jit enabled")
    with jax.disable_jit():
        battery("disable_jit")


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print-floor",
        action="store_true",
        help="print the jax version pyproject.toml declares and exit (used by CI to pin pip)",
    )
    args = parser.parse_args(argv)

    floor = declared_floor()
    if args.print_floor:
        print(floor)
        return 0

    def say(message: str) -> None:
        # Flushed, so the ordered narrative survives being interleaved with the
        # stderr failure line in a CI log.
        print(message, flush=True)

    try:
        installed = check_installed_is_floor(floor)
        say(f"jax {installed} == declared floor (pyproject.toml: jax>={floor})")
        symbols = check_symbols()
        say(f"symbol availability: {len(symbols)} derived jax symbols from src/ + tests/, present")
        check_primitive_pattern()
        say(
            "primitive pattern: jit / grad / jvp / linear_transpose / vmap-in-grad / "
            "grad(grad) / disable_jit all agree with the dense 3x3"
        )
    except ProbeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    say(f"jax floor probe: OK at jax {floor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
