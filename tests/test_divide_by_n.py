"""``divide_by_n`` on both operators (issue #20): the exact adjoint pair.

Through v0.1.2 the ``1/n`` factor was hard-wired: ``dirty2vis`` never applied
it (ducc's ``divide_by_n=False``) and ``vis2dirty`` always did
(``divide_by_n=True``). That mixed pair is *not* an adjoint pair. The identity
that survives it is ``Re<A x, y> = <n x, A^H y>``, and even that holds only
inside the unit disc: outside it the forward evaluates the analytic extension
``n - 1 = -sqrt(l^2 + m^2 - 1) - 1`` (so ``n < 0``) while the adjoint zeroes
those pixels. On the EDA2 full-sky fixture (64^2 at a 120-degree FoV, 1155 of
4096 pixels outside the disc) that leaves the mixed pair off by a relative
2.7e-1 in the ``n x`` form and 6.3e-1 in the plain form -- measured here, and
reproduced with ducc0 called the same way. Anyone using ``vis2dirty`` as the
gradient of ``dirty2vis`` on a wide field gets that error in the gradient.

Issue #20 exposes the flag on both operators, keyword-only and static, with
the *defaults unchanged* (``False`` forward, ``True`` adjoint). With **equal**
flags the pair is exactly adjoint:

    Re<A x, y> = <x, A^H y>                       (both flags, any field)

measured at 1.3e-15 .. 1.9e-12 over the **single-channel float64** fixtures of
section 2, hence the eps-independent ``1e-11`` bound this module gates that
population at -- the same bound ``tests/test_adjoint.py`` uses for the
reduction-order comparison it makes. Other populations in this module read
differently and are scoped where they are stated: the section-6 two-channel
matrix measures 8.1e-16 .. 6.6e-13 in float64 and 5.5e-8 .. 4.2e-7 in float32.
Issue #21's ``custom_vjp`` needs exactly that pair, so these tests are its
contract too.

Because it is #21's contract, section 6 enumerates the axes that select the
operators' **numerical path** -- 4 ``w_strategy`` x 2 ``channel_strategy`` x 2
``hermitian`` x 2 flag values x 2 precisions, on a two-channel plan -- across
**three** mutually non-redundant tests, none of which gates the claim alone:

  * the **identity** cannot see a defect applied consistently to *both*
    operators: that is just the other flag's pair, and still perfectly
    adjoint;
  * the **composition** test catches such a defect only when its pinned
    ``(dense_scan, scan)`` reference is spared by it. A symmetric defect the
    reference *shares* -- both paths using the same wrong but self-adjoint
    diagonal -- leaves every cell agreeing with an equally wrong reference,
    and passes;
  * so a per-cell **semantic oracle** states what the diagonal actually is,
    against a ``1/n`` built from ``(l, m)`` alone in this file. It shares no
    construction with ``src/``, which is what makes it able to fail when both
    of the others cannot.

That is the *numerical-path* enumeration, and it is not the same thing as
"every argument either operator takes". Four things the operators dispatch or
branch on are deliberately **outside** it, listed so the claim is not read
wider than it is:

  * ``nthreads`` -- a static argument, and ``_resolve_nthreads`` returns ``1``
    for every fixture in this module, so nothing here would notice a defect
    gated on ``nthreads != 1``. One cell covers it
    (``test_the_diagonal_survives_an_explicit_nthreads``) rather than another
    factor of two across the matrix; it changes FINUFFT's threading, not the
    arithmetic ``divide_by_n`` participates in.
  * ``weights`` -- multiplied into the visibilities, i.e. on the far side of
    the operator from this image-side diagonal. ``tests/test_against_ducc.py``
    covers it at the defaults.
  * complex images on ``hermitian=False`` plans -- ``divide_by_n`` multiplies a
    complex image by a real diagonal either way, and the fold restriction is
    issue #17's contract, pinned in ``tests/test_hermitian.py``.
  * the 2-D broadcast image -- deliberately unused here, since it makes the
    forward a different operator from the one ``vis2dirty`` adjoints (see
    ``_Problem.image``). Sections 1-5 exercise it at ``n_chan == 1``.

Also outside section 6, and covered separately because it is a property of the
plan's *geometry* rather than of dispatch: the diagonal's l/m axis assignment,
which no square isotropic fixture can falsify. See ``_ANISO_PIXSIZE_M_RATIO``.

The identity is scored **per channel** rather than on the channel sum; see
:func:`_dot_product_residual` for why summing the blocks first would hide
one-sided defects behind a cancellation between them.

What each flag means, asserted separately from the identity in section 4:

  * forward with ``divide_by_n=True`` multiplies the image by ``1/n`` inside
    the disc and by **zero** outside -- the output is then insensitive to
    every pixel outside it;
  * adjoint with ``divide_by_n=False`` returns the analytic-extension result
    *without* the ``1/n``, so those pixels carry their (large) values rather
    than zeros.

Both are checked against ducc0 (public API only, black-box oracle) at the
repo's ``3 * eps`` bound in section 5, and the outside-disc values against an
exact DFT in the repo's sign convention.

Precision: the identity, the default check, both flag-semantics checks, the
traceability check and **all three section-6 tests** are parametrised over both
legs via ``_PRECISIONS``; the float64 entry carries ``requires_x64`` so
``JAX_ENABLE_X64=0`` skips it rather than failing, and the float32 entry runs
in both legs. Float32 plans are built with ``dtype=jnp.float32`` and
``epsilon = 1e-5``.

Backends
--------

**Every tolerance in the table below is justified against two backends: CPU
(macOS arm64, 10-core) and one NVIDIA GH200.** That is two machines, not a
general claim about GPUs -- another accelerator, or another XLA version, may
sit elsewhere, and a bound whose binding case is the GH200 is the one to
re-measure first if that changes. Measured maxima, with the binding backend in
the last column:

    quantity              CPU        GPU x64    GPU x32    bound   binds
    dft_all       f64   8.740e-07      ?          --       2e-06   ?     2.29x
    dft_outside   f64   5.325e-07      ?          --       2e-06   ?     3.76x
    ducc (28)     f64   24.4% of bd    ?          --       3*eps   ?     4.09x
    identity sec2 f64   1.936e-12  5.506e-15      --       1e-11   CPU   5.2x
    identity sec6 f64   6.574e-13  2.384e-15      --       1e-11   CPU  15.2x
    strategy_err  f64   1.299e-13  1.667e-16      --       1e-11   CPU  77.0x
    fold_err      f64   1.070e-06  1.070e-06      --       4e-06   both  3.7x
    oracle        f64   3.338e-15  3.983e-16      --       1e-12   CPU 299.6x
    same-exec     f64   0.000e+00  1.604e-16      --       1e-11   GPU  6.2e4x
    identity sec2 f32   1.775e-07  7.482e-07  1.456e-06    5e-06   GPU   3.4x
    identity sec6 f32   4.222e-07  4.375e-07  4.013e-07    5e-06   GPU  11.4x
    strategy_err  f32   9.901e-08  7.894e-08  7.819e-08    1e-06   CPU  10.1x
    fold_err      f32   9.132e-06  9.185e-06  9.185e-06    4e-05   GPU   4.4x
    oracle        f32   1.839e-07  1.949e-07  1.949e-07    1e-05   GPU  51.3x
    same-exec     f32   0.000e+00  7.931e-08  7.634e-08    1e-04   GPU  1.3e3x

The first three rows carry ``?`` because they are **not yet measured on the
GH200**: ``DFT_TOL_FACTOR`` and ``DUCC_TOL_FACTOR`` are ``requires_x64``, so
they run on that machine's float64 leg and nobody has read the numbers off it.
They run device code like every other row and belong here; until those cells
are filled, their CPU figures are one backend's evidence and should be read as
such. **``dft_all`` at 2.29x is the tightest bound in this module and the entry
to re-measure first on new hardware.**

Read that table before touching any constant in it, because **CPU evidence
alone ranks these bounds close to backwards**:

  * of the rows measured on both backends, the tightest is ``identity sec2`` in
    float32, at **3.4x** on the GPU's ``JAX_ENABLE_X64=0`` leg -- the
    population that looks *roomiest* on CPU, at 28.2x. The two disagree by 8x.
    A bound fitted to CPU would have been set near 2e-6 and this leg would fail
    it;
  * ``fold_err`` looks like the risk at 3.7x, and is in fact the safest entry
    here: it measures 1.070e-06 on **both** backends, agreeing to four
    significant figures (9.185e-06 against 9.132e-06 in float32). The quantity
    does not move between backends, so its headroom is the whole story rather
    than a sample of one;
  * the float64 identity's 5.2x is the **CPU** being the worse backend by 350x
    (1.9e-12 against 5.5e-15), not a latent GPU risk.

The two same-executable comparisons are the reason the ``same-exec`` row
exists: they are held at ``SAME_EXECUTABLE_TOL_*`` rather than at bit-equality
because XLA:GPU reductions are not run-to-run deterministic. CPU measures
exactly 0.0 there and the GPU does not, which is precisely the shape of
assumption that CPU-only evidence cannot test.

Only three things stay float64-only, and they say so with ``requires_x64``:
the ducc0 parity comparison, the exact-DFT comparison and the mixed-default
measurements, whose bounds (``3 * eps`` at eps=1e-6 and ``2 * eps``) single
precision cannot reach, so restating them at a float32 tolerance would gate
nothing. The signature checks carry no precision at all.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from jax_nufft import dirty2vis, make_plan, vis2dirty, wgridder
from jax_nufft._utils import SPEED_OF_LIGHT
from tests.conftest import EDA2, MWA_COMPACT, Telescope, requires_x64, synthetic_uvw

# The **strategy-equivalence** bound: how far two runs of the operators that
# differ only in reduction order (w_strategy, channel_strategy) may sit apart.
# Matches tests/test_strategies_equivalent.py and tests/test_adjoint.py's
# reduction-order comparison (issue #10).
#
# Kept distinct from IDENTITY_TOL_* below, which used to share these constants.
# The two are different quantities: this one compares the operators to
# themselves under a reordering and measures 9.9e-8 in float32 on the section-6
# matrix, while the identity residual measures round-off in a relation that is
# exact on paper and runs several times larger and backend-dependent. Sharing
# one constant would drag whichever is tighter up to the other's level for a
# reason that does not apply to it.
#
# Binding case: **CPU** on both legs -- 1.299e-13 float64 and 9.901e-08
# float32, against a GH200's 1.667e-16 and 7.894e-08. Reordering a reduction is
# what this quantity measures, so it is mildly surprising that the GPU is the
# quieter of the two; the reason is that it is a comparison *within* one
# backend, so the backend's own reduction order largely cancels. 77x and 10.1x
# of headroom respectively.
DOT_TOL_F64 = 1e-11
DOT_TOL_F32 = 1e-6
# The section-6 semantic oracle's bounds: ``divide_by_n=True`` against the
# *other* flag fed an image (or output) scaled by an independently built masked
# ``1/n``. Measured maxima over its 64 cells: 3.338e-15 (CPU) / 3.983e-16 (GPU)
# in float64, and 1.839e-07 (CPU) / 1.949e-07 (GPU) in float32. The GPU binds
# on the float32 leg, by 6%; both legs keep 50x or more of headroom, so this is
# among the least backend-sensitive bounds here. Same numbers as section 4's
# inline bounds, which make the same comparison on one strategy; named here
# because section 6 makes it 64 times and the two should not drift apart.
ORACLE_TOL_F64 = 1e-12
ORACLE_TOL_F32 = 1e-5
# The dot-product identity's own bounds, kept separate from the
# strategy-equivalence ones above because they are a different quantity with a
# different backend sensitivity, not the same quantity on a different fixture.
#
# The identity is exact on paper, so its residual is pure round-off in the
# operators' own reductions -- and reduction order is exactly what changes
# between backends. Measured maxima over this module's two identity
# populations (six section-2 cases; 32 section-6 two-channel cells, scored per
# channel):
#
#                     CPU        GPU x64    GPU x32     binds
#   sec2 float64   1.936e-12   5.506e-15      --        CPU     5.2x
#   sec6 float64   6.574e-13   2.384e-15      --        CPU    15.2x
#   sec2 float32   1.775e-07   7.482e-07  1.456e-06     GPU     3.4x
#   sec6 float32   4.222e-07   4.375e-07  4.013e-07     GPU    11.4x
#
# Two things in that table are worth the space they take.
#
# The **float32 binding case is the GPU's x32 leg at 1.456e-06**, in the
# population whose CPU number (1.775e-07) is the roomiest in the module. CPU
# and GPU disagree by 8x there, in the direction that matters. An earlier
# version of this constant was 1e-6, fitted to CPU, and the GH200 exceeded it
# by 5%; fitted to CPU evidence *including* that correction it would have
# landed near 2e-6, and this leg would still fail. 5e-6 covers it with 3.4x --
# the tightest margin in this module, and the entry to re-measure first on new
# hardware.
#
# The **float64 binding case is the CPU**, by 350x (1.936e-12 against
# 5.506e-15). Its 5.2x is therefore not a latent GPU risk but the worse of two
# backends already measured, and 1e-11 stands.
#
# What fixes the float32 value is detection, not headroom: the uniform 5e-5
# one-sided scaling defect (``MUT-E``) must keep failing, and it exceeds even
# the worst measured cell by 34x. A bound is only honest here if a defect of
# the size this test exists to catch still fails it with margin.
#
# The strategy-equivalence use in section 6's composition test deliberately
# does NOT move to this number: it compares two runs at the same precision
# differing only in reduction order, measures 9.9e-8, and giving it a 5e-6
# bound would loosen a tight quantity for a reason that belongs to another.
IDENTITY_TOL_F64 = 1e-11
IDENTITY_TOL_F32 = 5e-6

# Two calls that route to the *same* JIT cache entry are not bit-identical on
# every backend. XLA:GPU reductions are not run-to-run deterministic, so the
# same executable on the same inputs can differ in the last bits -- measured on
# a GH200 at 1.137e-13 absolute on values of order 10 (~1e-14 relative) for a
# float64 adjoint, across 37% of pixels. An earlier version of this module
# asserted bit-equality here and passed everywhere on CPU while being wrong
# about what it was entitled to assume.
#
# So the *numeric* half of a same-executable comparison is held at round-off,
# and the claim it used to carry -- that the no-argument call routes to the
# declared default rather than somewhere else -- is gated separately and
# exactly, at the JIT boundary, where it is a property of the dispatch rather
# than of the arithmetic.
#
# Measured maxima over the module's three same-executable comparisons: CPU
# **exactly 0.000e+00** on both legs, against a GH200's 1.604e-16 (float64) and
# 7.931e-08 (float32). The GPU binds trivially, with 6.2e4x and 1.3e3x of
# headroom. That CPU column is the whole lesson of this constant: a backend
# that is bit-reproducible cannot distinguish "deterministic by construction"
# from "deterministic here", so no amount of CPU evidence could have shown the
# bit-equality assertion these bounds replaced to be unsound.
SAME_EXECUTABLE_TOL_F64 = 1e-11
SAME_EXECUTABLE_TOL_F32 = 1e-4
# ...and however tight that bound is in absolute terms, it must stay far below
# the contrast the same test measures between the two flag values, or the
# comparison stops discriminating. Asserted as a ratio so it scales with the
# fixture instead of trusting two independently chosen constants.
SAME_EXECUTABLE_SEPARATION = 1e3
# ducc0 parity contract, mirroring tests/test_against_ducc.py::DUCC_TOL_FACTOR.
# Worst of this module's 28 ducc cells on CPU: 24.4% of the bound (4.09x).
# Not yet measured on the GH200 -- see the module docstring's table.
DUCC_TOL_FACTOR = 3.0
# Exact-DFT contract, mirroring tests/test_adjoint.py::DFT_TOL_FACTOR.
# CPU: 8.740e-07 whole-image (2.29x) and 5.325e-07 outside-disc (3.76x) against
# 2*eps at eps=1e-6. The whole-image leg is the **tightest bound in this
# module** -- tighter than anything in the two-backend table -- and it too is
# unmeasured on the GH200. Both are ``requires_x64``, which is why they were
# absent from that table for a round; running device code, they belong in it.
DFT_TOL_FACTOR = 2.0
# Folded vs unfolded agreement, mirroring
# tests/test_hermitian.py::CROSS_PATH_TOL_FACTOR (issue #17): the triangle
# inequality on the 2*eps DFT contract each path meets separately.
#
# That derivation holds only in **float64** in this repository. The suites that
# establish the 2*eps DFT contract and the folded/unfolded agreement
# (``test_against_dft.py``, ``test_adjoint.py``, ``test_hermitian.py``) are in
# ``conftest.collect_ignore`` for the ``JAX_ENABLE_X64=0`` leg, so in single
# precision there is no such contract for the triangle inequality to be applied
# to. The float32 use of this factor therefore rests on the measurement in the
# module docstring's table (9.132e-06 CPU / 9.185e-06 GH200 against 4e-05, a
# 4.4x margin), not on the derivation -- worth knowing before treating the
# float32 bound as principled rather than empirical.
#
# The 3.7x headroom this gives in float64 (1.070e-06 against 4e-06) is the
# smallest number in the module docstring's table, and it is nonetheless the
# entry least in need of watching: it is **measured identical on both
# backends** -- 1.070e-06 on CPU and on a GH200, agreeing to four significant
# figures, and 9.132e-06 against 9.185e-06 in float32. Folded and unfolded
# plans are two approximations of the same operator, and the gap between them
# is set by the w-plane geometry rather than by reduction order, so it does not
# move with the backend. Its headroom is therefore the whole story, not a
# sample of one; contrast the float32 identity above it, whose CPU headroom
# overstates the truth by 8x.
CROSS_PATH_TOL_FACTOR = 4.0

W_STRATEGIES = ("dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap")
CHANNEL_STRATEGIES = ("scan", "vmap")
FLAG_VALUES = (False, True)

# As ``_PRECISIONS``, but carrying the identity's own bounds. Used by every
# test that asserts ``Re<A x, y> == <x, A^H y>``; ``_PRECISIONS`` stays with
# the tests that compare two runs of the operators to each other.
_IDENTITY_PRECISIONS = [
    pytest.param(jnp.float64, 1e-6, IDENTITY_TOL_F64, id="float64", marks=requires_x64),
    pytest.param(jnp.float32, 1e-5, IDENTITY_TOL_F32, id="float32"),
]


# The geometries that can falsify the ``divide_by_n`` diagonal's **axis
# assignment**, and the reason they have to exist.
#
# ``divide_by_n`` puts a geometry-indexed diagonal into both operators, and
# ``n = sqrt(1 - l^2 - m^2)`` is symmetric under swapping l and m exactly when
# the grid is. Every fixture in this repository is square with isotropic
# pixels, and on such a plan ``plan.n_minus_1`` is transpose-symmetric
# *bit-for-bit* -- ``max|nm1 - nm1.T| == 0.0``, not merely to round-off. So
# ``n_grid.T`` in place of ``n_grid`` is a literal no-op there, and every test
# in this module passed with that mutation in place: the identity (both
# operators transposed alike, so still adjoint), the composition matrix (the
# pinned reference transposed too), and even the ``(l, m)``-built oracle, which
# inherits the blindness structurally because a single scalar pixel size cannot
# express an asymmetric grid.
#
# It is not a harmless mutation. On the anisotropic plan below,
# ``vis2dirty(divide_by_n=True)`` against ducc0 goes from 3.5e-07 -- inside the
# repo's 3*eps contract -- to 1.0e+00.
#
# Two geometries, because they fail a transposed diagonal in different ways:
#
#   * ``_ANISO_SQUARE`` keeps ``n_l == n_m`` and makes the *pixel sizes*
#     differ, so a transpose stays shape-valid and has to be caught
#     numerically. This is the sharp one.
#   * ``_ANISO_NONSQUARE`` makes the shape itself asymmetric, where a
#     transposed diagonal cannot even broadcast. Cheaper to catch, and it
#     covers an axis-swap that a square grid would hide behind broadcasting.
#
# Deliberately the minimum needed to falsify *this* diagonal. Systematic
# odd / non-square / anisotropic coverage is issue #14's scope, not this one's.
_ANISO_PIXSIZE_M_RATIO = 0.6
_ANISO_SQUARE: dict[str, Any] = {"pixsize_m_ratio": _ANISO_PIXSIZE_M_RATIO}
_ANISO_NONSQUARE: dict[str, Any] = {"shape": (48, 64)}
_ANISO_GEOMETRIES = [
    pytest.param(_ANISO_SQUARE, id="square_pixsize_m_0.6"),
    pytest.param(_ANISO_NONSQUARE, id="nonsquare_48x64"),
]


# The channel count and per-channel frequencies the section-6 matrices use.
#
# What two channels actually buy, stated narrowly. A *single*-channel plan can
# already expose plenty: with ``channel_strategy`` parametrised it exercises
# both branches of the dispatch and both lowerings, and it would catch MUT-C
# (the symmetric channel-strategy defect) on its own. What one channel cannot
# exercise is anything depending on the channel *axis* having extent --
# cross-channel indexing, and the association of ``inv_lambda[c]`` with image
# plane ``c`` and visibility column ``c``. A transposed or off-by-one
# association is invisible at ``n_chan == 1`` because there is only one thing
# to associate. That, and only that, is why the matrices use two.
#
# The frequencies are distinct for the same narrow reason: at equal
# frequencies every channel shares one ``inv_lambda``, so a mis-association
# still lands on the right value and the axis goes untested. Equal-frequency
# channels would *not* be vacuous in general -- their images and visibilities
# still differ -- but they cannot test the association, which is what a second
# channel is here for.
#
# 1.25 rather than something wilder: the ratio multiplies every baseline in
# wavelengths, so it drives ``n_w`` and the runtime of a 64-cell matrix.
_MATRIX_N_CHAN = 2
_CHANNEL_FREQ_RATIOS = (1.0, 1.25)

# Both precision legs. The float64 entry is skipped (not failed) under
# JAX_ENABLE_X64=0; the float32 entry runs in *both* legs, since a float32
# plan is legal either way.
_PRECISIONS = [
    pytest.param(jnp.float64, 1e-6, DOT_TOL_F64, id="float64", marks=requires_x64),
    pytest.param(jnp.float32, 1e-5, DOT_TOL_F32, id="float32"),
]

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Problem:
    plan: Any
    uvw: np.ndarray
    freq: np.ndarray
    # Separate per-axis pixel sizes, and ``shape`` may be non-square. Both
    # exist for one reason: ``divide_by_n`` puts a *geometry-indexed* diagonal
    # into both operators, and ``n = sqrt(1 - l^2 - m^2)`` is symmetric in l
    # and m exactly when the grid is. On every square isotropic fixture in this
    # repository ``plan.n_minus_1`` is bit-for-bit transpose-symmetric
    # (``max|nm1 - nm1.T| == 0.0``), so transposing the diagonal is a no-op and
    # the axis assignment is unfalsifiable. See ``_ANISO`` below.
    pixsize_l: float
    pixsize_m: float
    shape: tuple[int, int]
    # (n_l, n_m) real at ``n_chan == 1``; (n_chan, n_l, n_m) above it. The
    # multi-channel image is deliberately *not* the 2-D broadcast form: a 2-D
    # image makes the forward a map from one plane to ``n_chan`` visibility
    # columns, whose adjoint sums over channels, so it is a different operator
    # from the one ``vis2dirty`` implements and the dot-product identity below
    # would not even be shape-compatible with it.
    image: np.ndarray
    vis: np.ndarray  # (n_rows, n_chan) complex, in the plan's complex dtype
    eps: float


_CACHE: dict[tuple, _Problem] = {}


def _problem(
    tel: Telescope,
    zenith_angle_deg: float,
    *,
    eps: float,
    dtype: DTypeLike = jnp.float64,
    hermitian: bool = True,
    uvw_seed: int = 0,
    data_seed: int = 7,
    n_chan: int = 1,
    pixsize_m_ratio: float = 1.0,
    shape: tuple[int, int] | None = None,
) -> _Problem:
    """Build (and cache) a plan plus a real image and complex visibilities.

    Cached because several tests below want the same plan and building one is
    pure host work; the arrays handed out are read-only by convention (every
    caller that modifies one takes a copy first).

    ``n_chan`` defaults to 1, which reproduces the single-channel fixture
    exactly -- same frequency (``tel.freq_hz * 1.0``), same 2-D image, same
    ``(n_rows, 1)`` visibilities, same draws in the same order from the same
    generator. Above 1 the frequencies come from ``_CHANNEL_FREQ_RATIOS`` and
    the image gains a leading channel axis.

    ``pixsize_m_ratio`` and ``shape`` likewise default to the square isotropic
    fixture. Passing either builds a plan whose ``n`` grid is *not*
    transpose-symmetric, which is the only way to falsify the l/m axis
    assignment of the ``divide_by_n`` diagonal -- see ``_ANISO``.
    """
    real_dtype = np.dtype(jnp.dtype(dtype))
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    key = (
        tel.name,
        zenith_angle_deg,
        eps,
        real_dtype.name,
        hermitian,
        uvw_seed,
        data_seed,
        n_chan,
        pixsize_m_ratio,
        shape,
    )
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    uvw = synthetic_uvw(tel, zenith_angle_deg, seed=uvw_seed)
    freq = tel.freq_hz * np.asarray(_CHANNEL_FREQ_RATIOS[:n_chan], dtype=np.float64)
    pixsize_l = tel.pixsize
    pixsize_m = tel.pixsize * pixsize_m_ratio
    shape = (tel.n_pix, tel.n_pix) if shape is None else shape
    rng = np.random.default_rng(data_seed)
    image = rng.standard_normal(shape if n_chan == 1 else (n_chan, *shape)).astype(real_dtype)
    vis = (
        rng.standard_normal((tel.n_rows, n_chan)) + 1j * rng.standard_normal((tel.n_rows, n_chan))
    ).astype(complex_dtype)
    plan = make_plan(uvw, freq, shape, pixsize_l, pixsize_m, eps, dtype=dtype, hermitian=hermitian)
    problem = _Problem(
        plan=plan,
        uvw=uvw,
        freq=freq,
        pixsize_l=pixsize_l,
        pixsize_m=pixsize_m,
        shape=shape,
        image=image,
        vis=vis,
        eps=eps,
    )
    _CACHE[key] = problem
    return problem


def _n_grid(plan: Any) -> np.ndarray:
    """``n = (n - 1) + 1`` on the plan's own grid, in the plan's own dtype.

    Formed the way the operators form it -- ``plan.n_minus_1`` plus one, in
    ``plan.real_dtype`` -- so the ``n_grid > 0`` disc mask this module uses is
    the mask the operators use, bit for bit, on a float32 plan as well.
    """
    nm1 = np.asarray(plan.n_minus_1)
    return np.asarray(nm1 + nm1.dtype.type(1.0), dtype=np.float64)


def _outside_disc(plan: Any) -> np.ndarray:
    return _n_grid(plan) <= 0.0


def _independent_one_over_n(problem: _Problem) -> np.ndarray:
    """``1/n`` inside the unit disc and ``0`` outside, from ``(l, m)`` alone.

    The oracle for section 6's semantic check, and the point of it is what it
    does **not** touch: not ``plan.n_minus_1``, not ``plan.real_dtype``, not
    any helper in ``src/``. Only the image geometry the caller passed to
    ``make_plan`` -- the pixel size and the ``(i - n/2) * pixsize`` grid
    convention documented in the README -- and ``n = sqrt(1 - l^2 - m^2)``
    straight from the measurement equation. Always float64, whatever the plan's
    precision, so the oracle is not limited by the thing it is checking.

    Why an independent construction rather than ``_n_grid(plan)``, which is
    right there and cheaper: the two other section-6 tests are both blind to a
    diagonal that is *wrong but self-adjoint and shared*. The identity cannot
    see it (a real diagonal applied to both operators keeps the pair adjoint
    whatever it contains), and the composition test cannot see it either once
    its pinned reference has the same defect -- every cell then agrees with an
    equally wrong reference. Only a statement of what the diagonal should
    actually *be*, built from something the implementation does not supply, can
    fail there. Reading the factor off the plan would reintroduce exactly the
    shared-source problem, one level down.

    ``test_the_independent_one_over_n_matches_the_plans_own_grid`` pins that
    this construction and the operators' own disc mask agree, so a boundary
    pixel disagreeing shows up as one clear failure rather than as noise
    spread over 64 cells.
    """
    n_l, n_m = problem.shape
    ll = (np.arange(n_l) - n_l // 2) * problem.pixsize_l
    mm = (np.arange(n_m) - n_m // 2) * problem.pixsize_m
    lgrid, mgrid = np.meshgrid(ll, mm, indexing="ij")
    rho2 = lgrid**2 + mgrid**2
    inside = rho2 < 1.0
    n_grid = np.sqrt(np.where(inside, 1.0 - rho2, 0.0))
    ok = inside & (n_grid > 0.0)
    return np.where(ok, 1.0 / np.where(ok, n_grid, 1.0), 0.0)


def _forward(problem: _Problem, image: np.ndarray, **kwargs: Any) -> np.ndarray:
    return np.asarray(dirty2vis(problem.plan, jnp.asarray(image), **kwargs))


def _adjoint(problem: _Problem, **kwargs: Any) -> np.ndarray:
    return np.asarray(vis2dirty(problem.plan, jnp.asarray(problem.vis), **kwargs))


def _relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 difference ``||a - b|| / ||b||``, in float64.

    The scale-free form both same-executable comparisons use, in place of the
    bit-equality they used to assert. See ``SAME_EXECUTABLE_TOL_F64``.
    """
    a64 = np.asarray(a, dtype=np.complex128 if np.iscomplexobj(a) else np.float64)
    b64 = np.asarray(b, dtype=np.complex128 if np.iscomplexobj(b) else np.float64)
    return float(np.linalg.norm(a64 - b64) / np.linalg.norm(b64))


def _dot_product_residual(
    problem: _Problem,
    *,
    image: np.ndarray | None = None,
    image_rhs: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[float, float, float]:
    """Worst **per-channel** relative residual of ``Re<A x, y> == <x_rhs, A^H y>``.

    Scored channel by channel and reduced with ``max``, never summed over the
    channel axis first. That is a correctness requirement, not extra detail.

    Both operators are block-diagonal over channels -- channel ``c`` sees only
    ``inv_lambda[c]`` and its own image plane and visibility column -- so the
    identity holds per block, and the aggregate is just the sum of the blocks.
    But the blocks can carry opposite signs and nearly equal magnitudes: on the
    two-channel EDA2 fixture they are ``+3909.9`` and ``-3837.0``, so **1.86%**
    of either survives the sum. Scoring the sum then divides an error that
    scales with 3.9e3 by a denominator of order 7e1.

    How much that inflates the residual depends on the cell, and the spread is
    too wide for a single figure -- measured aggregate-over-per-channel across
    the 32-cell matrix: 0.63x .. 1.04x (float64, ``divide_by_n=False``),
    27.5x .. 107.3x (float64, ``True``), 2.90x .. 4.46x (float32, ``False``),
    2.24x .. 86.9x (float32, ``True``). The ``True`` legs are where it bites,
    consistent with the ``1/n`` factor being what makes the two blocks nearly
    cancel in the first place.

    The consequence is *not* that any cancellation-absorbing bound is blind to
    a one-sided defect; that overstates it. Under aggregate scoring this
    population reaches 1.229e-05, so a bound must clear that, and set tightly
    at 2e-05 it would still catch ``MUT-E``'s uniform 5e-5 error (5.766e-05
    aggregate) by 2.9x. What is true, and is what actually happened, is
    narrower: the bound chosen on that scoring was 1e-4 and ``MUT-E`` passed
    it. Absorbing the cancellation turns a 50x detection margin into a
    single-digit one and makes the bound's honesty rest on a headroom
    judgement that has nothing to do with the operators.

    Per-channel scoring removes the cancellation entirely: each block is
    divided by its own magnitude, so the residual measures the operators
    rather than how nearly the blocks annihilate. At ``n_chan == 1`` it is
    bit-identical to the aggregate form it replaces, so nothing single-channel
    moved.

    ``image_rhs`` defaults to ``image``: passing a different array is how the
    mixed-default tests state the ``n x`` correction on the right-hand side
    only. Everything is accumulated in float64 so the residual measures the
    operators, not the test's own summation.

    Returns the worst channel's ``(residual, lhs, rhs)``.
    """
    x = problem.image if image is None else image
    rhs_x = x if image_rhs is None else image_rhs
    ax = _forward(problem, x, **kwargs).astype(np.complex128)  # (n_rows, n_chan)
    ay = _adjoint(problem, **kwargs).astype(np.float64)  # (n_chan, n_l, n_m)
    vis = problem.vis.astype(np.complex128)  # (n_rows, n_chan)
    rhs_arr = np.broadcast_to(np.asarray(rhs_x, dtype=np.float64), ay.shape)

    worst = (-1.0, 0.0, 0.0)
    for c in range(ay.shape[0]):
        lhs_c = float(np.vdot(ax[:, c], vis[:, c]).real)
        rhs_c = float(np.vdot(rhs_arr[c].ravel(), ay[c].ravel()))
        residual_c = abs(lhs_c - rhs_c) / max(abs(lhs_c), abs(rhs_c))
        if residual_c > worst[0]:
            worst = (residual_c, lhs_c, rhs_c)
    return worst


def _reference_adjoint_no_divide(
    vis: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    shape: tuple[int, int],
    pixsize_l: float,
    pixsize_m: float,
) -> np.ndarray:
    """Exact DFT adjoint **without** the ``1/n`` and without the disc mask.

    The repo's sign convention (AGENTS.md section 1) with ``n - 1`` taken on
    the analytic extension outside the unit disc, i.e. what ``vis2dirty(...,
    divide_by_n=False)`` is supposed to return everywhere. Same structure as
    ``tests/test_adjoint.py::_reference_adjoint``, minus its final division
    and mask; written independently of ``src/`` so it is an oracle rather than
    a restatement.
    """
    n_l, n_m = shape
    ll = (np.arange(n_l) - n_l // 2) * pixsize_l
    mm = (np.arange(n_m) - n_m // 2) * pixsize_m
    lgrid, mgrid = np.meshgrid(ll, mm, indexing="ij")
    rho2 = lgrid**2 + mgrid**2
    inside = rho2 <= 1.0
    nm1 = np.where(
        inside,
        np.sqrt(np.where(inside, 1.0 - rho2, 0.0)) - 1.0,
        -np.sqrt(np.where(inside, 0.0, rho2 - 1.0)) - 1.0,
    )
    out = np.zeros(shape, dtype=np.float64)
    scale = freq[0] / SPEED_OF_LIGHT
    u = uvw[:, 0] * scale
    v = uvw[:, 1] * scale
    w = uvw[:, 2] * scale
    vis64 = vis.astype(np.complex128)
    for r in range(uvw.shape[0]):
        phase = 2j * np.pi * (u[r] * lgrid + v[r] * mgrid - w[r] * nm1)
        out += (vis64[r, 0] * np.exp(phase)).real
    return out


def _eda2_full_sky(dtype: DTypeLike = jnp.float64, eps: float = 1e-6, **kw: Any) -> _Problem:
    """EDA2 at a 120-degree FoV: the fixture whose image runs past the disc.

    Every test that says "outside the disc" uses this one, and asserts the
    region is non-empty and carries signal before asserting anything about it
    -- a narrow-field fixture would make those tests vacuously true.
    """
    return _problem(EDA2, 0.0, eps=eps, dtype=dtype, **kw)


def _assert_outside_disc_is_loaded(problem: _Problem) -> np.ndarray:
    """Guard: the fixture must actually have signal outside the unit disc."""
    outside = _outside_disc(problem.plan)
    n_out = int(outside.sum())
    assert 0 < n_out < outside.size, (
        f"fixture has {n_out} of {outside.size} pixels outside the unit disc: this test "
        "says nothing unless the outside-disc region is non-empty and not the whole image"
    )
    # ``[..., outside]`` rather than ``[outside]``: the mask is (n_l, n_m) and
    # must land on the image's trailing two axes, which is the whole array for
    # a 2-D image and every channel of a 3-D one. Indexed from the front, a
    # multi-channel image would raise instead.
    signal = float(np.linalg.norm(np.asarray(problem.image, dtype=np.float64)[..., outside]))
    assert signal > 0.0, "the outside-disc pixels must carry signal, not zeros"
    return outside


# ---------------------------------------------------------------------------
# 1. the declared defaults (issue #20 keeps today's behaviour by default)
# ---------------------------------------------------------------------------


def _declared(fn: Callable[..., Any]) -> inspect.Parameter:
    params = inspect.signature(fn).parameters
    assert "divide_by_n" in params, (
        f"{fn.__name__} has no divide_by_n parameter; issue #20 exposes it on **both** "
        "operators, and the pair is only adjoint when both accept it"
    )
    return params["divide_by_n"]


def test_dirty2vis_declares_divide_by_n_false_and_keyword_only() -> None:
    """The forward's declared default is the bool ``False``, keyword-only.

    ``is False`` rather than ``== False`` or ``not ...``: a truthy/falsy
    non-bool (``0``, ``None``) would normalise to the same behaviour through
    any ``bool()`` the implementation applies, so behaviour alone cannot see
    it -- but it is wrong in the public signature and, since the flag is a
    static JIT argument, it changes the cache key's type. Keyword-only because
    positional acceptance would silently reinterpret an existing
    ``dirty2vis(plan, image, w_strategy)``-style call.
    """
    param = _declared(dirty2vis)
    assert param.default is False, (
        f"dirty2vis's declared divide_by_n default is {param.default!r}, not the bool False. "
        "Issue #20 must not change what a caller who says nothing gets: the forward has "
        "never applied the 1/n factor (ducc's divide_by_n=False) and the existing DFT and "
        "ducc parity tests are written against that."
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"dirty2vis's divide_by_n is {param.kind}, not keyword-only"
    )


def test_vis2dirty_declares_divide_by_n_true_and_keyword_only() -> None:
    """The adjoint's declared default is the bool ``True``, keyword-only."""
    param = _declared(vis2dirty)
    assert param.default is True, (
        f"vis2dirty's declared divide_by_n default is {param.default!r}, not the bool True. "
        "The adjoint has always applied 1/n on its output (ducc's divide_by_n=True); "
        "issue #20 exposes the flag without moving the default."
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"vis2dirty's divide_by_n is {param.kind}, not keyword-only"
    )


def test_the_declared_defaults_are_not_swapped() -> None:
    """Both defaults in one assertion, because the pair is the claim.

    The two operators' defaults are deliberately *different*, which is exactly
    the shape of edit that gets made backwards. Pinning them one file-section
    apart lets a swap pass one test while failing another; this states the
    pair.
    """
    fwd = _declared(dirty2vis).default
    adj = _declared(vis2dirty).default
    assert (fwd, adj) == (False, True), (
        f"divide_by_n defaults are (dirty2vis={fwd!r}, vis2dirty={adj!r}); issue #20 ships "
        "(False, True) -- the mixed pair that reproduces pre-#20 behaviour. Swapping them "
        "would silently change every existing caller's answer by a factor of n."
    )
    assert fwd is False and adj is True


def _spy_jit(monkeypatch: pytest.MonkeyPatch, jit_name: str) -> list[dict]:
    """Record the kwargs reaching ``wgridder.<jit_name>``; return a dummy array.

    The pattern ``tests/test_jax_integration.py`` and
    ``tests/test_default_w_strategy.py`` use for the same purpose. Both public
    wrappers return the JIT function's result unchanged, so any return value
    does.
    """
    calls: list[dict] = []

    def stub(*args: object, **kwargs: object) -> jax.Array:
        calls.append(kwargs)
        return jnp.zeros(())

    monkeypatch.setattr(wgridder, jit_name, stub)
    return calls


@pytest.mark.parametrize(
    "op, jit_name, declared",
    [
        ("dirty2vis", "_dirty2vis_jit", False),
        ("vis2dirty", "_vis2dirty_jit", True),
    ],
)
def test_the_no_argument_call_routes_the_declared_default_to_the_jit_boundary(
    monkeypatch: pytest.MonkeyPatch, op: str, jit_name: str, declared: bool
) -> None:
    """The routing half of the default check, gated exactly and without arithmetic.

    The numeric test below can only say the no-argument call produces *nearly*
    the same answer as the explicit default, because two runs of one executable
    are not bit-identical on every backend (see ``SAME_EXECUTABLE_TOL_F64``).
    That leaves a gap a numeric comparison cannot close on its own: a call that
    routed to a different-but-similar path would also land inside any tolerance
    loose enough to absorb backend round-off.

    So the routing is asserted where it is exact -- the value handed to the JIT
    boundary -- and asserted with ``is``, which additionally pins the *type*.
    ``divide_by_n`` is a static argument, so an int ``1`` in place of ``True``
    would behave identically downstream while occupying a separate JIT cache
    entry; the signature tests in this section pin the declaration, and this
    pins what the wrapper actually forwards.

    Backend-independent by construction: no transform runs at all, the JIT
    function being stubbed out.
    """
    problem = _problem(EDA2, 0.0, eps=1e-5, dtype=jnp.float32)
    calls = _spy_jit(monkeypatch, jit_name)

    if op == "dirty2vis":
        dirty2vis(problem.plan, jnp.asarray(problem.image))
        dirty2vis(problem.plan, jnp.asarray(problem.image), divide_by_n=declared)
    else:
        vis2dirty(problem.plan, jnp.asarray(problem.vis))
        vis2dirty(problem.plan, jnp.asarray(problem.vis), divide_by_n=declared)

    assert len(calls) == 2, f"expected two {jit_name} calls, saw {len(calls)}"
    from_default, from_explicit = calls[0]["divide_by_n"], calls[1]["divide_by_n"]
    assert from_default is declared, (
        f"{op} with no divide_by_n forwarded {from_default!r} to {jit_name}, not the "
        f"declared default {declared!r}. The signature can say one thing and the wrapper "
        "forward another; this is the half that catches that."
    )
    assert from_explicit is declared, (
        f"{op} with divide_by_n={declared!r} forwarded {from_explicit!r} to {jit_name}"
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_omitting_the_flag_reproduces_the_declared_default(
    op: str, dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Behavioural half of the default check, on a fixture where it can fail.

    The signature test above cannot see an implementation that declares the
    right default and then ignores it, or one that forwards a constant to the
    JIT boundary. So: the no-argument call must be bit-identical to the
    explicit default and materially different from the other value. EDA2
    full-sky is the fixture that makes "materially different" true -- on a
    narrow field with ``n ~ 1`` the two values are close, and this test would
    pass while gating nothing.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    _assert_outside_disc_is_loaded(problem)
    if op == "dirty2vis":
        default = _forward(problem, problem.image)
        same = _forward(problem, problem.image, divide_by_n=False)
        other = _forward(problem, problem.image, divide_by_n=True)
    else:
        default = _adjoint(problem)
        same = _adjoint(problem, divide_by_n=True)
        other = _adjoint(problem, divide_by_n=False)
    is_f64 = np.dtype(jnp.dtype(dtype)) == np.float64
    same_path = _relative_difference(default, same)
    same_path_tol = SAME_EXECUTABLE_TOL_F64 if is_f64 else SAME_EXECUTABLE_TOL_F32
    assert same_path < same_path_tol, (
        f"{op} with no divide_by_n differs by {same_path:.3e} from the explicit declared "
        f"default (tol {same_path_tol:.1e}) -- more than two runs of one executable can "
        "differ by, so the no-argument call is reaching different arithmetic"
    )
    contrast = np.linalg.norm(default - other) / np.linalg.norm(other)
    assert contrast > 1e-2, (
        f"{op}'s two divide_by_n values differ by only {contrast:.3e} on EDA2 full-sky: "
        "either the flag is being ignored or this fixture cannot tell the two apart, and "
        "the agreement asserted above is then vacuous"
    )
    assert same_path * SAME_EXECUTABLE_SEPARATION < contrast, (
        f"{op}: the no-argument call sits {same_path:.3e} from the declared default while "
        f"the two flag values sit {contrast:.3e} apart -- less than "
        f"{SAME_EXECUTABLE_SEPARATION:g}x of separation, so this fixture can no longer "
        "tell 'same path' from 'other path' and the tolerance above is doing nothing"
    )


# ---------------------------------------------------------------------------
# 2. the headline gate: an exact adjoint pair under equal flags
# ---------------------------------------------------------------------------


_IDENTITY_FIXTURES = [
    # (telescope, zenith angle, expects pixels outside the unit disc)
    pytest.param(EDA2, 0.0, True, id="EDA2_zenith_full_sky"),
    pytest.param(MWA_COMPACT, 0.0, False, id="MWA_compact_zenith"),
    pytest.param(MWA_COMPACT, 30.0, False, id="MWA_compact_off30"),
]


@pytest.mark.parametrize("tel, zenith_deg, has_outside_disc", _IDENTITY_FIXTURES)
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("dtype, eps, dot_tol", _IDENTITY_PRECISIONS)
def test_dot_product_identity_under_equal_flags(
    tel: Telescope,
    zenith_deg: float,
    has_outside_disc: bool,
    divide_by_n: bool,
    dtype: DTypeLike,
    eps: float,
    dot_tol: float,
) -> None:
    """``Re<A x, y> = <x, A^H y>``, for **both** flag values, per fixture.

    This is issue #20's definition of done and issue #21's precondition: a
    ``custom_vjp`` is only correct if the pair is exactly adjoint, and it is
    exactly adjoint only with equal flags. Asserted per fixture rather than
    aggregated, so the wide-field case (EDA2 at 120 degrees, where the mixed
    default pair is off by 2.7e-1) cannot be averaged out by the narrow ones.

    No ``n`` correction anywhere in this identity: that factor is an artefact
    of the *mixed* pair and is what section 3 below pins separately.
    """
    problem = _problem(tel, zenith_deg, eps=eps, dtype=dtype)
    outside = int(_outside_disc(problem.plan).sum())
    assert (outside > 0) == has_outside_disc, (
        f"{tel.name} zen={zenith_deg} has {outside} pixels outside the unit disc, "
        f"expected {'some' if has_outside_disc else 'none'}: the fixture set must keep "
        "covering both regimes for this parametrisation to mean anything"
    )
    residual, lhs, rhs = _dot_product_residual(problem, divide_by_n=divide_by_n)
    assert residual < dot_tol, (
        f"{tel.name} zen={zenith_deg} divide_by_n={divide_by_n} dtype={np.dtype(jnp.dtype(dtype)).name}: "
        f"dot-product residual {residual:.3e} exceeds {dot_tol:.1e} "
        f"(lhs={lhs!r}, rhs={rhs!r}). With equal flags the pair must be exactly adjoint -- "
        "this is what issue #21's custom_vjp is allowed to assume."
    )


@requires_x64
def test_the_mixed_default_pair_is_not_an_adjoint_pair_on_the_full_sky() -> None:
    """Why the flag exists, pinned as a measurement rather than prose.

    The defaults are unchanged by issue #20, so on a wide field they still do
    **not** form an adjoint pair -- neither in the plain form nor with the
    ``n x`` correction, because outside the unit disc the forward evaluates
    the analytic extension while the adjoint zeroes those pixels. Measured
    here: 6.3e-1 plain, 2.7e-1 with ``n x``. Equal flags on the same fixture
    and the same data land at 1e-15 .. 1e-13.

    If someone "fixes" the wide-field gradient by flipping a default instead
    of adding the flag, this test is what says the defaults moved.
    """
    problem = _eda2_full_sky()
    _assert_outside_disc_is_loaded(problem)
    n_grid = _n_grid(problem.plan)
    image = np.asarray(problem.image, dtype=np.float64)

    plain, _, _ = _dot_product_residual(problem)
    corrected, _, _ = _dot_product_residual(problem, image_rhs=image * n_grid)
    assert plain > 1e-2 and corrected > 1e-2, (
        f"the default (mixed) pair now looks adjoint on EDA2 full-sky: plain residual "
        f"{plain:.3e}, n*x residual {corrected:.3e}. That is not something issue #20 "
        "delivers -- it keeps the defaults as they were and fixes the pair via equal "
        "flags. A default was probably changed."
    )
    for flag in FLAG_VALUES:
        equal, _, _ = _dot_product_residual(problem, divide_by_n=flag)
        assert equal < IDENTITY_TOL_F64, (
            f"equal flags divide_by_n={flag} on the same fixture: residual {equal:.3e}"
        )


# ---------------------------------------------------------------------------
# 3. the documented behaviour of the mixed default pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tel, zenith_deg, mask_to_disc",
    [
        pytest.param(MWA_COMPACT, 0.0, False, id="MWA_compact_zenith_whole_image"),
        pytest.param(MWA_COMPACT, 30.0, False, id="MWA_compact_off30_whole_image"),
        pytest.param(EDA2, 0.0, True, id="EDA2_full_sky_disc_masked"),
    ],
)
@requires_x64
def test_mixed_default_pair_keeps_the_documented_n_correction(
    tel: Telescope, zenith_deg: float, mask_to_disc: bool
) -> None:
    """``Re<A x, y> = <n x, A^H y>`` for the defaults, restricted to the disc.

    This is the identity ``tests/test_adjoint.py`` asserts today, kept here so
    that issue #20 pins the documented behaviour of the *defaults* rather than
    silently changing it. Neither operator is passed a flag: the point is what
    a caller who says nothing gets.

    On EDA2 the image is masked to the unit disc first, which is precisely the
    restriction the docstring claims -- with the mask the identity holds at
    4e-16, without it at 2.7e-1 (the test above).
    """
    problem = _problem(tel, zenith_deg, eps=1e-6)
    n_grid = _n_grid(problem.plan)
    image = np.asarray(problem.image, dtype=np.float64)
    if mask_to_disc:
        assert int((n_grid <= 0.0).sum()) > 0, "EDA2 must have pixels outside the disc"
        image = np.where(n_grid > 0.0, image, 0.0)
    else:
        assert int((n_grid <= 0.0).sum()) == 0, (
            f"{tel.name} was expected to lie entirely inside the unit disc"
        )
    residual, lhs, rhs = _dot_product_residual(problem, image=image, image_rhs=image * n_grid)
    assert residual < IDENTITY_TOL_F64, (
        f"{tel.name} zen={zenith_deg}: default-pair residual {residual:.3e} in the n*x form "
        f"exceeds {IDENTITY_TOL_F64:.1e} (lhs={lhs!r}, rhs={rhs!r}). The defaults' documented "
        "behaviour has changed."
    )


# ---------------------------------------------------------------------------
# 4. what each flag *means*, independent of the identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_forward_with_divide_by_n_ignores_every_pixel_outside_the_disc(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """``divide_by_n=True`` multiplies by **zero** outside the unit disc.

    Stated as insensitivity, which is the sharp version: two images that
    differ only outside the disc must give bit-identical visibilities under
    ``True``. The same pair under ``False`` must differ materially -- without
    that half the test would also pass on a forward that zeroed the whole
    image, or on a fixture with nothing outside the disc.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    outside = _assert_outside_disc_is_loaded(problem)
    rng = np.random.default_rng(2024)
    perturbed = np.array(problem.image, copy=True)
    perturbed[outside] += rng.standard_normal(int(outside.sum())).astype(perturbed.dtype) * 5.0

    with_flag = _forward(problem, problem.image, divide_by_n=True)
    with_flag_perturbed = _forward(problem, perturbed, divide_by_n=True)
    # Same-executable comparison, so held at round-off rather than bit-for-bit:
    # the two calls run one executable on two images that the mask makes
    # identical, and on a nondeterministic backend that still does not
    # guarantee identical bits. The claim survives intact because the control
    # leg below moves by more than 1e-2 on exactly the same perturbation -- the
    # discriminating power is in the ratio, not in the exactness.
    is_f64 = np.dtype(jnp.dtype(dtype)) == np.float64
    insensitivity = _relative_difference(with_flag_perturbed, with_flag)
    tol = SAME_EXECUTABLE_TOL_F64 if is_f64 else SAME_EXECUTABLE_TOL_F32
    assert insensitivity < tol, (
        f"dirty2vis(divide_by_n=True) moved by {insensitivity:.3e} (tol {tol:.1e}) when "
        "only the pixels OUTSIDE the unit disc changed. Those pixels must be multiplied "
        "by zero (n < 0 there, so 1/n is not a gain the measurement equation defines); "
        "leaving them at the analytic extension is what breaks the adjoint pair on wide "
        "fields."
    )
    without_flag = _forward(problem, problem.image, divide_by_n=False)
    without_flag_perturbed = _forward(problem, perturbed, divide_by_n=False)
    contrast = np.linalg.norm(without_flag - without_flag_perturbed) / np.linalg.norm(without_flag)
    assert insensitivity * SAME_EXECUTABLE_SEPARATION < contrast, (
        f"divide_by_n=True moved {insensitivity:.3e} under the perturbation while "
        f"divide_by_n=False moved {contrast:.3e} -- less than "
        f"{SAME_EXECUTABLE_SEPARATION:g}x apart, so 'insensitive' and 'sensitive' are no "
        "longer distinguishable on this fixture"
    )
    assert contrast > 1e-2, (
        f"the control leg moved by only {contrast:.3e}: divide_by_n=False must stay "
        "sensitive to the outside-disc pixels (it uses the analytic extension there), "
        "otherwise the insensitivity asserted above is a property of the fixture"
    )
    assert np.linalg.norm(with_flag - without_flag) / np.linalg.norm(without_flag) > 1e-2, (
        "the two flag values give the same forward output on a full-sky fixture"
    )


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_the_forward_selects_the_zero_rather_than_multiplying_by_one(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Outside the disc the image is *selected away*, not scaled to zero.

    ``_dirty2vis_jit`` writes ``jnp.where(inside, image / safe_n, 0.0)`` and
    its comment says the ``where`` is there so the zeros are exact "regardless
    of what the caller put there". That reason was stated and never tested:
    multiplying by a zeroed reciprocal passes every other test in this module,
    because on finite input ``x * 0.0`` and ``select(False, ..., 0.0)`` agree
    exactly.

    They differ on non-finite input, which is the whole claim. With the
    outside-disc pixels set to NaN a select returns finite visibilities -- the
    NaNs never enter the transform -- while a multiply returns NaN everywhere,
    since ``NaN * 0.0`` is NaN and the NUFFT spreads it across every output.
    A caller handing in a padded image with NaN outside the field is doing
    nothing unreasonable, and under ``divide_by_n=True`` those pixels are
    documented as ignored.

    Compared against the same call on a zero-filled image at
    ``SAME_EXECUTABLE_TOL_*`` rather than bit-for-bit: two runs of one
    executable are not bit-identical on every backend.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    outside = _assert_outside_disc_is_loaded(problem)

    with_nan = np.array(problem.image, copy=True)
    with_nan[..., outside] = np.nan
    zero_filled = np.array(problem.image, copy=True)
    zero_filled[..., outside] = 0.0

    got = _forward(problem, with_nan, divide_by_n=True)
    assert np.all(np.isfinite(np.asarray(got))), (
        "dirty2vis(divide_by_n=True) returned non-finite visibilities for an image "
        "carrying NaN OUTSIDE the unit disc. Those pixels are multiplied by zero by "
        "contract, so they must be selected away (jnp.where) rather than scaled away "
        "(image * zeroed_reciprocal) -- NaN * 0.0 is NaN, and the NUFFT then spreads it "
        "over every output visibility."
    )
    want = _forward(problem, zero_filled, divide_by_n=True)
    is_f64 = np.dtype(jnp.dtype(dtype)) == np.float64
    tol = SAME_EXECUTABLE_TOL_F64 if is_f64 else SAME_EXECUTABLE_TOL_F32
    diff = _relative_difference(got, want)
    assert diff < tol, (
        f"the NaN-outside image gave visibilities {diff:.3e} from the zero-filled one "
        f"(tol {tol:.1e}): the outside-disc pixels are reaching the transform"
    )


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_forward_with_divide_by_n_applies_one_over_n_inside_the_disc(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Inside the disc the flag is exactly a ``1/n`` pre-multiplication.

    The companion to the test above: insensitivity outside would also be
    satisfied by a forward that zeroed everything, so the inside-disc half has
    to be pinned too. Compared against the *other* flag value fed the explicitly
    scaled image, so the two paths differ only in where the factor is applied.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    _assert_outside_disc_is_loaded(problem)
    n_grid = _n_grid(problem.plan)
    scaled = np.where(
        n_grid > 0.0,
        np.asarray(problem.image, dtype=np.float64) / np.where(n_grid > 0.0, n_grid, 1.0),
        0.0,
    ).astype(problem.image.dtype)

    got = _forward(problem, problem.image, divide_by_n=True).astype(np.complex128)
    want = _forward(problem, scaled, divide_by_n=False).astype(np.complex128)
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    tol = ORACLE_TOL_F64 if np.dtype(jnp.dtype(dtype)) == np.float64 else ORACLE_TOL_F32
    assert err < tol, (
        f"dirty2vis(divide_by_n=True) differs by {err:.3e} from dirty2vis on an image "
        f"pre-multiplied by the masked 1/n (tol {tol:.1e}). The flag must apply 1/n to the "
        "IMAGE inside the disc -- applying n instead, or applying it to the output "
        "visibilities, both land here."
    )


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_adjoint_without_divide_by_n_keeps_the_outside_disc_pixels(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """``divide_by_n=False`` returns the analytic extension, not zeros.

    The default (``True``) zeroes every pixel with ``n <= 0`` -- there is no
    ``1/n`` to apply there -- and those zeros are exactly what breaks the
    mixed pair against a forward that keeps evaluating the extension. With the
    flag off the adjoint must return those values.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    outside = _assert_outside_disc_is_loaded(problem)

    divided = _adjoint(problem, divide_by_n=True)[0]
    undivided = _adjoint(problem, divide_by_n=False)[0]

    np.testing.assert_array_equal(
        divided[outside],
        np.zeros(int(outside.sum()), dtype=divided.dtype),
        err_msg="divide_by_n=True must still zero the pixels outside the unit disc",
    )
    inside_scale = float(np.max(np.abs(divided)))
    outside_signal = float(np.max(np.abs(undivided[outside])))
    assert outside_signal > 1e-3 * inside_scale, (
        f"vis2dirty(divide_by_n=False) returned {outside_signal:.3e} at most outside the "
        f"unit disc against an in-disc scale of {inside_scale:.3e}: those pixels are still "
        "being zeroed, so the flag is not returning the analytic-extension result and the "
        "pair stays non-adjoint on wide fields"
    )


@requires_x64
def test_adjoint_without_divide_by_n_matches_the_exact_dft_outside_the_disc() -> None:
    """The outside-disc values are *the right* values, not merely non-zero.

    Checked against an exact DFT written in the repo's sign convention with
    ``n - 1`` on the analytic extension and no division -- an oracle
    independent of ``src/``. Held to the repo's ``2 * eps`` DFT contract, on
    the outside-disc pixels alone (they carry about half the image norm here,
    so restricting the ratio to them does not inflate it).
    """
    eps = 1e-6
    problem = _problem(EDA2, 0.0, eps=eps, uvw_seed=1, data_seed=11)
    outside = _outside_disc(problem.plan)
    assert int(outside.sum()) > 0

    got = _adjoint(problem, divide_by_n=False)[0]
    want = _reference_adjoint_no_divide(
        problem.vis, problem.uvw, problem.freq, problem.shape, problem.pixsize_l, problem.pixsize_m
    )
    err_out = np.linalg.norm(got[outside] - want[outside]) / np.linalg.norm(want[outside])
    assert err_out < DFT_TOL_FACTOR * eps, (
        f"outside-disc relative error {err_out:.3e} exceeds {DFT_TOL_FACTOR * eps:.3e}: "
        "vis2dirty(divide_by_n=False) does not reproduce the analytic-extension adjoint "
        "there"
    )
    err_all = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err_all < DFT_TOL_FACTOR * eps, (
        f"whole-image relative error {err_all:.3e} exceeds {DFT_TOL_FACTOR * eps:.3e}"
    )


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_the_two_adjoint_flag_values_differ_by_exactly_one_over_n_inside_the_disc(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Inside the disc, ``False`` is ``True`` times ``n``: nothing else moved.

    Pins that the flag changes the output factor and *only* the output factor
    -- an implementation that (say) also dropped the ``w0`` screen conjugation
    on the ``False`` path would satisfy the outside-disc test above and fail
    here.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    n_grid = _n_grid(problem.plan)
    inside = n_grid > 0.0
    assert int(inside.sum()) > 0

    divided = _adjoint(problem, divide_by_n=True)[0].astype(np.float64)
    undivided = _adjoint(problem, divide_by_n=False)[0].astype(np.float64)
    got = undivided[inside] / n_grid[inside]
    want = divided[inside]
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    tol = ORACLE_TOL_F64 if np.dtype(jnp.dtype(dtype)) == np.float64 else ORACLE_TOL_F32
    assert err < tol, (
        f"inside the disc, vis2dirty(divide_by_n=False)/n differs from "
        f"vis2dirty(divide_by_n=True) by {err:.3e} (tol {tol:.1e}): the flag changed more "
        "than the 1/n output factor"
    )


# ---------------------------------------------------------------------------
# 5. ducc0 parity for the two new flag combinations (black-box oracle)
# ---------------------------------------------------------------------------


@requires_x64
@pytest.mark.parametrize("w_strategy", ["dense_scan", "windowed_scan"])
@pytest.mark.parametrize("eps", [1e-4, 1e-6])
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_ducc_parity_for_the_new_flag_combinations(
    short_telescope_pointing: tuple[Telescope, float],
    op: str,
    eps: float,
    w_strategy: str,
) -> None:
    """``dirty2vis(divide_by_n=True)`` and ``vis2dirty(divide_by_n=False)``.

    The two combinations the operators could not express before issue #20,
    against ducc0 configured the same way, at the repo's ``3 * eps`` bound
    (``tests/test_against_ducc.py``). ducc0 is used through its public Python
    API only. The fixture set includes EDA2 at 120 degrees, so the parity is
    asserted on a field that runs past the unit disc, where the two libraries
    have to agree about the outside-disc convention as well as the accuracy.
    """
    ducc0_wgridder = pytest.importorskip("ducc0.wgridder")
    tel, zen_deg = short_telescope_pointing
    problem = _problem(tel, zen_deg, eps=eps, uvw_seed=0, data_seed=7)
    common = dict(
        uvw=problem.uvw,
        freq=problem.freq,
        pixsize_x=problem.pixsize_l,
        pixsize_y=problem.pixsize_m,
        epsilon=eps,
        do_wgridding=True,
        nthreads=1,
    )
    if op == "dirty2vis":
        got = _forward(problem, problem.image, divide_by_n=True, w_strategy=w_strategy)
        want = ducc0_wgridder.dirty2vis(
            dirty=np.ascontiguousarray(problem.image, dtype=np.float64),
            divide_by_n=True,
            **common,
        )
    else:
        got = _adjoint(problem, divide_by_n=False, w_strategy=w_strategy)[0]
        want = ducc0_wgridder.vis2dirty(
            vis=problem.vis,
            npix_x=problem.shape[0],
            npix_y=problem.shape[1],
            divide_by_n=False,
            **common,
        )
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < DUCC_TOL_FACTOR * eps, (
        f"{tel.name} zen={zen_deg} eps={eps:g} {w_strategy} {op}: relative error {err:.3e} "
        f"exceeds {DUCC_TOL_FACTOR:g}*eps={DUCC_TOL_FACTOR * eps:.3e} for the new "
        "divide_by_n combination"
    )


@requires_x64
@pytest.mark.parametrize("geometry", _ANISO_GEOMETRIES)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_ducc_parity_on_an_asymmetric_grid(op: str, geometry: dict[str, Any]) -> None:
    """ducc0 parity with ``divide_by_n=True`` where ``n`` is not symmetric.

    The oracle above is an internal consistency statement -- ``True`` against
    ``False`` plus a factor this file builds. This is the external one, and it
    is what turns "the two flags agree with each other" into "the diagonal is
    the one ducc0 computes". Both operators run with ``divide_by_n=True``,
    since that is the setting that uses the diagonal at all; the adjoint's
    ``False`` path applies no factor and would pass a transposed one.

    Measured on the square anisotropic plan: a diagonal indexed ``[m, l]``
    takes ``vis2dirty(divide_by_n=True)`` from 3.5e-07 against ducc0 -- inside
    the repo's ``3 * eps`` contract -- to 1.0e+00.
    """
    ducc0_wgridder = pytest.importorskip("ducc0.wgridder")
    eps = 1e-6
    problem = _problem(EDA2, 0.0, eps=eps, **geometry)
    common = dict(
        uvw=problem.uvw,
        freq=problem.freq,
        pixsize_x=problem.pixsize_l,
        pixsize_y=problem.pixsize_m,
        epsilon=eps,
        do_wgridding=True,
        nthreads=1,
    )
    if op == "dirty2vis":
        got = _forward(problem, problem.image, divide_by_n=True)
        want = ducc0_wgridder.dirty2vis(
            dirty=np.ascontiguousarray(problem.image, dtype=np.float64),
            divide_by_n=True,
            **common,
        )
    else:
        got = _adjoint(problem, divide_by_n=True)[0]
        want = ducc0_wgridder.vis2dirty(
            vis=problem.vis,
            npix_x=problem.shape[0],
            npix_y=problem.shape[1],
            divide_by_n=True,
            **common,
        )
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < DUCC_TOL_FACTOR * eps, (
        f"{op} shape={problem.shape} pixsize_l={problem.pixsize_l:.6g} "
        f"pixsize_m={problem.pixsize_m:.6g}: relative error {err:.3e} exceeds "
        f"{DUCC_TOL_FACTOR:g}*eps={DUCC_TOL_FACTOR * eps:.3e} against ducc0 at the same "
        "asymmetric geometry"
    )


# ---------------------------------------------------------------------------
# 6. composition with the flags already in the plan
# ---------------------------------------------------------------------------


def _matrix_run(
    op: str,
    problem: _Problem,
    w_strategy: str,
    channel_strategy: str,
    divide_by_n: bool,
) -> np.ndarray:
    """One operator call in the section-6 matrix, widened to float64.

    Widened *after* the call, never before: the plan's own precision does the
    arithmetic, and the comparison is done in float64 so the test's own
    subtraction and norms are not what the tolerance is measuring.
    """
    if op == "dirty2vis":
        return _forward(
            problem,
            problem.image,
            divide_by_n=divide_by_n,
            w_strategy=w_strategy,
            channel_strategy=channel_strategy,
        ).astype(np.complex128)
    return _adjoint(
        problem,
        divide_by_n=divide_by_n,
        w_strategy=w_strategy,
        channel_strategy=channel_strategy,
    ).astype(np.float64)


_REFERENCE_CACHE: dict[tuple, np.ndarray] = {}


def _matrix_reference(
    op: str,
    problem: _Problem,
    divide_by_n: bool,
    hermitian: bool,
    dtype: DTypeLike,
    eps: float,
) -> np.ndarray:
    """The ``(dense_scan, scan)`` reference for a section-6 cell, memoised.

    **Pinned on both strategy axes.** The cell varies ``w_strategy`` and
    ``channel_strategy``; the reference does not. That is the whole point: a
    reference computed with the cell's own strategies would move with any
    defect conditioned on them and the comparison would be vacuous -- which is
    precisely how a channel-strategy-specific defect passed the pre-existing
    matrix.

    Memoised because 128 cells share only 16 distinct references (2 operators
    x 2 flags x 2 fold settings x 2 precisions), and recomputing each one
    eight times is the difference between a fast module and a slow one. The
    cache key names every input the reference depends on; ``hermitian`` and
    ``dtype`` are in it even though they are already implied by ``problem``,
    because a stale entry here would silently weaken every cell that read it.
    """
    key = (op, divide_by_n, hermitian, np.dtype(jnp.dtype(dtype)).name, eps, problem.plan.n_chan)
    hit = _REFERENCE_CACHE.get(key)
    if hit is None:
        hit = _matrix_run(op, problem, "dense_scan", "scan", divide_by_n)
        _REFERENCE_CACHE[key] = hit
    return hit


@pytest.mark.parametrize("dtype, eps, dot_tol", _PRECISIONS)
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize("channel_strategy", CHANNEL_STRATEGIES)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_both_flag_values_compose_with_every_strategy_and_fold_setting(
    op: str,
    w_strategy: str,
    channel_strategy: str,
    hermitian: bool,
    divide_by_n: bool,
    dtype: DTypeLike,
    eps: float,
    dot_tol: float,
) -> None:
    """Both loops x both fold settings x both flags x both precisions.

    Enumerated, not sampled, on every axis: ``W_STRATEGIES`` (four),
    ``CHANNEL_STRATEGIES`` (both), ``hermitian`` (both), the flag (both) and
    ``_PRECISIONS`` (both), because "the flag composes with the strategy
    family" is a universally quantified claim and one witness would not gate
    it. 128 cells per run; the plans are cached and the two reference outputs
    are memoised across cells (:func:`_matrix_reference`), so the marginal
    cost of a cell is the one call it is actually about.

    Three references, because the three axes are different kinds of
    equivalence and the repo already prices them differently:

      * against ``dense_scan`` **at the same fold setting and the same
        precision**: the four strategies are the same operator in a different
        accumulation order, so the 1e-11 strategy-equivalence bound applies in
        float64 (``tests/test_strategies_equivalent.py``); the float32 leg
        cannot reach that and is held at ``DOT_TOL_F32``;
      * the **same reference for both channel strategies**: the reference is
        pinned to ``channel_strategy="scan"`` whatever the cell uses, so a
        defect that mishandles the vmap channel loop cannot hide by moving the
        reference with it. This is the check that makes the channel axis
        independently falsifiable, and it is the *only* one that catches a
        defect applied consistently to both operators (which stays adjoint,
        and so is invisible to the identity matrix below);
      * against ``dense_scan`` on an ``hermitian=False`` plan: folded and
        unfolded are different approximations of the same operator and agree
        to ``4 * eps``, the triangle inequality on the ``2 * eps`` DFT
        contract each meets (``tests/test_hermitian.py``,
        ``CROSS_PATH_TOL_FACTOR``).

    All are needed. A flag ignored on *every* folded strategy would keep the
    same-fold check green (the reference is wrong in the same way) and is
    caught only by the cross-fold one; a flag mishandled by one strategy or
    one channel loop is caught only by the same-fold one. The fold is the
    interesting geometric axis -- it conjugates the forward's *output* and the
    adjoint's *input*, while ``divide_by_n`` acts on the image end of both --
    and the channel loop is the interesting structural one, being the only
    axis on which a defect can be introduced without touching the maths.

    Two channels (``_MATRIX_N_CHAN``): at one channel the scan and vmap
    channel loops both run their body once and the axis is vacuous.
    """
    unfolded = _problem(EDA2, 0.0, eps=eps, dtype=dtype, hermitian=False, n_chan=_MATRIX_N_CHAN)
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, hermitian=hermitian, n_chan=_MATRIX_N_CHAN)
    assert problem.plan.hermitian is hermitian
    assert problem.plan.n_chan == _MATRIX_N_CHAN and problem.image.ndim == 3, (
        "the channel axis must be real: a single-channel plan runs both channel "
        "strategies' bodies exactly once and cannot falsify anything about them"
    )
    n_negative = int((problem.uvw[:, 2] < 0).sum())
    assert 0 < n_negative < problem.uvw.shape[0], (
        "the fixture must have mixed-sign w or the hermitian=True leg is the "
        "hermitian=False leg under another name"
    )

    got = _matrix_run(op, problem, w_strategy, channel_strategy, divide_by_n)
    # Reference pinned to (dense_scan, scan) on *both* strategy axes.
    same_fold = _matrix_reference(op, problem, divide_by_n, hermitian, dtype, eps)
    cross_fold = _matrix_reference(op, unfolded, divide_by_n, False, dtype, eps)

    label = (
        f"{op} w_strategy={w_strategy} channel_strategy={channel_strategy} "
        f"hermitian={hermitian} divide_by_n={divide_by_n} "
        f"dtype={np.dtype(jnp.dtype(dtype)).name}"
    )
    strategy_err = np.linalg.norm(got - same_fold) / np.linalg.norm(same_fold)
    assert strategy_err < dot_tol, (
        f"{label}: relative difference {strategy_err:.3e} against (dense_scan, scan) on "
        f"the same plan exceeds the {dot_tol:.1e} strategy-equivalence bound -- "
        "divide_by_n must compose with every w_strategy AND every channel_strategy, not "
        "just the dense scan ones."
    )
    fold_err = np.linalg.norm(got - cross_fold) / np.linalg.norm(cross_fold)
    fold_bound = CROSS_PATH_TOL_FACTOR * eps
    assert fold_err < fold_bound, (
        f"{label}: relative difference {fold_err:.3e} against (dense_scan, scan) on an "
        f"unfolded plan exceeds {fold_bound:.3e} ({CROSS_PATH_TOL_FACTOR:g}*eps) -- the "
        "flag must mean the same thing on a folded plan as on an unfolded one, and be "
        "applied on the image side of the fold's per-row conjugation rather than skipped "
        "there."
    )


# The guard's own bound. float64 measures 7.622e-16; float32 measures 3.380e-07,
# which is not round-off in the comparison but the plan's grid genuinely being
# float32 while ``_independent_one_over_n`` is always float64 -- a difference of
# order the float32 epsilon, amplified near the disc edge where n -> 0. That
# mismatch is exactly what this guard exists to bound, so the float32 leg is
# the one that matters and it must not be skipped.
GUARD_TOL_F64 = 1e-13
GUARD_TOL_F32 = 1e-5


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_the_independent_one_over_n_matches_the_plans_own_grid(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Guard on the oracle: the two constructions must agree before it is used.

    :func:`_independent_one_over_n` is built from ``(l, m)`` alone while the
    operators build their diagonal from ``plan.n_minus_1``. They should be the
    same function of the same geometry, but they are computed by different code
    along different paths, and the interesting place is the disc boundary --
    one pixel where a tiny positive ``n`` in one construction is a zero in the
    other would turn ``1/n`` into a large number on one side and a zero on the
    other.

    Pinned here, once, rather than discovered as unexplained noise spread over
    the oracle's 64 cells. Failing here means the *oracle* needs looking at;
    failing there means the implementation does.

    Run on **both** precision legs, and the float32 one is the point: there the
    plan's grid is float32 while this file's construction is float64, so the
    two genuinely differ (3.380e-07 measured, against 7.622e-16 in float64).
    Skipping the float32 leg would leave the oracle's 32 float32 cells running
    against an unchecked factor -- the precise mismatch this guard is for.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, n_chan=_MATRIX_N_CHAN)
    independent = _independent_one_over_n(problem)
    n_grid = _n_grid(problem.plan)
    inside_plan = n_grid > 0.0

    np.testing.assert_array_equal(
        independent > 0.0,
        inside_plan,
        err_msg=(
            "the (l, m)-only disc mask and the operators' n > 0 mask disagree, so the "
            "oracle would be scoring a different set of pixels than the implementation "
            "divides"
        ),
    )
    assert 0 < int(inside_plan.sum()) < inside_plan.size
    from_plan = np.where(inside_plan, 1.0 / np.where(inside_plan, n_grid, 1.0), 0.0)
    rel = np.abs(independent - from_plan)[inside_plan] / np.abs(from_plan)[inside_plan]
    is_f64 = np.dtype(jnp.dtype(dtype)) == np.float64
    guard_tol = GUARD_TOL_F64 if is_f64 else GUARD_TOL_F32
    assert float(rel.max()) < guard_tol, (
        f"the two 1/n constructions differ by {float(rel.max()):.3e} relative inside the "
        "disc; the oracle is only an oracle while it agrees with the geometry the plan "
        "was built from"
    )


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize("channel_strategy", CHANNEL_STRATEGIES)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_every_cell_applies_the_documented_one_over_n_diagonal(
    op: str,
    w_strategy: str,
    channel_strategy: str,
    hermitian: bool,
    dtype: DTypeLike,
    eps: float,
    _dot_tol: float,
) -> None:
    """Per cell: the flag applies *this* diagonal, not merely a self-adjoint one.

    The third of section 6's three tests, and the only one that can fail when
    the other two cannot. Both of those are relational -- the identity relates
    the two operators to each other, the composition test relates a cell to a
    reference cell -- so a diagonal that is wrong in the same way everywhere
    satisfies both. Concretely: replace ``1/n`` with ``1/n^2`` on both
    operators and the pair is still exactly adjoint (a real diagonal is
    self-adjoint whatever it contains) *and* every cell still agrees with a
    reference that shares the substitution. Nothing above notices.

    So this states the flag's meaning directly, per cell, against a ``1/n``
    built from ``(l, m)`` in :func:`_independent_one_over_n`:

      * forward: ``dirty2vis(x, True) == dirty2vis(x * (1/n), False)``
      * adjoint: ``vis2dirty(v, True) == (1/n) * vis2dirty(v, False)``

    Both sides use the *same* ``w_strategy`` and ``channel_strategy``, so the
    two runs differ only in where the factor is applied -- which is what makes
    this a statement about the flag rather than about the strategy. Section 4
    makes the same comparison on the default strategy and single channel; this
    is that check lifted onto every cell of the matrix, which is where a
    strategy- or channel-conditioned substitution would live.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, hermitian=hermitian, n_chan=_MATRIX_N_CHAN)
    _assert_outside_disc_is_loaded(problem)
    factor = _independent_one_over_n(problem)
    kwargs: dict[str, Any] = {"w_strategy": w_strategy, "channel_strategy": channel_strategy}

    if op == "dirty2vis":
        got = _forward(problem, problem.image, divide_by_n=True, **kwargs).astype(np.complex128)
        # Scaled in float64 and narrowed once, so the comparison is not limited
        # by the test's own arithmetic on a float32 plan.
        scaled = (np.asarray(problem.image, dtype=np.float64) * factor).astype(problem.image.dtype)
        want = _forward(problem, scaled, divide_by_n=False, **kwargs).astype(np.complex128)
    else:
        got = _adjoint(problem, divide_by_n=True, **kwargs).astype(np.float64)
        want = factor * _adjoint(problem, divide_by_n=False, **kwargs).astype(np.float64)

    is_f64 = np.dtype(jnp.dtype(dtype)) == np.float64
    tol = ORACLE_TOL_F64 if is_f64 else ORACLE_TOL_F32
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < tol, (
        f"{op} w_strategy={w_strategy} channel_strategy={channel_strategy} "
        f"hermitian={hermitian} dtype={np.dtype(jnp.dtype(dtype)).name}: "
        f"divide_by_n=True differs by {err:.3e} (tol {tol:.1e}) from the same call with "
        "the flag off and an independently built masked 1/n applied by hand. The flag "
        "must apply exactly that diagonal -- a different but still self-adjoint one "
        "(1/n^2, 1/(n+c), the shifted grid) passes both other section-6 tests and fails "
        "only here."
    )


@requires_x64
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_the_diagonal_survives_an_explicit_nthreads(op: str) -> None:
    """The oracle relation once at ``nthreads=0``, the value nothing else uses.

    ``nthreads`` is a static argument on the same JIT boundary as
    ``divide_by_n``, and ``_resolve_nthreads`` returns ``1`` for every fixture
    in this module (all are far below ``_NTHREADS_SMALL_N_ROWS``). So a defect
    conditioned on ``nthreads != 1`` would pass every other test here. This is
    one cell rather than a fifth axis across the matrix: threading changes how
    FINUFFT accumulates, not the image-side diagonal, so the interesting
    question is whether the flag survives a different value at all -- not
    whether it survives one per strategy.

    ``0`` means "let FINUFFT decide" and is the one other value
    ``_resolve_nthreads`` can return.
    """
    problem = _eda2_full_sky()
    _assert_outside_disc_is_loaded(problem)
    factor = _independent_one_over_n(problem)
    kwargs: dict[str, Any] = {"nthreads": 0}

    if op == "dirty2vis":
        got = _forward(problem, problem.image, divide_by_n=True, **kwargs).astype(np.complex128)
        scaled = (np.asarray(problem.image, dtype=np.float64) * factor).astype(problem.image.dtype)
        want = _forward(problem, scaled, divide_by_n=False, **kwargs).astype(np.complex128)
    else:
        got = _adjoint(problem, divide_by_n=True, **kwargs).astype(np.float64)
        want = factor * _adjoint(problem, divide_by_n=False, **kwargs).astype(np.float64)

    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < ORACLE_TOL_F64, (
        f"{op} at nthreads=0: divide_by_n=True differs by {err:.3e} from the "
        f"independently scaled call (tol {ORACLE_TOL_F64:.1e})"
    )


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
@pytest.mark.parametrize("geometry", _ANISO_GEOMETRIES)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_the_diagonal_is_indexed_l_by_m_and_not_m_by_l(
    op: str,
    geometry: dict[str, Any],
    dtype: DTypeLike,
    eps: float,
    _dot_tol: float,
) -> None:
    """The oracle again, on a grid where ``n`` is *not* transpose-symmetric.

    Everything else in this module runs on square plans with isotropic pixels,
    where ``plan.n_minus_1`` equals its own transpose bit-for-bit. A diagonal
    indexed ``[m, l]`` instead of ``[l, m]`` is therefore literally the same
    array there, and all four of this module's mutation classes -- including
    the ``(l, m)``-built oracle -- pass with the transpose in place. See
    ``_ANISO_PIXSIZE_M_RATIO`` for the measurement.

    Same relation as
    ``test_every_cell_applies_the_documented_one_over_n_diagonal``, on the two
    geometries that break the symmetry. One ``w_strategy`` and one
    ``channel_strategy`` are enough here and the enumeration is deliberately
    not repeated: the factor is applied outside both loops (ahead of the
    strategy branch in the forward, after it in the adjoint), so the axis
    assignment cannot depend on either. What this cell adds is geometry, not
    dispatch.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, **geometry)
    _assert_outside_disc_is_loaded(problem)
    n_l, n_m = problem.shape
    assert (n_l != n_m) or (problem.pixsize_l != problem.pixsize_m), (
        "this geometry is square and isotropic, so n is transpose-symmetric and the "
        "test cannot tell an [l, m] diagonal from an [m, l] one"
    )
    factor = _independent_one_over_n(problem)
    kwargs: dict[str, Any] = {"w_strategy": "dense_scan", "channel_strategy": "scan"}

    if op == "dirty2vis":
        got = _forward(problem, problem.image, divide_by_n=True, **kwargs).astype(np.complex128)
        scaled = (np.asarray(problem.image, dtype=np.float64) * factor).astype(problem.image.dtype)
        want = _forward(problem, scaled, divide_by_n=False, **kwargs).astype(np.complex128)
    else:
        got = _adjoint(problem, divide_by_n=True, **kwargs).astype(np.float64)
        want = factor * _adjoint(problem, divide_by_n=False, **kwargs).astype(np.float64)

    is_f64 = np.dtype(jnp.dtype(dtype)) == np.float64
    tol = ORACLE_TOL_F64 if is_f64 else ORACLE_TOL_F32
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < tol, (
        f"{op} shape={problem.shape} pixsize_l={problem.pixsize_l:.6g} "
        f"pixsize_m={problem.pixsize_m:.6g} dtype={np.dtype(jnp.dtype(dtype)).name}: "
        f"divide_by_n=True differs by {err:.3e} (tol {tol:.1e}) from the independently "
        "scaled call. On an asymmetric grid this catches a diagonal indexed [m, l] "
        "instead of [l, m] -- which every square isotropic fixture in this repository "
        "accepts silently."
    )


@pytest.mark.parametrize("dtype, eps, dot_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize("channel_strategy", CHANNEL_STRATEGIES)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_the_pair_stays_adjoint_for_every_strategy_and_fold_setting(
    w_strategy: str,
    channel_strategy: str,
    hermitian: bool,
    divide_by_n: bool,
    dtype: DTypeLike,
    eps: float,
    dot_tol: float,
) -> None:
    """The identity itself, over the full strategy x fold x flag enumeration.

    Section 2 gates the identity on the shipped defaults for those axes; this
    gates it everywhere, which is what issue #21 needs -- a ``custom_vjp`` is
    chosen per call, and a caller who passes ``windowed_vmap`` with
    ``channel_strategy="vmap"`` on a folded float32 plan must get the same
    guarantee as one who passes nothing.

    64 cells: 4 ``w_strategy`` x 2 ``channel_strategy`` x 2 ``hermitian`` x 2
    flags x 2 precisions, on a two-channel plan. Every axis the operators
    dispatch on is enumerated, because "the pair is adjoint" is the claim
    issue #21 builds on and it is stated without qualification.

    **This matrix is necessary but not sufficient on its own**, and the reason
    is worth stating where the next reader will look for it: a defect applied
    *consistently to both operators* -- say, both silently ignoring
    ``divide_by_n`` on one channel strategy -- leaves the pair perfectly
    adjoint, because it is then simply the other flag's pair. The identity
    cannot see it by construction. What catches it is the pinned-reference
    check in ``test_both_flag_values_compose_with_every_strategy_and_fold_
    setting`` above, whose reference stays on ``(dense_scan, scan)`` while the
    cell moves. The two tests are a pair; neither alone gates the claim.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, hermitian=hermitian, n_chan=_MATRIX_N_CHAN)
    assert problem.plan.n_chan == _MATRIX_N_CHAN and problem.image.ndim == 3
    residual, lhs, rhs = _dot_product_residual(
        problem,
        divide_by_n=divide_by_n,
        w_strategy=w_strategy,
        channel_strategy=channel_strategy,
    )
    assert residual < dot_tol, (
        f"w_strategy={w_strategy} channel_strategy={channel_strategy} "
        f"hermitian={hermitian} divide_by_n={divide_by_n} "
        f"dtype={np.dtype(jnp.dtype(dtype)).name}: "
        f"dot-product residual {residual:.3e} exceeds {dot_tol:.1e} "
        f"(lhs={lhs!r}, rhs={rhs!r})"
    )


# ---------------------------------------------------------------------------
# 7. the flag is static: JIT / grad traceability (issue #21's precondition)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_both_flag_values_stay_traceable_through_jit_and_grad(
    divide_by_n: bool, dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Both operators, both flag values, under ``jax.jit`` and ``jax.grad``.

    Issue #21 replaces the reverse-mode rule with a ``custom_vjp`` built on
    this pair, so the flag has to survive tracing (i.e. be a static argument,
    not a traced one) and produce finite gradients at both values.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    image = jnp.asarray(problem.image)
    vis = jnp.asarray(problem.vis)

    fwd = jax.jit(lambda im: dirty2vis(problem.plan, im, divide_by_n=divide_by_n))
    adj = jax.jit(lambda v: vis2dirty(problem.plan, v, divide_by_n=divide_by_n))
    assert np.all(np.isfinite(np.asarray(fwd(image))))
    assert np.all(np.isfinite(np.asarray(adj(vis))))

    grad_fwd = jax.grad(lambda im: jnp.sum(jnp.abs(fwd(im)) ** 2))(image)
    grad_adj = jax.grad(lambda v: jnp.sum(adj(v) ** 2))(vis.real)
    assert np.all(np.isfinite(np.asarray(grad_fwd)))
    assert np.all(np.isfinite(np.asarray(grad_adj)))
    assert float(np.linalg.norm(np.asarray(grad_fwd, dtype=np.float64))) > 0.0
