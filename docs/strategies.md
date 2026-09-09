# Strategies, threads and the memory/compute dial

How `w_strategy`, `channel_strategy`, `w_chunk` and `nthreads` interact, what
the `"auto"` heuristic decides and why, and which knob to reach for when a
problem does not fit. See [README.md](../README.md) for the summary.

<!--
Moved out of README.md by the 0.2.0 documentation pass. The text is unchanged:
every figure in it was verified against the code or the committed benchmark
JSON during issue #33, and several are recomputed by
tests/test_benchmark_claims.py, so it is moved rather than rewritten.
-->

## Strategy options

`w_strategy` selects how the w-plane loop is structured. There are six
canonical choices plus the `"auto"` resolver that picks between the first
four, and the two v0.1 names are kept as deprecated aliases:

| `w_strategy`      | Per-plane work               | Peak transient memory       | `grad` memory               | Notes                                          |
|-------------------|------------------------------|-----------------------------|-----------------------------|------------------------------------------------|
| `"dense_scan"`    | `n_rows * W^2`               | `O(image_size + n_rows)`    | 1.48-1.50x its own forward  | the default through v0.1.2; v0.1 `"scan"` is a deprecated alias. |
| `"dense_vmap"`    | `n_rows * W^2`               | `O(n_w * image_size)`       | 0.98-1.98x its own forward  | v0.1 `"vmap"` is a deprecated alias.           |
| `"windowed_scan"` | fwd `max_window_size * W^2`, adj `bucket_length * W^2` | `O(image_size + n_rows)`    | 1.46-1.50x its own forward  | v0.1.1; helps on adjoint when `n_w >> W`.      |
| `"windowed_vmap"` | fwd `max_window_size * W^2`, adj `bucket_length * W^2` | `O(n_w * image_size)`       | 0.98-1.98x its own forward  | v0.1.1; rare wins, mostly for completeness.    |
| `"chunked"`       | `n_rows * W^2`               | `O(w_chunk * image_size)`   | 1.03-1.96x its own forward  | 0.2.0 (#25); takes `w_chunk` (default 32).    |
| `"windowed_chunked"` | fwd `max_window_size * W^2`, adj `bucket_length * W^2` | `O(w_chunk * image_size)`   | 1.03-1.33x its own forward  | 0.2.0 (#25); the windowed half of the same knob. |
| `"auto"`          | resolves to one of the first four | matches the resolved choice | matches the resolved choice | v0.1.2; the default since #46. Platform-aware heuristic. |

`channel_strategy` is independently `"scan"` (default) or `"vmap"`.

### `w_chunk`: the memory/compute knob

The last two rows are not a fifth and sixth algorithm. They are the general
form of the other four, and the other four are values of `w_chunk`: the
w-plane loop scans over chunks of at most `w_chunk` planes with a `vmap`
inside each chunk, and `dense_scan` / `windowed_scan` **are** `w_chunk = 1`
while `dense_vmap` / `windowed_vmap` **are** `w_chunk = n_w`. They share the
code, not merely the answer, so a call at either end is bit-identical to the
old name for it on a deterministic backend *at equal `nthreads`*. The one
exception is worth stating: on a constant-w plan (`n_w == 1`) with
`n_rows >= 100_000` and `nthreads` left at its default, the `chunked` name
resolves to `1` thread and `dense_vmap` to `0`, and two thread counts is two
reductions — measured on a coplanar-uvw plan at 64², eps 1e-6, float64,
120 000 rows, `chunked(32)` and `dense_vmap` differ by a relative 1.8e-13
there. Pass an explicit `nthreads` if you depend on the identity.

```python
# 32 planes live at once instead of all n_w of them.
vis = dirty2vis(plan, image, w_strategy="chunked", w_chunk=32)
```

`w_chunk` is keyword-only and static (part of the JIT cache key, since it
sets the shape of every intermediate in the plane loop), must be a positive
`int`, and is clamped to `plan.n_w`. It is an *upper bound* on the planes
live at once, not an exact count: the loop runs `ceil(n_w / w_chunk)` chunks
of `ceil(n_w / n_chunks) <= w_chunk` planes, so at most `n_chunks - 1` planes
of padding are run and thrown away rather than up to `w_chunk - 1`.

Since 0.2.0 (#26) the windowed **adjoint** slices `bucket_length`, not
`max_window_size`: each channel's planes are sorted into at most four size
classes and each class is a sub-loop with its own static slice length, so a
plane whose window holds 8 rows does not read 155. Measured adjoint
`temp_size_in_bytes` on a 4000-row, 16², 138-plane fixture (float64,
eps 1e-6, `max_window_size` 1072 of 4000 rows), before against after:
`windowed_vmap` 2,932,224 B → 1,038,464 B and `windowed_chunked(32)`
1,143,392 B → 787,616 B. A scan holds one class's slice at a time, so
`windowed_scan`'s peak is set by the widest class and is unchanged
(106,568 B → 106,760 B).

**The windowed forward is unchanged**, and deliberately: bucketing it as
well ran 20.3x to 58.4x slower on a GH200 on the plans `auto` sends to
`windowed_vmap`, so it keeps the pre-#26 code, the `max_window_size` slice
and the `window_padding_overhead` figure. Its transients on that fixture are
byte-identical to the previous release's.

Those figures, and every other timing quoted for #26, are single-channel.
A slice length is a static shape, so the adjoint's channel axis can only be
mapped over channels that bucket identically, and at a realistic frequency
spread no two of them do — the compiled body count equals `n_chan`. Measured
on EDA2 off30 with `freq = f * linspace(0.95, 1.05, n_chan)`, the compile
time of a jitted `windowed_scan` adjoint is 0.083 / 0.224 / 0.397 / 0.656 s
at `n_chan` 1 / 4 / 8 / 16 against a flat 0.047-0.070 s for `dense_scan`;
run time is unchanged, and the adjoint's transient (which concatenates one
image cube per group) is 2,035,712 B at 16 channels against 1,202,376 B for
`dense_scan` on the same plan. The forward does not group and is unaffected.
On a wide spectral cube, prefer a dense or chunked strategy for the adjoint,
or expect a one-off compile of a second or two. Retuning `auto` for channel
count is [#34](https://github.com/chrisfinlay/jax-nufft/issues/34).

Whether the default clamps is a property of the plan. Measured over every
telescope in `tests/conftest.py` at both pointings (seed 0, eps 1e-6,
float64, hermitian, one channel), `n_w` is EDA2 11 / **56**, GH200_large
9 / 26, MWA_compact 8 / 12, MWA_extended 11 / **134**, MeerKAT 8 / 13
(zenith / off30) — so `w_chunk = 32` clamps to `dense_vmap` on eight of the
ten and runs a genuine 2-chunk loop (2 x 28, no padding) on EDA2 off30 or a
5-chunk loop (5 x 27, one padded plane) on MWA_extended off30, on the
other two. At the realistic sizes below it never clamps.

Measured on MWA_extended off30 (256&sup2;, 600 rows, `n_w = 134`, float64,
eps 1e-6, `nthreads=1`, single channel, `memory_analysis().temp_size_in_bytes`
on the CPU backend), in units of one complex image, with wall-clock as a
ratio against `dense_vmap` on the same machine (10-core Apple M-series, plan
and warm-up outside the timer, median of the interleaved rounds within a
pass, then the span over **7 independent passes** — 4 of 11 rounds and 3 of
15):

| | `dense_scan` | `chunked(8)` | `chunked(16)` | `chunked(32)` | `dense_vmap` |
|---|---:|---:|---:|---:|---:|
| forward temp | 2.01x | 9.08x | 16.15x | 28.26x | 135.23x |
| adjoint temp | 2.01x | 9.01x | 16.01x | 28.01x | 268.00x |
| forward time | 1.49-1.70x | 1.10-1.31x | 1.02-1.22x | 1.01-1.12x | 1.00x |
| adjoint time | 1.29-1.45x | 1.02-1.26x | 0.92-1.14x | 0.94-1.26x | 1.00x |

**Read the time rows to one significant figure, not two.** The suite carries
its own control — `chunked(1)` is the *same compiled program* as
`dense_scan`, so their ratio measures nothing but the instrument. Over the
same 7 passes that ratio spans 0.92-1.00 here, and 0.72-1.06 on the small
MeerKAT off30 fixture. A ±0.1 resolution on a 256&sup2; / 600-row problem is
what this machine gives, so the honest reading of the table is "`dense_scan`
costs about 1.5x, the chunked cells about 1.0-1.3x and get closer to
`dense_vmap` as `w_chunk` grows", not any particular two-decimal figure. The
memory rows carry no such caveat: `memory_analysis` is exact and reproduces
to the byte.

(The two-pass ranges published in the first cut of this table —
`chunked(8)` 1.23-1.26x forward, `chunked(16)` 1.09-1.17x — sit inside the
7-pass spans above but implied a precision of ±0.015 that the measurement
does not have. Widening them, and stating the control, is the correction.)

#### The CPU timing gate, and what it actually demonstrates

Issue #25's definition of done asks for `chunked(32)` within 1.2x of
`dense_vmap` on the five timing fixtures. Measured over the same 7 passes,
ratio against `dense_vmap` in the same run (median per pass, span across
passes):

| fixture | `n_w` | forward | adjoint | |
|---|---:|---:|---:|---|
| MWA_extended off30 | 134 | 1.01-1.12x | 0.94-1.26x | real chunking (5 x 27) |
| MeerKAT off30 | 13 | 0.96-1.08x | 0.96-1.02x | *clamped* — is `dense_vmap` |
| MWA_compact off30 | 12 | 0.97-1.09x | 0.93-1.03x | *clamped* — is `dense_vmap` |
| EDA2 zenith | 11 | 0.92-1.03x | 0.91-1.11x | *clamped* — is `dense_vmap` |
| MWA_extended zenith | 11 | 0.91-1.02x | 0.95-1.02x | *clamped* — is `dense_vmap` |

The gate passes, and **four of the five cells cannot fail it**: their `n_w`
is under 32, so `w_chunk` clamps to `n_w`, `chunked(32)` *is* `dense_vmap`
bit for bit, and 1.00x is arithmetic rather than measurement. Only
MWA_extended off30 tests anything, and there the medians are 1.06x forward
and 0.96x adjoint — comfortably inside 1.2x. (One adjoint pass of seven read
1.26x; with a control that spans 0.92-1.00 on the identical program, that is
the instrument, not the strategy.)

Where chunking is real on all five — `w_chunk = 8` — the picture is
different and worth stating plainly: forward medians 1.20-1.32x and adjoint
1.08-1.41x, with individual passes reaching 1.65x (EDA2 zenith, forward) and
1.74x (EDA2 zenith, adjoint). **A small chunk on a CI-sized plan does not
meet 1.2x.** That is the trade the knob exists to offer — the caller asked
for `n_w/8` times less memory — but the DoD's single 1.2x figure describes
`w_chunk = 32` on plans where 32 exceeds `n_w`, and should not be read as a
property of chunking in general.

#### On a GPU, at a realistic size

The 256&sup2; table above is the shape of the curve, not its stakes. Measured
on one GH200 (Daint `nid006544`, single device, eps 1e-6, float64, single
channel, `hermitian` and `nthreads` at their defaults; `n_pix` chosen as
`FoV / (lambda / (3 B_max))` rounded to the next 5-smooth size and
`n_rows = 150 x N_baselines`; temp is
`memory_analysis().temp_size_in_bytes`, time is the median of 5 with warm-up
outside the timer and `block_until_ready`):

**MWA_extended off30, 3600&sup2; / 1 219 200 rows, `n_w = 140`** (complex
image 197.8 MB) — the cell the issue was opened about:

| | temp | vs `dense_vmap` | forward time | adjoint time |
|---|---:|---:|---:|---:|
| `dense_scan`    |    414 MB | **73.1x less** | 2.35x | 1.94x |
| `chunked(8)`    |  1 929 MB | 15.7x less | 1.73x | 1.53x |
| `chunked(16)`   |  3 659 MB |  8.3x less | 1.35x | 1.24x |
| `chunked(32)`   |  6 256 MB |  4.8x less | **1.27x** | 1.17x |
| `chunked(64)`   | 10 367 MB |  2.9x less | 1.10x | 1.06x |
| `dense_vmap`    | 30 290 MB |  1.0x      | 1.00x | 1.00x |

The curve is the deliverable: 30 GB of transient — which is why this cell
needed a 96 GB GH200 at all — becomes **6.3 GB at `w_chunk = 32`** for 27%
more forward time, or 1.9 GB at `w_chunk = 8` for 73% more, or 0.4 GB on
`dense_scan` for 2.35x. Pick the point your device has room for.

**The definition of done's GPU gate is breached on that one cell, as
written.** The gate asks for `chunked(32)` within 1.2x of `dense_vmap` on
every cell of the sweep; the MWA_extended off30 *forward* measures 1.27x.
Its adjoint (1.17x) and every other cell measured pass. The gate is not
restated to fit — 1.27x is the number, and whether 27% of the time for 4.8x
of the memory is the right trade is a judgement the curve above lets a
reader make for themselves.

Two of the four fixtures cannot inform that gate either way, for the same
reason two of the five CPU timing fixtures cannot: **MWA_extended zenith**
(`n_w = 13`) and **MeerKAT off30** (`n_w = 14`) clamp every `w_chunk >= 16`
to `n_w`, so their `chunked(32)` *is* `dense_vmap` and reads 1.00x by
construction rather than by measurement. Only MWA_extended off30
(`n_w = 140`) and **EDA2 off30** (150&sup2; / 4 896 000 rows, `n_w = 60`)
exercise chunking at the default. EDA2 off30 passes the gate on the real
path — `chunked(32)` 1.07x forward / 1.01x adjoint at 2.0x less memory, and
`chunked(8)` 1.65x / 1.16x at 7.5x less — and it is also where `dense_scan`
is worst, at 5.97x forward, because its 4.9M rows make the per-plane
re-entry dominate.

`w_strategy="auto"` never resolves to a chunked strategy: choosing a chunk
size needs a memory budget the heuristic is not given.

The `grad` column is new in 0.2.0 (issue #21) and is a *ratio against the
same strategy's forward*, not an absolute size: both operators are now bound
as linear primitives whose transposes are each other, so reverse mode is one
call to the other operator at the forward's own settings rather than a
transposed replay of the w-plane loop.

That mechanism is also why the vmap and chunked rows carry a *range* rather
than a flat `~1x`: since `grad` is one call to the **other** operator, the
ratio it reports is that operator's transient over this one's, and the two
are only equal where the two operators allocate alike. Measured with the same
`memory_analysis().temp_size_in_bytes` protocol at eps 1e-6, float64,
`nthreads=1`, single channel, over EDA2 zenith, MWA_compact off30, MeerKAT
off30 and MWA_extended off30:

| strategy | `grad(dirty2vis)` / forward | `grad(vis2dirty)` / adjoint |
|---|---|---|
| `dense_scan` / `windowed_scan` | 1.46–1.50x | 1.01–1.14x |
| `dense_vmap` / `windowed_vmap` | 0.98–1.98x | 1.00–1.16x |
| `chunked` (`w_chunk` 2, 4, 8, 16, 32, 64) | 1.03–1.96x | 1.00–1.33x |
| `windowed_chunked` (`w_chunk` 2, 8, 32) | 1.03–1.33x | 1.03–1.34x |

The chunked spread has both ends, and they have different causes. At small
`w_chunk` the *forward* transient is small enough that the adjoint's fixed
extra costs show: `w_chunk = 2` measures 1.32x on MWA_extended off30, 1.25x
on EDA2 zenith. At the *large* end the adjoint's own transient stops being
`~1 · chunk · image` and becomes `~2 · chunk · image` — on MWA_extended off30
(`n_w = 134`) that happens between the balanced chunk 27 (`w_chunk = 32`,
1.03x) and the balanced chunk 45 (`w_chunk = 64`, **1.96x**, the same place
`dense_vmap` sits at 1.98x). The `~1x` the middle of the curve gives —
1.10x at `w_chunk = 8`, 1.05x at 16, 1.03x at 32 — is the default's regime,
not the whole of it. Before that change the scan strategies'
gradient cost `O(n_w * image_size)` like the vmap ones — measured with
`memory_analysis().temp_size_in_bytes` on `grad(0.5 ||A x||^2)` at eps 1e-6 in
float64, 1.72 MB against a 0.138 MB forward on EDA2 zenith (`n_w` = 11), 7.21
against 0.534 MB on MWA_compact off30, 285.5 against 2.11 MB on MWA_extended
off30 at 256² (`n_w` = 134) and 897.8 against 33.9 MB at 1024² / 20k rows;
after it, 0.20, 0.80, 3.16 and 50.7 MB, i.e. 12.5-135x down to 1.48-1.50x.
The vmap rows moved from 1.89-2.11x to 0.98-1.18x on the same measurement, but
they were never where the issue lived: their forward already allocates
`n_w * image_size`, so their gradient carried no per-plane residual stack to
remove. "Already inside 2x" would be too kind to them — on MWA_compact off30
the `vis2dirty` vmap cells read 2.07-2.11x before the change on both precision
legs, i.e. just outside it, while the scan cells read 13.5x. The forward
numerics are untouched — the accuracy sweep's worst cell is unmoved at 1.48x
eps, and both operators return bit-for-bit what they returned before.

`jax.vmap` of either operator remains a single batched call rather than a
per-element loop: the primitives carry the batch in a `batch_shape` parameter,
so batched throughput and batched transient memory are unchanged from v0.1.2
(measured identical at batch 1, 2, 4, 8 and 16).

`jax.jvp`, `jax.linear_transpose`, `jax.hessian` and `vmap` of a gradient all
work on both operators; `jax.linear_transpose` is new with the same change.
The reverse-mode convention is JAX's own, i.e. the plain transpose rather than
the Hermitian adjoint: `jax.vjp(dirty2vis)(y)` is `A^T y` (for a real image,
`Re(A^T y) == vis2dirty(plan, conj(y))` at equal `divide_by_n`), and
`jax.vjp(vis2dirty)(u)` is `conj(dirty2vis(plan, u))`.

For the windowed strategies, the plan exposes

```
# the forward's: every plane slices the plan's widest window
plan.window_padding_overhead = n_chan * n_w * max_window_size / plan.live_row_count

# the adjoint's: every plane slices its own size class (0.2.0, #26)
plan.window_padding_overhead_adjoint = (
    sum(slice_length * n_planes for c in channels for slice_length, n_planes in
        plan.window_buckets[c])
    / plan.live_row_count
)
```

as diagnostics: the factor by which a windowed traversal's row-work exceeds
the irreducible minimum. The numerator is what the traversal actually touches
— each `(channel, plane)` step slices a *static* number of rows, since the
shape has to be static for `lax.scan` / `vmap` — and the denominator, shared
by both, counts only the rows inside a plane's nominal `w`-kernel support.
Both are bounded below by 1.0, attaining it on the constant-`w` fast path
where the single plane holds every row and none of the slice is padding, and
the adjoint's is never above the forward's.

There are two of them because 0.2.0 (#26) bucketed the plane slices of the
windowed **adjoint** only. On the review fixtures that took the adjoint's
ratio to 1.00-1.41 against the forward's unchanged 1.08-5.08 over the
forty-cell calibration grid — five fixtures, two pointings, four epsilons
(1e-3, 1e-6, 1e-9, 1e-12), float64, seed 0, `hermitian=True`, single channel.
(The eps = 1e-6 slice alone reads 1.00-1.38 and 1.14-4.94; earlier drafts
quoted that slice while saying "the review fixtures", which is the whole
grid.) So the regime where
a high padding figure sends the `auto` selector to a dense strategy is no
longer reached by any of them on the adjoint. The forward keeps `ab7fbbd`'s
code and `ab7fbbd`'s figure: bucketing it as well was measured at 20.3x to
58.4x *slower* on a GH200 and reverted.

The denominator changed in 0.2.0. Through v0.1.2 it was the mean of the
per-`(channel, plane)` window lengths, which are measured *after* the builder
widens each window by `window_boundary_margin` and by one further row at each
end. Those rows are real work but lie outside nominal support, so counting
them as irreducible understated the waste — by up to 17% on the review
fixtures, worst where the windows are narrowest and the padding is therefore
relatively largest. The two scales are not convertible after the fact: the per-plane
window lengths are plan-time locals and were never stored.

### `w_strategy="auto"` (the shipped default)

`"auto"` resolves to one of the four canonical names before the JIT
boundary, using `plan.n_w`, `plan.w_kernel_width`, one of
`plan.window_padding_overhead` (forward) /
`plan.window_padding_overhead_adjoint` (adjoint), and (on GPU)
`plan.n_rows`, plus the operator's adjoint flag. Resolution happens in
the public wrapper, so an `"auto"` call shares a JIT cache entry with the
explicit equivalent on the same plan. The heuristic is **platform-aware**
(`jax.devices()[0].platform`):

- **CPU.** Conservative: never picks a windowed forward (no measured
  win on the v0.1.1 algorithm), and only picks `windowed_scan` on the
  adjoint when `n_w / w_kernel_width > 2` and the windowed padding
  overhead is below 6x. In practice only the first conjunct can bind: since
  #26 bucketed the adjoint's slices, no repository fixture reaches an adjoint
  overhead above 1.41 at any epsilon on the shipped folded geometry (1.62
  unfolded), so the 6x test passes for all of them
  and never changes a pick. Otherwise `dense_scan`. (That cutoff was 5x
  through v0.1.2, against the pre-0.2.0 denominator; it was restated so
  that redefining the diagnostic changes no decision on the calibration
  grid, where the worst fixture reads 5.78 on the new scale against 4.93
  on the old. Since #26 the *adjoint* leg of this comparison reads
  `window_padding_overhead_adjoint` instead, on which that same worst
  fixture is 1.41 folded (1.62 unfolded) and no repository fixture reaches either cutoff at any
  epsilon — so on the adjoint this branch no longer fires on one, which is
  the intended effect, the padding it guards against being what bucketing
  removes. The forward leg is unchanged.)
- **GPU** (tuned on the GH200 baseline sweep). Never picks a `_scan`
  variant: across the sweep's 160 scan/vmap pairs the scan family is
  slower in every one, by 1.45x to 32.7x (median 6.1x). Picks
  `windowed_vmap` only on large-row plans (`n_rows >= 10000`) with
  padding overhead below 3x -- on the adjoint at any pointing, and on
  the forward when `n_w` is low (`n_w <= 3 * w_kernel_width`).
  Otherwise `dense_vmap`.
- **Other platforms** (e.g. TPU) fall back to the CPU heuristic.

```python
# The default: the heuristic picks per call and per platform.
vis   = dirty2vis(plan, image)
dirty = vis2dirty(plan, vis)

# For reproducing benchmarks, pin the strategy explicitly. An explicit
# w_strategy always overrides the heuristic.
vis   = dirty2vis(plan, image, w_strategy="dense_vmap")
```

The GPU gates are validated against `docs/benchmarks/v0.1.2-baseline-gpu.json`
by `tests/test_auto_strategy_acceptance.py`, which asserts the picked
strategy is within 15% of the best measured strategy for every
(operator, telescope) cell. The `1.45x`, `32.7x` and `6.1x` above — and every
other figure this repository quotes from that JSON — are recomputed from it by
`tests/test_benchmark_claims.py`, so a re-measurement that leaves a sentence
behind fails the suite (issue #49).

#### Why it became the default (issue #46)

Through v0.1.2 both operators defaulted to `"dense_scan"` and `"auto"` was
opt-in, which meant the heuristic was never reached unless a caller asked
for it by name. On GPU that made the shipped default the *worst* of the
four choices. (It becomes the default in 0.2.0, the release now in
development.) Measured on one
NVIDIA GH200 (Daint) against `ducc0` on the
72 Grace cores of the same node, at `epsilon = 1e-6`, float64, single
channel, forward / adjoint milliseconds (median of 9, warm-up outside the
timer; the two implementations agree to `0.75 * epsilon`, so this is
like-for-like):

| fixture | `n_w` | ducc0 (72 cores) | `dense_scan` (the old default) | `dense_vmap` |
|---|---|---|---|---|
| MWA_extended off30, 256&sup2;, 600 rows | 251 | 89.8 / 91.1 | 501.9 / 331.8 | 30.9 / 14.4 |
| MeerKAT off30, 256&sup2;, 600 rows | 19 | 17.2 / 17.9 | 39.4 / 26.2 | 4.9 / 3.3 |
| GH200_large off30, 2048&sup2;, 50k rows | 43 | 67.9 / 184.8 | 160.5 / 159.2 | 47.6 / 78.8 |

(Measured on the pre-#17 `hermitian=False` geometry, hence the `n_w` column
of 251 / 19 / 43; the shipped default now builds 134 / 13 / 26 planes on
these three fixtures and is correspondingly faster — see the upgrade table
further down. The comparison against ducc0 was not re-run under the fold, so
these are the numbers as measured, not current-default figures.)

The last column is `dense_vmap` throughout, which is what the heuristic
picks in five of those six cells but *not* in the sixth: on GH200_large
off30 the adjoint pick is `windowed_vmap`, so `78.8` is not the number a
defaulting caller gets there. The re-run below has that cell measured on
the pick the default actually makes.

So on five of those six cells the old default ran 1.4-5.6x *slower* than
ducc0 on that hardware, where `dense_vmap` runs 1.4-6.3x faster in all
six. That range is the `dense_vmap` column, which is the heuristic's pick
in five of the six but not the sixth: on the GH200_large off30 adjoint it
picks `windowed_vmap`, whose re-run figure of 53.1 ms is 3.5x faster than
ducc0 and so falls inside the same range without changing it. The exception is the GH200_large off30 adjoint, where
`dense_scan` at 159.2 ms was about 1.16x *faster* than ducc0's 184.8 ms —
the one cell in the table where the old default was not losing to ducc0
outright, though it was still 2.0x off `dense_vmap` and 3.0x off the
`windowed_vmap` the heuristic actually picks there. Either way the
heuristic was never the problem, only its reachability. Re-tuning the
thresholds themselves is a separate question (issue #34).

That table was measured when #46 was filed. Re-run on the same hardware
against the code this default actually ships with, the shipped default
against an explicit `dense_scan`, same protocol — **on the pre-#17
`hermitian=False` geometry, which is what the planner produced at the
time**:

| fixture | `n_w` | `"auto"` resolves to | default | `dense_scan` | speedup |
|---|---|---|---|---|---|
| MWA_extended off30 | 251 | `dense_vmap` / `dense_vmap` | 30.8 / 14.3 | 653.6 / 333.9 | 21.3x / 23.4x |
| MeerKAT off30 | 19 | `dense_vmap` / `dense_vmap` | 4.9 / 3.2 | 40.7 / 26.0 | 8.4x / 8.0x |
| GH200_large off30 | 43 | `dense_vmap` / `windowed_vmap` | 47.3 / 53.1 | 159.0 / 157.3 | 3.4x / 3.0x |

Kept as the historical record of what #46 bought, and labelled as such
because its `n_w` column no longer describes a default plan: issue #17's
Hermitian fold halves it, and the same three fixtures now build 134 / 13 /
26 planes. The picks are unchanged by that — `dense_vmap` / `dense_vmap`,
`dense_vmap` / `dense_vmap`, `dense_vmap` / `windowed_vmap` on the folded
geometry too — so the `"auto"`-over-`dense_scan` margins above still stand
qualitatively; the absolute millisecond columns belong to the unfolded
geometry and should be read as history.

What a caller receives from the fold itself is a separate comparison, and
the one that matters for an upgrade: the shipped default on the folded
geometry against the shipped default on the unfolded one, same GH200, same
protocol, `epsilon` as marked.

| fixture | eps | `n_w` | forward | adjoint |
|---|---|---|---|---|
| GH200_large zenith | 1e-6 | 10 &rarr; 9 | 1.06x | 1.09x |
| GH200_large zenith | 1e-9 | 13 &rarr; 12 | 1.06x | 1.07x |
| GH200_large off30 | 1e-6 | 43 &rarr; 26 | 1.26x | 1.52x |
| GH200_large off30 | 1e-9 | 46 &rarr; 29 | 1.12x | 1.47x |
| MWA_extended off30 | 1e-6 | 251 &rarr; 134 | 1.78x | 1.64x |
| MWA_extended off30 | 1e-9 | 254 &rarr; 137 | 1.70x | 1.70x |
| MeerKAT off30 | 1e-6 | 19 &rarr; 13 | 1.23x | 1.10x |
| MeerKAT off30 | 1e-9 | 22 &rarr; 16 | 1.19x | 1.13x |

Every cell faster, 1.06x to 1.78x. One pick in that run is imperfect
rather than wrong: the GH200_large off30 forward at `epsilon = 1e-9`
resolves to `windowed_vmap` at 43.3 ms where `dense_vmap` would be
37.6 ms, so the heuristic leaves about 14% there — still a 1.12x gain on
upgrade, and correcting it means retuning a threshold (issue #34).

One cell of the six shifted between the two runs: the `dense_scan`
forward on MWA_extended off30 read 653.6 ms against the issue's 501.9 ms,
30% slower. The other five reproduced within 3.3%, the adjoint on that
same fixture at the same `n_w` among them (333.9 against 331.8, 0.6%).
The cause was not isolated. A single shifted cell on a shared-node GPU
benchmark is at least as consistent with run-to-run variance as with any
code change, and there is no same-node A/B of the two revisions on this
branch to separate the two explanations — so the honest statement is that
the gap is wider on this run, not that some particular change widened it.
What does not depend on that cell: the default beats `dense_scan` in all
six of the re-run's cells, and the smallest margin — 2.96x on the
GH200_large adjoint, which the table rounds to 3.0x — is one the shifted
cell has no part in.

On CPU the strategy change is now almost nothing, and issue #17 is why.
The forward was always untouched — the CPU heuristic never picks a windowed
forward, so it resolves to `dense_scan` on every fixture in this
repository, exactly the old default. The adjoint used to move to
`windowed_scan` at every off-zenith pointing; folded, it does so only on
MWA_extended off30. Halving `n_w` halves the `n_w / w_kernel_width` ratio
the adjoint gate reads, and MWA_compact off30 (2.43 &rarr; 1.71) and
MeerKAT off30 (2.71 &rarr; 1.86) drop under it. Those are exactly the two
cells that had stopped paying: v0.1.1 Part 2 measured its 1.05-1.53x
adjoint win with those two as its endpoints, and both now time flat.

Timed on a 10-core Apple M-series at `epsilon = 1e-6`, float64, single
channel, with the plan built and warm-ups taken outside the timed window,
each number the median of 9 `block_until_ready()` calls, over two passes —
the shipped default against an explicit `dense_scan`, both on the folded
geometry:

| fixture | `n_w` | adjoint pick | forward | adjoint |
|---|---|---|---|---|
| MWA_compact off30 | 12 | `dense_scan` | 0.99x | 0.98-1.00x |
| MWA_extended off30 | 134 | `windowed_scan` | 0.99-1.00x | 1.03x |
| MeerKAT off30 | 13 | `dense_scan` | 1.00x | 0.99-1.00x |
| EDA2 zenith | 11 | `dense_scan` | 0.98-0.99x | 0.98-1.02x |
| MWA_extended zenith | 11 | `dense_scan` | 0.99-1.03x | 0.99-1.00x |

Every cell is inside this machine's &plusmn;4% noise floor, including the
one cell that still changes strategy. So on the folded geometry the CPU
`"auto"` default is, as far as this machine can resolve, a tie with
`dense_scan` — which is the honest reading, and a change from the
pre-#17 statement that it bought 1.14-1.24x on MWA_extended off30.

That is a statement about the *strategy*, not about the fold. What the
fold is worth on the same machine and protocol — shipped default folded
against shipped default unfolded, i.e. what an upgrading caller receives:

| fixture | `n_w` | forward | adjoint |
|---|---|---|---|
| MWA_compact off30 | 17 &rarr; 12 | 1.36-1.39x | 1.34-1.37x |
| MWA_extended off30 | 251 &rarr; 134 | 1.84-1.85x | 1.67-1.68x |
| MeerKAT off30 | 19 &rarr; 13 | 1.42-1.45x | 1.44-1.47x |
| EDA2 zenith | 14 &rarr; 11 | 1.25-1.27x | 1.26-1.28x |
| MWA_extended zenith | 14 &rarr; 11 | 1.26-1.31x | 1.25-1.26x |

Every cell faster, ranges across the two passes.

In float32 the folded picture differs from float64 in one cell: MeerKAT
off30 keeps `windowed_scan` on the adjoint (ratio 2.2) where its float64
counterpart falls to 1.86. The narrower float32 kernel, `W = 5` against
`W = 7`, is the *denominator* of that ratio, so it cannot rescue a halved
`n_w` — unfolded it pushed two extra fixtures over the cutoff, folded it
holds one back from falling under.

**Backward compatibility.** `w_strategy="dense_scan"` restores the
pre-#46 *code path* on both operators: the same strategy, so the same
reduction order. What it does not restore is the pre-#46 *numbers*,
because this release carries more than #46. #16, #23 and #43 also landed
in it, and #16 moved the numbers on its own — the worst cell against the
exact DFT went from `0.67 * epsilon` to `1.47 * epsilon` after its
`nshift` centring (see *Accuracy expectation* below). Pinning
`w_strategy` therefore removes the strategy change from the comparison
and nothing else; it is not a way to reproduce v0.1.2 output.

Scoped to the strategy change alone, what is promised is that the four
strategies accumulate the w-planes in a different order but agree to the
bound `tests/test_strategies_equivalent.py` pins — `1e-11` relative in
float64 — rather than to the last bit. In practice the gap is far
smaller: on the three GH200 fixtures above, the default and `dense_scan`
agree to `5e-16` relative, i.e. to float64 rounding. The `1e-11` is what
is *guaranteed*, not what is typical.

Naming `w_strategy` explicitly is worth doing, but not for bitwise
determinism: `"auto"` is already deterministic for an unchanged plan on an
unchanged platform, and FINUFFT's own reduction order can vary underneath
either choice, so pinning the strategy neither adds nor guarantees
run-to-run bit identity. What it buys is insulation from the *heuristic*
and the *platform* — a retune (issue #34) or the same code running on GPU
instead of CPU changes what `"auto"` selects, and an explicit name does
not move. Reproducing an older release's numbers needs that release
pinned; neither choice of `w_strategy` substitutes for it.

## `nthreads` (issue #24, R11/D4)

`nthreads: int | None = None` on both operators. The default resolves
*before* the JIT boundary to a strategy-aware choice, not a flat number:
`1` for `dense_scan` / `windowed_scan` (each w-plane makes its own FINUFFT
call, so `nthreads > 1` just re-spins the whole OpenMP pool on every plane
— 4.80x-8.51x slower than `nthreads=1` for the old flat default of `0`,
measured on `dense_scan` on a 10-core Apple M-series: MWA_extended off30 3343.6ms vs
696.0ms, MeerKAT off30 207.5ms vs 41.5ms, EDA2 zenith 36.6ms vs 4.3ms), and
`0` (let FINUFFT decide) for `dense_vmap` / `windowed_vmap` (one batched
FINUFFT call across all w-planes, which benefits from threads at large
enough `n_w`). The `chunked` family splits on `w_chunk`: `w_chunk == 1` is a
plane-at-a-time loop and gets `1`, any wider chunk is a batched call and gets
`0`. Below 100k rows every strategy gets `1` regardless, since the whole
plane loop is short enough that spinning up a pool isn't worth it. Pass an
explicit `int` (including `0`) to opt out of the strategy-aware default.

Those three timings are from one session on one laptop and no JSON for them
is committed, so treat them as an illustration of the effect's size rather
than a figure to tune against. What *is* pinned, by
`tests/test_nthreads_resolution.py`, is the resolution table itself — which
integer comes out of `_resolve_nthreads` for each (strategy, `n_rows`,
`w_chunk`) — not any wall-clock number. The 100k cutoff is exact and the
comparison is strict: `n_rows = 100_000` takes the strategy rule, not the
small-problem rule.

**Limitation:** every fixture in this repository is below the 100k-row
cutoff -- 400-600 rows for the CPU telescope fixtures, 50k for the
`GH200_LARGE` GPU fixture -- so on every benchmark and test in
this repo the small-`n_rows` override applies to *all four* strategies --
the vmap-family steady-state branch (`nthreads=0` above the cutoff) is
never exercised by anything measured here, only by unit tests that build a
plan above the cutoff explicitly. Measured directly at these repo-sized row
counts (`dense_vmap` / `windowed_vmap`, `nthreads=1` vs `nthreads=0`, same
timing protocol as above): `dense_vmap` is ~1.8-2.3x faster at `nthreads=0`
on MWA_extended off30 and MeerKAT off30, but ~1.2x faster at `nthreads=1`
on EDA2 zenith (too little batched work at that `n_w` to amortise a pool
spin-up); `windowed_vmap` on MWA_extended off30 is ~6-7x *faster* at
`nthreads=1` than `nthreads=0` (windowing shrinks each plane's per-call row
count, so it behaves like the scan family here, not like `dense_vmap` at
large `n_w`). That split verdict is why the small-`n_rows` override stays
strategy-blind rather than exempting the vmap family -- see the comment
above `_NTHREADS_SMALL_N_ROWS` in `wgridder.py` for the full numbers.

## Picking a strategy

The default `"auto"` picks for you, per plan and per platform; this is what
it is picking between, and what to pass if you would rather pin it.

- **`dense_scan`** keeps memory bounded at
  `O(image_size + n_rows)` regardless of `n_w` and `n_chan`. Recommended
  on CPU and as a safe baseline for any problem; it was the default
  through v0.1.2, so it is also what to pass to reproduce that behaviour.
- **`dense_vmap`** allocates `O(n_w * image_size)` but is usually the
  fastest of the four on CPU at the tested scales.
- **`windowed_scan`** matches `dense_scan` memory and helps on the
  adjoint when `n_w >> W`. Since #26 its *adjoint*'s slice is the plane's
  own size class rather than the plan's widest window, so a clumped
  `w`-distribution no longer costs the adjoint what it used to; its
  forward still slices `max_window_size`, and
  `plan.window_padding_overhead < ~3` is still the guide there.
- **`windowed_vmap`** is the high-memory variant of the windowed path;
  marginal wins on most cases, kept primarily for GPU parity.

