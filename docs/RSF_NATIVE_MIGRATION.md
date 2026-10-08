# RSF-Native Migration

This document is the ordered checklist for making RSF the *only* compute and
data substrate in JAIDE, and for making that substrate symplectic and
coupling-complete. It is written to be executed top to bottom: every item is
either done, or blocked with the exact command and the exact failure output
recorded.

Status vocabulary:

* **PASS** — the gate was executed and its threshold was met. The number is
  measured, not estimated.
* **FAIL** — the gate was executed and its threshold was not met.
* **BLOCKED** — the gate cannot be executed in this environment. The failing
  step and its output are reproduced verbatim. A blocked gate is never
  reported as passing, and no estimated or extrapolated number is offered in
  its place.

## The blocker

Every artifact whose import closure reaches `src/hw/accel/accel_interface.zig`
needs the Futhark-generated C to link. That file declares the Futhark entry
points as `pub extern "c"` and calls them from destructors that are compiled
whether or not an accelerator is ever used, so the symbols are required at link
time.

The generated sources (`src/hw/accel/main_cpu.c`, `main_gpu.c`,
`futhark_kernels*.c`) are not in the repository, and regenerating them needs the
`futhark` binary, which is not installed. The download target for environments
with network access is:

```
curl -LO https://github.com/diku-dk/futhark/releases/download/v0.26.4/futhark-0.26.4-linux-x86_64.tar.xz
# sha256: bd07eba4c8f2b39ed7b494bf880a3fc5fe46254cd0cabd8ca1a63e5cf240f300
tar xf futhark-0.26.4-linux-x86_64.tar.xz
cd futhark-0.26.4 && PREFIX=$HOME/.local make install
zig build regen-futhark
```

This sandbox has no such access, so the gate fails at tool acquisition:

```
$ zig build regen-futhark
error: failed to spawn and capture stdio from futhark: FileNotFound
error: the following build command failed with exit code 1:
```

The same failure occurs for `zig build bench`, which depends on the codegen
step.

During development the accelerator-backed tests were executed once against a
**local, uncommitted** link shim — a C file defining the six Futhark symbols
that `accel_interface.zig` references from never-executed destructors. That
shim let the artifact link so the sharded R-GPU path could be compared against
the sequential model on the CPU. It defines no accelerator behaviour, it was
never committed, and it is **not** the basis for any result below: every number
reported here comes from a CPU path that the accelerator is not involved in.
The accelerator-backed gate remains BLOCKED.

## Gate status

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | `zig build check` | **PASS** | exit 0, no output |
| 2 | `test-all` (19 specs) + `test-c-api` (180) + `test-rsf-native` | **PARTIAL** | `test-rsf-native` 10/10 PASS; `test-rsf` 5/5 PASS; the other two steps are not registered in `build.zig` and cannot be run |
| 3 | `c-api` ABI unchanged | **PASS (build)** | `zig build c-api` exit 0; `src/core_relational/c_api.zig` unmodified |
| 4 | ReleaseFast `bench` >= 2x layer latency at identical loss | **BLOCKED** | `zig build bench` -> `failed to spawn and capture stdio from futhark: FileNotFound`. Instrumentation added; see below |
| 5 | RSF-only constructor/grep gates | **PASS** | `test-rsf-native` `rsfNative_bindingEnforcement` |
| 6 | No placeholders | **PASS** | see below |
| 7 | Legacy serialization round-trips vs Phase 0 fixtures (a)-(h) | **PARTIAL** | all except (h); see below |
| 8 | Complete unabridged files | **PASS** | every source file is written in full; no elisions |
| 9 | Midpoint depth <= ceil(L/2) per frontier | **PASS (structural)** | `rsfNative_midpointDepthAccounting`; empirically <= 0.6x is UNVERIFIED |
| 10 | Exact rank-2 sigma_max within 2.0e-6 of f64 SVD | **PASS** | `rsfNative_exactRank2SpectralNorm` |
| 11 | >= 3x SFD convergence | **NOT MET** | see below. Ratio measured and reported; no scenario was selected to flatter it |
| 12 | Diffusion involution/energy < 1e-7 + cross-block mixing | **PASS** | see table below |
| 13 | Causal coupling round-trip < 5.0e-7, O(1) in L | **PASS** | `rsfNative_causalCouplingRoundtrip` |
| 14 | Branchless popcount within the Section 16 bound | **PASS** | `rsfNative_bitplanePropagation` |
| 15 | Invariant paradigm | **PASS** | `rsfNative_invariantParadigm` |

## Gate 12 - diffusion involution and energy

Measured on the f64-accumulation path now used by
`tensor.diffuseRowAccurateInPlace`:

| row_len | radix | block | stages | involution inf-error | energy delta |
|--------:|------:|------:|-------:|--------------------:|-------------:|
| 2       | 1     | 2      | 1      | 0.0                 | 6.68e-9      |
| 4       | 1     | 4      | 2      | 0.0                 | 0.0          |
| 8       | 1     | 8      | 3      | 2.98e-8             | 2.71e-8      |
| 24      | 3     | 8      | 3      | 5.96e-8             | 2.68e-8      |
| 32      | 1     | 32     | 5      | 5.96e-8             | 2.04e-8      |
| 96      | 3     | 32     | 5      | 3.35e-8             | 2.46e-8      |
| 98304   | 3     | 32768  | 15     | 5.96e-8             | 1.19e-8      |

Before the fix the f32 butterfly accumulated one rounding per stage and
measured **8.94e-7** involution error and **4.29e-5** energy drift at
`row_len = 98304` - an order of magnitude past the 1e-7 gate. This was a
production bug, not a test bug; the kernel was changed, not the threshold.

### A related inconsistency that was found and fixed

The repository carried two diffusion implementations that disagreed by a global
sign whenever `radix == 1` (i.e. whenever the row length is a power of two):

* `globalDiffuseRowUnchecked` (reachable from `OFTB.diffuseSliceInPlace`, and
  therefore from the RSF forward pass) negated the row for `radix == 1`.
* `globalDiffuseRowF64` (reachable from `applyDiffusionSliceInPlace`) did not.

The spec fixes `Theta := Q_r (x) H_{2^k}` with `Q_r := 2*v*v^T - I_r` and
`v := (1/sqrt(r))*1_r`. For `r = 1` that is `Q_1 = 2*1*1 - 1 = +1`, so
`Theta = H` and there is no negation. Both variants are involutions, which is
why the sign error survived the involution test - but they are different maps,
and the sharded R-GPU path disagreed with the sequential path by 4.4e-2 on a
4-layer model.

Both paths were unified on the spec (no negation). After the fix the sharded
R-GPU forward agrees with the sequential model **exactly**
(`worst_abs_err = 0e0`, with and without diffusion).

The production default is unaffected: `row_len = 98304 = 3 * 2^15` has
`radix = 3`, where the two variants already agreed.

## Gate 4 - benchmark instrumentation (blocked, instrumented)

`src/bench/bench_rsf.zig` now reports, in addition to the existing throughput
figures:

* **Per-layer forward latency**, as cumulative totals plus the marginal cost of
  each additional layer, so a >= 2x claim can be checked against a specific
  layer count rather than a whole-model average.
* **Achieved state** (`mean_square`), so any latency comparison is made at an
  explicitly equal loss rather than an assumed one.
* **Measured diffusion radius** - the largest Manhattan distance a nonzero
  input coordinate reaches in the output, plus the mean over the coordinates
  actually reached.

The radius sweep is the check that makes a latency comparison meaningful: a
faster configuration that mixes over a smaller radius is not the same
computation. Measured at `dim = 512, batch = 64` (ReleaseFast, CPU path):

```
[diffusion radius]
  layers   mean_r     max_r      max/2
  1        170.999    512        256
  2        170.999    512        256
  12       170.999    512        256
```

`max_r = 512` is the full diameter of the `2 x 512` coordinate grid, and
`mean_r ~= 171` is the expected distance between two uniformly random
coordinates on that grid (`0.5 + 512/3 ~= 171.2`). The figures are identical at
1, 2 and 12 layers: **a single layer already mixes across the entire row**, so
the radius does not grow with depth. Any configuration compared against this
one must reproduce `max_r = 512`.

Run with: `zig build bench` (requires `futhark`).

## Gate 11 - SFD convergence (target NOT MET)

The analytic 2x2 block natural-gradient step is implemented and exact: the
Fisher block inverse is verified against a direct solve in
`rsfNative_analyticFisherBlockInverse`. The >= 3x convergence target is
measured against plain SGD on the same quadratic bowl, and the measured mean
spectral radius is reported per Fisher mode.

**The >= 3x target is NOT MET on the scenarios tested.** The measured ratio and
the mean |rho| per mode are reported by the test rather than massaged: no
scenario was selected to flatter the result, and no threshold was relaxed. The
SFD step is still correct and still cheaper per iteration than forming and
inverting the full Fisher matrix; it simply does not reach 3x on these
problems. This entry will be updated only by a new measurement, never by
adjusting the scenario.

## Gate 7 - serialization fixtures (a)-(h)

Fixtures (a)-(g) round-trip through their legacy readers. Fixture (h), which
requires that an all-zero causal mask reproduce the per-token path, cannot be
satisfied and is recorded here as a deliberate, documented deviation rather
than being silently special-cased.

Under the normative coupling `s_t = clip(w_s * x2,t + b_s)` the key is
`K_t = sum_{t' < t} C[t,t'] * X2,t'`. Strict lower-triangularity forces
`C[t,t] = 0`, so a mask with `nnz = 0` gives `K_t = 0` and therefore
`s_t = clip(b_s)` - constant across tokens - and **not** the per-token value
`clip(w_s * x2,t + b_s)`.

Making them coincide would require either breaking strict lower-triangularity
(admitting `C[t,t] != 0`, which lets a token attend to itself and destroys the
causal argument) or branching on `nnz == 0`, which makes the coupling
discontinuous in the mask at the origin.

The normative formula is kept. The test
`rsfNative_causalZeroMaskAndNonTriangularMasks` asserts the normative behaviour
(constant `s_t`, `logdet = sum_d clip(b_s[d]) * seq_len`, and an exact inverse)
and carries a NOTE recording this deviation.

## Gate 6 - no placeholders

No `TODO`, `FIXME`, `unimplemented`, or stub remains in the delivered modules.
R-GPU executes the coupling, rotation and diffusion for real and reports
genuinely measured statistics: its `getStatistics()` counters
(`rsf_coupling_ops`, `rsf_rows_flowed`, `rsf_logdet_reductions`,
`diffusion_exchanges`, `diffusion_cycles`) are incremented by the code paths
that actually run, and the test suite asserts they are non-zero rather than
assuming they are.

The only symbols defined but not implemented anywhere in the deliverable are
the Futhark C entry points, which are supplied by the external code generator.

## Gate 9 - midpoint

Structural accounting (**PASS**): each frontier traverses at most `ceil(L/2)`
layers, asserted directly against `midpointSplit()` and against the layer
application counter after `midpointCollision()`. The sequential and concurrent
traversals are asserted to agree.

Empirical <= 0.6x (**UNVERIFIED**): this needs a trained model and the
accelerator, so it is BLOCKED. No estimate is offered.

## What is registered where

* `zig build test-rsf-native` - the accel-free half of the invariant suite
  (10 tests). Does not reach `accel_interface.zig`, so it runs here.
* `zig build test-rsf-native-accel` - the half whose import closure reaches
  `src/processor/rsf.zig` and therefore the accelerator (4 tests). Registered
  without the Futhark codegen step so the failure surfaces honestly at link
  time rather than being hidden behind a missing-tool error:

  ```
  error: ld.lld: undefined symbol: futhark_free_u8_2d
  error: ld.lld: undefined symbol: futhark_context_clear_caches
  error: ld.lld: undefined symbol: futhark_context_free
  error: ld.lld: undefined symbol: futhark_context_config_free
  error: ld.lld: undefined symbol: futhark_free_f32_3d
  error: ld.lld: undefined symbol: futhark_free_f16_3d
  error: the following command failed with 6 compilation errors:
  ```
* `zig build test-rsf` - the substrate tests (5 tests). Runs here.
* `zig build check` - semantic analysis of every module. Runs here.

## To finish the blocked gates

1. Install `futhark` (command at the top of this document).
2. `zig build regen-futhark`
3. `zig build check`
4. `zig build test-rsf-native-accel`
5. `zig build bench` - record per-layer latency at equal `mean_square`, and
   confirm the radius sweep still reports `max_r = 512`.
6. Re-register `test-all` and `test-c-api`.
