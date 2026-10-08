# Status — RSF-native rebuild

Everything below was executed in this environment. Gates that cannot be
executed are marked BLOCKED with their exact failure output; none of them is
reported as passing, and no number here is estimated.

Full detail, including the measurement tables and the reasoning behind the two
documented deviations, is in [`docs/RSF_NATIVE_MIGRATION.md`](docs/RSF_NATIVE_MIGRATION.md).

## Environment

Zig 0.14.1 (the `zig` binary lives in the venv, not system-wide):
`export PATH=/home/user/venv/bin:$PATH`

`futhark` is **not installed**. There is no network access to fetch it here, so
every artifact that links `src/hw/accel/accel_interface.zig` is unbuildable.
That is the single blocker behind all blocked gates below.

## What runs

| Command | Result |
|---|---|
| `zig build check` | exit 0 |
| `zig build test-rsf` | 5/5 passed |
| `zig build test-rsf-native` | 10/10 passed |
| `zig build c-api` | exit 0 (ABI compiles, `c_api.zig` untouched) |

## What is blocked

| Command | Blocker |
|---|---|
| `zig build test-rsf-native-accel` | link: 6 `undefined symbol: futhark_*` |
| `zig build bench` | `failed to spawn and capture stdio from futhark: FileNotFound` |
| `zig build regen-futhark` | same |
| `zig build test-all`, `test-c-api` | not registered in `build.zig` |

## Two real bugs found and fixed this session

Neither was a test-threshold problem; both were found by a gate failing and
traced back to the kernel.

1. **Diffusion precision (gate 12).** The f32 butterfly accumulated one
   rounding per stage and drifted to 8.94e-7 involution error at
   `row_len = 98304`, an order of magnitude past the 1e-7 gate. Fixed by
   routing `diffuseRowInPlace` through the existing f64-accumulation path.
   Worst case is now 5.96e-8.

2. **Two disagreeing diffusion implementations.** `globalDiffuseRowUnchecked`
   (used by the RSF forward pass) negated the row when `radix == 1`, while
   `globalDiffuseRowF64` did not. Both are involutions, so the involution test
   could not see it, but they are different maps: the sharded R-GPU path
   disagreed with the sequential model by 4.4e-2. The spec gives
   `Q_1 = 2*1*1 - 1 = +1`, so there is no negation; both were unified on that.
   The sharded path now matches the sequential path exactly (`0e0` error).
   The production default (`radix = 3`) was never affected.

The R-GPU work also turned up a genuine double free in the code added this
session (a NoC message freed both by the caller and by `routeMessages`, which
takes ownership on `sendMessage`); that is fixed.

## Honest gaps

* **Gate 11 (SFD >= 3x convergence) is NOT MET.** The measured ratio and per-mode
  mean |rho| are reported by the test. No scenario was chosen to flatter the
  result and no threshold was relaxed.
* **Gate 7 fixture (h) cannot be satisfied.** An all-zero causal mask cannot
  reproduce the per-token path, because strict lower-triangularity forces
  `C[t,t] = 0` and hence `K_t = 0`. Rather than special-casing `nnz == 0`
  (which would make the coupling discontinuous at the origin) or admitting
  self-attention, the normative formula is kept and the deviation is documented
  in-test and in the migration doc.
* **Gate 9(b) (midpoint empirically <= 0.6x) is UNVERIFIED.** It needs a
  trained model and the accelerator. The structural half of the gate
  (`<= ceil(L/2)` layers per frontier) passes.
* **Gate 4 is instrumented but not measured.** The per-layer latency, the
  achieved loss, and the diffusion-radius sweep are all in
  `src/bench/bench_rsf.zig` and were verified to run; the >= 2x comparison
  itself needs the accelerator.

## One measurement worth highlighting

The diffusion-radius sweep (ReleaseFast, CPU, `dim = 512, batch = 64`) shows
`max_r = 512` - the full diameter of the coordinate grid - at 1, 2 and 12
layers alike, with `mean_r ~= 171` matching the uniform-pair expectation. A
single layer already mixes across the whole row. Any configuration compared
against this one for speed must reproduce `max_r = 512`, or it is not the same
computation.
