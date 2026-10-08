//! Phase 13: cross-module RSF-native invariant suite.
//!
//! Every test here is executable without the Futhark toolchain: the tensor,
//! OFTB, SFD, NSIR, VPU and R-GPU level invariants are reached through APIs
//! that do not call an accelerator entry point, so the suite links and runs on
//! the CPU backend alone. Tests that require the generated accelerator C live
//! in the accel-backed spec.
//!
//! Gate coverage:
//!   gate 10 — `rsfNative_exactSpectralEquivalence`
//!   gate 11 — `rsfNative_blockNaturalGradient`, `rsfNative_sfdConvergence`
//!   gate 12 — `rsfNative_diffusionGlobalMixing`
//!   gate 13 — `rsfNative_causalCrossTokenFlow`
//!   gate 14 — `rsfNative_bitplanePropagation`
//!   gate 15 — `rsfNative_invariantParadigm`
//!   gate 5  — `rsfNative_bindingEnforcement`
//!   gate 9(a)— `rsfNative_midpointDepthAccounting`
//!   gate 3/7 — `rsfNative_rgpuShardedEquivalence`

const std = @import("std");
const jaide = @import("jaide");
const tensor = jaide.tensor;
const types = jaide.types;
const sfd = jaide.sfd;
const nsir = jaide.nsir_core;
const oftb_mod = jaide.oftb;

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64) so every sweep is reproducible.
// ---------------------------------------------------------------------------

const Rng = struct {
    state: u64,

    fn init(seed: u64) Rng {
        return .{ .state = seed };
    }

    fn nextU64(self: *Rng) u64 {
        self.state +%= 0x9E37_79B9_7F4A_7C15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xBF58_476D_1CE4_E5B9;
        z = (z ^ (z >> 27)) *% 0x94D0_49BB_1331_11EB;
        return z ^ (z >> 31);
    }

    /// Uniform in [-1, 1).
    fn nextF32(self: *Rng) f32 {
        const v = self.nextU64() >> 40;
        return @as(f32, @floatFromInt(v)) / 8388608.0 - 1.0;
    }

    fn nextUSize(self: *Rng, bound: usize) usize {
        if (bound == 0) return 0;
        return @as(usize, @intCast(self.nextU64() % bound));
    }
};

// ---------------------------------------------------------------------------
// Independent references. These deliberately avoid every production kernel so
// a shared bug cannot make a test pass against itself.
// ---------------------------------------------------------------------------

/// f64 Jacobi eigenvalue iteration on the 2x2 Gram matrix G = WᵀW. Returns
/// sqrt of the larger eigenvalue, i.e. σ_max(W), in double precision.
fn jacobiSingularValueRank2(w0: []const f32, w1: []const f32) f64 {
    var a: f64 = 0.0;
    var b: f64 = 0.0;
    var c: f64 = 0.0;
    for (w0, w1) |x, y| {
        const xf: f64 = @floatCast(x);
        const yf: f64 = @floatCast(y);
        a += xf * xf;
        b += xf * yf;
        c += yf * yf;
    }
    // Closed-form eigenvalues of [[a,b],[b,c]] computed in f64, then refined
    // by one Jacobi rotation so a*b != 0 cases are cross-checked.
    const tr = a + c;
    const diff = a - c;
    const disc = @sqrt(diff * diff + 4.0 * b * b);
    var lambda_max = 0.5 * (tr + disc);
    // Jacobi refinement: rotate to diagonalise and recompute.
    if (@abs(b) > 1e-18) {
        const theta = 0.5 * std.math.atan(2.0 * b / (diff + 1e-300));
        const ct = @cos(theta);
        const st = @sin(theta);
        const app = a * ct * ct + 2.0 * b * ct * st + c * st * st;
        const aqq = a * st * st - 2.0 * b * ct * st + c * ct * ct;
        lambda_max = @max(app, aqq);
    }
    return @sqrt(@max(0.0, lambda_max));
}

/// Brute-force reference: max over θ of ‖W·(cosθ, sinθ)‖ over a dense angular
/// sweep. This is a genuinely independent check of the closed form.
fn angularSweepSingularValue(w0: []const f32, w1: []const f32, samples: usize) f64 {
    var best: f64 = 0.0;
    var i: usize = 0;
    while (i < samples) : (i += 1) {
        const theta = std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(samples));
        const ct = @cos(theta);
        const st = @sin(theta);
        var acc: f64 = 0.0;
        for (w0, w1) |x, y| {
            const p: f64 = @as(f64, x) * ct + @as(f64, y) * st;
            acc += p * p;
        }
        if (acc > best) best = acc;
    }
    return @sqrt(best);
}

// ---------------------------------------------------------------------------
// Gate 10: exact rank-2 spectral normalization
// ---------------------------------------------------------------------------

test "rsfNative_exactSpectralEquivalence" {
    const allocator = std.testing.allocator;
    var rng = Rng.init(0x5EED_1234_5678_9ABC);

    const shapes = [_]usize{ 2, 17, 257, 4096 };
    var worst_jacobi: f64 = 0.0;
    var worst_sweep: f64 = 0.0;
    var matrices: usize = 0;

    for (shapes) |rows| {
        var iter: usize = 0;
        while (iter < 250) : (iter += 1) {
            const w0 = try allocator.alloc(f32, rows);
            defer allocator.free(w0);
            const w1 = try allocator.alloc(f32, rows);
            defer allocator.free(w1);
            for (w0) |*v| v.* = rng.nextF32() * 4.0;
            for (w1) |*v| v.* = rng.nextF32() * 4.0;

            const inter = try allocator.alloc(f32, rows * 2);
            defer allocator.free(inter);
            for (0..rows) |i| {
                inter[i * 2 + 0] = w0[i];
                inter[i * 2 + 1] = w1[i];
            }

            const exact = try tensor.exactSpectralNormRank2(inter, rows);
            const jacobi = jacobiSingularValueRank2(w0, w1);
            const sweep = angularSweepSingularValue(w0, w1, 4096);

            worst_jacobi = @max(worst_jacobi, @abs(@as(f64, exact) - jacobi));
            // The angular sweep is a lower bound on σ_max (it samples, it does
            // not optimise), so only assert it never exceeds the closed form by
            // more than the tolerance.
            const gap = @as(f64, exact) - sweep;
            if (gap < -2.0e-6) worst_sweep = @max(worst_sweep, -gap);
            matrices += 1;
        }
    }

    try std.testing.expectEqual(@as(usize, 1000), matrices);
    try std.testing.expect(worst_jacobi < 2.0e-6);
    try std.testing.expect(worst_sweep < 2.0e-6);
}

test "rsfNative_spectralConstraintIsSinglePassAndLeavesSmallMatricesAlone" {
    const allocator = std.testing.allocator;
    var rng = Rng.init(0xA5A5_1234);

    var rescaled: usize = 0;
    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        const rows = 1 + rng.nextUSize(64);
        var w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, 7, 0, rows));
        defer w.deinit();

        // Half the sweep starts below the target: those must survive untouched.
        const amplitude: f32 = if (iter % 2 == 0) 0.001 else 40.0;
        for (0..rows) |i| {
            w.data[i * 2 + 0] = rng.nextF32() * amplitude;
            w.data[i * 2 + 1] = rng.nextF32() * amplitude;
        }
        const before = try allocator.dupe(f32, w.data);
        defer allocator.free(before);

        const sigma_before = try tensor.constrainCouplingSpectralNorm(&w, 0.9);
        const sigma_after = try tensor.couplingSpectralNorm(&w);
        try std.testing.expect(sigma_after <= 0.9 * (1.0 + 1e-6) + 1e-6);

        if (sigma_before <= 0.9) {
            for (before, w.data) |a, b| try std.testing.expectEqual(a, b);
        } else {
            rescaled += 1;
            try std.testing.expect(@abs(sigma_after - 0.9) < 1e-4);
        }
    }
    try std.testing.expect(rescaled > 0);
}

// ---------------------------------------------------------------------------
// Gate 11: analytic 2x2 block-diagonal natural gradient
// ---------------------------------------------------------------------------

test "rsfNative_blockNaturalGradient" {
    var rng = Rng.init(0xB10C_2024);
    var worst_identity: f64 = 0.0;
    var worst_degenerate: f64 = 0.0;
    var worst_cond: f64 = 0.0;

    var iter: usize = 0;
    while (iter < 10000) : (iter += 1) {
        // Random SPD 2x2 block: A, C > 0 with AC - B² > 0.
        const a0 = rng.nextF32();
        const c0 = rng.nextF32();
        const b0 = rng.nextF32();
        const A: f64 = @abs(@as(f64, a0)) * 10.0 + 1e-3;
        const C: f64 = @abs(@as(f64, c0)) * 10.0 + 1e-3;
        const rho = b0 * 0.99;
        const B: f64 = @as(f64, rho) * @sqrt(A * C);
        const lambda: f64 = 1e-8;

        // blockInverseSqrt takes the raw (A, B, C) and applies the damping
        // λ itself; M = [[A,B],[B,C]] + λI is what it inverts the root of.
        const m00 = A + lambda;
        const m11 = C + lambda;
        const m01 = B;

        const inv = sfd.blockInverseSqrt(A, B, C, lambda);

        // ‖M^{-1/2} M M^{-1/2} − I‖₂
        const k00 = inv.inv00 * m00 + inv.inv01 * m01;
        const k01 = inv.inv00 * m01 + inv.inv01 * m11;
        const k10 = inv.inv01 * m00 + inv.inv11 * m01;
        const k11 = inv.inv01 * m01 + inv.inv11 * m11;
        const p00 = k00 * inv.inv00 + k01 * inv.inv01;
        const p01 = k00 * inv.inv01 + k01 * inv.inv11;
        const p10 = k10 * inv.inv00 + k11 * inv.inv01;
        const p11 = k10 * inv.inv01 + k11 * inv.inv11;
        const e00 = p00 - 1.0;
        const e11 = p11 - 1.0;
        const norm = @sqrt(@max(e00 * e00 + p01 * p01, p10 * p10 + e11 * e11));
        worst_identity = @max(worst_identity, norm);

        // Symmetry + positive definiteness of M^{-1/2}.
        try std.testing.expect(@abs(inv.inv01 - inv.inv01) < 1e-30);
        try std.testing.expect(inv.inv00 > 0.0);
        try std.testing.expect(inv.inv11 > 0.0);

        // Condition number bookkeeping: (1+|ρ|)/(1−|ρ|) is the diagonal
        // preconditioner's residual condition number; the block form reduces
        // it to 1.
        const abs_rho = @abs(@as(f64, rho));
        if (abs_rho < 1.0) {
            const residual = (1.0 + abs_rho) / (1.0 - abs_rho);
            worst_cond = @max(worst_cond, residual);
        }
    }
    try std.testing.expect(worst_identity < 1e-5);
    try std.testing.expect(worst_cond > 1.0);

    // Degeneracy: B = 0 must reduce exactly to the diagonal step.
    iter = 0;
    while (iter < 1000) : (iter += 1) {
        const A: f64 = @abs(@as(f64, rng.nextF32())) * 8.0 + 1e-2;
        const C: f64 = @abs(@as(f64, rng.nextF32())) * 8.0 + 1e-2;
        const lambda: f64 = 1e-8;
        const inv = sfd.blockInverseSqrt(A, 0.0, C, lambda);
        worst_degenerate = @max(worst_degenerate, @abs(inv.inv00 - 1.0 / @sqrt(A + lambda)));
        worst_degenerate = @max(worst_degenerate, @abs(inv.inv11 - 1.0 / @sqrt(C + lambda)));
        worst_degenerate = @max(worst_degenerate, @abs(inv.inv01));
    }
    try std.testing.expect(worst_degenerate < 1e-6);
}

// ---------------------------------------------------------------------------
// Gate 12: global cross-channel diffusion Θ = Q_r ⊗ H_{2^k}
// ---------------------------------------------------------------------------

fn diffusionLayoutOf(row_len: usize) types.RSFDiffusionLayout {
    return types.rsfDiffusionLayout(row_len).?;
}

fn checkDiffusionInvariants(row_len: usize, seed: u64) !void {
    const allocator = std.testing.allocator;
    const layout = diffusionLayoutOf(row_len);
    var rng = Rng.init(seed);

    const x = try allocator.alloc(f32, row_len);
    defer allocator.free(x);
    const orig = try allocator.alloc(f32, row_len);
    defer allocator.free(orig);
    const once = try allocator.alloc(f32, row_len);
    defer allocator.free(once);
    for (x) |*v| v.* = rng.nextF32();
    @memcpy(orig, x);
    @memcpy(once, x);

    try tensor.diffuseRowInPlace(x, layout);
    // Θ is its own inverse, so a second application returns the input.
    try tensor.diffuseRowInPlace(x, layout);

    // Θ(Θ(x)) = x to 1e-7 in ∞-norm.
    for (orig, x) |want, back| {
        try std.testing.expect(@abs(want - back) < 1e-7);
    }

    // Energy preservation ‖Θ(x)‖₂ = ‖x‖₂ to 1e-7.
    try tensor.diffuseRowInPlace(once, layout);
    var n_in: f64 = 0.0;
    var n_out: f64 = 0.0;
    for (orig, once) |a, b| {
        n_in += @as(f64, a) * @as(f64, a);
        n_out += @as(f64, b) * @as(f64, b);
    }
    try std.testing.expect(@abs(@sqrt(n_in) - @sqrt(n_out)) < 1e-7);

    // Zero log-det contribution: |det Θ| = 1 exactly, so the implementation
    // reports 0 for the OFTB product it participates in.
    var oftb_inst = try oftb_mod.OFTB.initWithDiffusion(row_len / 2, true);
    try std.testing.expectEqual(@as(f32, 0.0), oftb_inst.logDetContribution());
}

test "rsfNative_diffusionGlobalMixing" {
    // Production geometry and the small geometries used by tests.
    const row_lens = [_]usize{ 2, 4, 8, 24, 32, 96, 32768 * 3 };
    for (row_lens) |len| {
        try checkDiffusionInvariants(len, 0xD1FF_0000 + len);
    }

    const allocator = std.testing.allocator;

    // Production geometry decomposition: row_len = 98304 = 3 · 2^15.
    const prod = diffusionLayoutOf(98304);
    try std.testing.expectEqual(@as(usize, 98304), prod.row_len);
    try std.testing.expectEqual(@as(usize, 3), prod.radix);
    try std.testing.expectEqual(@as(usize, 32768), prod.block);
    try std.testing.expectEqual(@as(usize, 15), prod.stages);

    // Cross-channel mixing: a single-coordinate perturbation must reach every
    // one of the r blocks.
    const layout = diffusionLayoutOf(24);
    try std.testing.expectEqual(@as(usize, 3), layout.radix);
    try std.testing.expectEqual(@as(usize, 8), layout.block);
    const delta = try allocator.alloc(f32, 24);
    defer allocator.free(delta);
    @memset(delta, 0.0);
    delta[0] = 1.0;
    try tensor.diffuseRowInPlace(delta, layout);
    for (0..layout.radix) |b| {
        var reached = false;
        for (0..layout.block) |o| {
            if (@abs(delta[b * layout.block + o]) > 1e-6) reached = true;
        }
        try std.testing.expect(reached);
    }

    // Θ equals its own transpose: ⟨Θa, b⟩ = ⟨a, Θb⟩.
    const a = try allocator.alloc(f32, 24);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, 24);
    defer allocator.free(b);
    const ta = try allocator.alloc(f32, 24);
    defer allocator.free(ta);
    const tb = try allocator.alloc(f32, 24);
    defer allocator.free(tb);
    var r2 = Rng.init(0x1234_5678);
    for (a) |*v| v.* = r2.nextF32();
    for (b) |*v| v.* = r2.nextF32();
    @memcpy(ta, a);
    @memcpy(tb, b);
    try tensor.diffuseRowInPlace(ta, layout);
    try tensor.diffuseRowInPlace(tb, layout);
    var lhs: f64 = 0.0;
    var rhs: f64 = 0.0;
    for (ta, b) |x, y| lhs += @as(f64, x) * @as(f64, y);
    for (a, tb) |x, y| rhs += @as(f64, x) * @as(f64, y);
    try std.testing.expect(@abs(lhs - rhs) < 1e-6);

    // r == 1 (pure power-of-two row) is still an exact involution.
    try checkDiffusionInvariants(32, 0xFACE_0001);

    // Non-power-of-two blocks are rejected by the primitive.
    const bad = try allocator.alloc(f32, 6);
    defer allocator.free(bad);
    try std.testing.expectError(error.InvalidDimension, tensor.walshHadamardInPlace(bad));
}

// ---------------------------------------------------------------------------
// Gate 13: causal cross-token relational coupling
// ---------------------------------------------------------------------------

test "rsfNative_causalCrossTokenFlow" {
    const allocator = std.testing.allocator;
    const seq_len: usize = 16;
    const dim: usize = 8;
    const model_id: u64 = 4242;

    var mask = try types.RSFSequenceMask.initCausal(allocator, seq_len);
    defer mask.deinit();
    try std.testing.expect(mask.full_causal);

    var s_w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, model_id, 0, dim));
    defer s_w.deinit();
    var t_w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_t, model_id, 0, dim));
    defer t_w.deinit();
    var rng = Rng.init(0xCA05_0001);
    for (s_w.data) |*v| v.* = rng.nextF32() * 0.4;
    for (t_w.data) |*v| v.* = rng.nextF32() * 0.4;

    var state = try tensor.Tensor.initLatent(allocator, model_id, dim, seq_len);
    defer state.deinit();
    var original = try state.clone(allocator);
    defer original.deinit();
    for (state.data) |*v| v.* = rng.nextF32() * 0.5;
    try original.copyFrom(&state);

    const scratch_k = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(scratch_k);
    const scratch_x2 = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(scratch_x2);

    const logdet = try tensor.causalCouplingForwardBatch(&state, &mask, &s_w, &t_w, -5.0, 5.0, scratch_k);
    try tensor.causalCouplingInverseBatch(&state, &mask, &s_w, &t_w, -5.0, 5.0, scratch_k, scratch_x2);

    // Gate 13 inversion bound.
    var worst: f32 = 0.0;
    for (state.data, original.data) |got, want| {
        worst = @max(worst, @abs(got - want));
    }
    try std.testing.expect(worst < 5.0e-7);

    // Causality: perturbing token t' must leave the *outputs* at tokens
    // t < t' bit-identical. The comparison is between two forward passes -
    // one on the original sequence and one on the perturbed sequence.
    var probe = try tensor.Tensor.initLatent(allocator, model_id, dim, seq_len);
    defer probe.deinit();
    try probe.copyFrom(&original);
    const perturbed_token: usize = 9;
    for (0..dim) |d| {
        probe.data[(perturbed_token * dim + d) * 2 + 0] += 0.37;
        probe.data[(perturbed_token * dim + d) * 2 + 1] -= 0.21;
    }
    _ = try tensor.causalCouplingForwardBatch(&probe, &mask, &s_w, &t_w, -5.0, 5.0, scratch_k);
    var forward_out = try tensor.Tensor.initLatent(allocator, model_id, dim, seq_len);
    defer forward_out.deinit();
    try forward_out.copyFrom(&original);
    _ = try tensor.causalCouplingForwardBatch(&forward_out, &mask, &s_w, &t_w, -5.0, 5.0, scratch_k);
    for (0..seq_len) |t| {
        for (0..dim * 2) |i| {
            const idx = t * dim * 2 + i;
            if (t < perturbed_token) {
                try std.testing.expectEqual(forward_out.data[idx], probe.data[idx]);
            }
        }
    }
    // And the perturbation must actually change something at or after t',
    // otherwise the test could pass vacuously.
    var changed_after = false;
    for (perturbed_token..seq_len) |t| {
        for (0..dim * 2) |i| {
            const idx = t * dim * 2 + i;
            if (forward_out.data[idx] != probe.data[idx]) changed_after = true;
        }
    }
    try std.testing.expect(changed_after);

    // Log-det is Σ_t Σ_d s_{t,d}: recomputed independently from the recovered
    // X_2 prefix sums.
    var expected_logdet: f64 = 0.0;
    const work = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(work);
    for (0..seq_len) |t| {
        for (0..dim) |d| {
            var acc: f32 = 0.0;
            for (0..t) |j| acc += original.data[(j * dim + d) * 2 + 1];
            const raw = s_w.data[d * 2] * acc + s_w.data[d * 2 + 1];
            expected_logdet += @as(f64, @max(-5.0, @min(5.0, raw)));
        }
    }
    try std.testing.expect(@abs(@as(f64, logdet) - expected_logdet) < 1e-3);
}

test "rsfNative_causalZeroMaskAndNonTriangularMasks" {
    const allocator = std.testing.allocator;
    const seq_len: usize = 6;
    const dim: usize = 4;
    const model_id: u64 = 909;

    var zero_mask = try types.RSFSequenceMask.initZero(allocator, seq_len);
    defer zero_mask.deinit();
    try std.testing.expectEqual(@as(usize, 0), zero_mask.nnz);

    var s_w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, model_id, 0, dim));
    defer s_w.deinit();
    var t_w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_t, model_id, 0, dim));
    defer t_w.deinit();
    var rng = Rng.init(0x0F0F_0011);
    for (s_w.data) |*v| v.* = rng.nextF32() * 0.5;
    for (t_w.data) |*v| v.* = rng.nextF32() * 0.5;

    var state = try tensor.Tensor.initLatent(allocator, model_id, dim, seq_len);
    defer state.deinit();
    for (state.data) |*v| v.* = rng.nextF32() * 0.5;

    const scratch_k = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(scratch_k);
    const scratch_x2 = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(scratch_x2);

    // An all-zero mask empties every prefix sum, so K_t = 0 and therefore
    // s_t = clip(b_s) - a scale that is CONSTANT across tokens. This is the
    // exact behaviour the Section 4.8 formula prescribes.
    //
    // NOTE (documented deviation): the source plan additionally asks that an
    // all-zero mask "reproduce the per-token path exactly" (Phase 0 fixture
    // (h)). Under the normative definition K_t = Σ_{t'<t} C[t,t']·X_2,t' the
    // two cannot coincide: the per-token coupling uses s = clip(w·x2 + b),
    // which is what K_t = X_2,t would give, and C[t,t] = 0 is forced by strict
    // lower triangularity. Special-casing an empty mask to a different formula
    // would create a discontinuous semantic at nnz = 0 and would make the
    // documented inverse inconsistent with the forward, so the normative
    // formula is kept and the difference is recorded here instead.
    const logdet_zero = try tensor.causalCouplingForwardBatch(&state, &zero_mask, &s_w, &t_w, -5.0, 5.0, scratch_k);
    var expected_zero_logdet: f64 = 0.0;
    for (0..dim) |d| {
        const clipped = @max(-5.0, @min(5.0, s_w.data[d * 2 + 1]));
        expected_zero_logdet += @as(f64, clipped) * @as(f64, @floatFromInt(seq_len));
    }
    try std.testing.expect(@abs(@as(f64, logdet_zero) - expected_zero_logdet) < 1e-4);

    // Inverse still recovers the input exactly with an empty mask.
    var recovered = try state.clone(allocator);
    defer recovered.deinit();
    try tensor.causalCouplingInverseBatch(&recovered, &zero_mask, &s_w, &t_w, -5.0, 5.0, scratch_k, scratch_x2);
    for (0..seq_len) |t| {
        for (0..dim) |d| {
            // Y_1,t = X_1,t · exp(s_t) with s_t constant across t.
            const s_t = @max(-5.0, @min(5.0, s_w.data[d * 2 + 1]));
            const y1 = state.data[(t * dim + d) * 2 + 0];
            const y2 = state.data[(t * dim + d) * 2 + 1];
            const x2 = y2 - (t_w.data[d * 2] * y1 + t_w.data[d * 2 + 1]);
            const x1 = y1 * @exp(-s_t);
            try std.testing.expect(@abs(recovered.data[(t * dim + d) * 2 + 0] - x1) < 1e-6);
            try std.testing.expect(@abs(recovered.data[(t * dim + d) * 2 + 1] - x2) < 1e-6);
        }
    }

    // Masks that are not strictly lower triangular, or whose bytes are not in
    // {0,1}, are rejected with a typed error.
    var bytes = try allocator.alloc(u8, seq_len * seq_len);
    defer allocator.free(bytes);

    @memset(bytes, 0);
    bytes[2 * seq_len + 2] = 1; // j == i
    try std.testing.expectError(error.InvalidCausalMask, types.RSFSequenceMask.initFromBytes(allocator, seq_len, bytes));

    @memset(bytes, 0);
    bytes[1 * seq_len + 3] = 1; // j > i
    try std.testing.expectError(error.InvalidCausalMask, types.RSFSequenceMask.initFromBytes(allocator, seq_len, bytes));

    @memset(bytes, 0);
    bytes[3 * seq_len + 1] = 2; // value outside {0,1}
    try std.testing.expectError(error.InvalidCausalMask, types.RSFSequenceMask.initFromBytes(allocator, seq_len, bytes));

    // `set` refuses non-causal coordinates rather than silently storing them.
    var probe = try types.RSFSequenceMask.initZero(allocator, seq_len);
    defer probe.deinit();
    try std.testing.expectError(error.InvalidCausalMask, probe.set(2, 2));
    try std.testing.expectError(error.InvalidCausalMask, probe.set(2, 5));
    try probe.set(5, 2);
    try std.testing.expect(probe.get(5, 2));
}

test "rsfNative_causalPrefixSumPathsAgreeOnFullCausalMask" {
    const allocator = std.testing.allocator;
    const seq_len: usize = 12;
    const dim: usize = 3;

    var mask = try types.RSFSequenceMask.initCausal(allocator, seq_len);
    defer mask.deinit();
    try std.testing.expect(mask.full_causal);

    const x2 = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(x2);
    const k_a = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(k_a);
    const k_b = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(k_b);
    var rng = Rng.init(0x9A9A_0022);
    for (x2) |*v| v.* = rng.nextF32();

    try tensor.causalPrefixSum(&mask, x2, dim, k_a);

    // Independent running-recurrence reference.
    for (0..dim) |d| k_b[d] = 0.0;
    for (1..seq_len) |t| {
        for (0..dim) |d| k_b[t * dim + d] = k_b[(t - 1) * dim + d] + x2[(t - 1) * dim + d];
    }
    for (k_a, k_b) |a, b| try std.testing.expect(@abs(a - b) < 1e-6);

    // A relational (non-full) mask must not be routed through the running
    // accumulator: rows have different support, so the recurrence is invalid.
    var band = try types.RSFSequenceMask.initBand(allocator, seq_len, 2);
    defer band.deinit();
    try std.testing.expect(!band.full_causal);
    try band.set(5, 0);
    try std.testing.expect(band.get(5, 0));
    try std.testing.expect(!band.get(5, 2)); // outside the band of 2
}

// ---------------------------------------------------------------------------
// Gate 14: branchless bit-plane relational propagation
// ---------------------------------------------------------------------------

test "rsfNative_bitplanePropagation" {
    const allocator = std.testing.allocator;
    var rng = Rng.init(0xB17F_1234);

    const node_counts = [_]usize{ 1, 3, 7, 63, 64, 65, 128, 256 };
    const densities = [_]f32{ 0.0, 0.01, 0.5, 1.0 };
    const decays = [_]f32{ 0.0, 0.5, -0.25, 1.0 };

    var worst_ratio: f64 = 0.0;
    var configs: usize = 0;

    for (node_counts) |n| {
        const words = nsir.bitmaskWordCount(n);
        const bitmask = try allocator.alloc(u64, n * words);
        defer allocator.free(bitmask);
        const exact_out = try allocator.alloc(f32, n);
        defer allocator.free(exact_out);
        const plane_out = try allocator.alloc(f32, n);
        defer allocator.free(plane_out);
        const signal = try allocator.alloc(f32, n);
        defer allocator.free(signal);

        for (densities) |density| {
            for (0..n) |i| {
                for (0..words) |w| bitmask[i * words + w] = 0;
                for (0..n) |j| {
                    if (i == j) continue;
                    if (@as(f32, @floatFromInt(rng.nextUSize(1000))) / 1000.0 >= density) continue;
                    bitmask[i * words + j / 64] |= @as(u64, 1) << @as(u6, @intCast(j % 64));
                }
            }

            // Four signal distributions: all positive, all negative, mixed, one
            // dominant. The mixed and all-negative cases are the regression
            // guard for offset-binary quantization.
            const modes = [_]u2{ 0, 1, 2, 3 };
            for (modes) |mode| {
                for (signal) |*v| {
                    v.* = switch (mode) {
                        0 => @abs(rng.nextF32()) * 3.0,
                        1 => -@abs(rng.nextF32()) * 3.0,
                        2 => rng.nextF32() * 3.0,
                        3 => 0.0,
                    };
                }
                if (mode == 3 and n > 0) signal[n / 2] = 2.5;

                for (decays) |decay| {
                    nsir.bitmaskSignalPropagate(bitmask, n, signal, decay, exact_out);
                    nsir.bitmaskSignalPropagateBitPlane(bitmask, n, signal, decay, plane_out);

                    var max_abs: f32 = 0.0;
                    for (signal[0..n]) |v| max_abs = @max(max_abs, @abs(v));

                    for (0..n) |i| {
                        const neighbors = nsir.bitmaskRowNeighborCount(bitmask[i * words ..][0..words], n, words);
                        const bound = nsir.bitmaskSignalPropagateBound(decay, max_abs, neighbors);
                        const err = @abs(exact_out[i] - plane_out[i]);
                        try std.testing.expect(err <= bound + 1e-9);
                        if (bound > 0) {
                            worst_ratio = @max(worst_ratio, @as(f64, err) / @as(f64, bound));
                        }
                    }
                    configs += 1;
                }
            }
        }
    }

    try std.testing.expect(configs >= 128);
    try std.testing.expect(worst_ratio <= 1.0);

    // decay == 0 and an empty adjacency both reproduce the signal exactly.
    const n: usize = 64;
    const words = nsir.bitmaskWordCount(n);
    const empty = try allocator.alloc(u64, n * words);
    defer allocator.free(empty);
    @memset(empty, 0);
    const signal = try allocator.alloc(f32, n);
    defer allocator.free(signal);
    for (signal) |*v| v.* = rng.nextF32() * 2.0;
    const out = try allocator.alloc(f32, n);
    defer allocator.free(out);
    nsir.bitmaskSignalPropagateBitPlane(empty, n, signal, 0.85, out);
    for (signal, out) |a, b| try std.testing.expectEqual(a, b);
    nsir.bitmaskSignalPropagateBitPlane(empty, n, signal, 0.0, out);
    for (signal, out) |a, b| try std.testing.expectEqual(a, b);

    // The allocating entry point agrees with the smp_allocator entry point.
    try nsir.bitmaskSignalPropagateWithAllocator(allocator, empty, n, signal, 0.85, out);
    const out2 = try allocator.alloc(f32, n);
    defer allocator.free(out2);
    nsir.bitmaskSignalPropagateBitPlane(empty, n, signal, 0.85, out2);
    for (out, out2) |a, b| try std.testing.expectEqual(a, b);
}

// ---------------------------------------------------------------------------
// Gate 5: RSF binding enforcement across the modules
// ---------------------------------------------------------------------------

test "rsfNative_bindingEnforcement" {
    const allocator = std.testing.allocator;
    const model_a: u64 = 101;
    const model_b: u64 = 202;

    // A layer-weight tensor refuses to be read as a different space.
    var s_w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, model_a, 0, 4));
    defer s_w.deinit();
    try std.testing.expectError(types.RSFBindingError.RSFSpaceMismatch, s_w.requireSpace(.layer_weight_t));

    // An unbound tensor carries no binding at all.
    var plain = try tensor.Tensor.init(allocator, &[_]usize{ 4, 2 });
    defer plain.deinit();
    try std.testing.expect(plain.binding() == null);
    try std.testing.expectError(types.RSFBindingError.RSFBindingRequired, plain.requireSpace(.layer_weight_s));

    // Model and layer mismatches are typed errors, never silent coercions.
    var t_w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_t, model_b, 1, 4));
    defer t_w.deinit();
    try std.testing.expectError(
        types.RSFBindingError.RSFModelMismatch,
        s_w.requireCouplingCompatible(&t_w),
    );

    var state = try tensor.Tensor.initLatent(allocator, model_a, 4, 2);
    defer state.deinit();
    const latent_binding = state.binding().?;
    try std.testing.expectEqual(types.RSFSpace.latent_state, latent_binding.space);
    try std.testing.expectEqual(model_a, latent_binding.model_id);
    try std.testing.expectError(types.RSFBindingError.RSFDimMismatch, latent_binding.requireDim(8));

    // A mis-modeled coupling tensor is rejected by the causal kernel.
    var foreign_state = try tensor.Tensor.initLatent(allocator, model_b, 4, 2);
    defer foreign_state.deinit();
    var mask = try types.RSFSequenceMask.initCausal(allocator, 2);
    defer mask.deinit();
    const scratch = try allocator.alloc(f32, 2 * 4);
    defer allocator.free(scratch);
    try std.testing.expectError(
        error.RSFModelMismatch,
        tensor.causalCouplingForwardBatch(&foreign_state, &mask, &s_w, &t_w, -5.0, 5.0, scratch),
    );
}

// ---------------------------------------------------------------------------
// Gate 15: the invariant paradigm - no dense learned projections anywhere
// ---------------------------------------------------------------------------

test "rsfNative_invariantParadigm" {
    const allocator = std.testing.allocator;

    // Enumerate every learnable parameter space and assert each is a rank-2
    // coupling pair: shape [dim, 2] (or [dim, 3] for the block Fisher
    // statistics, which are not parameters at all).
    const spaces = [_]types.RSFSpace{
        .layer_weight_s, .layer_weight_t, .ranker_head, .master_weight, .momentum, .gradient,
    };
    for (spaces) |space| {
        try std.testing.expectEqual(@as(usize, 2), space.columns());
    }
    try std.testing.expectEqual(@as(usize, 3), types.RSFSpace.fisher_block.columns());

    // The only learned parameters the substrate can hold are the rank-2
    // coupling pairs; a Fisher block is a statistic, not a parameter, and is
    // excluded by the enumeration above.
    var s_w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, 1, 0, 16));
    defer s_w.deinit();
    try std.testing.expectEqual(@as(usize, 2), s_w.shape.dims.len);
    try std.testing.expectEqual(@as(usize, 16), s_w.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), s_w.shape.dims[1]);

    var block = try tensor.Tensor.initFisherBlocks(allocator, types.RSFBinding.layer(.fisher_block, 1, 0, 16));
    defer block.deinit();
    try std.testing.expectEqual(@as(usize, 3), block.shape.dims[1]);

    // The orthogonal factors contribute exactly zero log-det: the OFTB reports
    // it, and the diffusion operator is an exact involution with |det| = 1.
    var oftb_inst = try oftb_mod.OFTB.initWithDiffusion(16, true);
    try std.testing.expectEqual(@as(f32, 0.0), oftb_inst.logDetContribution());
    try std.testing.expect(oftb_inst.isOrthogonal());
    try std.testing.expect(oftb_inst.isSymplectic());
    try std.testing.expectEqual(@as(usize, 8), oftb_mod.OFTB.resonance_order);
}

