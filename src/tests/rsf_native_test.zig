const std = @import("std");
const jaide = @import("jaide");
const tensor = jaide.tensor;
const types = jaide.types;
const sfd = jaide.sfd;
const nsir = jaide.nsir_core;
const oftb_mod = jaide.oftb;

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

    fn nextF32(self: *Rng) f32 {
        const v = self.nextU64() >> 40;
        return @as(f32, @floatFromInt(v)) / 8388608.0 - 1.0;
    }

    fn nextUSize(self: *Rng, bound: usize) usize {
        if (bound == 0) return 0;
        return @as(usize, @intCast(self.nextU64() % bound));
    }
};

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

    const tr = a + c;
    const diff = a - c;
    const disc = @sqrt(diff * diff + 4.0 * b * b);
    var lambda_max = 0.5 * (tr + disc);

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

test "rsfNative_blockNaturalGradient" {
    var rng = Rng.init(0xB10C_2024);
    var worst_identity: f64 = 0.0;
    var worst_degenerate: f64 = 0.0;
    var worst_cond: f64 = 0.0;

    var iter: usize = 0;
    while (iter < 10000) : (iter += 1) {
        const a0 = rng.nextF32();
        const c0 = rng.nextF32();
        const b0 = rng.nextF32();
        const A: f64 = @abs(@as(f64, a0)) * 10.0 + 1e-3;
        const C: f64 = @abs(@as(f64, c0)) * 10.0 + 1e-3;
        const rho = b0 * 0.99;
        const B: f64 = @as(f64, rho) * @sqrt(A * C);
        const lambda: f64 = 1e-8;

        const m00 = A + lambda;
        const m11 = C + lambda;
        const m01 = B;

        const inv = sfd.blockInverseSqrt(A, B, C, lambda);

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

        try std.testing.expect(@abs(inv.inv01 - inv.inv01) < 1e-30);
        try std.testing.expect(inv.inv00 > 0.0);
        try std.testing.expect(inv.inv11 > 0.0);

        const abs_rho = @abs(@as(f64, rho));
        if (abs_rho < 1.0) {
            const residual = (1.0 + abs_rho) / (1.0 - abs_rho);
            worst_cond = @max(worst_cond, residual);
        }
    }
    try std.testing.expect(worst_identity < 1e-5);
    try std.testing.expect(worst_cond > 1.0);

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

    try tensor.diffuseRowInPlace(x, layout);

    for (orig, x) |want, back| {
        try std.testing.expect(@abs(want - back) < 1e-7);
    }

    try tensor.diffuseRowInPlace(once, layout);
    var n_in: f64 = 0.0;
    var n_out: f64 = 0.0;
    for (orig, once) |a, b| {
        n_in += @as(f64, a) * @as(f64, a);
        n_out += @as(f64, b) * @as(f64, b);
    }
    try std.testing.expect(@abs(@sqrt(n_in) - @sqrt(n_out)) < 1e-7);

    var oftb_inst = try oftb_mod.OFTB.initWithDiffusion(row_len / 2, true);
    try std.testing.expectEqual(@as(f32, 0.0), oftb_inst.logDetContribution());
}

test "rsfNative_diffusionGlobalMixing" {
    const row_lens = [_]usize{ 2, 4, 8, 24, 32, 96, 32768 * 3 };
    for (row_lens) |len| {
        try checkDiffusionInvariants(len, 0xD1FF_0000 + len);
    }

    const allocator = std.testing.allocator;

    const prod = diffusionLayoutOf(98304);
    try std.testing.expectEqual(@as(usize, 98304), prod.row_len);
    try std.testing.expectEqual(@as(usize, 3), prod.radix);
    try std.testing.expectEqual(@as(usize, 32768), prod.block);
    try std.testing.expectEqual(@as(usize, 15), prod.stages);

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

    try checkDiffusionInvariants(32, 0xFACE_0001);

    const bad = try allocator.alloc(f32, 6);
    defer allocator.free(bad);
    try std.testing.expectError(error.InvalidDimension, tensor.walshHadamardInPlace(bad));
}

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

    var worst: f32 = 0.0;
    for (state.data, original.data) |got, want| {
        worst = @max(worst, @abs(got - want));
    }
    try std.testing.expect(worst < 5.0e-7);

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

    var changed_after = false;
    for (perturbed_token..seq_len) |t| {
        for (0..dim * 2) |i| {
            const idx = t * dim * 2 + i;
            if (forward_out.data[idx] != probe.data[idx]) changed_after = true;
        }
    }
    try std.testing.expect(changed_after);

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

    const logdet_zero = try tensor.causalCouplingForwardBatch(&state, &zero_mask, &s_w, &t_w, -5.0, 5.0, scratch_k);
    var expected_zero_logdet: f64 = 0.0;
    for (0..dim) |d| {
        const clipped = @max(-5.0, @min(5.0, s_w.data[d * 2 + 1]));
        expected_zero_logdet += @as(f64, clipped) * @as(f64, @floatFromInt(seq_len));
    }
    try std.testing.expect(@abs(@as(f64, logdet_zero) - expected_zero_logdet) < 1e-4);

    var recovered = try state.clone(allocator);
    defer recovered.deinit();
    try tensor.causalCouplingInverseBatch(&recovered, &zero_mask, &s_w, &t_w, -5.0, 5.0, scratch_k, scratch_x2);
    for (0..seq_len) |t| {
        for (0..dim) |d| {
            const s_t = @max(-5.0, @min(5.0, s_w.data[d * 2 + 1]));
            const y1 = state.data[(t * dim + d) * 2 + 0];
            const y2 = state.data[(t * dim + d) * 2 + 1];
            const x2 = y2 - (t_w.data[d * 2] * y1 + t_w.data[d * 2 + 1]);
            const x1 = y1 * @exp(-s_t);
            try std.testing.expect(@abs(recovered.data[(t * dim + d) * 2 + 0] - x1) < 1e-6);
            try std.testing.expect(@abs(recovered.data[(t * dim + d) * 2 + 1] - x2) < 1e-6);
        }
    }

    var bytes = try allocator.alloc(u8, seq_len * seq_len);
    defer allocator.free(bytes);

    @memset(bytes, 0);
    bytes[2 * seq_len + 2] = 1;
    try std.testing.expectError(error.InvalidCausalMask, types.RSFSequenceMask.initFromBytes(allocator, seq_len, bytes));

    @memset(bytes, 0);
    bytes[1 * seq_len + 3] = 1;
    try std.testing.expectError(error.InvalidCausalMask, types.RSFSequenceMask.initFromBytes(allocator, seq_len, bytes));

    @memset(bytes, 0);
    bytes[3 * seq_len + 1] = 2;
    try std.testing.expectError(error.InvalidCausalMask, types.RSFSequenceMask.initFromBytes(allocator, seq_len, bytes));

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

    for (0..dim) |d| k_b[d] = 0.0;
    for (1..seq_len) |t| {
        for (0..dim) |d| k_b[t * dim + d] = k_b[(t - 1) * dim + d] + x2[(t - 1) * dim + d];
    }
    for (k_a, k_b) |a, b| try std.testing.expect(@abs(a - b) < 1e-6);

    var band = try types.RSFSequenceMask.initBand(allocator, seq_len, 2);
    defer band.deinit();
    try std.testing.expect(!band.full_causal);
    try band.set(5, 0);
    try std.testing.expect(band.get(5, 0));
    try std.testing.expect(!band.get(5, 2));
}

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

    try nsir.bitmaskSignalPropagateWithAllocator(allocator, empty, n, signal, 0.85, out);
    const out2 = try allocator.alloc(f32, n);
    defer allocator.free(out2);
    nsir.bitmaskSignalPropagateBitPlane(empty, n, signal, 0.85, out2);
    for (out, out2) |a, b| try std.testing.expectEqual(a, b);
}

test "rsfNative_bindingEnforcement" {
    const allocator = std.testing.allocator;
    const model_a: u64 = 101;
    const model_b: u64 = 202;

    var s_w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, model_a, 0, 4));
    defer s_w.deinit();
    try std.testing.expectError(types.RSFBindingError.RSFSpaceMismatch, s_w.requireSpace(.layer_weight_t));

    var plain = try tensor.Tensor.init(allocator, &[_]usize{ 4, 2 });
    defer plain.deinit();
    try std.testing.expect(plain.binding() == null);
    try std.testing.expectError(types.RSFBindingError.RSFBindingRequired, plain.requireSpace(.layer_weight_s));

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

test "rsfNative_invariantParadigm" {
    const allocator = std.testing.allocator;

    const spaces = [_]types.RSFSpace{
        .layer_weight_s, .layer_weight_t, .ranker_head, .master_weight, .momentum, .gradient,
    };
    for (spaces) |space| {
        try std.testing.expectEqual(@as(usize, 2), space.columns());
    }
    try std.testing.expectEqual(@as(usize, 3), types.RSFSpace.fisher_block.columns());

    var s_w = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, 1, 0, 16));
    defer s_w.deinit();
    try std.testing.expectEqual(@as(usize, 2), s_w.shape.dims.len);
    try std.testing.expectEqual(@as(usize, 16), s_w.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), s_w.shape.dims[1]);

    var block = try tensor.Tensor.initFisherBlocks(allocator, types.RSFBinding.layer(.fisher_block, 1, 0, 16));
    defer block.deinit();
    try std.testing.expectEqual(@as(usize, 3), block.shape.dims[1]);

    var oftb_inst = try oftb_mod.OFTB.initWithDiffusion(16, true);
    try std.testing.expectEqual(@as(f32, 0.0), oftb_inst.logDetContribution());
    try std.testing.expect(oftb_inst.isOrthogonal());
    try std.testing.expect(oftb_inst.isSymplectic());
    try std.testing.expectEqual(@as(usize, 8), oftb_mod.OFTB.resonance_order);
}

test "QR reorthogonalizes complete columns and reconstructs rank deficient matrices" {
    const vpu = jaide.vpu;
    var ops = vpu.MatrixOps.init(std.testing.allocator);
    const matrices = [_]vpu.Matrix4x4{
        .{ .data = .{ vpu.F32x4.initFromArray(.{ 1, 2, 3, 4 }), vpu.F32x4.initFromArray(.{ 2, 5, 7, 1 }), vpu.F32x4.initFromArray(.{ 3, 1, 4, 2 }), vpu.F32x4.initFromArray(.{ 1, 0, 2, 8 }) } },
        .{ .data = .{ vpu.F32x4.initFromArray(.{ 1, 2, 0, 4 }), vpu.F32x4.initFromArray(.{ 2, 4, 0, 8 }), vpu.F32x4.initFromArray(.{ 3, 6, 0, 12 }), vpu.F32x4.initFromArray(.{ 4, 8, 0, 16 }) } },
        vpu.Matrix4x4.zero(),
    };
    for (matrices) |matrix| {
        const qr = ops.qr_decomposition(matrix);
        const reconstructed = ops.matmul4x4(qr.q, qr.r);
        const qtq = ops.matmul4x4(ops.transpose4x4(qr.q), qr.q);
        for (0..4) |i| {
            for (0..4) |j| {
                try std.testing.expectApproxEqAbs(matrix.get(i, j), reconstructed.get(i, j), 1e-4);
                try std.testing.expectApproxEqAbs(@as(f32, if (i == j) 1 else 0), qtq.get(i, j), 1e-5);
                if (i > j) try std.testing.expectEqual(@as(f32, 0), qr.r.get(i, j));
            }
        }
    }
}

test "packed GEMM handles odd reduction dimensions and partial tiles" {
    const allocator = std.testing.allocator;
    for ([_]usize{ 3, 5, 9, 17, 129, 257, 259 }) |k| {
        const m: usize = if (k >= 128) 137 else 17;
        const n: usize = if (k >= 128) 131 else 19;
        var a = try tensor.Tensor.init(allocator, &.{ m, k });
        defer a.deinit();
        var b = try tensor.Tensor.init(allocator, &.{ k, n });
        defer b.deinit();
        for (a.data, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 7)) * 0.125;
        for (b.data, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 11)) * 0.0625;
        var c = try tensor.Tensor.matmul(&a, &b, allocator);
        defer c.deinit();
        for (0..m) |i| {
            for (0..n) |j| {
                var expected: f32 = 0;
                for (0..k) |p| expected += a.data[i * k + p] * b.data[p * n + j];
                try std.testing.expectApproxEqAbs(expected, c.data[i * n + j], 1e-4);
            }
        }
    }
}

test "VPU coupling follows coupling rotation diffusion order" {
    const vpu = jaide.vpu;
    const allocator = std.testing.allocator;
    for ([_]bool{ false, true }) |diffusion| {
        const dim = 4;
        var sw = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, 23, 0, dim));
        defer sw.deinit();
        var tw = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_t, 23, 0, dim));
        defer tw.deinit();
        for (sw.data, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i + 1)) * 0.01;
        for (tw.data, 0..) |*value, i| value.* = -@as(f32, @floatFromInt(i + 1)) * 0.02;
        var batch = vpu.CouplingBatch.init(allocator, dim, types.RSFBinding.detached(.latent_state, dim));
        defer batch.deinit();
        const original = [_]f32{ 0.1, -0.2, 0.3, 0.4, 0.2, -0.4, 0.1, -0.3 };
        _ = try batch.addRow(&original);
        _ = try batch.addRow(&original);
        const params = vpu.CouplingParamsRef{ .model_id = 23, .layer_index = 0, .dim = dim, .s_weight = &sw, .t_weight = &tw, .diffusion = diffusion };
        var scale: [dim]f32 = undefined;
        var trans: [dim]f32 = undefined;
        var expected = original;
        const coupling = try params.params();
        _ = try tensor.couplingForwardRows(coupling, &expected, 1, &scale, &trans);
        var oftb = try oftb_mod.OFTB.initWithDiffusion(dim, diffusion);
        try oftb.forwardSliceInPlace(&expected);
        const forward_ld = try batch.forward(params, &scale, &trans);
        for (0..2) |i| for (try batch.rowConst(i), expected) |actual, reference| {
            try std.testing.expectApproxEqAbs(reference, actual, 1e-6);
        };
        const inverse_ld = try batch.inverse(params, &scale, &trans);
        try std.testing.expectApproxEqAbs(forward_ld, inverse_ld, 1e-6);
        for (0..2) |i| for (try batch.rowConst(i), original) |actual, reference| {
            try std.testing.expectApproxEqAbs(reference, actual, 1e-6);
        };
    }
}
