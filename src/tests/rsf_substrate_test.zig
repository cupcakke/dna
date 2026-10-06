const std = @import("std");
const jaide = @import("jaide");
const tensor = jaide.tensor;
const types = jaide.types;

fn makeWeights(allocator: std.mem.Allocator, model_id: u64) !struct { s: tensor.Tensor, t: tensor.Tensor } {
    var s = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, model_id, 0, 2));
    errdefer s.deinit();
    var t = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_t, model_id, 0, 2));
    errdefer t.deinit();
    s.data[0] = 0.2;
    s.data[1] = 0.1;
    s.data[2] = -0.1;
    s.data[3] = -0.2;
    t.data[0] = 0.3;
    t.data[1] = 0.05;
    t.data[2] = 0.15;
    t.data[3] = -0.04;
    return .{ .s = s, .t = t };
}

test "rank two spectral norm is exact and bound aware" {
    const allocator = std.testing.allocator;
    var weights = try tensor.Tensor.initCoupling(allocator, types.RSFBinding.layer(.layer_weight_s, 11, 0, 2));
    defer weights.deinit();
    weights.data[0] = 3.0;
    weights.data[1] = 0.0;
    weights.data[2] = 4.0;
    weights.data[3] = 0.0;
    const sigma = try tensor.couplingSpectralNorm(&weights);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), sigma, 1.0e-5);
    const before = try tensor.constrainCouplingSpectralNorm(&weights, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), before, 1.0e-5);
    const after = try tensor.couplingSpectralNorm(&weights);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), after, 1.0e-5);
}

test "strict causal batch forward and inverse round trip" {
    const allocator = std.testing.allocator;
    var mask = try types.RSFSequenceMask.initCausal(allocator, 4);
    defer mask.deinit();
    var state = try tensor.Tensor.initLatent(allocator, 17, 2, 4);
    defer state.deinit();
    var original = try state.clone(allocator);
    defer original.deinit();
    var pair = try makeWeights(allocator, 17);
    defer pair.s.deinit();
    defer pair.t.deinit();
    for (state.data, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i + 1)) * 0.03;
    try original.copyFrom(&state);
    const key = try allocator.alloc(f32, 8);
    defer allocator.free(key);
    _ = try tensor.causalCouplingForwardBatch(&state, &mask, &pair.s, &pair.t, -5.0, 5.0, key);
    const inverse_key = try allocator.alloc(f32, 8);
    defer allocator.free(inverse_key);
    const x2 = try allocator.alloc(f32, 8);
    defer allocator.free(x2);
    try tensor.causalCouplingInverseBatch(&state, &mask, &pair.s, &pair.t, -5.0, 5.0, inverse_key, x2);
    for (state.data, original.data) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected, actual, 1.0e-5);
    }
}

test "sparse causal keys exclude the current token" {
    const allocator = std.testing.allocator;
    var mask = try types.RSFSequenceMask.initBand(allocator, 4, 2);
    defer mask.deinit();
    var x2 = [_]f32{ 1.0, 10.0, 2.0, 20.0, 4.0, 40.0, 8.0, 80.0 };
    var keys = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    try tensor.causalPrefixSum(&mask, &x2, 2, &keys);
    try std.testing.expectEqual(@as(f32, 0.0), keys[0]);
    try std.testing.expectEqual(@as(f32, 0.0), keys[1]);
    try std.testing.expectEqual(@as(f32, 1.0), keys[2]);
    try std.testing.expectEqual(@as(f32, 10.0), keys[3]);
    try std.testing.expectEqual(@as(f32, 3.0), keys[4]);
    try std.testing.expectEqual(@as(f32, 30.0), keys[5]);
    try std.testing.expectEqual(@as(f32, 6.0), keys[6]);
    try std.testing.expectEqual(@as(f32, 60.0), keys[7]);
}

test "causal adjoint scratch is caller-owned" {
    var scratch = try tensor.CausalAdjointScratch.init(std.testing.allocator, 4, 2);
    defer scratch.deinit();
    try std.testing.expectEqual(@as(usize, 8), scratch.ds.len);
    try std.testing.expectEqual(@as(usize, 4), scratch.ds_weight.len);
}

test "analytic Fisher block inverse is exact" {
    const sfd = jaide.sfd;
    const inverse = sfd.blockInverseFisher(2.0, 0.0, 4.0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), inverse.inv00, 1.0e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), inverse.inv11, 1.0e-12);
    try std.testing.expectEqual(@as(f64, 0.0), inverse.inv01);
}
