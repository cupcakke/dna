const std = @import("std");
const deps = @import("deps");
const core = deps.core_tensor;
const Tensor = deps.core_tensor.Tensor;
const RSF = deps.rsf.RSF;
const SpectralNormalizer = deps.sfd.SpectralNormalizer;

const QUANT_N: usize = 1 << 20;
const QUANT_ITERS: usize = 100;

const WEIGHT_DIM: usize = 512;
const SPECTRAL_ITERS: usize = 50;

fn quantizeFP4(value: f32) f32 {
    if (!std.math.isFinite(value)) return value;
    const clamped = std.math.clamp(value, -6.0, 6.0);
    const abs_v = if (clamped < 0) -clamped else clamped;
    const sign: f32 = if (clamped < 0) -1.0 else 1.0;
    const best: f32 = if (abs_v < 0.25) 0.0
        else if (abs_v < 0.75) 0.5
        else if (abs_v < 1.25) 1.0
        else if (abs_v < 1.75) 1.5
        else if (abs_v < 2.5) 2.0
        else if (abs_v < 3.5) 3.0
        else if (abs_v < 5.0) 4.0
        else 6.0;
    return sign * best;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            std.debug.print("RESULT: FAIL (memory leak detected)\n", .{});
        }
    }
    const allocator = gpa.allocator();

    const eff = core.effectiveCpuCount();
    const src = core.cgroupSource();
    std.debug.print("[env] effective_cpu={d} cgroup_source={s}\n", .{ eff, src });

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("BENCHMARK: SFD Optimizations (FP4 quantization + SpectralNorm)\n", .{});
    std.debug.print("================================================================================\n", .{});

    {
        std.debug.print("Config: quant_n={d}, iters={d}\n", .{ QUANT_N, QUANT_ITERS });
        std.debug.print("--------------------------------------------------------------------------------\n", .{});

        const input = try allocator.alloc(f32, QUANT_N);
        defer allocator.free(input);
        const output = try allocator.alloc(f32, QUANT_N);
        defer allocator.free(output);

        var i: usize = 0;
        while (i < QUANT_N) : (i += 1) {
            const t_val: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(QUANT_N - 1));
            input[i] = -6.0 + 12.0 * t_val;
        }

        var timer = try std.time.Timer.start();
        var iter: usize = 0;
        while (iter < QUANT_ITERS) : (iter += 1) {
            for (input, 0..) |v, idx| {
                output[idx] = quantizeFP4(v);
            }
        }
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const total_elements = @as(f64, @floatFromInt(QUANT_N)) * @as(f64, @floatFromInt(QUANT_ITERS));
        const ns_per_elem = @as(f64, @floatFromInt(elapsed_ns)) / total_elements;
        const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        const elems_per_sec = if (elapsed_secs > 0) total_elements / elapsed_secs else 0;

        std.debug.print("[FP4 quantization]\n", .{});
        std.debug.print("  Total time:        {d:.2} ms\n", .{elapsed_ms});
        std.debug.print("  ns per element:    {d:.2} ns\n", .{ns_per_elem});
        std.debug.print("  Throughput:        {d:.2} elements/sec\n", .{elems_per_sec});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});
    }

    {
        std.debug.print("Config: coupling_dim={d}, iterations={d}, method=exact_rank2\n", .{ WEIGHT_DIM, SPECTRAL_ITERS });
        std.debug.print("--------------------------------------------------------------------------------\n", .{});
        var model = try RSF.init(allocator, WEIGHT_DIM, 1);
        defer model.deinit();
        var weights = try Tensor.initCoupling(allocator, .{
            .space = .layer_weight_s,
            .model_id = model.id,
            .layer_index = 0,
            .dim = WEIGHT_DIM,
        });
        defer weights.deinit();
        var prng = std.Random.DefaultPrng.init(42);
        for (weights.data) |*value| value.* = prng.random().float(f32) * 2.0 - 1.0;
        var normalizer = SpectralNormalizer.init();
        var timer = try std.time.Timer.start();
        var iter: usize = 0;
        while (iter < SPECTRAL_ITERS) : (iter += 1) {
            for (weights.data, 0..) |*value, i| value.* = @as(f32, @floatFromInt((i % 17) + 1)) * 0.01;
            try normalizer.normalizeWeights(&weights, allocator);
        }
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter = elapsed_ms / @as(f64, @floatFromInt(SPECTRAL_ITERS));
        const sigma = try core.exactSpectralNormRank2(weights.data, WEIGHT_DIM);
        std.debug.print("[SpectralNorm exact_rank2]\n", .{});
        std.debug.print("  Total time:        {d:.2} ms\n", .{elapsed_ms});
        std.debug.print("  Per iteration:     {d:.2} ms\n", .{per_iter});
        std.debug.print("  Final exact sigma:  {d:.6}\n", .{sigma});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});
    }

    std.debug.print("RESULT: PASS\n", .{});
}
