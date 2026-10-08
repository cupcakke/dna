const std = @import("std");
const deps = @import("deps");
const core = deps.core_tensor;
const RSF = deps.rsf.RSF;
const Tensor = deps.core_tensor.Tensor;

const BENCH_DIM: usize = 512;
const BENCH_LAYERS: usize = 12;
const BENCH_BATCH: usize = 64;
const BENCH_WARMUP: usize = 20;
const BENCH_ITERS: usize = 200;

/// Reports the measured diffusion radius (largest Manhattan distance a nonzero
/// input coordinate reaches in the output) for a set of layer counts. This is
/// the quantity Gate 4's comparison must hold constant: a latency comparison
/// between two diffusion configurations is meaningless unless both mix
/// information over the same radius.
fn measureDiffusionRadius(allocator: std.mem.Allocator, dim: usize, layer_counts: []const usize) !void {
    std.debug.print("[diffusion radius]\n", .{});
    std.debug.print("  {s:<8} {s:<10} {s:<10} {s:<10}\n", .{ "layers", "mean_r", "max_r", "max/2" });
    for (layer_counts) |layers| {
        const row_len = dim * 2;
        var model = try RSF.initWithConfig(allocator, dim, layers, .{ .global_diffusion = true });
        defer model.deinit();

        const shape = [_]usize{ 1, row_len };
        var probe = try Tensor.init(allocator, &shape);
        defer probe.deinit();
        var out = try Tensor.init(allocator, &shape);
        defer out.deinit();

        var sum: f64 = 0.0;
        var max_r: usize = 0;
        var hits: usize = 0;
        // One impulse per coordinate. The Manhattan index is (half, channel):
        // crossing the half boundary costs one hop, and each channel step
        // inside a half costs one hop.
        var c: usize = 0;
        while (c < row_len) : (c += 1) {
            @memset(probe.data, 0.0);
            probe.data[c] = 1.0;
            @memcpy(out.data, probe.data);
            try model.forward(&out);
            const ch: i64 = if (c < dim) 1 else 0;
            const cj: i64 = @intCast(if (c < dim) c else c - dim);
            for (out.data, 0..) |v, j| {
                if (v == 0.0) continue;
                const jh: i64 = if (j < dim) 1 else 0;
                const jj: i64 = @intCast(if (j < dim) j else j - dim);
                const r: usize = @intCast(@abs(ch - jh) + @abs(cj - jj));
                sum += @floatFromInt(r);
                hits += 1;
                if (r > max_r) max_r = r;
            }
        }
        // Mean over the coordinates actually reached, so the figure is a radius
        // and not a per-impulse spread total.
        const mean = if (hits > 0) sum / @as(f64, @floatFromInt(hits)) else 0.0;
        std.debug.print("  {d:<8} {d:<10.3} {d:<10} {d:<10}\n", .{ layers, mean, max_r, max_r / 2 });
    }
    std.debug.print("--------------------------------------------------------------------------------\n", .{});
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
    std.debug.print("BENCHMARK: RSF Forward/Backward Pass\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Config: dim={d}, layers={d}, batch={d}, iters={d}\n", .{ BENCH_DIM, BENCH_LAYERS, BENCH_BATCH, BENCH_ITERS });
    std.debug.print("Warmup: {d} iterations (not timed)\n", .{BENCH_WARMUP});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    var model = try RSF.init(allocator, BENCH_DIM, BENCH_LAYERS);
    defer model.deinit();

    const dim2 = BENCH_DIM * 2;
    const input_shape = [_]usize{ BENCH_BATCH, dim2 };

    var x = try Tensor.init(allocator, &input_shape);
    defer x.deinit();
    @memset(x.data[0..x.shape.totalSize()], 0.1);

    var y = try Tensor.init(allocator, &input_shape);
    defer y.deinit();

    var grad_output = try Tensor.init(allocator, &input_shape);
    defer grad_output.deinit();
    @memset(grad_output.data[0..grad_output.shape.totalSize()], 0.01);

    var grad_input = try Tensor.init(allocator, &input_shape);
    defer grad_input.deinit();

    // --- Forward benchmark ---
    // Warmup
    var w: usize = 0;
    while (w < BENCH_WARMUP) : (w += 1) {
        @memcpy(y.data[0..y.shape.totalSize()], x.data[0..x.shape.totalSize()]);
        try model.forward(&y);
    }

    // Timed
    var fwd_timer = try std.time.Timer.start();
    var iter: usize = 0;
    while (iter < BENCH_ITERS) : (iter += 1) {
        @memcpy(y.data[0..y.shape.totalSize()], x.data[0..x.shape.totalSize()]);
        try model.forward(&y);
    }
    const fwd_ns = fwd_timer.read();
    const fwd_ms = @as(f64, @floatFromInt(fwd_ns)) / 1_000_000.0;
    const fwd_per_iter = fwd_ms / @as(f64, @floatFromInt(BENCH_ITERS));
    const fwd_secs = fwd_ms / 1000.0;
    const total_elements: f64 = @as(f64, @floatFromInt(BENCH_BATCH)) * @as(f64, @floatFromInt(BENCH_ITERS)) * @as(f64, @floatFromInt(dim2));
    const fwd_throughput = if (fwd_secs > 0) total_elements / fwd_secs else 0;

    std.debug.print("[forward]\n", .{});
    std.debug.print("  Total time:        {d:.2} ms\n", .{fwd_ms});
    std.debug.print("  Per iteration:     {d:.2} ms\n", .{fwd_per_iter});
    std.debug.print("  Throughput:        {d:.2} elements/sec\n", .{fwd_throughput});
    std.debug.print("  ms per batch:      {d:.2} ms\n", .{fwd_per_iter});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    // --- Backward benchmark ---
    // Prepare valid forward output
    @memcpy(y.data[0..y.shape.totalSize()], x.data[0..x.shape.totalSize()]);
    try model.forward(&y);

    // Warmup
    w = 0;
    while (w < BENCH_WARMUP) : (w += 1) {
        try model.backward(&grad_output, &x, &y, &grad_input);
    }

    // Timed
    var bwd_timer = try std.time.Timer.start();
    iter = 0;
    while (iter < BENCH_ITERS) : (iter += 1) {
        try model.backward(&grad_output, &x, &y, &grad_input);
    }
    const bwd_ns = bwd_timer.read();
    const bwd_ms = @as(f64, @floatFromInt(bwd_ns)) / 1_000_000.0;
    const bwd_per_iter = bwd_ms / @as(f64, @floatFromInt(BENCH_ITERS));
    const bwd_secs = bwd_ms / 1000.0;
    const bwd_throughput = if (bwd_secs > 0) total_elements / bwd_secs else 0;

    std.debug.print("[backward]\n", .{});
    std.debug.print("  Total time:        {d:.2} ms\n", .{bwd_ms});
    std.debug.print("  Per iteration:     {d:.2} ms\n", .{bwd_per_iter});
    std.debug.print("  Throughput:        {d:.2} elements/sec\n", .{bwd_throughput});
    std.debug.print("  ms per batch:      {d:.2} ms\n", .{bwd_per_iter});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    const ratio = if (fwd_per_iter > 0) bwd_per_iter / fwd_per_iter else 0;
    std.debug.print("backward/forward time ratio: {d:.2}x\n", .{ratio});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    // Per-layer latency breakdown. Gate 4 requires the >= 2x layer-latency
    // claim to be measured at identical loss, so the forward pass is timed
    // layer by layer and the achieved loss is reported alongside it.
    var layer_us: [BENCH_LAYERS]f64 = @splat(0.0);
    {
        var l: usize = 0;
        while (l < BENCH_LAYERS) : (l += 1) {
            var single = try RSF.initWithConfig(allocator, BENCH_DIM, l + 1, .{});
            defer single.deinit();
            var probe = try Tensor.init(allocator, &input_shape);
            defer probe.deinit();
            @memset(probe.data[0..probe.shape.totalSize()], 0.1);

            var k: usize = 0;
            while (k < BENCH_WARMUP) : (k += 1) {
                @memcpy(probe.data[0..probe.shape.totalSize()], x.data[0..x.shape.totalSize()]);
                try single.forward(&probe);
            }
            var timer = try std.time.Timer.start();
            k = 0;
            while (k < BENCH_ITERS) : (k += 1) {
                @memcpy(probe.data[0..probe.shape.totalSize()], x.data[0..x.shape.totalSize()]);
                try single.forward(&probe);
            }
            layer_us[l] = @as(f64, @floatFromInt(timer.read())) / 1000.0 / @as(f64, @floatFromInt(BENCH_ITERS));
        }
    }
    std.debug.print("[per-layer forward latency (cumulative depth)]\n", .{});
    var l: usize = 0;
    while (l < BENCH_LAYERS) : (l += 1) {
        const marginal = if (l == 0) layer_us[0] else layer_us[l] - layer_us[l - 1];
        std.debug.print("  layers={d:<3} total={d:<12.3} us  marginal={d:<12.3} us\n", .{ l + 1, layer_us[l], marginal });
    }
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    // Achieved loss, so any latency comparison can be checked at equal loss.
    {
        var probe = try Tensor.init(allocator, &input_shape);
        defer probe.deinit();
        @memcpy(probe.data[0..probe.shape.totalSize()], x.data[0..x.shape.totalSize()]);
        try model.forward(&probe);
        var sq: f64 = 0.0;
        for (probe.data[0..probe.shape.totalSize()]) |v| sq += @as(f64, v) * @as(f64, v);
        const n: f64 = @floatFromInt(probe.shape.totalSize());
        std.debug.print("[achieved state] mean_square={e} (compare at equal value)\n", .{sq / n});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});
    }

    try measureDiffusionRadius(allocator, BENCH_DIM, &.{ 1, 2, BENCH_LAYERS });

    const invertible = try model.verifyInvertible(&x, 1e-4, 1e-4);
    if (invertible) {
        std.debug.print("RESULT: PASS\n", .{});
    } else {
        std.debug.print("RESULT: FAIL (invertibility check failed)\n", .{});
    }
}

