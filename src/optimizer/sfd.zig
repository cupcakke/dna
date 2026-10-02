const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("../core/types.zig");
const tensor_mod = @import("../core/tensor.zig");
const Tensor = tensor_mod.Tensor;
const core_io = @import("../core/io.zig");
const rsf_mod = @import("../processor/rsf.zig");
const RSF = rsf_mod.RSF;
const RSFBranch = rsf_mod.RSFBranch;

const coupling_width: usize = 2;
const WEIGHT_COLUMN: usize = 0;
const BIAS_COLUMN: usize = 1;
const FISHER_LAYOUT: u32 = 3;
const SFD4_MAGIC: u32 = 0x53464434;
const SFD3_MAGIC: u32 = 0x53464433;
const MIN_DET: f64 = 1.0e-12;
const SPECTRAL_REPROJECT_REL: f64 = 1.0e-6;

pub const FisherMode = enum(u8) {
    rsf_gauss_newton = 0,
    gradient = 1,
    external = 2,
};

pub const SFDConfig = struct {
    beta1: f32 = 0.9,
    fisher_gamma: f32 = 0.99,
    fisher_epsilon: f32 = 1e-8,
    fisher_mode: FisherMode = .rsf_gauss_newton,
    eps: f32 = 1e-8,
    clip_threshold: f32 = 0.1,
    weight_floor: f32 = 1e-3,
    fisher_max: f32 = 1e6,
    warmup_steps: usize = 10,
    spectral_target: f32 = 0.9,
};

pub const StepStats = struct {
    step: u64,
    lr_effective: f32,
    grad_global_norm: f64,
    fisher_block_mean: f64,
    fisher_offdiag_mean_abs: f64,
    correlation_abs_mean: f64,
    condition_number_max: f64,
    clipped_fraction: f64,
    spectral_reprojected: bool,
};

pub const BlockInv = struct {
    inv00: f64,
    inv01: f64,
    inv11: f64,
};

pub fn blockInverseSqrt(A: f64, B: f64, C: f64, lambda: f64) BlockInv {
    const Ad = A + lambda;
    const Cd = C + lambda;
    const det_raw = Ad * Cd - B * B;
    const det = if (det_raw > MIN_DET) det_raw else MIN_DET;
    const sqrt_det = @sqrt(det);
    const S = @sqrt(Ad + Cd + 2.0 * sqrt_det);
    const denom = sqrt_det * S;
    const alpha = if (denom > 0.0) 1.0 / denom else 0.0;
    return .{
        .inv00 = alpha * (Cd + sqrt_det),
        .inv01 = -alpha * B,
        .inv11 = alpha * (Ad + sqrt_det),
    };
}

fn validateConfig(cfg: SFDConfig) !void {
    if (!std.math.isFinite(cfg.beta1) or cfg.beta1 < 0.0 or cfg.beta1 >= 1.0) return error.InvalidBeta1;
    if (!std.math.isFinite(cfg.fisher_gamma) or cfg.fisher_gamma < 0.0 or cfg.fisher_gamma >= 1.0) return error.InvalidFisherGamma;
    if (!std.math.isFinite(cfg.fisher_epsilon) or cfg.fisher_epsilon < 1.0e-12) return error.InvalidFisherEpsilon;
    if (!std.math.isFinite(cfg.eps) or cfg.eps <= 0.0) return error.InvalidEpsilon;
    if (!std.math.isFinite(cfg.clip_threshold) or cfg.clip_threshold <= 0.0 or cfg.clip_threshold > 1.0) return error.InvalidClipThreshold;
    if (!std.math.isFinite(cfg.weight_floor) or cfg.weight_floor <= 0.0) return error.InvalidWeightFloor;
    if (!std.math.isFinite(cfg.fisher_max) or cfg.fisher_max <= 0.0) return error.InvalidFisherMax;
    if (!std.math.isFinite(cfg.spectral_target) or !(cfg.spectral_target > 0.0)) return error.InvalidSpectralTarget;
}

fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return error.Overflow;
}

fn pairLen(dim: usize) !usize {
    return checkedMul(dim, coupling_width);
}

fn blockLen(dim: usize) !usize {
    return checkedMul(dim, 3);
}

fn crcU8(hasher: *std.hash.Crc32, v: u8) void {
    const b = [_]u8{v};
    hasher.update(&b);
}

fn crcU32LE(hasher: *std.hash.Crc32, v: u32) void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    hasher.update(&b);
}

fn crcU64LE(hasher: *std.hash.Crc32, v: u64) void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    hasher.update(&b);
}

fn crcF32LE(hasher: *std.hash.Crc32, v: f32) void {
    crcU32LE(hasher, @bitCast(v));
}

fn capDiag(value: f32, cap: f32) f32 {
    if (!std.math.isFinite(value) or value < 0.0) return 0.0;
    if (value > cap) return cap;
    return value;
}

fn capOff(value: f32, cap: f32) f32 {
    if (!std.math.isFinite(value)) return 0.0;
    if (value > cap) return cap;
    if (value < -cap) return -cap;
    return value;
}

fn safeGradient(g: f32) f32 {
    return if (std.math.isFinite(g)) g else 0.0;
}

fn warmupFactor(step: u64, warmup_steps: usize) f32 {
    if (warmup_steps == 0) return 1.0;
    if (step >= warmup_steps) return 1.0;
    return @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(warmup_steps));
}

fn conditionNumber(A: f64, B: f64, C: f64) f64 {
    const disc = (A - C) * (A - C) + 4.0 * B * B;
    const root = @sqrt(if (disc > 0.0) disc else 0.0);
    const lam_max = 0.5 * (A + C + root);
    const lam_min = 0.5 * (A + C - root);
    if (!(lam_min > 0.0)) return std.math.inf(f64);
    return lam_max / lam_min;
}

fn applyBlockVec(inv: BlockInv, x0: f64, x1: f64) struct { y0: f64, y1: f64 } {
    return .{
        .y0 = inv.inv00 * x0 + inv.inv01 * x1,
        .y1 = inv.inv01 * x0 + inv.inv11 * x1,
    };
}

pub const SpectralNormalizerConfig = struct {
    power_iterations: usize = 5,
    eps: f32 = 1e-12,
    max_singular_value: f32 = 1.0,
};

pub const SpectralNormalizer = struct {
    power_iterations: usize,
    eps: f32,
    max_singular_value: f32,

    pub fn init(power_iterations: usize) SpectralNormalizer {
        return .{
            .power_iterations = power_iterations,
            .eps = 1e-12,
            .max_singular_value = 1.0,
        };
    }

    pub fn initWithConfig(config: SpectralNormalizerConfig) SpectralNormalizer {
        return .{
            .power_iterations = config.power_iterations,
            .eps = config.eps,
            .max_singular_value = config.max_singular_value,
        };
    }

    pub fn normalizeWeights(self: *SpectralNormalizer, weights: *Tensor, allocator: Allocator) !void {
        if (weights.shape.dims.len != 2) return error.InvalidShape;
        const rows = weights.shape.dims[0];
        const cols = weights.shape.dims[1];
        if (rows == 0 or cols == 0) return error.InvalidShape;
        if (cols == coupling_width) {
            const sigma = try tensor_mod.constrainCouplingSpectralNorm(weights.data, rows, self.max_singular_value);
            _ = sigma;
            return;
        }
        const sigma = try denseSpectralNorm(weights.data, rows, cols, self.power_iterations, self.eps, allocator);
        if (!std.math.isFinite(sigma)) return error.NonFinite;
        if (sigma > self.max_singular_value and sigma > 0.0) {
            try weights.mulScalar(self.max_singular_value / sigma);
        }
    }

    pub fn lipschitzRegularization(_: *const SpectralNormalizer, loss: f32, spectral_norms: []const f32, lambda: f32) f32 {
        var reg_term: f32 = 0.0;
        for (spectral_norms) |sigma| {
            const deviation = sigma - 1.0;
            reg_term += deviation * deviation;
        }
        return loss + lambda * reg_term;
    }
};

fn denseSpectralNorm(data: []const f32, rows: usize, cols: usize, iterations: usize, eps: f32, allocator: Allocator) !f32 {
    const n = try checkedMul(rows, cols);
    if (data.len < n) return error.InvalidShape;
    const v = try allocator.alloc(f32, cols);
    defer allocator.free(v);
    const u = try allocator.alloc(f32, rows);
    defer allocator.free(u);
    const inv_sqrt: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(if (cols == 0) 1 else cols)));
    var j: usize = 0;
    while (j < cols) : (j += 1) v[j] = inv_sqrt;
    const iters = if (iterations == 0) @as(usize, 1) else iterations;
    var iter: usize = 0;
    while (iter < iters) : (iter += 1) {
        @memset(u, 0.0);
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            var acc: f64 = 0.0;
            j = 0;
            while (j < cols) : (j += 1) acc += @as(f64, data[i * cols + j]) * @as(f64, v[j]);
            u[i] = @floatCast(acc);
        }
        var un: f64 = 0.0;
        i = 0;
        while (i < rows) : (i += 1) un += @as(f64, u[i]) * @as(f64, u[i]);
        un = @sqrt(un);
        if (un > @as(f64, eps)) {
            const inv = 1.0 / un;
            i = 0;
            while (i < rows) : (i += 1) u[i] = @floatCast(@as(f64, u[i]) * inv);
        }
        @memset(v, 0.0);
        i = 0;
        while (i < rows) : (i += 1) {
            j = 0;
            while (j < cols) : (j += 1) v[j] += u[i] * data[i * cols + j];
        }
        var vn: f64 = 0.0;
        j = 0;
        while (j < cols) : (j += 1) vn += @as(f64, v[j]) * @as(f64, v[j]);
        vn = @sqrt(vn);
        if (vn > @as(f64, eps)) {
            const inv = 1.0 / vn;
            j = 0;
            while (j < cols) : (j += 1) v[j] = @floatCast(@as(f64, v[j]) * inv);
        }
    }
    var sigma: f64 = 0.0;
    var i: usize = 0;
    while (i < rows) : (i += 1) {
        var j2: usize = 0;
        while (j2 < cols) : (j2 += 1) {
            sigma += @as(f64, u[i]) * @as(f64, data[i * cols + j2]) * @as(f64, v[j2]);
        }
    }
    const out: f32 = @floatCast(if (sigma >= 0.0) sigma else -sigma);
    if (!std.math.isFinite(out)) return error.NonFinite;
    return out;
}

pub const KFACBlock = struct {
    A_block: Tensor,
    G_block: Tensor,
    damping: f32,
    alpha: f32,
    dim: usize,
    model_id: u64,
    layer: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, model: *const RSF, layer: usize, damping: f32) !KFACBlock {
        return initWithAlpha(allocator, model, layer, damping, 0.95);
    }

    pub fn initWithAlpha(allocator: Allocator, model: *const RSF, layer: usize, damping: f32, alpha: f32) !KFACBlock {
        if (!std.math.isFinite(damping) or damping < 0.0) return error.InvalidEpsilon;
        if (!std.math.isFinite(alpha) or alpha < 0.0 or alpha >= 1.0) return error.InvalidBeta1;
        const dim = try model.dim();
        const layers = try model.layerCount();
        if (layer >= layers) return error.LayerIndexOutOfBounds;
        if (dim == 0) return error.InvalidDimension;
        const blen = try blockLen(dim);
        _ = blen;
        var A = try Tensor.zeros(allocator, &[_]usize{ dim, 3 });
        errdefer A.deinit();
        var G = try Tensor.zeros(allocator, &[_]usize{ dim, 3 });
        errdefer G.deinit();
        return .{
            .A_block = A,
            .G_block = G,
            .damping = damping,
            .alpha = alpha,
            .dim = dim,
            .model_id = model.id,
            .layer = layer,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KFACBlock) void {
        self.A_block.deinit();
        self.G_block.deinit();
    }

    pub fn updateStatistics(self: *KFACBlock, x2_batch: *const Tensor, dy_batch: *const Tensor) !void {
        if (x2_batch.shape.dims.len != 2) return error.InvalidShape;
        if (dy_batch.shape.dims.len != 2) return error.InvalidShape;
        const batch = x2_batch.shape.dims[0];
        if (batch == 0) return error.InvalidBatchSize;
        if (x2_batch.shape.dims[1] != self.dim) return error.ShapeMismatch;
        const dim2 = try pairLen(self.dim);
        if (dy_batch.shape.dims[0] != batch or dy_batch.shape.dims[1] != dim2) return error.ShapeMismatch;
        const inv_b: f64 = 1.0 / @as(f64, @floatFromInt(batch));
        const keep: f64 = self.alpha;
        const mix: f64 = 1.0 - keep;
        var d: usize = 0;
        while (d < self.dim) : (d += 1) {
            var sum_x2sq: f64 = 0.0;
            var sum_x2: f64 = 0.0;
            var sum_dy2sq: f64 = 0.0;
            var sum_cross: f64 = 0.0;
            var sum_dy1sq: f64 = 0.0;
            var b: usize = 0;
            while (b < batch) : (b += 1) {
                const x2: f64 = x2_batch.data[b * self.dim + d];
                const dy1: f64 = dy_batch.data[b * dim2 + d];
                const dy2: f64 = dy_batch.data[b * dim2 + self.dim + d];
                sum_x2sq += x2 * x2;
                sum_x2 += x2;
                sum_dy2sq += dy2 * dy2;
                sum_cross += dy1 * dy2;
                sum_dy1sq += dy1 * dy1;
            }
            const a0: f32 = @floatCast(keep * @as(f64, self.A_block.data[d * 3 + 0]) + mix * sum_x2sq * inv_b);
            const a1: f32 = @floatCast(keep * @as(f64, self.A_block.data[d * 3 + 1]) + mix * sum_x2 * inv_b);
            const a2: f32 = @floatCast(keep * @as(f64, self.A_block.data[d * 3 + 2]) + mix * 1.0);
            const g0: f32 = @floatCast(keep * @as(f64, self.G_block.data[d * 3 + 0]) + mix * sum_dy2sq * inv_b);
            const g1: f32 = @floatCast(keep * @as(f64, self.G_block.data[d * 3 + 1]) + mix * sum_cross * inv_b);
            const g2: f32 = @floatCast(keep * @as(f64, self.G_block.data[d * 3 + 2]) + mix * sum_dy1sq * inv_b);
            if (!std.math.isFinite(a0) or !std.math.isFinite(a1) or !std.math.isFinite(a2)) return error.NonFinite;
            if (!std.math.isFinite(g0) or !std.math.isFinite(g1) or !std.math.isFinite(g2)) return error.NonFinite;
            self.A_block.data[d * 3 + 0] = a0;
            self.A_block.data[d * 3 + 1] = a1;
            self.A_block.data[d * 3 + 2] = a2;
            self.G_block.data[d * 3 + 0] = g0;
            self.G_block.data[d * 3 + 1] = g1;
            self.G_block.data[d * 3 + 2] = g2;
        }
    }

    pub fn preconditionGradient(self: *const KFACBlock, grad: *Tensor) !void {
        if (grad.shape.dims.len != 2) return error.InvalidShape;
        if (grad.shape.dims[0] != self.dim or grad.shape.dims[1] != coupling_width) return error.ShapeMismatch;
        const lambda: f64 = self.damping;
        var d: usize = 0;
        while (d < self.dim) : (d += 1) {
            const A = blockInverseSqrt(
                self.A_block.data[d * 3 + 0],
                self.A_block.data[d * 3 + 1],
                self.A_block.data[d * 3 + 2],
                lambda,
            );
            const G = blockInverseSqrt(
                self.G_block.data[d * 3 + 0],
                self.G_block.data[d * 3 + 1],
                self.G_block.data[d * 3 + 2],
                lambda,
            );
            const gw: f64 = grad.data[d * coupling_width + WEIGHT_COLUMN];
            const gb: f64 = grad.data[d * coupling_width + BIAS_COLUMN];
            const whitened = applyBlockVec(A, gw, gb);
            const out = applyBlockVec(G, whitened.y0, whitened.y1);
            const ow: f32 = @floatCast(out.y0);
            const ob: f32 = @floatCast(out.y1);
            if (!std.math.isFinite(ow) or !std.math.isFinite(ob)) return error.NonFinite;
            grad.data[d * coupling_width + WEIGHT_COLUMN] = ow;
            grad.data[d * coupling_width + BIAS_COLUMN] = ob;
        }
    }
};

pub const SFD = struct {
    fisher_blocks_s: []Tensor,
    fisher_blocks_t: []Tensor,
    momentum_s: []Tensor,
    momentum_t: []Tensor,
    master_s: []Tensor,
    master_t: []Tensor,
    model_id: u64,
    dim: usize,
    num_layers: usize,
    step_count: u64,
    cfg: SFDConfig,
    allocator: Allocator,
    initialized: bool,

    pub fn init(allocator: Allocator, model: *const RSF) !SFD {
        return initWithConfig(allocator, model, .{});
    }

    pub fn initWithConfig(allocator: Allocator, model: *const RSF, cfg: SFDConfig) !SFD {
        try validateConfig(cfg);
        const model_id = model.id;
        if (model_id == 0) return error.ModelMismatch;
        const dim = try model.dim();
        const num_layers = try model.layerCount();
        if (dim == 0) return error.InvalidDimension;
        if (num_layers == 0) return error.InvalidLayerCount;
        const plen = try pairLen(dim);
        var fisher_s = try allocator.alloc(Tensor, num_layers);
        errdefer allocator.free(fisher_s);
        var fisher_t = try allocator.alloc(Tensor, num_layers);
        errdefer allocator.free(fisher_t);
        var mom_s = try allocator.alloc(Tensor, num_layers);
        errdefer allocator.free(mom_s);
        var mom_t = try allocator.alloc(Tensor, num_layers);
        errdefer allocator.free(mom_t);
        var master_s = try allocator.alloc(Tensor, num_layers);
        errdefer allocator.free(master_s);
        var master_t = try allocator.alloc(Tensor, num_layers);
        errdefer allocator.free(master_t);
        var initialized_layers: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < initialized_layers) : (j += 1) {
                fisher_s[j].deinit();
                fisher_t[j].deinit();
                mom_s[j].deinit();
                mom_t[j].deinit();
                master_s[j].deinit();
                master_t[j].deinit();
            }
        }
        const s_tmp = try allocator.alloc(f32, plen);
        defer allocator.free(s_tmp);
        const t_tmp = try allocator.alloc(f32, plen);
        defer allocator.free(t_tmp);
        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            fisher_s[l] = try Tensor.zeros(allocator, &[_]usize{ dim, 3 });
            errdefer fisher_s[l].deinit();
            fisher_t[l] = try Tensor.zeros(allocator, &[_]usize{ dim, 3 });
            errdefer fisher_t[l].deinit();
            mom_s[l] = try Tensor.zeros(allocator, &[_]usize{ dim, 2 });
            errdefer mom_s[l].deinit();
            mom_t[l] = try Tensor.zeros(allocator, &[_]usize{ dim, 2 });
            errdefer mom_t[l].deinit();
            master_s[l] = try Tensor.zeros(allocator, &[_]usize{ dim, 2 });
            errdefer master_s[l].deinit();
            master_t[l] = try Tensor.zeros(allocator, &[_]usize{ dim, 2 });
            errdefer master_t[l].deinit();
            try model.readLayerWeights(l, s_tmp, t_tmp);
            @memcpy(master_s[l].data[0..plen], s_tmp);
            @memcpy(master_t[l].data[0..plen], t_tmp);
            _ = types.RSFBinding.layer(.fisher_block, model_id, l, dim);
            _ = types.RSFBinding.layer(.momentum, model_id, l, dim);
            _ = types.RSFBinding.layer(.master_weight, model_id, l, dim);
            initialized_layers += 1;
        }
        return .{
            .fisher_blocks_s = fisher_s,
            .fisher_blocks_t = fisher_t,
            .momentum_s = mom_s,
            .momentum_t = mom_t,
            .master_s = master_s,
            .master_t = master_t,
            .model_id = model_id,
            .dim = dim,
            .num_layers = num_layers,
            .step_count = 0,
            .cfg = cfg,
            .allocator = allocator,
            .initialized = true,
        };
    }

    pub fn deinit(self: *SFD) void {
        if (!self.initialized) return;
        var l: usize = 0;
        while (l < self.num_layers) : (l += 1) {
            self.fisher_blocks_s[l].deinit();
            self.fisher_blocks_t[l].deinit();
            self.momentum_s[l].deinit();
            self.momentum_t[l].deinit();
            self.master_s[l].deinit();
            self.master_t[l].deinit();
        }
        self.allocator.free(self.fisher_blocks_s);
        self.allocator.free(self.fisher_blocks_t);
        self.allocator.free(self.momentum_s);
        self.allocator.free(self.momentum_t);
        self.allocator.free(self.master_s);
        self.allocator.free(self.master_t);
        self.fisher_blocks_s = &.{};
        self.fisher_blocks_t = &.{};
        self.momentum_s = &.{};
        self.momentum_t = &.{};
        self.master_s = &.{};
        self.master_t = &.{};
        self.initialized = false;
    }

    fn requireModel(self: *const SFD, model: *const RSF) !void {
        if (!self.initialized) return error.NotInitialized;
        if (model.id != self.model_id) return error.ModelMismatch;
        const dim = try model.dim();
        const layers = try model.layerCount();
        if (dim != self.dim or layers != self.num_layers) return error.ModelMismatch;
    }

    fn mixFisher(F: []f32, d: usize, p_ww: f32, p_wb: f32, p_bb: f32, gamma: f32, cap: f32) void {
        const base = d * 3;
        const keep = gamma;
        const mix = 1.0 - gamma;
        const ww = keep * F[base + 0] + mix * p_ww;
        const wb = keep * F[base + 1] + mix * p_wb;
        const bb = keep * F[base + 2] + mix * p_bb;
        F[base + 0] = capDiag(ww, cap);
        F[base + 1] = capOff(wb, cap);
        F[base + 2] = capDiag(bb, cap);
    }

    pub fn step(self: *SFD, model: *RSF, lr: f32) !StepStats {
        try self.requireModel(model);
        if (!std.math.isFinite(lr) or lr < 0.0) return error.InvalidLearningRate;
        try model.ensureGradients();
        const plen = try pairLen(self.dim);
        const blen = try blockLen(self.dim);
        const allocator = self.allocator;
        const gs = try allocator.alloc(f32, plen);
        defer allocator.free(gs);
        const gt = try allocator.alloc(f32, plen);
        defer allocator.free(gt);
        const prod_s = try allocator.alloc(f32, blen);
        defer allocator.free(prod_s);
        const prod_t = try allocator.alloc(f32, blen);
        defer allocator.free(prod_t);
        const gn = try allocator.alloc(f32, blen);
        defer allocator.free(gn);
        const next_step = self.step_count + 1;
        const step_f: f32 = @floatFromInt(next_step);
        const m_corr = @max(1.0 - std.math.pow(f32, self.cfg.beta1, step_f), self.cfg.eps);
        const f_corr = @max(1.0 - std.math.pow(f32, self.cfg.fisher_gamma, step_f), self.cfg.eps);
        const lr_eff = lr * warmupFactor(next_step, self.cfg.warmup_steps);
        const lambda: f64 = self.cfg.fisher_epsilon;
        const gamma = self.cfg.fisher_gamma;
        const cap = self.cfg.fisher_max;
        var grad_sq: f64 = 0.0;
        var fisher_sum: f64 = 0.0;
        var fisher_n: u64 = 0;
        var off_sum: f64 = 0.0;
        var rho_sum: f64 = 0.0;
        var rho_n: u64 = 0;
        var cond_max: f64 = 0.0;
        var clipped: u64 = 0;
        var components: u64 = 0;
        var spectral_reprojected = false;
        var l: usize = 0;
        while (l < self.num_layers) : (l += 1) {
            try model.readLayerGradients(l, gs, gt);
            try model.readLayerGradientProducts(l, .s, prod_s);
            try model.readLayerGradientProducts(l, .t, prod_t);
            try model.readLayerGaussNewton(l, gn);
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                const gsw = safeGradient(gs[d * coupling_width + WEIGHT_COLUMN]);
                const gsb = safeGradient(gs[d * coupling_width + BIAS_COLUMN]);
                const gtw = safeGradient(gt[d * coupling_width + WEIGHT_COLUMN]);
                const gtb = safeGradient(gt[d * coupling_width + BIAS_COLUMN]);
                grad_sq += @as(f64, gsw) * @as(f64, gsw) + @as(f64, gsb) * @as(f64, gsb) + @as(f64, gtw) * @as(f64, gtw) + @as(f64, gtb) * @as(f64, gtb);
                switch (self.cfg.fisher_mode) {
                    .external => {},
                    .gradient => {
                        mixFisher(self.fisher_blocks_s[l].data, d, prod_s[d * 3 + 0], prod_s[d * 3 + 1], prod_s[d * 3 + 2], gamma, cap);
                        mixFisher(self.fisher_blocks_t[l].data, d, prod_t[d * 3 + 0], prod_t[d * 3 + 1], prod_t[d * 3 + 2], gamma, cap);
                    },
                    .rsf_gauss_newton => {
                        mixFisher(
                            self.fisher_blocks_s[l].data,
                            d,
                            prod_s[d * 3 + 0] + gn[d * 3 + 0],
                            prod_s[d * 3 + 1] + gn[d * 3 + 1],
                            prod_s[d * 3 + 2] + gn[d * 3 + 2],
                            gamma,
                            cap,
                        );
                        mixFisher(self.fisher_blocks_t[l].data, d, prod_t[d * 3 + 0], prod_t[d * 3 + 1], prod_t[d * 3 + 2], gamma, cap);
                    },
                }
                const branches = [_]*Tensor{ &self.fisher_blocks_s[l], &self.fisher_blocks_t[l] };
                const moms = [_]*Tensor{ &self.momentum_s[l], &self.momentum_t[l] };
                const masters = [_]*Tensor{ &self.master_s[l], &self.master_t[l] };
                const grads_w = [_]f32{ gsw, gtw };
                const grads_b = [_]f32{ gsb, gtb };
                const raw_w = [_]f32{ gs[d * coupling_width + WEIGHT_COLUMN], gt[d * coupling_width + WEIGHT_COLUMN] };
                const raw_b = [_]f32{ gs[d * coupling_width + BIAS_COLUMN], gt[d * coupling_width + BIAS_COLUMN] };
                var br: usize = 0;
                while (br < 2) : (br += 1) {
                    const F = branches[br].data;
                    const M = moms[br].data;
                    const W = masters[br].data;
                    const mw_old = M[d * coupling_width + WEIGHT_COLUMN];
                    const mb_old = M[d * coupling_width + BIAS_COLUMN];
                    const mw_cand = self.cfg.beta1 * mw_old + (1.0 - self.cfg.beta1) * grads_w[br];
                    const mb_cand = self.cfg.beta1 * mb_old + (1.0 - self.cfg.beta1) * grads_b[br];
                    const mw = if (std.math.isFinite(mw_cand)) mw_cand else mw_old;
                    const mb = if (std.math.isFinite(mb_cand)) mb_cand else mb_old;
                    M[d * coupling_width + WEIGHT_COLUMN] = mw;
                    M[d * coupling_width + BIAS_COLUMN] = mb;
                    const Fww = F[d * 3 + 0];
                    const Fwb = F[d * 3 + 1];
                    const Fbb = F[d * 3 + 2];
                    fisher_sum += @as(f64, Fww) + @as(f64, Fbb);
                    fisher_n += 2;
                    off_sum += @abs(@as(f64, Fwb));
                    const den_rho = @sqrt(@as(f64, Fww) * @as(f64, Fbb));
                    if (den_rho > 0.0) {
                        rho_sum += @abs(@as(f64, Fwb)) / den_rho;
                        rho_n += 1;
                    }
                    const Fhat_ww: f64 = @as(f64, Fww) / @as(f64, f_corr);
                    const Fhat_wb: f64 = @as(f64, Fwb) / @as(f64, f_corr);
                    const Fhat_bb: f64 = @as(f64, Fbb) / @as(f64, f_corr);
                    const Ad = Fhat_ww + lambda;
                    const Cd = Fhat_bb + lambda;
                    const cond = conditionNumber(Ad, Fhat_wb, Cd);
                    if (cond > cond_max) cond_max = cond;
                    const inv = blockInverseSqrt(Fhat_ww, Fhat_wb, Fhat_bb, lambda);
                    const mhw: f64 = @as(f64, mw) / @as(f64, m_corr);
                    const mhb: f64 = @as(f64, mb) / @as(f64, m_corr);
                    const ng = applyBlockVec(inv, mhw, mhb);
                    var dw: f32 = @floatCast(@as(f64, lr_eff) * ng.y0);
                    var db: f32 = @floatCast(@as(f64, lr_eff) * ng.y1);
                    const tw = W[d * coupling_width + WEIGHT_COLUMN];
                    const tb = W[d * coupling_width + BIAS_COLUMN];
                    const max_w = self.cfg.clip_threshold * @max(@abs(tw), self.cfg.weight_floor);
                    const max_b = self.cfg.clip_threshold * @max(@abs(tb), self.cfg.weight_floor);
                    components += 2;
                    if (@abs(dw) > max_w) {
                        dw = std.math.clamp(dw, -max_w, max_w);
                        clipped += 1;
                    }
                    if (@abs(db) > max_b) {
                        db = std.math.clamp(db, -max_b, max_b);
                        clipped += 1;
                    }
                    const both_grad_finite = std.math.isFinite(raw_w[br]) and std.math.isFinite(raw_b[br]);
                    if (both_grad_finite and std.math.isFinite(dw) and std.math.isFinite(db)) {
                        const nw = tw - dw;
                        const nb = tb - db;
                        if (std.math.isFinite(nw) and std.math.isFinite(nb)) {
                            W[d * coupling_width + WEIGHT_COLUMN] = nw;
                            W[d * coupling_width + BIAS_COLUMN] = nb;
                        }
                    }
                }
            }
            const sigma_s = try tensor_mod.constrainCouplingSpectralNorm(self.master_s[l].data, self.dim, self.cfg.spectral_target);
            const sigma_t = try tensor_mod.constrainCouplingSpectralNorm(self.master_t[l].data, self.dim, self.cfg.spectral_target);
            const target: f64 = self.cfg.spectral_target;
            if (sigma_s > target * (1.0 + SPECTRAL_REPROJECT_REL)) spectral_reprojected = true;
            if (sigma_t > target * (1.0 + SPECTRAL_REPROJECT_REL)) spectral_reprojected = true;
            try model.writeLayerWeights(l, self.master_s[l].data[0..plen], self.master_t[l].data[0..plen]);
        }
        try model.notifyWeightsChanged();
        self.step_count = next_step;
        const grad_norm = @sqrt(grad_sq);
        const fisher_mean = if (fisher_n == 0) 0.0 else fisher_sum / @as(f64, @floatFromInt(fisher_n));
        const off_mean = if (fisher_n == 0) 0.0 else off_sum / @as(f64, @floatFromInt(self.num_layers * self.dim * 2));
        const rho_mean = if (rho_n == 0) 0.0 else rho_sum / @as(f64, @floatFromInt(rho_n));
        const clip_frac = if (components == 0) 0.0 else @as(f64, @floatFromInt(clipped)) / @as(f64, @floatFromInt(components));
        if (!std.math.isFinite(grad_norm) or !std.math.isFinite(fisher_mean) or !std.math.isFinite(off_mean) or !std.math.isFinite(rho_mean) or !std.math.isFinite(clip_frac)) return error.NonFinite;
        return .{
            .step = self.step_count,
            .lr_effective = lr_eff,
            .grad_global_norm = grad_norm,
            .fisher_block_mean = fisher_mean,
            .fisher_offdiag_mean_abs = off_mean,
            .correlation_abs_mean = rho_mean,
            .condition_number_max = cond_max,
            .clipped_fraction = clip_frac,
            .spectral_reprojected = spectral_reprojected,
        };
    }

    pub fn zeroMomentum(self: *SFD) void {
        if (!self.initialized) return;
        var l: usize = 0;
        while (l < self.num_layers) : (l += 1) {
            @memset(self.momentum_s[l].data, 0.0);
            @memset(self.momentum_t[l].data, 0.0);
        }
    }

    pub fn resetFisher(self: *SFD) void {
        if (!self.initialized) return;
        var l: usize = 0;
        while (l < self.num_layers) : (l += 1) {
            @memset(self.fisher_blocks_s[l].data, 0.0);
            @memset(self.fisher_blocks_t[l].data, 0.0);
        }
    }

    pub fn clipGradNorm(self: *SFD, model: *RSF, max_norm: f32) !f32 {
        try self.requireModel(model);
        if (!std.math.isFinite(max_norm) or !(max_norm > 0.0)) return error.InvalidClipThreshold;
        const pre = try model.gradientL2Norm();
        if (!std.math.isFinite(pre)) return error.NonFinite;
        if (pre > max_norm) {
            const scale = max_norm / (pre + self.cfg.eps);
            try model.scaleLayerGradients(scale);
        }
        return pre;
    }

    fn flatLen(self: *const SFD) !usize {
        const plen = try pairLen(self.dim);
        const blen = try blockLen(self.dim);
        const masters = try checkedMul(try checkedMul(self.num_layers, plen), 4);
        const fishers = try checkedMul(try checkedMul(self.num_layers, blen), 2);
        return std.math.add(usize, masters, fishers) catch return error.Overflow;
    }

    pub fn exportFlatState(self: *const SFD, allocator: Allocator) ![]f32 {
        if (!self.initialized) return error.NotInitialized;
        const plen = try pairLen(self.dim);
        const blen = try blockLen(self.dim);
        const total = try self.flatLen();
        const out = try allocator.alloc(f32, total);
        errdefer allocator.free(out);
        var off: usize = 0;
        var l: usize = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(out[off .. off + plen], self.master_s[l].data[0..plen]);
            off += plen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(out[off .. off + plen], self.master_t[l].data[0..plen]);
            off += plen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(out[off .. off + plen], self.momentum_s[l].data[0..plen]);
            off += plen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(out[off .. off + plen], self.momentum_t[l].data[0..plen]);
            off += plen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(out[off .. off + blen], self.fisher_blocks_s[l].data[0..blen]);
            off += blen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(out[off .. off + blen], self.fisher_blocks_t[l].data[0..blen]);
            off += blen;
        }
        if (off != total) return error.InvalidModelState;
        return out;
    }

    pub fn importFlatState(self: *SFD, state: []const f32) !void {
        if (!self.initialized) return error.NotInitialized;
        const total = try self.flatLen();
        if (state.len != total) return error.InvalidDataLength;
        const plen = try pairLen(self.dim);
        const blen = try blockLen(self.dim);
        var off: usize = 0;
        var l: usize = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(self.master_s[l].data[0..plen], state[off .. off + plen]);
            off += plen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(self.master_t[l].data[0..plen], state[off .. off + plen]);
            off += plen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(self.momentum_s[l].data[0..plen], state[off .. off + plen]);
            off += plen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(self.momentum_t[l].data[0..plen], state[off .. off + plen]);
            off += plen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(self.fisher_blocks_s[l].data[0..blen], state[off .. off + blen]);
            off += blen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(self.fisher_blocks_t[l].data[0..blen], state[off .. off + blen]);
            off += blen;
        }
    }

    pub fn setExternalFisher(self: *SFD, blocks: []const f32) !void {
        if (!self.initialized) return error.NotInitialized;
        const blen = try blockLen(self.dim);
        const expected = try checkedMul(try checkedMul(self.num_layers, blen), 2);
        if (blocks.len != expected) return error.InvalidDataLength;
        var off: usize = 0;
        var l: usize = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(self.fisher_blocks_s[l].data[0..blen], blocks[off .. off + blen]);
            off += blen;
        }
        l = 0;
        while (l < self.num_layers) : (l += 1) {
            @memcpy(self.fisher_blocks_t[l].data[0..blen], blocks[off .. off + blen]);
            off += blen;
        }
    }

    pub fn saveState(self: *const SFD, path: []const u8) !void {
        if (!self.initialized) return error.NotInitialized;
        const flat = try self.exportFlatState(self.allocator);
        defer self.allocator.free(flat);
        var file = try core_io.createFilePath(path, .{ .mode = 0o600 });
        defer file.close();
        var buffered = std.io.bufferedWriter(file.writer());
        const writer = buffered.writer();
        var hasher = std.hash.Crc32.init();
        try writer.writeInt(u32, SFD4_MAGIC, .little);
        crcU32LE(&hasher, SFD4_MAGIC);
        try writer.writeInt(u64, self.model_id, .little);
        crcU64LE(&hasher, self.model_id);
        try writer.writeInt(u64, @intCast(self.dim), .little);
        crcU64LE(&hasher, @intCast(self.dim));
        try writer.writeInt(u64, @intCast(self.num_layers), .little);
        crcU64LE(&hasher, @intCast(self.num_layers));
        try writer.writeInt(u64, self.step_count, .little);
        crcU64LE(&hasher, self.step_count);
        try writer.writeInt(u32, @bitCast(self.cfg.beta1), .little);
        crcF32LE(&hasher, self.cfg.beta1);
        try writer.writeInt(u32, @bitCast(self.cfg.fisher_gamma), .little);
        crcF32LE(&hasher, self.cfg.fisher_gamma);
        try writer.writeInt(u32, @bitCast(self.cfg.fisher_epsilon), .little);
        crcF32LE(&hasher, self.cfg.fisher_epsilon);
        const mode_u8: u8 = @intFromEnum(self.cfg.fisher_mode);
        try writer.writeByte(mode_u8);
        crcU8(&hasher, mode_u8);
        try writer.writeInt(u32, @bitCast(self.cfg.eps), .little);
        crcF32LE(&hasher, self.cfg.eps);
        try writer.writeInt(u32, @bitCast(self.cfg.clip_threshold), .little);
        crcF32LE(&hasher, self.cfg.clip_threshold);
        try writer.writeInt(u32, @bitCast(self.cfg.weight_floor), .little);
        crcF32LE(&hasher, self.cfg.weight_floor);
        try writer.writeInt(u32, @bitCast(self.cfg.fisher_max), .little);
        crcF32LE(&hasher, self.cfg.fisher_max);
        try writer.writeInt(u64, @intCast(self.cfg.warmup_steps), .little);
        crcU64LE(&hasher, @intCast(self.cfg.warmup_steps));
        try writer.writeInt(u32, @bitCast(self.cfg.spectral_target), .little);
        crcF32LE(&hasher, self.cfg.spectral_target);
        try writer.writeInt(u32, FISHER_LAYOUT, .little);
        crcU32LE(&hasher, FISHER_LAYOUT);
        try writer.writeInt(u64, @intCast(flat.len), .little);
        crcU64LE(&hasher, @intCast(flat.len));
        for (flat) |v| {
            const bits: u32 = @bitCast(v);
            try writer.writeInt(u32, bits, .little);
            crcU32LE(&hasher, bits);
        }
        try writer.writeInt(u32, hasher.final(), .little);
        try buffered.flush();
    }

    pub fn loadState(self: *SFD, path: []const u8) !void {
        if (!self.initialized) return error.NotInitialized;
        const file = try core_io.openFilePath(path, .{ .mode = .read_only });
        defer file.close();
        var buffered = std.io.bufferedReader(file.reader());
        const reader = buffered.reader();
        const magic = try reader.readInt(u32, .little);
        if (magic == SFD3_MAGIC) {
            try self.loadLegacyV1(reader);
            return;
        }
        if (magic != SFD4_MAGIC) return error.InvalidStateFormat;
        var hasher = std.hash.Crc32.init();
        crcU32LE(&hasher, magic);
        const model_id = try reader.readInt(u64, .little);
        crcU64LE(&hasher, model_id);
        if (model_id != self.model_id) return error.ModelMismatch;
        const dim_u64 = try reader.readInt(u64, .little);
        crcU64LE(&hasher, dim_u64);
        const layers_u64 = try reader.readInt(u64, .little);
        crcU64LE(&hasher, layers_u64);
        if (dim_u64 != self.dim or layers_u64 != self.num_layers) return error.ModelMismatch;
        const loaded_step = try reader.readInt(u64, .little);
        crcU64LE(&hasher, loaded_step);
        const beta1: f32 = @bitCast(try reader.readInt(u32, .little));
        crcF32LE(&hasher, beta1);
        const fisher_gamma: f32 = @bitCast(try reader.readInt(u32, .little));
        crcF32LE(&hasher, fisher_gamma);
        const fisher_epsilon: f32 = @bitCast(try reader.readInt(u32, .little));
        crcF32LE(&hasher, fisher_epsilon);
        const mode_u8 = try reader.readByte();
        crcU8(&hasher, mode_u8);
        const eps: f32 = @bitCast(try reader.readInt(u32, .little));
        crcF32LE(&hasher, eps);
        const clip: f32 = @bitCast(try reader.readInt(u32, .little));
        crcF32LE(&hasher, clip);
        const weight_floor: f32 = @bitCast(try reader.readInt(u32, .little));
        crcF32LE(&hasher, weight_floor);
        const fisher_max: f32 = @bitCast(try reader.readInt(u32, .little));
        crcF32LE(&hasher, fisher_max);
        const warmup_u64 = try reader.readInt(u64, .little);
        crcU64LE(&hasher, warmup_u64);
        const spectral_target: f32 = @bitCast(try reader.readInt(u32, .little));
        crcF32LE(&hasher, spectral_target);
        const layout = try reader.readInt(u32, .little);
        crcU32LE(&hasher, layout);
        if (layout != FISHER_LAYOUT) return error.InvalidStateFormat;
        const n_u64 = try reader.readInt(u64, .little);
        crcU64LE(&hasher, n_u64);
        if (n_u64 > std.math.maxInt(usize)) return error.InvalidStateFormat;
        const n: usize = @intCast(n_u64);
        const expected = try self.flatLen();
        if (n != expected) return error.InvalidStateFormat;
        const flat = try self.allocator.alloc(f32, n);
        defer self.allocator.free(flat);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            crcU32LE(&hasher, bits);
            flat[i] = @bitCast(bits);
        }
        const stored_crc = try reader.readInt(u32, .little);
        if (stored_crc != hasher.final()) return error.ChecksumMismatch;
        const trailing = reader.readByte() catch |err| switch (err) {
            error.EndOfStream => null,
            else => return err,
        };
        if (trailing != null) return error.InvalidStateFormat;
        const loaded_cfg = SFDConfig{
            .beta1 = beta1,
            .fisher_gamma = fisher_gamma,
            .fisher_epsilon = fisher_epsilon,
            .fisher_mode = std.meta.intToEnum(FisherMode, mode_u8) catch return error.InvalidStateFormat,
            .eps = eps,
            .clip_threshold = clip,
            .weight_floor = weight_floor,
            .fisher_max = fisher_max,
            .warmup_steps = std.math.cast(usize, warmup_u64) orelse return error.InvalidStateFormat,
            .spectral_target = spectral_target,
        };
        try validateConfig(loaded_cfg);
        try self.importFlatState(flat);
        self.cfg = loaded_cfg;
        self.step_count = loaded_step;
    }

    fn loadLegacyV1(self: *SFD, reader: anytype) !void {
        const beta1: f32 = @bitCast(try reader.readInt(u32, .little));
        const beta2: f32 = @bitCast(try reader.readInt(u32, .little));
        _ = beta2;
        const eps: f32 = @bitCast(try reader.readInt(u32, .little));
        const clip: f32 = @bitCast(try reader.readInt(u32, .little));
        const weight_floor: f32 = @bitCast(try reader.readInt(u32, .little));
        const fisher_max: f32 = @bitCast(try reader.readInt(u32, .little));
        const warmup_u64 = try reader.readInt(u64, .little);
        const size_u64 = try reader.readInt(u64, .little);
        const step_u64 = try reader.readInt(u64, .little);
        if (!std.math.isFinite(beta1) or beta1 < 0.0 or beta1 >= 1.0) return error.InvalidStateFormat;
        if (!std.math.isFinite(eps) or eps <= 0.0) return error.InvalidStateFormat;
        if (!std.math.isFinite(clip) or clip <= 0.0 or clip > 1.0) return error.InvalidStateFormat;
        if (!std.math.isFinite(weight_floor) or weight_floor <= 0.0) return error.InvalidStateFormat;
        if (!std.math.isFinite(fisher_max) or fisher_max <= 0.0) return error.InvalidStateFormat;
        if (warmup_u64 > std.math.maxInt(usize) or size_u64 > std.math.maxInt(usize) or step_u64 > std.math.maxInt(usize)) return error.InvalidStateFormat;
        const param_size: usize = @intCast(size_u64);
        const expected = try checkedMul(try checkedMul(self.num_layers, self.dim), 4);
        if (param_size != expected) return error.InvalidStateFormat;
        var fisher = try Tensor.load(self.allocator, reader);
        defer fisher.deinit();
        var momentum = try Tensor.load(self.allocator, reader);
        defer momentum.deinit();
        if (fisher.data.len != param_size or momentum.data.len != param_size) return error.InvalidStateFormat;
        const trailing = reader.readByte() catch |err| switch (err) {
            error.EndOfStream => null,
            else => return err,
        };
        if (trailing != null) return error.InvalidStateFormat;
        const plen = try pairLen(self.dim);
        var offset: usize = 0;
        var l: usize = 0;
        while (l < self.num_layers) : (l += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                const iw = offset + d * coupling_width + WEIGHT_COLUMN;
                const ib = offset + d * coupling_width + BIAS_COLUMN;
                self.fisher_blocks_s[l].data[d * 3 + 0] = capDiag(fisher.data[iw], fisher_max);
                self.fisher_blocks_s[l].data[d * 3 + 1] = 0.0;
                self.fisher_blocks_s[l].data[d * 3 + 2] = capDiag(fisher.data[ib], fisher_max);
                self.momentum_s[l].data[d * coupling_width + WEIGHT_COLUMN] = if (std.math.isFinite(momentum.data[iw])) momentum.data[iw] else 0.0;
                self.momentum_s[l].data[d * coupling_width + BIAS_COLUMN] = if (std.math.isFinite(momentum.data[ib])) momentum.data[ib] else 0.0;
            }
            offset += plen;
            d = 0;
            while (d < self.dim) : (d += 1) {
                const iw = offset + d * coupling_width + WEIGHT_COLUMN;
                const ib = offset + d * coupling_width + BIAS_COLUMN;
                self.fisher_blocks_t[l].data[d * 3 + 0] = capDiag(fisher.data[iw], fisher_max);
                self.fisher_blocks_t[l].data[d * 3 + 1] = 0.0;
                self.fisher_blocks_t[l].data[d * 3 + 2] = capDiag(fisher.data[ib], fisher_max);
                self.momentum_t[l].data[d * coupling_width + WEIGHT_COLUMN] = if (std.math.isFinite(momentum.data[iw])) momentum.data[iw] else 0.0;
                self.momentum_t[l].data[d * coupling_width + BIAS_COLUMN] = if (std.math.isFinite(momentum.data[ib])) momentum.data[ib] else 0.0;
            }
            offset += plen;
        }
        self.cfg.beta1 = beta1;
        self.cfg.eps = eps;
        self.cfg.clip_threshold = clip;
        self.cfg.weight_floor = weight_floor;
        self.cfg.fisher_max = fisher_max;
        self.cfg.warmup_steps = @intCast(warmup_u64);
        self.step_count = @intCast(step_u64);
        try validateConfig(self.cfg);
    }
};

pub fn ampSchedule(step: usize, warmup: usize, total: usize) f32 {
    if (warmup > 0 and step < warmup) return @as(f32, @floatFromInt(step + 1)) / @as(f32, @floatFromInt(warmup));
    if (total <= warmup) return 1.0;
    const progress = @min(@as(f32, @floatFromInt(step -| warmup)) / @as(f32, @floatFromInt(total - warmup)), 1.0);
    return 0.5 * (1.0 + @cos(std.math.pi * progress));
}

pub fn adaptiveLR(grad_norm: f32, param_norm: f32) f32 {
    if (!std.math.isFinite(grad_norm) or grad_norm < 0.0 or !std.math.isFinite(param_norm) or param_norm < 0.0) return 1.0;
    const eps: f32 = 1.0e-8;
    const result = 1.0 / @sqrt(grad_norm / (param_norm + eps) + eps);
    return if (std.math.isFinite(result)) result else 1.0;
}

pub fn writeBackFP16(params: *const Tensor, dst: []f16) !void {
    if (dst.len != params.data.len) return error.ShapeMismatch;
    for (params.data, dst) |value, *target| target.* = @floatCast(std.math.clamp(value, @as(f32, -65504.0), @as(f32, 65504.0)));
}

pub fn loadFromFP16(src: []const f16, params: *Tensor) !void {
    if (src.len != params.data.len) return error.ShapeMismatch;
    for (src, params.data) |value, *target| target.* = @floatCast(value);
}
