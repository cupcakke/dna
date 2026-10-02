const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");
const Tensor = tensor.Tensor;
const memory = @import("../core/memory.zig");
const core_io = @import("../core/io.zig");
const accel = @import("../hw/accel/accel_interface.zig");
const OFTB = @import("oftb.zig").OFTB;
const types = @import("../core/types.zig");
const Thread = std.Thread;
const LAYER_TARGET_SPECTRAL_NORM: f32 = 0.9;
const WEIGHT_COLUMN: usize = 0;
const BIAS_COLUMN: usize = 1;
const GPU_VALIDATION_SEED: u64 = 0x5D9F_1C33_A17B_0E45;
const GPU_VALIDATION_BATCH: usize = 2;
const MODEL_CROSS_CHECK_ABS_TOL: f32 = 5.0e-2;
const MODEL_CROSS_CHECK_REL_TOL: f32 = 5.0e-2;
const SAVE_VERSION: u32 = 7;
const SAVE_VERSION_LEGACY: u32 = 6;
const TENSOR_RANK_TAG: u64 = 2;
const coupling_width: usize = accel.rsf_coupling_width;
comptime {
    if (coupling_width != 2) {
        @compileError("rsf.zig implements an affine coupling with exactly one weight column and one bias column; accel.rsf_coupling_width must be 2");
    }
    if (BIAS_COLUMN >= coupling_width or WEIGHT_COLUMN >= coupling_width) {
        @compileError("rsf.zig coupling column indices must be inside the coupling width");
    }
}
fn scratchAllocator() Allocator {
    return std.heap.smp_allocator;
}
fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return error.Overflow;
}
fn checkedMulU64(a: u64, b: u64) !u64 {
    return std.math.mul(u64, a, b) catch return error.Overflow;
}
fn checkedAddU64(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch return error.Overflow;
}
fn checkedCastU64ToUsize(v: u64) !usize {
    if (v > std.math.maxInt(usize)) return error.TooLarge;
    return @intCast(v);
}
fn constrainSpectralNorm(weight: *Tensor, rows: usize, cols: usize, target: f32) !void {
    if (!std.math.isFinite(target) or !(target > 0.0)) return error.InvalidConfig;
    if (rows == 0 or cols == 0) return error.InvalidDimension;
    if (cols != coupling_width) return error.ShapeMismatch;
    const expected = try checkedMul(rows, cols);
    if (weight.data.len < expected) return error.DataLengthMismatch;
    try ensureFiniteSlice(weight.data[0..expected]);
    const sigma = try tensor.constrainCouplingSpectralNorm(weight.data[0..expected], rows, target);
    const sigma_f32: f32 = @floatCast(sigma);
    if (!std.math.isFinite(sigma_f32)) return error.NonFinite;
}

pub fn exactSpectralNormRank2(weight: []const f32, dim: usize) !f32 {
    if (dim == 0) return error.InvalidDimension;
    const expected = try checkedMul(dim, coupling_width);
    if (weight.len < expected) return error.DataLengthMismatch;
    const sigma = try tensor.exactSpectralNormRank2(weight[0..expected], dim);
    const sigma_f32: f32 = @floatCast(sigma);
    if (!std.math.isFinite(sigma_f32)) return error.NonFinite;
    return sigma_f32;
}

pub fn layerTargetSpectralNorm() f32 {
    return LAYER_TARGET_SPECTRAL_NORM;
}

pub fn diffusionLayoutForRowLen(row_len: usize) !types.RSFDiffusionLayout {
    const layout = types.rsfDiffusionLayout(row_len) orelse return error.InvalidConfig;
    if (!tensor.diffusionLayoutIsApplicable(row_len, layout)) return error.InvalidConfig;
    return layout;
}

pub const RSFLayerConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    seed_offset: u64 = 0,
    grad_mean: bool = true,
};
pub const RSFConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    grad_mean: bool = true,
    max_dim: usize = 1 << 20,
    max_layers: usize = 1 << 20,
    global_diffusion: bool = true,
};
fn validateClipRange(clip_min: f32, clip_max: f32) !void {
    if (!std.math.isFinite(clip_min) or !std.math.isFinite(clip_max)) return error.NonFinite;
    if (!(clip_min < clip_max)) return error.InvalidConfig;
    if (clip_max > 20.0 or clip_min < -20.0) return error.InvalidConfig;
}
fn validateComparisonTolerances(abs_tol: f32, rel_tol: f32) !void {
    if (!std.math.isFinite(abs_tol) or !std.math.isFinite(rel_tol)) return error.InvalidTolerance;
    if (abs_tol < 0.0 or rel_tol < 0.0) return error.InvalidTolerance;
}
fn validateTensor2D(t: *const Tensor) !void {
    if (t.shape.dims.len != 2) return error.ShapeMismatch;
    const expected = try checkedMul(t.shape.dims[0], t.shape.dims[1]);
    if (t.data.len != expected) return error.DataLengthMismatch;
}
fn validateTensor2DShape(t: *const Tensor, rows: usize, cols: usize) !void {
    if (t.shape.dims.len != 2 or t.shape.dims[0] != rows or t.shape.dims[1] != cols) return error.ShapeMismatch;
    const expected = try checkedMul(rows, cols);
    if (t.data.len != expected) return error.DataLengthMismatch;
}
fn tensorHasShape(t: *const Tensor, rows: usize, cols: usize) bool {
    return t.shape.dims.len == 2 and t.shape.dims[0] == rows and t.shape.dims[1] == cols;
}
fn tensorsSameShape(a: *const Tensor, b: *const Tensor) bool {
    return a.shape.dims.len == 2 and b.shape.dims.len == 2 and a.shape.dims[0] == b.shape.dims[0] and a.shape.dims[1] == b.shape.dims[1];
}
fn ensureFiniteSlice(data: []const f32) !void {
    for (data) |v| {
        if (!std.math.isFinite(v)) return error.NonFinite;
    }
}
fn zeroTensor(t: *Tensor) void {
    @memset(t.data, @as(f32, 0.0));
}
fn tensorsOverlap(a: *const Tensor, b: *const Tensor) bool {
    if (a.data.len == 0 or b.data.len == 0) return false;
    const a_start: usize = @intFromPtr(a.data.ptr);
    const b_start: usize = @intFromPtr(b.data.ptr);
    const a_bytes = std.math.mul(usize, a.data.len, @sizeOf(f32)) catch return true;
    const b_bytes = std.math.mul(usize, b.data.len, @sizeOf(f32)) catch return true;
    const a_end = std.math.add(usize, a_start, a_bytes) catch return true;
    const b_end = std.math.add(usize, b_start, b_bytes) catch return true;
    return a_start < b_end and b_start < a_end;
}
fn tensorClone(allocator: Allocator, src: *const Tensor) !Tensor {
    try validateTensor2D(src);
    var dst = try Tensor.init(allocator, &.{ src.shape.dims[0], src.shape.dims[1] });
    errdefer dst.deinit();
    @memcpy(dst.data, src.data);
    return dst;
}
fn valuesWithinTolerance(a: f32, b: f32, abs_tol: f32, rel_tol: f32) bool {
    if (!std.math.isFinite(a) or !std.math.isFinite(b)) return false;
    const diff = @abs(a - b);
    const scale = @max(@abs(a), @abs(b));
    return diff <= abs_tol + rel_tol * scale;
}
fn tensorAllCloseEq(a: *const Tensor, b: *const Tensor, abs_tol: f32, rel_tol: f32) !bool {
    try validateComparisonTolerances(abs_tol, rel_tol);
    try validateTensor2D(a);
    try validateTensor2D(b);
    if (!tensorsSameShape(a, b)) return false;
    var i: usize = 0;
    while (i < a.data.len) : (i += 1) {
        if (!valuesWithinTolerance(a.data[i], b.data[i], abs_tol, rel_tol)) return false;
    }
    return true;
}
fn validateModelConfigValues(dim: usize, num_layers: usize, cfg: RSFConfig) !void {
    if (dim == 0) return error.InvalidDimension;
    if (num_layers == 0) return error.InvalidLayerCount;
    try validateClipRange(cfg.clip_min, cfg.clip_max);
    if (cfg.max_dim == 0 or cfg.max_layers == 0) return error.InvalidConfig;
    if (dim > cfg.max_dim or num_layers > cfg.max_layers) return error.InvalidConfig;
    if (cfg.global_diffusion) _ = try diffusionLayoutForRowLen(try checkedMul(dim, 2));
}

fn initOFTBForConfig(dim: usize, global_diffusion: bool) !OFTB {
    if (global_diffusion) {
        _ = try diffusionLayoutForRowLen(try checkedMul(dim, 2));
        return OFTB.initWithDiffusion(dim, true);
    }
    return OFTB.init(dim);
}
const LayerCore = struct {
    s_weight: Tensor,
    t_weight: Tensor,
    s_weight_grad: ?Tensor,
    t_weight_grad: ?Tensor,
    dim: usize,
    allocator: Allocator,
    clip_min: f32,
    clip_max: f32,
    grad_mean: bool,
    gn_block: ?[]f32,
    model_id: u64,
    layer_index: usize,
    rwlock: Thread.RwLock,
    fn initOwned(allocator: Allocator, dim: usize, config: RSFLayerConfig) !LayerCore {
        if (dim == 0) return error.InvalidDimension;
        try validateClipRange(config.clip_min, config.clip_max);
        _ = try checkedMul(dim, coupling_width);
        _ = try checkedMul(dim, 2);
        const fan_in: f32 = @floatFromInt(dim);
        const fan_out: f32 = @floatFromInt(dim);
        const fan_sum = fan_in + fan_out;
        if (!std.math.isFinite(fan_sum) or !(fan_sum > 0.0)) return error.InvalidDimension;
        const xavier_bound: f32 = @sqrt(6.0 / fan_sum);
        if (!std.math.isFinite(xavier_bound) or !(xavier_bound > 0.0)) return error.InvalidDimension;
        const weight_shape = [_]usize{ dim, coupling_width };
        const seed1 = try checkedAddU64(42, config.seed_offset);
        const seed2 = try checkedAddU64(43, config.seed_offset);
        var s_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed1);
        errdefer s_w.deinit();
        var t_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed2);
        errdefer t_w.deinit();
        for (0..dim) |d| {
            s_w.data[d * coupling_width + BIAS_COLUMN] = 0.0;
            t_w.data[d * coupling_width + BIAS_COLUMN] = 0.0;
        }
        try constrainSpectralNorm(&s_w, dim, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
        try constrainSpectralNorm(&t_w, dim, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
        return LayerCore{
            .s_weight = s_w,
            .t_weight = t_w,
            .s_weight_grad = null,
            .t_weight_grad = null,
            .dim = dim,
            .allocator = allocator,
            .clip_min = config.clip_min,
            .clip_max = config.clip_max,
            .grad_mean = config.grad_mean,
            .gn_block = null,
            .model_id = 0,
            .layer_index = 0,
            .rwlock = .{},
        };
    }
    fn deinitOwned(self: *LayerCore) void {
        self.s_weight.deinit();
        self.t_weight.deinit();
        if (self.s_weight_grad) |*g| g.deinit();
        if (self.t_weight_grad) |*g| g.deinit();
        if (self.gn_block) |blk| self.allocator.free(blk);
        self.s_weight_grad = null;
        self.t_weight_grad = null;
        self.gn_block = null;
    }
    pub fn ensureGradients(self: *LayerCore) !void {
        const need_swg = self.s_weight_grad == null;
        const need_twg = self.t_weight_grad == null;
        if (!(need_swg or need_twg)) return;
        const weight_shape = [_]usize{ self.dim, coupling_width };
        var swg_new: ?Tensor = null;
        var twg_new: ?Tensor = null;
        errdefer {
            if (swg_new) |*t| t.deinit();
            if (twg_new) |*t| t.deinit();
        }
        if (need_swg) swg_new = try Tensor.zeros(self.allocator, &weight_shape);
        if (need_twg) twg_new = try Tensor.zeros(self.allocator, &weight_shape);
        if (swg_new) |t| self.s_weight_grad = t;
        if (twg_new) |t| self.t_weight_grad = t;
        swg_new = null;
        twg_new = null;
    }
    fn zeroGradients(self: *LayerCore) void {
        if (self.s_weight_grad) |*g| zeroTensor(g);
        if (self.t_weight_grad) |*g| zeroTensor(g);
    }
    fn gradientSquaredSum(self: *const LayerCore) !f64 {
        var total: f64 = 0.0;
        if (self.s_weight_grad) |g| {
            try ensureFiniteSlice(g.data);
            for (g.data) |v| total += @as(f64, v) * @as(f64, v);
        }
        if (self.t_weight_grad) |g| {
            try ensureFiniteSlice(g.data);
            for (g.data) |v| total += @as(f64, v) * @as(f64, v);
        }
        return total;
    }
    fn applyGradientStep(self: *LayerCore, learning_rate: f32) !void {
        if (!std.math.isFinite(learning_rate)) return error.NonFinite;
        if (!(learning_rate > 0.0)) return error.InvalidLearningRate;
        const s_grad: *Tensor = if (self.s_weight_grad) |*g| g else return error.NoGradients;
        const t_grad: *Tensor = if (self.t_weight_grad) |*g| g else return error.NoGradients;
        try validateTensor2DShape(&self.s_weight, self.dim, coupling_width);
        try validateTensor2DShape(&self.t_weight, self.dim, coupling_width);
        try validateTensor2DShape(s_grad, self.dim, coupling_width);
        try validateTensor2DShape(t_grad, self.dim, coupling_width);
        try ensureFiniteSlice(s_grad.data);
        try ensureFiniteSlice(t_grad.data);
        var s_next = try tensorClone(self.allocator, &self.s_weight);
        errdefer s_next.deinit();
        var t_next = try tensorClone(self.allocator, &self.t_weight);
        errdefer t_next.deinit();
        var i: usize = 0;
        while (i < s_next.data.len) : (i += 1) {
            const updated = s_next.data[i] - learning_rate * s_grad.data[i];
            if (!std.math.isFinite(updated)) return error.NonFinite;
            s_next.data[i] = updated;
        }
        i = 0;
        while (i < t_next.data.len) : (i += 1) {
            const updated = t_next.data[i] - learning_rate * t_grad.data[i];
            if (!std.math.isFinite(updated)) return error.NonFinite;
            t_next.data[i] = updated;
        }
        try constrainSpectralNorm(&s_next, self.dim, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
        try constrainSpectralNorm(&t_next, self.dim, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
        try ensureFiniteSlice(s_next.data);
        try ensureFiniteSlice(t_next.data);
        self.s_weight.deinit();
        self.s_weight = s_next;
        self.t_weight.deinit();
        self.t_weight = t_next;
    }
    fn validatePair(self: *const LayerCore, a: *const Tensor, b: *const Tensor) !usize {
        try validateTensor2D(a);
        try validateTensor2D(b);
        if (a.shape.dims[1] != self.dim or b.shape.dims[1] != self.dim) return error.ShapeMismatch;
        if (a.shape.dims[0] != b.shape.dims[0]) return error.ShapeMismatch;
        const batch_size = a.shape.dims[0];
        if (batch_size == 0) return error.InvalidBatchSize;
        _ = try checkedMul(batch_size, self.dim);
        return batch_size;
    }
    fn couplingParams(self: *const LayerCore) !tensor.RSFCouplingParams {
        const expected = try checkedMul(self.dim, coupling_width);
        if (self.s_weight.data.len < expected) return error.DataLengthMismatch;
        if (self.t_weight.data.len < expected) return error.DataLengthMismatch;
        return tensor.RSFCouplingParams.init(self.s_weight.data[0..expected], self.t_weight.data[0..expected], self.dim, self.clip_min, self.clip_max);
    }
    fn couplingForwardRow(self: *const LayerCore, row: []f32, scale: []f32, trans: []f32) !void {
        const dim = self.dim;
        const total = try checkedMul(dim, 2);
        if (row.len != total) return error.DataLengthMismatch;
        if (scale.len < dim or trans.len < dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        _ = try tensor.couplingForwardHalves(params, row[0..dim], row[dim..total], scale, trans);
    }
    fn couplingForwardLogDetRow(self: *const LayerCore, row: []f32, scale: []f32, trans: []f32) !f32 {
        const dim = self.dim;
        const total = try checkedMul(dim, 2);
        if (row.len != total) return error.DataLengthMismatch;
        if (scale.len < dim or trans.len < dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        const logdet = try tensor.couplingForwardHalves(params, row[0..dim], row[dim..total], scale, trans);
        const logdet_f32: f32 = @floatCast(logdet);
        if (!std.math.isFinite(logdet_f32)) return error.NonFinite;
        return logdet_f32;
    }
    fn couplingInverseRow(self: *const LayerCore, row: []f32, scale: []f32, trans: []f32) !void {
        const dim = self.dim;
        const total = try checkedMul(dim, 2);
        if (row.len != total) return error.DataLengthMismatch;
        if (scale.len < dim or trans.len < dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        _ = try tensor.couplingInverseHalves(params, row[0..dim], row[dim..total], scale, trans);
    }
    fn couplingInverseLogDetRow(self: *const LayerCore, row: []f32, scale: []f32, trans: []f32) !f32 {
        const dim = self.dim;
        const total = try checkedMul(dim, 2);
        if (row.len != total) return error.DataLengthMismatch;
        if (scale.len < dim or trans.len < dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        const logdet = try tensor.couplingInverseHalves(params, row[0..dim], row[dim..total], scale, trans);
        const logdet_f32: f32 = @floatCast(logdet);
        if (!std.math.isFinite(logdet_f32)) return error.NonFinite;
        return logdet_f32;
    }
    fn forwardInPlace(self: *const LayerCore, x1: *Tensor, x2: *Tensor, scale: []f32, trans: []f32) !void {
        if (scale.len < self.dim or trans.len < self.dim) return error.DataLengthMismatch;
        if (tensorsOverlap(x1, x2)) return error.AliasedBuffers;
        const batch_size = try self.validatePair(x1, x2);
        const params = try self.couplingParams();
        _ = try tensor.couplingForwardStrided(params, x1.data, x2.data, batch_size, self.dim, self.dim, scale, trans);
    }
    fn inverseInPlace(self: *const LayerCore, y1: *Tensor, y2: *Tensor, scale: []f32, trans: []f32) !void {
        if (scale.len < self.dim or trans.len < self.dim) return error.DataLengthMismatch;
        if (tensorsOverlap(y1, y2)) return error.AliasedBuffers;
        const batch_size = try self.validatePair(y1, y2);
        const params = try self.couplingParams();
        _ = try tensor.couplingInverseStrided(params, y1.data, y2.data, batch_size, self.dim, self.dim, scale, trans);
    }
    fn ensureGaussNewton(self: *LayerCore) !void {
        if (self.gn_block != null) return;
        const len = try checkedMul(self.dim, 3);
        const blk = try self.allocator.alloc(f32, len);
        @memset(blk, 0.0);
        self.gn_block = blk;
    }
    fn zeroGaussNewton(self: *LayerCore) void {
        if (self.gn_block) |blk| @memset(blk, 0.0);
    }
    fn recordGaussNewtonRow(self: *LayerCore, x2_row: []const f32, params: tensor.RSFCouplingParams, logdet_adjoint: f32) !void {
        if (logdet_adjoint == 0.0) return;
        try self.ensureGaussNewton();
        const blk = self.gn_block orelse return error.NoGaussNewtonBlock;
        const dim = self.dim;
        var d: usize = 0;
        while (d < dim) : (d += 1) {
            const raw = params.scaleWeight(d) * x2_row[d] + params.scaleBias(d);
            const active: f32 = if (tensor.couplingSaturates(raw, params.clip_min, params.clip_max)) 0.0 else logdet_adjoint;
            const g_w = active * x2_row[d];
            blk[d * 3 + 0] += g_w * g_w;
            blk[d * 3 + 1] += g_w * active;
            blk[d * 3 + 2] += active * active;
        }
    }
    fn recordCausalGaussNewton(self: *LayerCore, params: tensor.RSFCouplingParams, mask: *const types.RSFSequenceMask, x2_in: []const f32, logdet_weight: f32, key: []f32) !void {
        if (logdet_weight == 0.0) return;
        try self.ensureGaussNewton();
        const blk = self.gn_block orelse return error.NoGaussNewtonBlock;
        const dim = self.dim;
        const seq_len = mask.seq_len;
        if (key.len < dim) return error.DataLengthMismatch;
        const required = try checkedMul(seq_len, dim);
        if (x2_in.len < required) return error.DataLengthMismatch;
        const k = key[0..dim];
        @memset(k, 0.0);
        var t: usize = 0;
        while (t < seq_len) : (t += 1) {
            tensor.causalKeyAccumulate(mask.*, x2_in[0..required], t, k);
            var d: usize = 0;
            while (d < dim) : (d += 1) {
                const raw = params.scaleWeight(d) * k[d] + params.scaleBias(d);
                const active: f32 = if (tensor.couplingSaturates(raw, params.clip_min, params.clip_max)) 0.0 else logdet_weight;
                const g_w = active * k[d];
                blk[d * 3 + 0] += g_w * g_w;
                blk[d * 3 + 1] += g_w * active;
                blk[d * 3 + 2] += active * active;
            }
        }
    }
    fn backwardFromInputsRow(
        self: *LayerCore,
        x1_row: []const f32,
        x2_row: []const f32,
        dy1_row: []const f32,
        dy2_row: []const f32,
        dx1_row_out: []f32,
        dx2_row_out: []f32,
        y1_buf: []f32,
        scale_buf: []f32,
        ds_weight_buf: []f32,
        dt_weight_buf: []f32,
        grad_scale: f32,
        logdet_adjoint: f32,
    ) !void {
        const dim = self.dim;
        if (!std.math.isFinite(grad_scale)) return error.NonFinite;
        if (!std.math.isFinite(logdet_adjoint)) return error.NonFinite;
        if (x1_row.len != dim or x2_row.len != dim) return error.ShapeMismatch;
        if (dy1_row.len != dim or dy2_row.len != dim) return error.ShapeMismatch;
        if (dx1_row_out.len != dim or dx2_row_out.len != dim) return error.ShapeMismatch;
        if (y1_buf.len != dim or scale_buf.len != dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        const expected = try checkedMul(dim, coupling_width);
        if (ds_weight_buf.len < expected or dt_weight_buf.len < expected) return error.DataLengthMismatch;
        @memset(ds_weight_buf[0..expected], 0.0);
        @memset(dt_weight_buf[0..expected], 0.0);
        var d: usize = 0;
        while (d < dim) : (d += 1) {
            const scale = tensor.couplingScaleFromInput(params, x2_row[d], d);
            scale_buf[d] = scale;
            y1_buf[d] = x1_row[d] * scale;
        }
        _ = try tensor.couplingBackwardHalves(
            params,
            x1_row,
            x2_row,
            y1_buf,
            dy1_row,
            dy2_row,
            logdet_adjoint,
            ds_weight_buf[0..expected],
            dt_weight_buf[0..expected],
            dx1_row_out,
            dx2_row_out,
        );
        try self.recordGaussNewtonRow(x2_row, params, logdet_adjoint);
        if (self.s_weight_grad) |*swg| {
            if (swg.data.len >= expected) {
                d = 0;
                while (d < expected) : (d += 1) swg.data[d] += grad_scale * ds_weight_buf[d];
            }
        }
        if (self.t_weight_grad) |*twg| {
            if (twg.data.len >= expected) {
                d = 0;
                while (d < expected) : (d += 1) twg.data[d] += grad_scale * dt_weight_buf[d];
            }
        }
    }
};
const LayerBindings = struct {
    s_weight: types.RSFBinding,
    t_weight: types.RSFBinding,
    gradient: types.RSFBinding,
    fisher_block: types.RSFBinding,
};

fn layerBindings(model_id: u64, layer_index: usize, dim: usize) LayerBindings {
    return .{
        .s_weight = types.RSFBinding.layer(.layer_weight_s, model_id, layer_index, dim),
        .t_weight = types.RSFBinding.layer(.layer_weight_t, model_id, layer_index, dim),
        .gradient = types.RSFBinding.layer(.gradient, model_id, layer_index, dim),
        .fisher_block = types.RSFBinding.layer(.fisher_block, model_id, layer_index, dim),
    };
}

fn checkedLayerIndex(core: *const RSFCore, layer: usize) !*LayerCore {
    const count = try checkedModelLayerCount(core);
    if (layer >= count) return error.LayerIndexOutOfBounds;
    return &core.layers[layer];
}

fn assignLayerBindings(core: *RSFCore, model_id: u64) void {
    for (core.layers, 0..) |*layer, index| {
        layer.model_id = model_id;
        layer.layer_index = index;
    }
}

const LayerRegistryEntry = struct {
    core: *LayerCore,
    active_ops: usize,
    destroyed: bool,
};
fn maybeShrinkRegistry(comptime EntryType: type, registry: *std.AutoHashMap(u64, EntryType)) void {
    if (registry.count() == 0) {
        registry.deinit();
        registry.* = std.AutoHashMap(u64, EntryType).init(std.heap.page_allocator);
    }
}
fn registerRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    next_id: *std.atomic.Value(u64),
    core: *CoreType,
) !u64 {
    mutex.lock();
    defer mutex.unlock();
    var id: u64 = 0;
    while (id == 0 or registry.contains(id)) {
        id = next_id.fetchAdd(1, .monotonic);
    }
    try registry.put(id, .{ .core = core, .active_ops = 0, .destroyed = false });
    return id;
}
fn acquireRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    id: u64,
) !*CoreType {
    if (id == 0) return error.NotInitialized;
    mutex.lock();
    defer mutex.unlock();
    const entry = registry.getPtr(id) orelse return error.NotInitialized;
    if (entry.destroyed) return error.NotInitialized;
    if (entry.active_ops == std.math.maxInt(usize)) return error.TooManyActiveOperations;
    entry.active_ops += 1;
    return entry.core;
}
fn releaseRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    id: u64,
    destroy_fn: *const fn (*CoreType) void,
) void {
    if (id == 0) return;
    var core_to_destroy: ?*CoreType = null;
    mutex.lock();
    if (registry.getPtr(id)) |entry| {
        if (entry.active_ops > 0) entry.active_ops -= 1;
        if (entry.destroyed and entry.active_ops == 0) {
            if (registry.fetchRemove(id)) |kv| {
                core_to_destroy = kv.value.core;
                maybeShrinkRegistry(EntryType, registry);
            }
        }
    }
    mutex.unlock();
    if (core_to_destroy) |core| destroy_fn(core);
}
fn requestDestroyRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    id: u64,
    destroy_fn: *const fn (*CoreType) void,
) void {
    if (id == 0) return;
    var core_to_destroy: ?*CoreType = null;
    mutex.lock();
    if (registry.getPtr(id)) |entry| {
        entry.destroyed = true;
        if (entry.active_ops == 0) {
            if (registry.fetchRemove(id)) |kv| {
                core_to_destroy = kv.value.core;
                maybeShrinkRegistry(EntryType, registry);
            }
        }
    }
    mutex.unlock();
    if (core_to_destroy) |core| destroy_fn(core);
}
fn handleId(id: u64) !u64 {
    if (id == 0) return error.NotInitialized;
    return id;
}
var g_layer_registry_mutex: Thread.Mutex = .{};
var g_layer_registry = std.AutoHashMap(u64, LayerRegistryEntry).init(std.heap.page_allocator);
var g_layer_next_id = std.atomic.Value(u64).init(1);
fn destroyLayerCore(core: *LayerCore) void {
    const allocator = core.allocator;
    core.deinitOwned();
    allocator.destroy(core);
}
fn registerLayerCore(core: *LayerCore) !u64 {
    return registerRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, &g_layer_next_id, core);
}
fn acquireLayerCore(id: u64) !*LayerCore {
    return acquireRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, id);
}
fn releaseLayerCore(id: u64) void {
    releaseRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, id, destroyLayerCore);
}
fn requestDestroyLayerCore(id: u64) void {
    requestDestroyRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, id, destroyLayerCore);
}
pub const RSFLayer = struct {
    id: u64 = 0,
    pub fn init(allocator: Allocator, dim: usize) !RSFLayer {
        return initWithConfig(allocator, dim, .{});
    }
    pub fn initWithArena(arena: *memory.ArenaAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(arena.allocator(), dim, config);
    }
    pub fn initWithPool(pool: *memory.PoolAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(pool.allocator(), dim, config);
    }
    pub fn initWithSlab(slab: *memory.SlabAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(slab.allocator(), dim, config);
    }
    pub fn initWithBuddy(buddy: *memory.BuddyAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(buddy.allocator(), dim, config);
    }
    pub fn initWithConfig(allocator: Allocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        const core = try allocator.create(LayerCore);
        errdefer allocator.destroy(core);
        core.* = try LayerCore.initOwned(allocator, dim, config);
        errdefer core.deinitOwned();
        const id = try registerLayerCore(core);
        return RSFLayer{ .id = id };
    }
    pub fn ensureGradients(self: *RSFLayer) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try core.ensureGradients();
    }
    pub fn deinit(self: *RSFLayer) void {
        const id = self.id;
        if (id == 0) return;
        self.id = 0;
        requestDestroyLayerCore(id);
    }
    pub fn zeroGradients(self: *RSFLayer) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        core.zeroGradients();
    }
    pub fn applyGradientStep(self: *RSFLayer, learning_rate: f32) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try core.applyGradientStep(learning_rate);
    }
    pub fn gradientL2Norm(self: *const RSFLayer) !f32 {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const total = try core.gradientSquaredSum();
        const result: f32 = @floatCast(@sqrt(total));
        if (!std.math.isFinite(result)) return error.NonFinite;
        return result;
    }
    pub fn forward(self: *const RSFLayer, x1: *Tensor, x2: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const allocator = scratchAllocator();
        const scale = try allocator.alloc(f32, core.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, core.dim);
        defer allocator.free(trans);
        try core.forwardInPlace(x1, x2, scale, trans);
    }
    pub fn inverse(self: *const RSFLayer, y1: *Tensor, y2: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const allocator = scratchAllocator();
        const scale = try allocator.alloc(f32, core.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, core.dim);
        defer allocator.free(trans);
        try core.inverseInPlace(y1, y2, scale, trans);
    }
    pub fn verifyInvertible(self: *const RSFLayer, x1: *const Tensor, x2: *const Tensor, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const allocator = scratchAllocator();
        var fx1 = try tensorClone(allocator, x1);
        defer fx1.deinit();
        var fx2 = try tensorClone(allocator, x2);
        defer fx2.deinit();
        const scale = try allocator.alloc(f32, core.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, core.dim);
        defer allocator.free(trans);
        try core.forwardInPlace(&fx1, &fx2, scale, trans);
        try core.inverseInPlace(&fx1, &fx2, scale, trans);
        const ok1 = try tensorAllCloseEq(x1, &fx1, abs_tol, rel_tol);
        if (!ok1) return false;
        const ok2 = try tensorAllCloseEq(x2, &fx2, abs_tol, rel_tol);
        return ok2;
    }
};
const RSFCore = struct {
    allocator: Allocator,
    dim: usize,
    num_layers: usize,
    layers: []LayerCore,
    cfg: RSFConfig,
    rwlock: Thread.RwLock,
    gpu_accel: ?accel.RSFAccelerator,
    gpu_available: std.atomic.Value(u8),
    gpu_weight_version: u64,
    cpu_weight_version: u64,
    f16_buf: ?[]f16,
    oftb: OFTB,
    layer_applications: std.atomic.Value(usize),
    frontier_depth: std.atomic.Value(usize),
};
const ModelRegistryEntry = struct {
    core: *RSFCore,
    active_ops: usize,
    destroyed: bool,
};
var g_model_registry_mutex: Thread.Mutex = .{};
var g_model_registry = std.AutoHashMap(u64, ModelRegistryEntry).init(std.heap.page_allocator);
var g_model_next_id = std.atomic.Value(u64).init(1);
fn destroyModelCore(core: *RSFCore) void {
    if (core.gpu_accel) |*ga| {
        ga.deinit();
        core.gpu_accel = null;
    }
    if (core.f16_buf) |buf| {
        core.allocator.free(buf);
        core.f16_buf = null;
    }
    core.gpu_available.store(0, .monotonic);
    core.oftb.deinit();
    const allocator = core.allocator;
    for (core.layers) |*layer| layer.deinitOwned();
    allocator.free(core.layers);
    allocator.destroy(core);
}
fn registerModelCore(core: *RSFCore) !u64 {
    return registerRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, &g_model_next_id, core);
}
fn acquireModelCore(id: u64) !*RSFCore {
    return acquireRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, id);
}
fn releaseModelCore(id: u64) void {
    releaseRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, id, destroyModelCore);
}
fn requestDestroyModelCore(id: u64) void {
    requestDestroyRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, id, destroyModelCore);
}
pub fn shutdownGlobalRegistries() !void {
    g_layer_registry_mutex.lock();
    defer g_layer_registry_mutex.unlock();
    g_model_registry_mutex.lock();
    defer g_model_registry_mutex.unlock();
    if (g_layer_registry.count() != 0 or g_model_registry.count() != 0) return error.ResourcesStillActive;
    g_layer_registry.deinit();
    g_layer_registry = std.AutoHashMap(u64, LayerRegistryEntry).init(std.heap.page_allocator);
    g_model_registry.deinit();
    g_model_registry = std.AutoHashMap(u64, ModelRegistryEntry).init(std.heap.page_allocator);
}
fn checkedModelLayerCount(core: *const RSFCore) !usize {
    if (core.num_layers != core.layers.len) return error.InvalidModelState;
    if (core.layers.len == 0) return error.InvalidLayerCount;
    return core.layers.len;
}
fn validateModelMetadata(core: *const RSFCore) !void {
    const layer_count = try checkedModelLayerCount(core);
    try validateModelConfigValues(core.dim, layer_count, core.cfg);
    if (core.oftb.dim != core.dim) return error.InvalidModelState;
    var i: usize = 0;
    while (i < layer_count) : (i += 1) {
        const layer = &core.layers[i];
        if (layer.dim != core.dim) return error.InvalidModelState;
        if (layer.clip_min != core.cfg.clip_min or layer.clip_max != core.cfg.clip_max or layer.grad_mean != core.cfg.grad_mean) return error.InvalidConfig;
        try validateTensor2DShape(&layer.s_weight, core.dim, coupling_width);
        try validateTensor2DShape(&layer.t_weight, core.dim, coupling_width);
    }
}
fn forwardOnCore(core: *const RSFCore, x: *Tensor) !void {
    try validateTensor2D(x);
    try validateModelMetadata(core);
    const layer_count = try checkedModelLayerCount(core);
    const dim2 = try checkedMul(core.dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = x.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    const allocator = scratchAllocator();
    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);
    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);
    var l: usize = 0;
    while (l < layer_count) : (l += 1) {
        const layer = &core.layers[l];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = x.data[b * dim2 .. b * dim2 + dim2];
            try layer.couplingForwardRow(row, scale, trans);
            core.oftb.forwardSliceInPlace(row);
        }
    }
}
fn inverseOnCore(core: *const RSFCore, y: *Tensor) !void {
    try validateTensor2D(y);
    try validateModelMetadata(core);
    const layer_count = try checkedModelLayerCount(core);
    const dim2 = try checkedMul(core.dim, 2);
    if (y.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = y.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    const allocator = scratchAllocator();
    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);
    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);
    var idx = layer_count;
    while (idx > 0) : (idx -= 1) {
        const layer = &core.layers[idx - 1];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = y.data[b * dim2 .. b * dim2 + dim2];
            core.oftb.inverseSliceInPlace(row);
            try layer.couplingInverseRow(row, scale, trans);
        }
    }
}
fn forwardLogDetOnCore(core: *const RSFCore, x: *Tensor, logdet_per_row: []f32) !void {
    try validateTensor2D(x);
    try validateModelMetadata(core);
    const layer_count = try checkedModelLayerCount(core);
    const dim2 = try checkedMul(core.dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = x.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    if (logdet_per_row.len < batch_size) return error.DataLengthMismatch;
    @memset(logdet_per_row[0..batch_size], 0.0);
    const allocator = scratchAllocator();
    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);
    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);
    var l: usize = 0;
    while (l < layer_count) : (l += 1) {
        const layer = &core.layers[l];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = x.data[b * dim2 .. b * dim2 + dim2];
            const row_logdet = try layer.couplingForwardLogDetRow(row, scale, trans);
            logdet_per_row[b] += row_logdet;
            core.oftb.forwardSliceInPlace(row);
        }
    }
}
fn inverseLogDetOnCore(core: *const RSFCore, y: *Tensor, logdet_per_row: []f32) !void {
    try validateTensor2D(y);
    try validateModelMetadata(core);
    const layer_count = try checkedModelLayerCount(core);
    const dim2 = try checkedMul(core.dim, 2);
    if (y.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = y.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    if (logdet_per_row.len < batch_size) return error.DataLengthMismatch;
    @memset(logdet_per_row[0..batch_size], 0.0);
    const allocator = scratchAllocator();
    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);
    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);
    var idx = layer_count;
    while (idx > 0) : (idx -= 1) {
        const layer = &core.layers[idx - 1];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = y.data[b * dim2 .. b * dim2 + dim2];
            core.oftb.inverseSliceInPlace(row);
            const row_logdet = try layer.couplingInverseLogDetRow(row, scale, trans);
            logdet_per_row[b] += row_logdet;
        }
    }
}

const MidpointSplit = struct {
    forward_layers: usize,
    backward_layers: usize,
};
fn midpointSplitLayers(layer_count: usize) MidpointSplit {
    const forward_layers = layer_count / 2;
    return .{ .forward_layers = forward_layers, .backward_layers = layer_count - forward_layers };
}
fn noteLayerApplication(core: *const RSFCore) void {
    _ = @constCast(core).layer_applications.fetchAdd(1, .monotonic);
}
fn noteFrontierDepth(core: *const RSFCore, depth: usize) void {
    var current = core.frontier_depth.load(.monotonic);
    while (depth > current) {
        current = @constCast(core).frontier_depth.cmpxchgWeak(current, depth, .monotonic, .monotonic) orelse break;
    }
}
fn resetLayerApplicationCounters(core: *RSFCore) void {
    core.layer_applications.store(0, .monotonic);
    core.frontier_depth.store(0, .monotonic);
}
fn blockedEnergy(blocked: *const Tensor) f64 {
    var acc: f64 = 0.0;
    for (blocked.data) |v| acc += @as(f64, v) * @as(f64, v);
    return acc;
}
fn blockedMaxAbs(blocked: *const Tensor) f64 {
    var acc: f64 = 0.0;
    for (blocked.data) |v| {
        const a = @abs(@as(f64, v));
        if (a > acc) acc = a;
    }
    return acc;
}
fn collisionLossBlocked(z: *const Tensor, w: *const Tensor, dim: usize) !f32 {
    if (!tensorsSameShape(z, w)) return error.ShapeMismatch;
    try validateTensor2D(z);
    const batch = z.shape.dims[0];
    const dim2 = try checkedMul(dim, 2);
    if (z.shape.dims[1] != dim2) return error.ShapeMismatch;
    const tokens_dim = try checkedMul(batch, dim);
    if (tokens_dim == 0) return error.InvalidBatchSize;
    const n = try checkedMul(batch, dim2);
    if (z.data.len < n or w.data.len < n) return error.DataLengthMismatch;
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const d = @as(f64, z.data[i]) - @as(f64, w.data[i]);
        sum += d * d;
    }
    const loss: f32 = @floatCast(sum / @as(f64, @floatFromInt(tokens_dim)));
    if (!std.math.isFinite(loss)) return error.NonFinite;
    return loss;
}
fn blockedMeanLogDetAndApplyForward(core: *const RSFCore, blocked: *Tensor, start: usize, end: usize) !f32 {
    try validateTensor2D(blocked);
    try validateModelMetadata(core);
    const dim = core.dim;
    const dim2 = try checkedMul(dim, 2);
    if (blocked.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch = blocked.shape.dims[0];
    if (batch == 0) return error.InvalidBatchSize;
    const layer_count = try checkedModelLayerCount(core);
    if (end < start or end > layer_count) return error.LayerIndexOutOfBounds;
    if (start == end) return 0.0;
    try ensureFiniteSlice(blocked.data[0..try checkedMul(batch, dim2)]);
    const allocator = scratchAllocator();
    const scale = try allocator.alloc(f32, dim);
    defer allocator.free(scale);
    const trans = try allocator.alloc(f32, dim);
    defer allocator.free(trans);
    var sum: f64 = 0.0;
    var l: usize = start;
    while (l < end) : (l += 1) {
        const layer = &core.layers[l];
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const row = blocked.data[b * dim2 .. b * dim2 + dim2];
            const row_logdet = try layer.couplingForwardLogDetRow(row, scale, trans);
            sum += @as(f64, row_logdet);
            core.oftb.forwardSliceInPlace(row);
        }
        noteLayerApplication(core);
    }
    const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(batch)));
    if (!std.math.isFinite(mean)) return error.NonFinite;
    return mean;
}
fn blockedMeanLogDetAndApplyInverse(core: *const RSFCore, blocked: *Tensor, start: usize, end: usize) !f32 {
    try validateTensor2D(blocked);
    try validateModelMetadata(core);
    const dim = core.dim;
    const dim2 = try checkedMul(dim, 2);
    if (blocked.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch = blocked.shape.dims[0];
    if (batch == 0) return error.InvalidBatchSize;
    const layer_count = try checkedModelLayerCount(core);
    if (end < start or end > layer_count) return error.LayerIndexOutOfBounds;
    if (start == end) return 0.0;
    try ensureFiniteSlice(blocked.data[0..try checkedMul(batch, dim2)]);
    const allocator = scratchAllocator();
    const scale = try allocator.alloc(f32, dim);
    defer allocator.free(scale);
    const trans = try allocator.alloc(f32, dim);
    defer allocator.free(trans);
    var sum: f64 = 0.0;
    var idx = end;
    while (idx > start) {
        idx -= 1;
        const layer = &core.layers[idx];
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const row = blocked.data[b * dim2 .. b * dim2 + dim2];
            core.oftb.inverseSliceInPlace(row);
            const row_logdet = try layer.couplingInverseLogDetRow(row, scale, trans);
            sum += @as(f64, row_logdet);
        }
        noteLayerApplication(core);
    }
    const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(batch)));
    if (!std.math.isFinite(mean)) return error.NonFinite;
    return mean;
}
fn addLayerParamGrads(layer: *LayerCore, ds: []const f32, dt: []const f32) !void {
    try layer.ensureGradients();
    const expected = try checkedMul(layer.dim, coupling_width);
    if (ds.len < expected or dt.len < expected) return error.DataLengthMismatch;
    const s_grad = if (layer.s_weight_grad) |*g| g else return error.NoGradients;
    const t_grad = if (layer.t_weight_grad) |*g| g else return error.NoGradients;
    var k: usize = 0;
    while (k < expected) : (k += 1) {
        const ns = s_grad.data[k] + ds[k];
        const nt = t_grad.data[k] + dt[k];
        if (!std.math.isFinite(ns) or !std.math.isFinite(nt)) return error.NonFinite;
        s_grad.data[k] = ns;
        t_grad.data[k] = nt;
    }
}
fn midpointForwardAdjointOnCore(
    core: *RSFCore,
    z_blocked: *const Tensor,
    grad: []f32,
    grad_input_blocked: *Tensor,
    start: usize,
    end: usize,
    logdet_adjoint: f32,
) !void {
    try validateTensor2D(z_blocked);
    try validateTensor2D(grad_input_blocked);
    const dim = core.dim;
    const dim2 = try checkedMul(dim, 2);
    const batch = z_blocked.shape.dims[0];
    if (batch == 0) return error.InvalidBatchSize;
    const n = try checkedMul(batch, dim2);
    if (z_blocked.data.len < n or grad.len < n) return error.DataLengthMismatch;
    if (grad_input_blocked.data.len < n) return error.DataLengthMismatch;
    if (!std.math.isFinite(logdet_adjoint)) return error.NonFinite;
    if (start == end) {
        @memcpy(grad_input_blocked.data[0..n], grad[0..n]);
        return;
    }
    const allocator = scratchAllocator();
    const state = try allocator.alloc(f32, n);
    defer allocator.free(state);
    @memcpy(state, z_blocked.data[0..n]);
    var scratch = try tensor.InvertedFlowScratch.init(allocator, dim);
    defer scratch.deinit();
    const expected = try checkedMul(dim, coupling_width);
    const ds_buf = try allocator.alloc(f32, expected);
    defer allocator.free(ds_buf);
    const dt_buf = try allocator.alloc(f32, expected);
    defer allocator.free(dt_buf);
    const scale = try allocator.alloc(f32, dim);
    defer allocator.free(scale);
    const trans = try allocator.alloc(f32, dim);
    defer allocator.free(trans);
    const dx = try allocator.alloc(f32, dim2);
    defer allocator.free(dx);
    var l = end;
    while (l > start) {
        l -= 1;
        const layer = &core.layers[l];
        @memset(ds_buf, 0.0);
        @memset(dt_buf, 0.0);
        const params = try layer.couplingParams();
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const row = state[b * dim2 .. b * dim2 + dim2];
            const grow = grad[b * dim2 .. b * dim2 + dim2];
            core.oftb.inverseSliceInPlace(row);
            try layer.couplingInverseRow(row, scale, trans);
            core.oftb.backwardSliceInPlace(grow);
            _ = try tensor.couplingAdjointRow(
                params,
                row[0..dim],
                row[dim..dim2],
                grow[0..dim],
                grow[dim..dim2],
                dx[0..dim],
                dx[dim..dim2],
                ds_buf,
                dt_buf,
                1.0,
                logdet_adjoint,
                &scratch,
            );
            @memcpy(grow, dx);
        }
        try addLayerParamGrads(layer, ds_buf, dt_buf);
        noteLayerApplication(core);
    }
    @memcpy(grad_input_blocked.data[0..n], grad[0..n]);
}
fn midpointBackwardAdjointOnCore(
    core: *RSFCore,
    w_blocked: *const Tensor,
    grad: []f32,
    start: usize,
    end: usize,
    ld_shift: f32,
) !void {
    try validateTensor2D(w_blocked);
    const dim = core.dim;
    const dim2 = try checkedMul(dim, 2);
    const batch = w_blocked.shape.dims[0];
    if (batch == 0) return error.InvalidBatchSize;
    const n = try checkedMul(batch, dim2);
    if (w_blocked.data.len < n or grad.len < n) return error.DataLengthMismatch;
    if (!std.math.isFinite(ld_shift)) return error.NonFinite;
    if (start == end) return;
    const allocator = scratchAllocator();
    const state = try allocator.alloc(f32, n);
    defer allocator.free(state);
    @memcpy(state, w_blocked.data[0..n]);
    var scratch = try tensor.InvertedFlowScratch.init(allocator, dim);
    defer scratch.deinit();
    const expected = try checkedMul(dim, coupling_width);
    const ds_buf = try allocator.alloc(f32, expected);
    defer allocator.free(ds_buf);
    const dt_buf = try allocator.alloc(f32, expected);
    defer allocator.free(dt_buf);
    const scale = try allocator.alloc(f32, dim);
    defer allocator.free(scale);
    const trans = try allocator.alloc(f32, dim);
    defer allocator.free(trans);
    const u = try allocator.alloc(f32, dim2);
    defer allocator.free(u);
    const o = try allocator.alloc(f32, dim2);
    defer allocator.free(o);
    const gu = try allocator.alloc(f32, dim2);
    defer allocator.free(gu);
    var l: usize = start;
    while (l < end) : (l += 1) {
        const layer = &core.layers[l];
        @memset(ds_buf, 0.0);
        @memset(dt_buf, 0.0);
        const params = try layer.couplingParams();
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const row = state[b * dim2 .. b * dim2 + dim2];
            const grow = grad[b * dim2 .. b * dim2 + dim2];
            @memcpy(u, row);
            try layer.couplingForwardRow(u, scale, trans);
            @memcpy(o, u);
            core.oftb.forwardSliceInPlace(o);
            _ = try tensor.couplingInvertedFlowAdjointRow(
                params,
                u[0..dim],
                u[dim..dim2],
                grow[0..dim],
                grow[dim..dim2],
                gu[0..dim],
                gu[dim..dim2],
                ds_buf,
                dt_buf,
                1.0,
                ld_shift,
                &scratch,
            );
            @memcpy(grow, gu);
            core.oftb.forwardSliceInPlace(grow);
            @memcpy(row, o);
        }
        try addLayerParamGrads(layer, ds_buf, dt_buf);
        noteLayerApplication(core);
    }
}
fn seedCollisionGrads(z: *const Tensor, w: *const Tensor, gz: []f32, gw: []f32, dim: usize, grad_scale: f32) !void {
    if (!tensorsSameShape(z, w)) return error.ShapeMismatch;
    try validateTensor2D(z);
    const batch = z.shape.dims[0];
    const dim2 = try checkedMul(dim, 2);
    const n = try checkedMul(batch, dim2);
    if (gz.len < n or gw.len < n) return error.DataLengthMismatch;
    if (!std.math.isFinite(grad_scale)) return error.NonFinite;
    const tokens_dim = try checkedMul(batch, dim);
    const inv_td = grad_scale / @as(f32, @floatFromInt(tokens_dim));
    if (!std.math.isFinite(inv_td)) return error.NonFinite;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const g = 2.0 * (z.data[i] - w.data[i]) * inv_td;
        if (!std.math.isFinite(g)) return error.NonFinite;
        gz[i] = g;
        gw[i] = -g;
    }
}
fn midpointCollisionOnCore(
    core: *const RSFCore,
    input: *const RSFLatentState,
    target: *const RSFLatentState,
    allocator: Allocator,
    parallel: bool,
) !MidpointResult {
    try validateModelMetadata(core);
    const layer_count = try checkedModelLayerCount(core);
    const split = midpointSplitLayers(layer_count);
    var z = try input.clone(allocator);
    errdefer z.deinit();
    var w = try target.clone(allocator);
    errdefer w.deinit();
    var z_blocked = try allocBlockedLatentTensor(allocator, &z);
    defer z_blocked.deinit();
    var w_blocked = try allocBlockedLatentTensor(allocator, &w);
    defer w_blocked.deinit();
    try latentToBlockedTensor(&z, &z_blocked);
    try latentToBlockedTensor(&w, &w_blocked);
    var logdet_forward: f32 = 0.0;
    var logdet_backward: f32 = 0.0;
    if (parallel) {
        const Job = struct {
            fn fwd(c: *const RSFCore, t: *Tensor, start: usize, end: usize, out: *f32, err_slot: *?anyerror) void {
                out.* = blockedMeanLogDetAndApplyForward(c, t, start, end) catch |e| {
                    err_slot.* = e;
                    return;
                };
            }
            fn bwd(c: *const RSFCore, t: *Tensor, start: usize, end: usize, out: *f32, err_slot: *?anyerror) void {
                out.* = blockedMeanLogDetAndApplyInverse(c, t, start, end) catch |e| {
                    err_slot.* = e;
                    return;
                };
            }
        };
        var fwd_err: ?anyerror = null;
        var bwd_err: ?anyerror = null;
        const fwd_thread = try Thread.spawn(.{}, Job.fwd, .{ core, &z_blocked, @as(usize, 0), split.forward_layers, &logdet_forward, &fwd_err });
        const bwd_thread = Thread.spawn(.{}, Job.bwd, .{ core, &w_blocked, split.forward_layers, layer_count, &logdet_backward, &bwd_err }) catch |spawn_err| {
            fwd_thread.join();
            return spawn_err;
        };
        fwd_thread.join();
        bwd_thread.join();
        if (fwd_err) |e| return e;
        if (bwd_err) |e| return e;
    } else {
        logdet_forward = try blockedMeanLogDetAndApplyForward(core, &z_blocked, 0, split.forward_layers);
        logdet_backward = try blockedMeanLogDetAndApplyInverse(core, &w_blocked, split.forward_layers, layer_count);
    }
    const depth = if (split.forward_layers > split.backward_layers) split.forward_layers else split.backward_layers;
    noteFrontierDepth(core, depth);
    const collision_loss = try collisionLossBlocked(&z_blocked, &w_blocked, core.dim);
    try blockedTensorToLatent(&z_blocked, &z);
    try blockedTensorToLatent(&w_blocked, &w);
    z.log_det = logdet_forward;
    w.log_det = logdet_backward;
    const logdet_total = logdet_forward + logdet_backward;
    if (!std.math.isFinite(logdet_total)) return error.NonFinite;
    return .{
        .z = z,
        .w = w,
        .collision_loss = collision_loss,
        .logdet_forward = logdet_forward,
        .logdet_backward = logdet_backward,
        .logdet_total = logdet_total,
        .forward_layers = split.forward_layers,
        .backward_layers = split.backward_layers,
    };
}
fn midpointBackwardOnCore(
    core: *RSFCore,
    z_blocked: *const Tensor,
    w_blocked: *const Tensor,
    grad_input_blocked: *Tensor,
    start: usize,
    end: usize,
    layer_count: usize,
    grad_scale: f32,
    logdet_weight: f32,
    parallel: bool,
) !void {
    try validateTensor2D(z_blocked);
    try validateTensor2D(w_blocked);
    try validateTensor2D(grad_input_blocked);
    if (!tensorsSameShape(z_blocked, w_blocked)) return error.ShapeMismatch;
    if (!tensorsSameShape(z_blocked, grad_input_blocked)) return error.ShapeMismatch;
    if (!std.math.isFinite(grad_scale) or !std.math.isFinite(logdet_weight)) return error.NonFinite;
    const dim = core.dim;
    const dim2 = try checkedMul(dim, 2);
    const batch = z_blocked.shape.dims[0];
    const n = try checkedMul(batch, dim2);
    const allocator = scratchAllocator();
    const gz = try allocator.alloc(f32, n);
    defer allocator.free(gz);
    const gw = try allocator.alloc(f32, n);
    defer allocator.free(gw);
    try seedCollisionGrads(z_blocked, w_blocked, gz, gw, dim, grad_scale);
    const tokens: f32 = @floatFromInt(batch);
    const ld_shift = logdet_weight / tokens;
    if (!std.math.isFinite(ld_shift)) return error.NonFinite;
    const logdet_adjoint = -ld_shift;
    var li: usize = 0;
    while (li < layer_count) : (li += 1) try core.layers[li].ensureGradients();
    if (parallel) {
        const Job = struct {
            fn fwd(
                c: *RSFCore,
                z: *const Tensor,
                g: []f32,
                gin: *Tensor,
                s: usize,
                e: usize,
                adj: f32,
                err_slot: *?anyerror,
            ) void {
                midpointForwardAdjointOnCore(c, z, g, gin, s, e, adj) catch |err| {
                    err_slot.* = err;
                };
            }
            fn bwd(
                c: *RSFCore,
                w: *const Tensor,
                g: []f32,
                s: usize,
                e: usize,
                shift: f32,
                err_slot: *?anyerror,
            ) void {
                midpointBackwardAdjointOnCore(c, w, g, s, e, shift) catch |err| {
                    err_slot.* = err;
                };
            }
        };
        var fwd_err: ?anyerror = null;
        var bwd_err: ?anyerror = null;
        const fwd_thread = try Thread.spawn(.{}, Job.fwd, .{ core, z_blocked, gz, grad_input_blocked, start, end, logdet_adjoint, &fwd_err });
        const bwd_thread = Thread.spawn(.{}, Job.bwd, .{ core, w_blocked, gw, end, layer_count, ld_shift, &bwd_err }) catch |spawn_err| {
            fwd_thread.join();
            return spawn_err;
        };
        fwd_thread.join();
        bwd_thread.join();
        if (fwd_err) |e| return e;
        if (bwd_err) |e| return e;
    } else {
        try midpointForwardAdjointOnCore(core, z_blocked, gz, grad_input_blocked, start, end, logdet_adjoint);
        try midpointBackwardAdjointOnCore(core, w_blocked, gw, end, layer_count, ld_shift);
    }
    const depth = if (end - start > layer_count - end) end - start else layer_count - end;
    noteFrontierDepth(core, depth);
}
fn backwardOnCore(core: *RSFCore, grad_output: *const Tensor, input: *const Tensor, output: *const Tensor, grad_input_out: *Tensor, logdet_weight: f32) !void {
    try validateModelMetadata(core);
    try validateTensor2D(grad_output);
    try validateTensor2D(input);
    try validateTensor2D(output);
    try validateTensor2D(grad_input_out);
    const layer_count = try checkedModelLayerCount(core);
    const dim = core.dim;
    const dim2 = try checkedMul(dim, 2);
    if (input.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (!tensorsSameShape(grad_output, input)) return error.ShapeMismatch;
    if (!tensorsSameShape(output, input)) return error.ShapeMismatch;
    if (!tensorsSameShape(grad_input_out, input)) return error.ShapeMismatch;
    if (!std.math.isFinite(logdet_weight)) return error.NonFinite;
    if (tensorsOverlap(grad_input_out, grad_output)) return error.AliasedBuffers;
    if (tensorsOverlap(grad_input_out, input)) return error.AliasedBuffers;
    if (tensorsOverlap(grad_input_out, output)) return error.AliasedBuffers;
    try ensureFiniteSlice(input.data);
    try ensureFiniteSlice(output.data);
    try ensureFiniteSlice(grad_output.data);
    const batch_size = input.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    var li: usize = 0;
    while (li < layer_count) : (li += 1) try core.layers[li].ensureGradients();
    const grad_scale: f32 = blk: {
        if (!core.cfg.grad_mean) break :blk 1.0;
        const s = 1.0 / @as(f32, @floatFromInt(batch_size));
        break :blk if (std.math.isFinite(s)) s else 1.0;
    };
    const allocator = scratchAllocator();
    const state_len = try checkedMul(layer_count, dim2);
    const states = try allocator.alloc(f32, state_len);
    defer allocator.free(states);
    const cur = try allocator.alloc(f32, dim2);
    defer allocator.free(cur);
    const dy = try allocator.alloc(f32, dim2);
    defer allocator.free(dy);
    const dx = try allocator.alloc(f32, dim2);
    defer allocator.free(dx);
    const scale_buf = try allocator.alloc(f32, dim);
    defer allocator.free(scale_buf);
    const y1_buf = try allocator.alloc(f32, dim);
    defer allocator.free(y1_buf);
    const trans_buf = try allocator.alloc(f32, dim);
    defer allocator.free(trans_buf);
    const per_layer_params = try checkedMul(dim, coupling_width);
    const ds_weight_buf = try allocator.alloc(f32, per_layer_params);
    defer allocator.free(ds_weight_buf);
    const dt_weight_buf = try allocator.alloc(f32, per_layer_params);
    defer allocator.free(dt_weight_buf);
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        @memcpy(cur, input.data[b * dim2 .. b * dim2 + dim2]);
        var l: usize = 0;
        while (l < layer_count) : (l += 1) {
            @memcpy(states[l * dim2 .. l * dim2 + dim2], cur);
            try core.layers[l].couplingForwardRow(cur, scale_buf, trans_buf);
            core.oftb.forwardSliceInPlace(cur);
        }
        var check: usize = 0;
        while (check < dim2) : (check += 1) {
            if (!valuesWithinTolerance(cur[check], output.data[b * dim2 + check], MODEL_CROSS_CHECK_ABS_TOL, MODEL_CROSS_CHECK_REL_TOL)) return error.StateMismatch;
        }
        @memcpy(dy, grad_output.data[b * dim2 .. b * dim2 + dim2]);
        var idx = layer_count;
        while (idx > 0) : (idx -= 1) {
            core.oftb.backwardSliceInPlace(dy);
            const state = states[(idx - 1) * dim2 .. (idx - 1) * dim2 + dim2];
            try core.layers[idx - 1].backwardFromInputsRow(
                state[0..dim],
                state[dim..dim2],
                dy[0..dim],
                dy[dim..dim2],
                dx[0..dim],
                dx[dim..dim2],
                y1_buf,
                scale_buf,
                ds_weight_buf,
                dt_weight_buf,
                grad_scale,
                logdet_weight,
            );
            @memcpy(dy, dx);
        }
        @memcpy(grad_input_out.data[b * dim2 .. b * dim2 + dim2], dy);
    }
}
const SequenceShape = struct {
    num_sequences: usize,
    seq_len: usize,
    dim: usize,
    dim2: usize,
    tokens: usize,

    fn seqDim(self: SequenceShape) !usize {
        return checkedMul(self.seq_len, self.dim);
    }

    fn stateLen(self: SequenceShape) !usize {
        return checkedMul(self.seq_len, self.dim2);
    }
};

fn sequenceShapeOf(x: *const Tensor, mask: *const types.RSFSequenceMask, core: *const RSFCore) !SequenceShape {
    try validateTensor2D(x);
    const dim = core.dim;
    const dim2 = try checkedMul(dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    const tokens = x.shape.dims[0];
    const seq_len = mask.seq_len;
    if (seq_len == 0 or tokens == 0) return error.DimensionMismatch;
    if (tokens % seq_len != 0) return error.DimensionMismatch;
    return .{
        .num_sequences = tokens / seq_len,
        .seq_len = seq_len,
        .dim = dim,
        .dim2 = dim2,
        .tokens = tokens,
    };
}

const SequenceScratch = struct {
    allocator: Allocator,
    x1: []f32,
    x2: []f32,
    y1: []f32,
    y2: []f32,
    key: []f32,
    scale: []f32,
    trans: []f32,
    row: []f32,
    ds_weight: []f32,
    dt_weight: []f32,
    ds_scratch: []f32,

    fn init(allocator: Allocator, shape: SequenceShape) !SequenceScratch {
        const seq_dim = try shape.seqDim();
        const params_len = try checkedMul(shape.dim, coupling_width);
        const x1 = try allocator.alloc(f32, seq_dim);
        errdefer allocator.free(x1);
        const x2 = try allocator.alloc(f32, seq_dim);
        errdefer allocator.free(x2);
        const y1 = try allocator.alloc(f32, seq_dim);
        errdefer allocator.free(y1);
        const y2 = try allocator.alloc(f32, seq_dim);
        errdefer allocator.free(y2);
        const key = try allocator.alloc(f32, shape.dim);
        errdefer allocator.free(key);
        const scale = try allocator.alloc(f32, shape.dim);
        errdefer allocator.free(scale);
        const trans = try allocator.alloc(f32, shape.dim);
        errdefer allocator.free(trans);
        const row = try allocator.alloc(f32, shape.dim2);
        errdefer allocator.free(row);
        const ds_weight = try allocator.alloc(f32, params_len);
        errdefer allocator.free(ds_weight);
        const dt_weight = try allocator.alloc(f32, params_len);
        errdefer allocator.free(dt_weight);
        const ds_scratch = try allocator.alloc(f32, seq_dim);
        errdefer allocator.free(ds_scratch);
        return .{
            .allocator = allocator,
            .x1 = x1,
            .x2 = x2,
            .y1 = y1,
            .y2 = y2,
            .key = key,
            .scale = scale,
            .trans = trans,
            .row = row,
            .ds_weight = ds_weight,
            .dt_weight = dt_weight,
            .ds_scratch = ds_scratch,
        };
    }

    fn deinit(self: *SequenceScratch) void {
        self.allocator.free(self.x1);
        self.allocator.free(self.x2);
        self.allocator.free(self.y1);
        self.allocator.free(self.y2);
        self.allocator.free(self.key);
        self.allocator.free(self.scale);
        self.allocator.free(self.trans);
        self.allocator.free(self.row);
        self.allocator.free(self.ds_weight);
        self.allocator.free(self.dt_weight);
        self.allocator.free(self.ds_scratch);
    }

    fn assembleRow(self: *SequenceScratch, shape: SequenceShape, t: usize) void {
        @memcpy(self.row[0..shape.dim], self.x1[t * shape.dim ..][0..shape.dim]);
        @memcpy(self.row[shape.dim..shape.dim2], self.x2[t * shape.dim ..][0..shape.dim]);
    }

    fn splitRow(self: *SequenceScratch, shape: SequenceShape, t: usize) void {
        @memcpy(self.x1[t * shape.dim ..][0..shape.dim], self.row[0..shape.dim]);
        @memcpy(self.x2[t * shape.dim ..][0..shape.dim], self.row[shape.dim..shape.dim2]);
    }

    fn assembleGradientRow(self: *SequenceScratch, shape: SequenceShape, t: usize, g1: []const f32, g2: []const f32) void {
        @memcpy(self.row[0..shape.dim], g1[t * shape.dim ..][0..shape.dim]);
        @memcpy(self.row[shape.dim..shape.dim2], g2[t * shape.dim ..][0..shape.dim]);
    }

    fn splitGradientRow(self: *SequenceScratch, shape: SequenceShape, t: usize, g1: []f32, g2: []f32) void {
        @memcpy(g1[t * shape.dim ..][0..shape.dim], self.row[0..shape.dim]);
        @memcpy(g2[t * shape.dim ..][0..shape.dim], self.row[shape.dim..shape.dim2]);
    }
};

fn gatherSequence(x: *const Tensor, shape: SequenceShape, sequence: usize, even: []f32, odd: []f32) !void {
    const seq_dim = try shape.seqDim();
    if (even.len < seq_dim or odd.len < seq_dim) return error.DataLengthMismatch;
    const base_token = try checkedMul(sequence, shape.seq_len);
    var t: usize = 0;
    while (t < shape.seq_len) : (t += 1) {
        const row = x.data[(base_token + t) * shape.dim2 ..][0..shape.dim2];
        @memcpy(even[t * shape.dim ..][0..shape.dim], row[0..shape.dim]);
        @memcpy(odd[t * shape.dim ..][0..shape.dim], row[shape.dim..shape.dim2]);
    }
}

fn scatterSequence(x: *Tensor, shape: SequenceShape, sequence: usize, even: []const f32, odd: []const f32) !void {
    const seq_dim = try shape.seqDim();
    if (even.len < seq_dim or odd.len < seq_dim) return error.DataLengthMismatch;
    try ensureFiniteSlice(even[0..seq_dim]);
    try ensureFiniteSlice(odd[0..seq_dim]);
    const base_token = try checkedMul(sequence, shape.seq_len);
    var t: usize = 0;
    while (t < shape.seq_len) : (t += 1) {
        const row = x.data[(base_token + t) * shape.dim2 ..][0..shape.dim2];
        @memcpy(row[0..shape.dim], even[t * shape.dim ..][0..shape.dim]);
        @memcpy(row[shape.dim..shape.dim2], odd[t * shape.dim ..][0..shape.dim]);
    }
}

fn storeSequenceState(states: []f32, shape: SequenceShape, layer_count: usize, slot: usize, even: []const f32, odd: []const f32) !void {
    if (slot > layer_count) return error.LayerIndexOutOfBounds;
    const state_len = try shape.stateLen();
    const seq_dim = try shape.seqDim();
    const base = try checkedMul(slot, state_len);
    if (states.len < base + state_len) return error.DataLengthMismatch;
    if (even.len < seq_dim or odd.len < seq_dim) return error.DataLengthMismatch;
    var t: usize = 0;
    while (t < shape.seq_len) : (t += 1) {
        const row = states[base + t * shape.dim2 ..][0..shape.dim2];
        @memcpy(row[0..shape.dim], even[t * shape.dim ..][0..shape.dim]);
        @memcpy(row[shape.dim..shape.dim2], odd[t * shape.dim ..][0..shape.dim]);
    }
}

fn loadSequenceState(states: []const f32, shape: SequenceShape, layer_count: usize, slot: usize, even: []f32, odd: []f32) !void {
    if (slot > layer_count) return error.LayerIndexOutOfBounds;
    const state_len = try shape.stateLen();
    const seq_dim = try shape.seqDim();
    const base = try checkedMul(slot, state_len);
    if (states.len < base + state_len) return error.DataLengthMismatch;
    if (even.len < seq_dim or odd.len < seq_dim) return error.DataLengthMismatch;
    var t: usize = 0;
    while (t < shape.seq_len) : (t += 1) {
        const row = states[base + t * shape.dim2 ..][0..shape.dim2];
        @memcpy(even[t * shape.dim ..][0..shape.dim], row[0..shape.dim]);
        @memcpy(odd[t * shape.dim ..][0..shape.dim], row[shape.dim..shape.dim2]);
    }
}

fn forwardSequenceOnCore(core: *const RSFCore, x: *Tensor, mask: *const types.RSFSequenceMask, logdet_per_sequence: ?[]f32) !void {
    try validateModelMetadata(core);
    const shape = try sequenceShapeOf(x, mask, core);
    const layer_count = try checkedModelLayerCount(core);
    if (logdet_per_sequence) |buf| {
        if (buf.len < shape.num_sequences) return error.DataLengthMismatch;
        @memset(buf[0..shape.num_sequences], 0.0);
    }
    try ensureFiniteSlice(x.data);
    const seq_dim = try shape.seqDim();
    const allocator = scratchAllocator();
    var scratch = try SequenceScratch.init(allocator, shape);
    defer scratch.deinit();
    var seq: usize = 0;
    while (seq < shape.num_sequences) : (seq += 1) {
        try gatherSequence(x, shape, seq, scratch.x1, scratch.x2);
        var l: usize = 0;
        while (l < layer_count) : (l += 1) {
            const params = try core.layers[l].couplingParams();
            const logdet = try tensor.causalCouplingForward(params, mask.*, scratch.x1, scratch.x2, scratch.y1, scratch.y2, scratch.key, scratch.scale, scratch.trans);
            if (logdet_per_sequence) |buf| {
                const value: f32 = @floatCast(logdet);
                if (!std.math.isFinite(value)) return error.NonFinite;
                buf[seq] += value;
            }
            @memcpy(scratch.x1[0..seq_dim], scratch.y1[0..seq_dim]);
            @memcpy(scratch.x2[0..seq_dim], scratch.y2[0..seq_dim]);
            var t: usize = 0;
            while (t < shape.seq_len) : (t += 1) {
                scratch.assembleRow(shape, t);
                core.oftb.forwardSliceInPlace(scratch.row);
                scratch.splitRow(shape, t);
            }
        }
        try scatterSequence(x, shape, seq, scratch.x1, scratch.x2);
    }
}

fn inverseSequenceOnCore(core: *const RSFCore, y: *Tensor, mask: *const types.RSFSequenceMask, logdet_per_sequence: ?[]f32) !void {
    try validateModelMetadata(core);
    const shape = try sequenceShapeOf(y, mask, core);
    const layer_count = try checkedModelLayerCount(core);
    if (logdet_per_sequence) |buf| {
        if (buf.len < shape.num_sequences) return error.DataLengthMismatch;
        @memset(buf[0..shape.num_sequences], 0.0);
    }
    try ensureFiniteSlice(y.data);
    const seq_dim = try shape.seqDim();
    const allocator = scratchAllocator();
    var scratch = try SequenceScratch.init(allocator, shape);
    defer scratch.deinit();
    var seq: usize = 0;
    while (seq < shape.num_sequences) : (seq += 1) {
        try gatherSequence(y, shape, seq, scratch.x1, scratch.x2);
        var idx = layer_count;
        while (idx > 0) : (idx -= 1) {
            var t: usize = 0;
            while (t < shape.seq_len) : (t += 1) {
                scratch.assembleRow(shape, t);
                core.oftb.inverseSliceInPlace(scratch.row);
                scratch.splitRow(shape, t);
            }
            const params = try core.layers[idx - 1].couplingParams();
            const logdet = try tensor.causalCouplingInverse(params, mask.*, scratch.x1, scratch.x2, scratch.y1, scratch.y2, scratch.key, scratch.scale, scratch.trans);
            if (logdet_per_sequence) |buf| {
                const value: f32 = @floatCast(logdet);
                if (!std.math.isFinite(value)) return error.NonFinite;
                buf[seq] += value;
            }
            @memcpy(scratch.x1[0..seq_dim], scratch.y1[0..seq_dim]);
            @memcpy(scratch.x2[0..seq_dim], scratch.y2[0..seq_dim]);
        }
        try scatterSequence(y, shape, seq, scratch.x1, scratch.x2);
    }
}

fn backwardSequenceOnCore(
    core: *RSFCore,
    grad_output: *const Tensor,
    input: *const Tensor,
    output: *const Tensor,
    mask: *const types.RSFSequenceMask,
    grad_input_out: *Tensor,
    logdet_weight: f32,
) !void {
    try validateModelMetadata(core);
    try validateTensor2D(grad_output);
    try validateTensor2D(input);
    try validateTensor2D(output);
    try validateTensor2D(grad_input_out);
    const shape = try sequenceShapeOf(input, mask, core);
    if (!tensorsSameShape(grad_output, input)) return error.ShapeMismatch;
    if (!tensorsSameShape(output, input)) return error.ShapeMismatch;
    if (!tensorsSameShape(grad_input_out, input)) return error.ShapeMismatch;
    if (!std.math.isFinite(logdet_weight)) return error.NonFinite;
    if (tensorsOverlap(grad_input_out, grad_output)) return error.AliasedBuffers;
    if (tensorsOverlap(grad_input_out, input)) return error.AliasedBuffers;
    if (tensorsOverlap(grad_input_out, output)) return error.AliasedBuffers;
    try ensureFiniteSlice(input.data);
    try ensureFiniteSlice(output.data);
    try ensureFiniteSlice(grad_output.data);
    const layer_count = try checkedModelLayerCount(core);
    var li: usize = 0;
    while (li < layer_count) : (li += 1) try core.layers[li].ensureGradients();
    const grad_scale: f32 = blk: {
        if (!core.cfg.grad_mean) break :blk 1.0;
        const value = 1.0 / @as(f32, @floatFromInt(shape.tokens));
        break :blk if (std.math.isFinite(value)) value else 1.0;
    };
    const seq_dim = try shape.seqDim();
    const state_len = try shape.stateLen();
    const allocator = scratchAllocator();
    var scratch = try SequenceScratch.init(allocator, shape);
    defer scratch.deinit();
    const states_len = try checkedMul(layer_count + 1, state_len);
    const states = try allocator.alloc(f32, states_len);
    defer allocator.free(states);
    const g1 = try allocator.alloc(f32, seq_dim);
    defer allocator.free(g1);
    const g2 = try allocator.alloc(f32, seq_dim);
    defer allocator.free(g2);
    const dx1 = try allocator.alloc(f32, seq_dim);
    defer allocator.free(dx1);
    const dx2 = try allocator.alloc(f32, seq_dim);
    defer allocator.free(dx2);
    const recovered1 = try allocator.alloc(f32, seq_dim);
    defer allocator.free(recovered1);
    const recovered2 = try allocator.alloc(f32, seq_dim);
    defer allocator.free(recovered2);
    const params_len = try checkedMul(shape.dim, coupling_width);

    var seq: usize = 0;
    while (seq < shape.num_sequences) : (seq += 1) {
        try gatherSequence(input, shape, seq, scratch.x1, scratch.x2);
        try storeSequenceState(states, shape, layer_count, 0, scratch.x1, scratch.x2);
        var l: usize = 0;
        while (l < layer_count) : (l += 1) {
            const params = try core.layers[l].couplingParams();
            _ = try tensor.causalCouplingForward(params, mask.*, scratch.x1, scratch.x2, scratch.y1, scratch.y2, scratch.key, scratch.scale, scratch.trans);
            @memcpy(scratch.x1[0..seq_dim], scratch.y1[0..seq_dim]);
            @memcpy(scratch.x2[0..seq_dim], scratch.y2[0..seq_dim]);
            var t: usize = 0;
            while (t < shape.seq_len) : (t += 1) {
                scratch.assembleRow(shape, t);
                core.oftb.forwardSliceInPlace(scratch.row);
                scratch.splitRow(shape, t);
            }
            try storeSequenceState(states, shape, layer_count, l + 1, scratch.x1, scratch.x2);
        }
        var check: usize = 0;
        while (check < state_len) : (check += 1) {
            var expected: f32 = undefined;
            const token = check / shape.dim2;
            const within = check % shape.dim2;
            expected = output.data[(seq * shape.seq_len + token) * shape.dim2 + within];
            const produced = states[layer_count * state_len + check];
            if (!valuesWithinTolerance(produced, expected, MODEL_CROSS_CHECK_ABS_TOL, MODEL_CROSS_CHECK_REL_TOL)) return error.StateMismatch;
        }
        try gatherSequence(grad_output, shape, seq, g1, g2);
        var idx = layer_count;
        while (idx > 0) : (idx -= 1) {
            const layer = &core.layers[idx - 1];
            var t: usize = 0;
            while (t < shape.seq_len) : (t += 1) {
                scratch.assembleGradientRow(shape, t, g1, g2);
                core.oftb.backwardSliceInPlace(scratch.row);
                scratch.splitGradientRow(shape, t, g1, g2);
            }
            try loadSequenceState(states, shape, layer_count, idx - 1, scratch.x1, scratch.x2);
            try loadSequenceState(states, shape, layer_count, idx, recovered1, recovered2);
            t = 0;
            while (t < shape.seq_len) : (t += 1) {
                scratch.assembleGradientRow(shape, t, recovered1, recovered2);
                core.oftb.inverseSliceInPlace(scratch.row);
                scratch.splitGradientRow(shape, t, recovered1, recovered2);
            }
            const params = try layer.couplingParams();
            @memset(scratch.ds_weight[0..params_len], 0.0);
            @memset(scratch.dt_weight[0..params_len], 0.0);
            _ = try tensor.causalCouplingBackward(
                params,
                mask.*,
                scratch.x2,
                recovered1,
                g1,
                g2,
                logdet_weight,
                scratch.ds_weight,
                scratch.dt_weight,
                dx1,
                dx2,
                scratch.key,
                scratch.ds_scratch,
            );
            try layer.recordCausalGaussNewton(params, mask, scratch.x2, logdet_weight, scratch.key);
            layer.rwlock.lock();
            defer layer.rwlock.unlock();
            if (layer.s_weight_grad) |*swg| {
                if (swg.data.len >= params_len) {
                    var k: usize = 0;
                    while (k < params_len) : (k += 1) swg.data[k] += grad_scale * scratch.ds_weight[k];
                }
            }
            if (layer.t_weight_grad) |*twg| {
                if (twg.data.len >= params_len) {
                    var k: usize = 0;
                    while (k < params_len) : (k += 1) twg.data[k] += grad_scale * scratch.dt_weight[k];
                }
            }
            @memcpy(g1[0..seq_dim], dx1[0..seq_dim]);
            @memcpy(g2[0..seq_dim], dx2[0..seq_dim]);
        }
        try scatterSequence(grad_input_out, shape, seq, g1, g2);
    }
}

fn meanSequenceLogDet(per_sequence: []const f32, num_sequences: usize) !f32 {
    return meanLogDet(per_sequence, num_sequences);
}

fn layerGPUCompatible(layer: *const LayerCore, cfg: *const RSFConfig, dim: usize) bool {
    if (layer.dim != dim) return false;
    if (layer.clip_min != cfg.clip_min or layer.clip_max != cfg.clip_max or layer.grad_mean != cfg.grad_mean) return false;
    return std.math.isFinite(layer.clip_min) and std.math.isFinite(layer.clip_max) and layer.clip_min < layer.clip_max;
}
fn modelGPUCompatible(core: *const RSFCore) bool {
    if (comptime !accel.gpu_enabled) return false;
    if (core.layers.len == 0) return false;
    if (core.num_layers != core.layers.len) return false;
    for (core.layers) |*layer| {
        if (!layerGPUCompatible(layer, &core.cfg, core.dim)) return false;
    }
    return true;
}
fn disableGPU(core: *RSFCore) void {
    core.gpu_available.store(0, .monotonic);
    if (core.gpu_accel) |*ga| {
        ga.deinit();
        core.gpu_accel = null;
    }
    if (core.f16_buf) |buf| {
        core.allocator.free(buf);
        core.f16_buf = null;
    }
    core.gpu_weight_version = 0;
}
fn validateF16Convertible(data: []const f32) !void {
    const max_f16: f32 = @floatCast(std.math.floatMax(f16));
    for (data) |v| {
        if (!std.math.isFinite(v)) return error.NonFinite;
        if (@abs(v) > max_f16) return error.NumericFailure;
    }
}
fn uploadLayerToAccel(core: *const RSFCore, layer: *const LayerCore, layer_index: usize, ga: *accel.RSFAccelerator, f16_buf: []f16) !void {
    const per_layer = try checkedMul(core.dim, coupling_width);
    if (f16_buf.len < per_layer) return error.DataLengthMismatch;
    try validateTensor2DShape(&layer.s_weight, core.dim, coupling_width);
    try validateTensor2DShape(&layer.t_weight, core.dim, coupling_width);
    try validateF16Convertible(layer.s_weight.data);
    try validateF16Convertible(layer.t_weight.data);
    var i: usize = 0;
    while (i < per_layer) : (i += 1) f16_buf[i] = @floatCast(layer.s_weight.data[i]);
    try ga.setLayerWeightsS(layer_index, f16_buf[0..per_layer], core.dim, coupling_width);
    i = 0;
    while (i < per_layer) : (i += 1) f16_buf[i] = @floatCast(layer.t_weight.data[i]);
    try ga.setLayerWeightsT(layer_index, f16_buf[0..per_layer], core.dim, coupling_width);
}
fn validateAcceleratorAgainstCPU(core: *const RSFCore, ga: *accel.RSFAccelerator) !void {
    const dim2 = try checkedMul(core.dim, 2);
    const allocator = scratchAllocator();
    var probe = try Tensor.init(allocator, &.{ GPU_VALIDATION_BATCH, dim2 });
    defer probe.deinit();
    var prng = types.PRNG.init(GPU_VALIDATION_SEED);
    for (probe.data) |*v| v.* = prng.float() * 0.5 - 0.25;
    var expected = try tensorClone(allocator, &probe);
    defer expected.deinit();
    try forwardOnCore(core, &expected);
    try ensureFiniteSlice(expected.data);
    var produced = ga.forwardFromTensor(&probe, allocator) catch return error.GPUValidationFailed;
    defer produced.deinit();
    if (!tensorHasShape(&produced, GPU_VALIDATION_BATCH, dim2)) return error.GPUValidationFailed;
    if (produced.data.len != expected.data.len) return error.GPUValidationFailed;
    try ensureFiniteSlice(produced.data);
    if (!(try tensorAllCloseEq(&expected, &produced, MODEL_CROSS_CHECK_ABS_TOL, MODEL_CROSS_CHECK_REL_TOL))) return error.GPUNumericMismatch;
}
fn syncAllLayersGPU(core: *RSFCore) !void {
    var success = false;
    defer if (!success) disableGPU(core);
    if (comptime !accel.gpu_enabled) return error.GPUUnsupportedConfiguration;
    try validateModelMetadata(core);
    if (!modelGPUCompatible(core)) return error.GPUUnsupportedConfiguration;
    const layer_count = try checkedModelLayerCount(core);
    const per_layer = try checkedMul(core.dim, coupling_width);
    for (core.layers) |*layer| {
        try ensureFiniteSlice(layer.s_weight.data);
        try ensureFiniteSlice(layer.t_weight.data);
        try validateF16Convertible(layer.s_weight.data);
        try validateF16Convertible(layer.t_weight.data);
    }
    const local_f16 = try core.allocator.alloc(f16, per_layer);
    var local_f16_owned = true;
    errdefer if (local_f16_owned) core.allocator.free(local_f16);
    const accel_model_dim = try checkedMul(core.dim, 2);
    var staged_accel = accel.RSFAccelerator.initMultiLayer(accel_model_dim, layer_count, core.allocator) catch return error.NoGPUAvailable;
    var staged_owned = true;
    errdefer if (staged_owned) staged_accel.deinit();
    try staged_accel.setClipRange(@floatCast(core.cfg.clip_min), @floatCast(core.cfg.clip_max));
    var index: usize = 0;
    while (index < layer_count) : (index += 1) {
        try uploadLayerToAccel(core, &core.layers[index], index, &staged_accel, local_f16);
    }
    try validateAcceleratorAgainstCPU(core, &staged_accel);
    if (core.gpu_accel) |*ga| ga.deinit();
    if (core.f16_buf) |buf| core.allocator.free(buf);
    core.gpu_accel = staged_accel;
    staged_owned = false;
    core.f16_buf = local_f16;
    local_f16_owned = false;
    core.gpu_weight_version = core.cpu_weight_version;
    core.gpu_available.store(1, .monotonic);
    success = true;
}
fn tryForwardGPU(core: *RSFCore, x: *Tensor) bool {
    if (comptime !accel.gpu_enabled) return false;
    if (!modelGPUCompatible(core)) {
        disableGPU(core);
        return false;
    }
    if (core.gpu_available.load(.monotonic) == 0) return false;
    if (core.gpu_weight_version != core.cpu_weight_version) return false;
    if (core.gpu_accel) |*ga| {
        if (ga.numLayers() != core.layers.len) {
            disableGPU(core);
            return false;
        }
        const allocator = scratchAllocator();
        if (ga.forwardFromTensor(x, allocator)) |result| {
            var gpu_result = result;
            defer gpu_result.deinit();
            if (!tensorHasShape(&gpu_result, x.shape.dims[0], x.shape.dims[1]) or gpu_result.data.len != x.data.len) {
                disableGPU(core);
                return false;
            }
            ensureFiniteSlice(gpu_result.data) catch {
                disableGPU(core);
                return false;
            };
            @memcpy(x.data, gpu_result.data);
            return true;
        } else |_| {
            disableGPU(core);
            return false;
        }
    }
    return false;
}
fn gpuPathSelected(core: *const RSFCore) bool {
    if (comptime !accel.gpu_enabled) return false;
    return core.gpu_available.load(.monotonic) != 0 or modelGPUCompatible(core);
}
fn forwardDispatchOnCore(core: *RSFCore, x: *Tensor) !void {
    try validateTensor2D(x);
    const dim2 = try checkedMul(core.dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (x.shape.dims[0] == 0) return error.InvalidBatchSize;
    if (comptime accel.gpu_enabled) {
        if (modelGPUCompatible(core)) {
            if (tryForwardGPU(core, x)) return;
            syncAllLayersGPU(core) catch {};
            if (tryForwardGPU(core, x)) return;
        } else if (core.gpu_available.load(.monotonic) != 0 or core.gpu_accel != null or core.f16_buf != null or core.gpu_weight_version != 0) {
            disableGPU(core);
        }
    }
    try forwardOnCore(core, x);
}
fn bumpWeightVersion(core: *RSFCore) void {
    core.cpu_weight_version +%= 1;
    if (core.cpu_weight_version == 0) core.cpu_weight_version = 1;
}
fn refreshGPUAfterWeightChange(core: *RSFCore) void {
    if (comptime !accel.gpu_enabled) {
        disableGPU(core);
        return;
    }
    if (!modelGPUCompatible(core)) {
        disableGPU(core);
        return;
    }
    syncAllLayersGPU(core) catch disableGPU(core);
}
const SavedLayerSnapshot = struct {
    clip_min: f32,
    clip_max: f32,
    grad_mean: bool,
    s_weight: Tensor,
    t_weight: Tensor,
    gn_block: []f32,
};
const SavedModelSnapshot = struct {
    allocator: Allocator,
    dim: usize,
    num_layers: usize,
    cfg: RSFConfig,
    layers: []SavedLayerSnapshot,
    fn deinit(self: *SavedModelSnapshot) void {
        const layers = self.layers;
        self.layers = layers[0..0];
        for (layers) |*layer| {
            layer.s_weight.deinit();
            layer.t_weight.deinit();
            self.allocator.free(layer.gn_block);
        }
        if (layers.len != 0) self.allocator.free(layers);
    }
};
fn snapshotModelForSave(allocator: Allocator, core: *const RSFCore) !SavedModelSnapshot {
    try validateModelMetadata(core);
    const layer_count = core.layers.len;
    const layers = try allocator.alloc(SavedLayerSnapshot, layer_count);
    errdefer allocator.free(layers);
    const gn_len = try checkedMul(core.dim, 3);
    var initialized: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < initialized) : (i += 1) {
            layers[i].s_weight.deinit();
            layers[i].t_weight.deinit();
            allocator.free(layers[i].gn_block);
        }
    }
    var i: usize = 0;
    while (i < layer_count) : (i += 1) {
        const layer = &core.layers[i];
        try validateClipRange(layer.clip_min, layer.clip_max);
        try ensureFiniteSlice(layer.s_weight.data);
        try ensureFiniteSlice(layer.t_weight.data);
        var sw = try tensorClone(allocator, &layer.s_weight);
        errdefer sw.deinit();
        var tw = try tensorClone(allocator, &layer.t_weight);
        errdefer tw.deinit();
        const gn = try allocator.alloc(f32, gn_len);
        errdefer allocator.free(gn);
        @memset(gn, 0.0);
        if (layer.gn_block) |blk| {
            if (blk.len >= gn_len) {
                try ensureFiniteSlice(blk[0..gn_len]);
                @memcpy(gn, blk[0..gn_len]);
            }
        }
        layers[i] = .{
            .clip_min = layer.clip_min,
            .clip_max = layer.clip_max,
            .grad_mean = layer.grad_mean,
            .s_weight = sw,
            .t_weight = tw,
            .gn_block = gn,
        };
        initialized += 1;
    }
    return .{
        .allocator = allocator,
        .dim = core.dim,
        .num_layers = core.num_layers,
        .cfg = core.cfg,
        .layers = layers,
    };
}
pub const RSFBranch = enum { s, t };

const LatentShape = struct {
    batch: usize,
    dim: usize,
};

fn latentShapeOf(state: *const RSFLatentState) !LatentShape {
    if (state.data.shape.dims.len != 3) return error.ShapeMismatch;
    if (state.data.shape.dims[2] != 2) return error.ShapeMismatch;
    const batch = state.data.shape.dims[0];
    const dim = state.data.shape.dims[1];
    if (batch == 0) return error.InvalidBatchSize;
    if (dim == 0) return error.InvalidDimension;
    _ = try checkedMul(try checkedMul(batch, dim), 2);
    return .{ .batch = batch, .dim = dim };
}

pub const LatentHalfView = struct {
    data: []f32,
    batch: usize,
    dim: usize,
    offset: usize,

    pub fn index(self: *const LatentHalfView, b: usize, d: usize) !usize {
        if (b >= self.batch or d >= self.dim) return error.OutOfBounds;
        return (b * self.dim + d) * 2 + self.offset;
    }

    pub fn get(self: *const LatentHalfView, b: usize, d: usize) !f32 {
        return self.data[try self.index(b, d)];
    }

    pub fn set(self: *const LatentHalfView, b: usize, d: usize, value: f32) !void {
        if (!std.math.isFinite(value)) return error.NonFinite;
        self.data[try self.index(b, d)] = value;
    }

    pub fn copyRowTo(self: *const LatentHalfView, b: usize, out: []f32) !void {
        if (b >= self.batch) return error.OutOfBounds;
        if (out.len < self.dim) return error.DataLengthMismatch;
        var d: usize = 0;
        while (d < self.dim) : (d += 1) out[d] = self.data[(b * self.dim + d) * 2 + self.offset];
    }

    pub fn copyRowFrom(self: *const LatentHalfView, b: usize, src: []const f32) !void {
        if (b >= self.batch) return error.OutOfBounds;
        if (src.len < self.dim) return error.DataLengthMismatch;
        try ensureFiniteSlice(src[0..self.dim]);
        var d: usize = 0;
        while (d < self.dim) : (d += 1) self.data[(b * self.dim + d) * 2 + self.offset] = src[d];
    }
};

fn latentToBlockedTensor(state: *const RSFLatentState, blocked: *Tensor) !void {
    const shape = try latentShapeOf(state);
    const dim2 = try checkedMul(shape.dim, 2);
    if (blocked.shape.dims.len != 2) return error.ShapeMismatch;
    if (blocked.shape.dims[0] != shape.batch or blocked.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (blocked.data.len < try checkedMul(shape.batch, dim2)) return error.DataLengthMismatch;
    var b: usize = 0;
    while (b < shape.batch) : (b += 1) {
        const dst = blocked.data[b * dim2 .. b * dim2 + dim2];
        var d: usize = 0;
        while (d < shape.dim) : (d += 1) {
            const src = (b * shape.dim + d) * 2;
            dst[d] = state.data.data[src];
            dst[shape.dim + d] = state.data.data[src + 1];
        }
    }
}

fn blockedTensorToLatent(blocked: *const Tensor, state: *RSFLatentState) !void {
    const shape = try latentShapeOf(state);
    const dim2 = try checkedMul(shape.dim, 2);
    if (blocked.shape.dims.len != 2) return error.ShapeMismatch;
    if (blocked.shape.dims[0] != shape.batch or blocked.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (blocked.data.len < try checkedMul(shape.batch, dim2)) return error.DataLengthMismatch;
    try ensureFiniteSlice(blocked.data[0 .. shape.batch * dim2]);
    var b: usize = 0;
    while (b < shape.batch) : (b += 1) {
        const src = blocked.data[b * dim2 .. b * dim2 + dim2];
        var d: usize = 0;
        while (d < shape.dim) : (d += 1) {
            const dst = (b * shape.dim + d) * 2;
            state.data.data[dst] = src[d];
            state.data.data[dst + 1] = src[shape.dim + d];
        }
    }
}

fn allocBlockedLatentTensor(allocator: Allocator, state: *const RSFLatentState) !Tensor {
    const shape = try latentShapeOf(state);
    return Tensor.init(allocator, &[_]usize{ shape.batch, try checkedMul(shape.dim, 2) });
}

fn meanLogDet(per_row: []const f32, batch: usize) !f32 {
    if (per_row.len < batch or batch == 0) return error.DataLengthMismatch;
    var sum: f64 = 0.0;
    for (per_row[0..batch]) |v| sum += @as(f64, @floatCast(v));
    const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(batch)));
    if (!std.math.isFinite(mean)) return error.NonFinite;
    return mean;
}

pub const RSFLatentState = struct {
    data: Tensor,
    log_det: f32,
    binding: types.RSFBinding,

    pub fn init(allocator: Allocator, model: *const RSF, batch: usize) !RSFLatentState {
        if (batch == 0) return error.InvalidBatchSize;
        const dim = try model.dim();
        const model_id = try handleId(model.id);
        var data = try Tensor.init(allocator, &[_]usize{ batch, dim, 2 });
        errdefer data.deinit();
        return .{ .data = data, .log_det = 0.0, .binding = types.RSFBinding.model(.latent_state, model_id, dim) };
    }

    pub fn fromHalves(allocator: Allocator, model: *const RSF, x1: *const Tensor, x2: *const Tensor) !RSFLatentState {
        try validateTensor2D(x1);
        try validateTensor2D(x2);
        const dim = try model.dim();
        if (x1.shape.dims[1] != dim or x2.shape.dims[1] != dim) return error.ShapeMismatch;
        if (x1.shape.dims[0] != x2.shape.dims[0]) return error.ShapeMismatch;
        const batch = x1.shape.dims[0];
        try ensureFiniteSlice(x1.data);
        try ensureFiniteSlice(x2.data);
        var state = try init(allocator, model, batch);
        errdefer state.deinit();
        const even = try state.evenView();
        const odd = try state.oddView();
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            try even.copyRowFrom(b, x1.data[b * dim .. b * dim + dim]);
            try odd.copyRowFrom(b, x2.data[b * dim .. b * dim + dim]);
        }
        return state;
    }

    pub fn deinit(self: *RSFLatentState) void {
        self.data.deinit();
        self.log_det = 0.0;
    }

    pub fn clone(self: *const RSFLatentState, allocator: Allocator) !RSFLatentState {
        const shape = try latentShapeOf(self);
        var data = try Tensor.init(allocator, &[_]usize{ shape.batch, shape.dim, 2 });
        errdefer data.deinit();
        try ensureFiniteSlice(self.data.data);
        @memcpy(data.data, self.data.data);
        return .{ .data = data, .log_det = self.log_det, .binding = self.binding };
    }

    pub fn evenView(self: *RSFLatentState) !LatentHalfView {
        const shape = try latentShapeOf(self);
        return .{ .data = self.data.data, .batch = shape.batch, .dim = shape.dim, .offset = 0 };
    }

    pub fn oddView(self: *RSFLatentState) !LatentHalfView {
        const shape = try latentShapeOf(self);
        return .{ .data = self.data.data, .batch = shape.batch, .dim = shape.dim, .offset = 1 };
    }

    pub fn requireModel(self: *const RSFLatentState, model: *const RSF) !void {
        const model_id = try handleId(model.id);
        const dim = try model.dim();
        try self.binding.requireSpace(.latent_state);
        try self.binding.requireModel(model_id);
        try self.binding.requireDim(dim);
        const shape = try latentShapeOf(self);
        if (shape.dim != dim) return types.RSFBindingError.RSFDimMismatch;
    }

    pub fn blockedLength(self: *const RSFLatentState) !usize {
        const shape = try latentShapeOf(self);
        return try checkedMul(shape.batch, try checkedMul(shape.dim, 2));
    }

    pub fn forwardThrough(self: *RSFLatentState, model: *RSF) !void {
        self.log_det += try model.forwardLatentWithLogDet(self);
    }

    pub fn inverseThrough(self: *RSFLatentState, model: *RSF) !void {
        self.log_det -= try model.inverseLatentWithLogDet(self);
    }

    pub fn roundtripError(self: *const RSFLatentState, model: *RSF, allocator: Allocator) !f32 {
        var working = try self.clone(allocator);
        defer working.deinit();
        try working.forwardThrough(model);
        try working.inverseThrough(model);
        const total = try self.blockedLength();
        var num: f64 = 0.0;
        var den: f64 = 0.0;
        var k: usize = 0;
        while (k < total) : (k += 1) {
            const original: f64 = @floatCast(self.data.data[k]);
            const recovered: f64 = @floatCast(working.data.data[k]);
            const diff = recovered - original;
            num += diff * diff;
            den += original * original;
        }
        if (den <= 0.0) return @as(f32, @floatCast(@sqrt(num)));
        return @as(f32, @floatCast(@sqrt(num / den)));
    }
};

pub const MidpointResult = struct {
    z: RSFLatentState,
    w: RSFLatentState,
    collision_loss: f32,
    logdet_forward: f32,
    logdet_backward: f32,
    logdet_total: f32,
    forward_layers: usize,
    backward_layers: usize,

    pub fn deinit(self: *MidpointResult) void {
        self.z.deinit();
        self.w.deinit();
        self.collision_loss = 0.0;
        self.logdet_forward = 0.0;
        self.logdet_backward = 0.0;
        self.logdet_total = 0.0;
        self.forward_layers = 0;
        self.backward_layers = 0;
    }
};

pub const RSF = struct {
    id: u64 = 0,
    ctrl: ?*RSFCore = null,
    pub fn init(allocator: Allocator, model_dim: usize, num_layers: usize) !RSF {
        return initWithConfig(allocator, model_dim, num_layers, .{});
    }
    pub fn initWithConfig(allocator: Allocator, model_dim: usize, num_layers: usize, cfg: RSFConfig) !RSF {
        try validateModelConfigValues(model_dim, num_layers, cfg);
        _ = try checkedMul(model_dim, coupling_width);
        _ = try checkedMul(model_dim, 2);
        const core = try allocator.create(RSFCore);
        errdefer allocator.destroy(core);
        core.* = .{
            .allocator = allocator,
            .dim = model_dim,
            .num_layers = num_layers,
            .layers = try allocator.alloc(LayerCore, num_layers),
            .cfg = cfg,
            .rwlock = .{},
            .gpu_accel = null,
            .gpu_available = std.atomic.Value(u8).init(0),
            .gpu_weight_version = 0,
            .cpu_weight_version = 1,
            .f16_buf = null,
            .oftb = try initOFTBForConfig(model_dim, cfg.global_diffusion),
            .layer_applications = std.atomic.Value(usize).init(0),
            .frontier_depth = std.atomic.Value(usize).init(0),
        };
        errdefer {
            if (core.gpu_accel) |*ga| {
                ga.deinit();
                core.gpu_accel = null;
            }
            if (core.f16_buf) |buf| {
                allocator.free(buf);
                core.f16_buf = null;
            }
            core.gpu_available.store(0, .monotonic);
            core.oftb.deinit();
        }
        errdefer allocator.free(core.layers);
        var initialized: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < initialized) : (j += 1) core.layers[j].deinitOwned();
        }
        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            const seed_base = try checkedMulU64(@as(u64, @intCast(l)), 10007);
            const layer_cfg = RSFLayerConfig{
                .clip_min = cfg.clip_min,
                .clip_max = cfg.clip_max,
                .seed_offset = seed_base,
                .grad_mean = cfg.grad_mean,
            };
            core.layers[l] = try LayerCore.initOwned(allocator, model_dim, layer_cfg);
            initialized += 1;
        }
        try validateModelMetadata(core);
        if (modelGPUCompatible(core)) {
            syncAllLayersGPU(core) catch disableGPU(core);
        }
        const id = try registerModelCore(core);
        assignLayerBindings(core, id);
        return RSF{ .id = id, .ctrl = core };
    }
    pub fn deinit(self: *RSF) void {
        const id = self.id;
        if (id == 0) return;
        self.id = 0;
        self.ctrl = null;
        requestDestroyModelCore(id);
    }
    pub fn isGPUAvailable(self: *const RSF) bool {
        const id = handleId(self.id) catch return false;
        const core = acquireModelCore(id) catch return false;
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        return modelGPUCompatible(core) and core.gpu_available.load(.monotonic) != 0 and core.gpu_weight_version == core.cpu_weight_version and core.gpu_accel != null;
    }
    pub fn syncWeightsToGPU(self: *RSF) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try syncAllLayersGPU(core);
    }
    pub fn ensureGradients(self: *RSF) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const layer_count = try checkedModelLayerCount(core);
        var i: usize = 0;
        while (i < layer_count) : (i += 1) try core.layers[i].ensureGradients();
    }
    pub fn zeroGradients(self: *RSF) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        for (core.layers) |*layer| layer.zeroGradients();
    }
    pub fn gradientL2Norm(self: *const RSF) !f32 {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const layer_count = try checkedModelLayerCount(core);
        var total: f64 = 0.0;
        var i: usize = 0;
        while (i < layer_count) : (i += 1) {
            total += try core.layers[i].gradientSquaredSum();
        }
        const result: f32 = @floatCast(@sqrt(total));
        if (!std.math.isFinite(result)) return error.NonFinite;
        return result;
    }
    pub fn applyGradientStep(self: *RSF, learning_rate: f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const layer_count = try checkedModelLayerCount(core);
        bumpWeightVersion(core);
        core.gpu_available.store(0, .monotonic);
        defer refreshGPUAfterWeightChange(core);
        var i: usize = 0;
        while (i < layer_count) : (i += 1) {
            try core.layers[i].applyGradientStep(learning_rate);
        }
        try validateModelMetadata(core);
    }
    pub fn forwardCPU(self: *RSF, x: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try forwardOnCore(core, x);
    }
    pub fn forward(self: *RSF, x: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        try validateTensor2D(x);
        const dim2 = try checkedMul(core.dim, 2);
        if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
        if (x.shape.dims[0] == 0) return error.InvalidBatchSize;
        if (gpuPathSelected(core)) {
            core.rwlock.lock();
            defer core.rwlock.unlock();
            try forwardDispatchOnCore(core, x);
        } else {
            core.rwlock.lockShared();
            defer core.rwlock.unlockShared();
            try forwardOnCore(core, x);
        }
    }
    pub fn inverse(self: *RSF, y: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try inverseOnCore(core, y);
    }
    pub fn backward(self: *RSF, grad_output: *const Tensor, input: *const Tensor, output: *const Tensor, grad_input_out: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try backwardOnCore(core, grad_output, input, output, grad_input_out, 0.0);
    }
    pub fn forwardWithLogDet(self: *RSF, x: *Tensor, logdet_per_row: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try forwardLogDetOnCore(core, x, logdet_per_row);
    }
    pub fn meanLogDetJacobian(self: *RSF, x: *const Tensor) !f32 {
        try validateTensor2D(x);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const batch_size = x.shape.dims[0];
        if (batch_size == 0) return error.InvalidBatchSize;
        const allocator = scratchAllocator();
        var probe = try tensorClone(allocator, x);
        defer probe.deinit();
        const logdet_per_row = try allocator.alloc(f32, batch_size);
        defer allocator.free(logdet_per_row);
        try forwardLogDetOnCore(core, &probe, logdet_per_row);
        var total: f32 = 0.0;
        var b: usize = 0;
        while (b < batch_size) : (b += 1) total += logdet_per_row[b];
        return total / @as(f32, @floatFromInt(batch_size));
    }
    pub fn backwardWithLogDet(self: *RSF, grad_output: *const Tensor, input: *const Tensor, output: *const Tensor, grad_input_out: *Tensor, logdet_weight: f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try backwardOnCore(core, grad_output, input, output, grad_input_out, logdet_weight);
    }
    pub fn notifyWeightsChanged(self: *RSF) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        bumpWeightVersion(core);
        core.gpu_available.store(0, .monotonic);
        try validateModelMetadata(core);
        refreshGPUAfterWeightChange(core);
    }
    pub fn verifyInvertible(self: *RSF, x: *const Tensor, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const allocator = scratchAllocator();
        var y = try tensorClone(allocator, x);
        defer y.deinit();
        if (gpuPathSelected(core)) {
            core.rwlock.lock();
            defer core.rwlock.unlock();
            try forwardDispatchOnCore(core, &y);
            try inverseOnCore(core, &y);
        } else {
            core.rwlock.lockShared();
            defer core.rwlock.unlockShared();
            try forwardOnCore(core, &y);
            try inverseOnCore(core, &y);
        }
        return tensorAllCloseEq(x, &y, abs_tol, rel_tol);
    }
    pub fn dim(self: *const RSF) !usize {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return core.dim;
    }

    pub fn layerCount(self: *const RSF) !usize {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return try checkedModelLayerCount(core);
    }

    pub fn globalDiffusionEnabled(self: *const RSF) !bool {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return core.cfg.global_diffusion and core.oftb.diffusionEnabled();
    }

    pub fn diffusionLayout(self: *const RSF) !types.RSFDiffusionLayout {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return core.oftb.diffusionLayout() orelse error.DiffusionDisabled;
    }

    pub fn latentBinding(self: *const RSF) !types.RSFBinding {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return types.RSFBinding.model(.latent_state, id, core.dim);
    }

    pub fn layerBindingsFor(self: *RSF, layer: usize) !LayerBindings {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const target = try checkedLayerIndex(core, layer);
        if (target.model_id != id or target.layer_index != layer) {
            core.rwlock.lock();
            defer core.rwlock.unlock();
            target.model_id = id;
            target.layer_index = layer;
        }
        return layerBindings(id, layer, core.dim);
    }

    pub fn readLayerWeights(self: *const RSF, layer: usize, s_out: []f32, t_out: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, coupling_width);
        if (s_out.len < expected or t_out.len < expected) return error.DataLengthMismatch;
        target.rwlock.lockShared();
        defer target.rwlock.unlockShared();
        try validateTensor2DShape(&target.s_weight, core.dim, coupling_width);
        try validateTensor2DShape(&target.t_weight, core.dim, coupling_width);
        try ensureFiniteSlice(target.s_weight.data[0..expected]);
        try ensureFiniteSlice(target.t_weight.data[0..expected]);
        @memcpy(s_out[0..expected], target.s_weight.data[0..expected]);
        @memcpy(t_out[0..expected], target.t_weight.data[0..expected]);
    }

    pub fn writeLayerWeights(self: *RSF, layer: usize, s_in: []const f32, t_in: []const f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, coupling_width);
        if (s_in.len < expected or t_in.len < expected) return error.DataLengthMismatch;
        try ensureFiniteSlice(s_in[0..expected]);
        try ensureFiniteSlice(t_in[0..expected]);
        target.rwlock.lock();
        defer target.rwlock.unlock();
        try validateTensor2DShape(&target.s_weight, core.dim, coupling_width);
        try validateTensor2DShape(&target.t_weight, core.dim, coupling_width);
        @memcpy(target.s_weight.data[0..expected], s_in[0..expected]);
        @memcpy(target.t_weight.data[0..expected], t_in[0..expected]);
        bumpWeightVersion(core);
        core.gpu_available.store(0, .monotonic);
        try validateModelMetadata(core);
        refreshGPUAfterWeightChange(core);
    }

    pub fn readLayerGradients(self: *const RSF, layer: usize, s_out: []f32, t_out: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, coupling_width);
        if (s_out.len < expected or t_out.len < expected) return error.DataLengthMismatch;
        const s_grad = target.s_weight_grad orelse return error.NoGradients;
        const t_grad = target.t_weight_grad orelse return error.NoGradients;
        target.rwlock.lockShared();
        defer target.rwlock.unlockShared();
        try validateTensor2DShape(&s_grad, core.dim, coupling_width);
        try validateTensor2DShape(&t_grad, core.dim, coupling_width);
        try ensureFiniteSlice(s_grad.data[0..expected]);
        try ensureFiniteSlice(t_grad.data[0..expected]);
        @memcpy(s_out[0..expected], s_grad.data[0..expected]);
        @memcpy(t_out[0..expected], t_grad.data[0..expected]);
    }

    pub fn readLayerGradientProducts(self: *const RSF, layer: usize, branch: RSFBranch, out: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const target = try checkedLayerIndex(core, layer);
        const model_dim = core.dim;
        const expected = try checkedMul(model_dim, 3);
        if (out.len < expected) return error.InvalidDataLength;
        const grad = switch (branch) {
            .s => target.s_weight_grad orelse return error.NoGradients,
            .t => target.t_weight_grad orelse return error.NoGradients,
        };
        target.rwlock.lockShared();
        defer target.rwlock.unlockShared();
        try validateTensor2DShape(&grad, model_dim, coupling_width);
        try ensureFiniteSlice(grad.data);
        var d: usize = 0;
        while (d < model_dim) : (d += 1) {
            const g_w = grad.data[d * coupling_width + WEIGHT_COLUMN];
            const g_b = grad.data[d * coupling_width + BIAS_COLUMN];
            out[d * 3 + 0] = g_w * g_w;
            out[d * 3 + 1] = g_w * g_b;
            out[d * 3 + 2] = g_b * g_b;
        }
    }

    pub fn readLayerGaussNewton(self: *const RSF, layer: usize, s_block_out: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, 3);
        if (s_block_out.len != expected) return error.InvalidDataLength;
        target.rwlock.lockShared();
        defer target.rwlock.unlockShared();
        if (target.gn_block) |blk| {
            if (blk.len < expected) return error.InvalidModelState;
            try ensureFiniteSlice(blk[0..expected]);
            @memcpy(s_block_out[0..expected], blk[0..expected]);
            return;
        }
        @memset(s_block_out[0..expected], 0.0);
    }

    pub fn zeroLayerGaussNewton(self: *RSF, layer: usize) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const target = try checkedLayerIndex(core, layer);
        target.rwlock.lock();
        defer target.rwlock.unlock();
        target.zeroGaussNewton();
    }

    pub fn scaleLayerGradients(self: *RSF, scale: f32) !void {
        if (!std.math.isFinite(scale)) return error.NonFinite;
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const layer_count = try checkedModelLayerCount(core);
        var i: usize = 0;
        while (i < layer_count) : (i += 1) {
            const target = &core.layers[i];
            try target.ensureGradients();
            const s_grad = target.s_weight_grad orelse return error.NoGradients;
            const t_grad = target.t_weight_grad orelse return error.NoGradients;
            try ensureFiniteSlice(s_grad.data);
            try ensureFiniteSlice(t_grad.data);
            for (s_grad.data) |v| {
                if (!std.math.isFinite(v * scale)) return error.NonFinite;
            }
            for (t_grad.data) |v| {
                if (!std.math.isFinite(v * scale)) return error.NonFinite;
            }
            target.rwlock.lock();
            defer target.rwlock.unlock();
            for (s_grad.data) |*v| v.* *= scale;
            for (t_grad.data) |*v| v.* *= scale;
        }
    }

    pub fn accumulateLayerGradients(self: *RSF, layer: usize, s_in: []const f32, t_in: []const f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, coupling_width);
        if (s_in.len < expected or t_in.len < expected) return error.DataLengthMismatch;
        try ensureFiniteSlice(s_in[0..expected]);
        try ensureFiniteSlice(t_in[0..expected]);
        try target.ensureGradients();
        const s_grad = target.s_weight_grad orelse return error.NoGradients;
        const t_grad = target.t_weight_grad orelse return error.NoGradients;
        try validateTensor2DShape(&s_grad, core.dim, coupling_width);
        try validateTensor2DShape(&t_grad, core.dim, coupling_width);
        try ensureFiniteSlice(s_grad.data);
        try ensureFiniteSlice(t_grad.data);
        var k: usize = 0;
        while (k < expected) : (k += 1) {
            if (!std.math.isFinite(s_grad.data[k] + s_in[k])) return error.NonFinite;
            if (!std.math.isFinite(t_grad.data[k] + t_in[k])) return error.NonFinite;
        }
        target.rwlock.lock();
        defer target.rwlock.unlock();
        k = 0;
        while (k < expected) : (k += 1) {
            s_grad.data[k] += s_in[k];
            t_grad.data[k] += t_in[k];
        }
    }

    pub fn forwardSequence(self: *RSF, x: *Tensor, mask: *const types.RSFSequenceMask) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try forwardSequenceOnCore(core, x, mask, null);
    }

    pub fn forwardSequenceWithLogDet(self: *RSF, x: *Tensor, mask: *const types.RSFSequenceMask, logdet_per_sequence: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try forwardSequenceOnCore(core, x, mask, logdet_per_sequence);
    }

    pub fn inverseSequence(self: *RSF, y: *Tensor, mask: *const types.RSFSequenceMask) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try inverseSequenceOnCore(core, y, mask, null);
    }

    pub fn inverseSequenceWithLogDet(self: *RSF, y: *Tensor, mask: *const types.RSFSequenceMask, logdet_per_sequence: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try inverseSequenceOnCore(core, y, mask, logdet_per_sequence);
    }

    pub fn meanLogDetJacobianSequence(self: *RSF, x: *const Tensor, mask: *const types.RSFSequenceMask) !f32 {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const shape = try sequenceShapeOf(x, mask, core);
        const allocator = scratchAllocator();
        var working = try tensorClone(allocator, x);
        defer working.deinit();
        const per_sequence = try allocator.alloc(f32, shape.num_sequences);
        defer allocator.free(per_sequence);
        try forwardSequenceOnCore(core, &working, mask, per_sequence);
        return try meanSequenceLogDet(per_sequence, shape.num_sequences);
    }

    pub fn verifySequenceInvertible(self: *RSF, x: *const Tensor, mask: *const types.RSFSequenceMask, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const allocator = scratchAllocator();
        var y = try tensorClone(allocator, x);
        defer y.deinit();
        try forwardSequenceOnCore(core, &y, mask, null);
        try inverseSequenceOnCore(core, &y, mask, null);
        return tensorAllCloseEq(x, &y, abs_tol, rel_tol);
    }

    pub fn backwardSequence(self: *RSF, grad_output: *const Tensor, input: *const Tensor, output: *const Tensor, mask: *const types.RSFSequenceMask, grad_input_out: *Tensor, logdet_weight: f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try backwardSequenceOnCore(core, grad_output, input, output, mask, grad_input_out, logdet_weight);
    }

    pub fn inverseWithLogDet(self: *RSF, y: *Tensor, logdet_per_row: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try inverseLogDetOnCore(core, y, logdet_per_row);
    }

    pub fn forwardLatentWithLogDet(self: *RSF, state: *RSFLatentState) !f32 {
        try state.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const shape = try latentShapeOf(state);
        const allocator = scratchAllocator();
        var blocked = try allocBlockedLatentTensor(allocator, state);
        defer blocked.deinit();
        try latentToBlockedTensor(state, &blocked);
        const per_row = try allocator.alloc(f32, shape.batch);
        defer allocator.free(per_row);
        core.rwlock.lockShared();
        try forwardLogDetOnCore(core, &blocked, per_row);
        core.rwlock.unlockShared();
        try blockedTensorToLatent(&blocked, state);
        return try meanLogDet(per_row, shape.batch);
    }

    pub fn inverseLatentWithLogDet(self: *RSF, state: *RSFLatentState) !f32 {
        try state.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const shape = try latentShapeOf(state);
        const allocator = scratchAllocator();
        var blocked = try allocBlockedLatentTensor(allocator, state);
        defer blocked.deinit();
        try latentToBlockedTensor(state, &blocked);
        const per_row = try allocator.alloc(f32, shape.batch);
        defer allocator.free(per_row);
        core.rwlock.lockShared();
        try inverseLogDetOnCore(core, &blocked, per_row);
        core.rwlock.unlockShared();
        try blockedTensorToLatent(&blocked, state);
        return try meanLogDet(per_row, shape.batch);
    }

    pub fn forwardLatent(self: *RSF, state: *RSFLatentState) !void {
        _ = try self.forwardLatentWithLogDet(state);
    }

    pub fn inverseLatent(self: *RSF, state: *RSFLatentState) !void {
        _ = try self.inverseLatentWithLogDet(state);
    }

    pub fn backwardLatent(self: *RSF, grad_output: *const RSFLatentState, input: *const RSFLatentState, output: *const RSFLatentState, grad_input_out: *RSFLatentState, logdet_weight: f32) !void {
        try grad_output.requireModel(self);
        try input.requireModel(self);
        try output.requireModel(self);
        try grad_input_out.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const allocator = scratchAllocator();
        var g_out = try allocBlockedLatentTensor(allocator, grad_output);
        defer g_out.deinit();
        var x_in = try allocBlockedLatentTensor(allocator, input);
        defer x_in.deinit();
        var y_out = try allocBlockedLatentTensor(allocator, output);
        defer y_out.deinit();
        var dx = try allocBlockedLatentTensor(allocator, grad_input_out);
        defer dx.deinit();
        try latentToBlockedTensor(grad_output, &g_out);
        try latentToBlockedTensor(input, &x_in);
        try latentToBlockedTensor(output, &y_out);
        try backwardOnCore(core, &g_out, &x_in, &y_out, &dx, logdet_weight);
        try blockedTensorToLatent(&dx, grad_input_out);
    }

    pub fn midpointSplit(self: *const RSF) !struct { forward_layers: usize, backward_layers: usize } {
        const layers = try self.layerCount();
        const split = midpointSplitLayers(layers);
        return .{ .forward_layers = split.forward_layers, .backward_layers = split.backward_layers };
    }

    pub fn resetLayerApplicationCounter(self: *RSF) void {
        const id = handleId(self.id) catch return;
        const core = acquireModelCore(id) catch return;
        defer releaseModelCore(id);
        resetLayerApplicationCounters(core);
    }

    pub fn layerApplicationCount(self: *const RSF) usize {
        const id = handleId(self.id) catch return 0;
        const core = acquireModelCore(id) catch return 0;
        defer releaseModelCore(id);
        return core.layer_applications.load(.monotonic);
    }

    pub fn lastFrontierDepth(self: *const RSF) usize {
        const id = handleId(self.id) catch return 0;
        const core = acquireModelCore(id) catch return 0;
        defer releaseModelCore(id);
        return core.frontier_depth.load(.monotonic);
    }

    pub fn midpointForwardFrontier(self: *RSF, input: *const RSFLatentState, out_z: *RSFLatentState) !void {
        try input.requireModel(self);
        try out_z.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const split = midpointSplitLayers(try checkedModelLayerCount(core));
        const allocator = scratchAllocator();
        var blocked = try allocBlockedLatentTensor(allocator, input);
        defer blocked.deinit();
        try latentToBlockedTensor(input, &blocked);
        const logdet = try blockedMeanLogDetAndApplyForward(core, &blocked, 0, split.forward_layers);
        noteFrontierDepth(core, split.forward_layers);
        try blockedTensorToLatent(&blocked, out_z);
        out_z.log_det = logdet;
    }

    pub fn midpointBackwardFrontier(self: *RSF, target: *const RSFLatentState, out_w: *RSFLatentState) !void {
        try target.requireModel(self);
        try out_w.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const layer_count = try checkedModelLayerCount(core);
        const split = midpointSplitLayers(layer_count);
        const allocator = scratchAllocator();
        var blocked = try allocBlockedLatentTensor(allocator, target);
        defer blocked.deinit();
        try latentToBlockedTensor(target, &blocked);
        const logdet = try blockedMeanLogDetAndApplyInverse(core, &blocked, split.forward_layers, layer_count);
        noteFrontierDepth(core, split.backward_layers);
        try blockedTensorToLatent(&blocked, out_w);
        out_w.log_det = logdet;
    }

    pub fn midpointCollision(self: *RSF, input: *const RSFLatentState, target: *const RSFLatentState, allocator: Allocator) !MidpointResult {
        try input.requireModel(self);
        try target.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        return midpointCollisionOnCore(core, input, target, allocator, false);
    }

    pub fn midpointCollisionParallel(self: *RSF, input: *const RSFLatentState, target: *const RSFLatentState, allocator: Allocator) !MidpointResult {
        try input.requireModel(self);
        try target.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        return midpointCollisionOnCore(core, input, target, allocator, true);
    }

    pub fn midpointBackward(self: *RSF, result: *const MidpointResult, grad_scale: f32, logdet_weight: f32, grad_input_out: *RSFLatentState) !void {
        try result.z.requireModel(self);
        try result.w.requireModel(self);
        try grad_input_out.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const layer_count = try checkedModelLayerCount(core);
        const split = midpointSplitLayers(layer_count);
        const allocator = scratchAllocator();
        var z_blocked = try allocBlockedLatentTensor(allocator, &result.z);
        defer z_blocked.deinit();
        var w_blocked = try allocBlockedLatentTensor(allocator, &result.w);
        defer w_blocked.deinit();
        var gin = try allocBlockedLatentTensor(allocator, grad_input_out);
        defer gin.deinit();
        try latentToBlockedTensor(&result.z, &z_blocked);
        try latentToBlockedTensor(&result.w, &w_blocked);
        try midpointBackwardOnCore(core, &z_blocked, &w_blocked, &gin, 0, split.forward_layers, layer_count, grad_scale, logdet_weight, false);
        try blockedTensorToLatent(&gin, grad_input_out);
    }

    pub fn midpointBackwardParallel(self: *RSF, result: *const MidpointResult, grad_scale: f32, logdet_weight: f32, grad_input_out: *RSFLatentState) !void {
        try result.z.requireModel(self);
        try result.w.requireModel(self);
        try grad_input_out.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const layer_count = try checkedModelLayerCount(core);
        const split = midpointSplitLayers(layer_count);
        const allocator = scratchAllocator();
        var z_blocked = try allocBlockedLatentTensor(allocator, &result.z);
        defer z_blocked.deinit();
        var w_blocked = try allocBlockedLatentTensor(allocator, &result.w);
        defer w_blocked.deinit();
        var gin = try allocBlockedLatentTensor(allocator, grad_input_out);
        defer gin.deinit();
        try latentToBlockedTensor(&result.z, &z_blocked);
        try latentToBlockedTensor(&result.w, &w_blocked);
        try midpointBackwardOnCore(core, &z_blocked, &w_blocked, &gin, 0, split.forward_layers, layer_count, grad_scale, logdet_weight, true);
        try blockedTensorToLatent(&gin, grad_input_out);
    }

    pub fn resonanceDrift(self: *RSF, state: *const RSFLatentState, allocator: Allocator) !f32 {
        try state.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const layer_count = try checkedModelLayerCount(core);
        const order = OFTB.resonance_order;
        var working = try state.clone(allocator);
        defer working.deinit();
        var blocked = try allocBlockedLatentTensor(allocator, &working);
        defer blocked.deinit();
        try latentToBlockedTensor(&working, &blocked);
        const e0: f64 = blockedEnergy(&blocked);
        var worst: f64 = 0.0;
        const denom = if (e0 > 1.0e-30) e0 else 1.0e-30;
        var l: usize = 0;
        while (l < layer_count) {
            _ = try blockedMeanLogDetAndApplyForward(core, &blocked, l, l + 1);
            l += 1;
            if (l % order == 0 or l == layer_count) {
                const e = blockedEnergy(&blocked);
                const drift = @abs(e - e0) / denom;
                if (drift > worst) worst = drift;
            }
        }
        const out: f32 = @floatCast(worst);
        if (!std.math.isFinite(out)) return error.NonFinite;
        return out;
    }

    pub fn stateGrowthReport(self: *RSF, state: *const RSFLatentState, allocator: Allocator) ![]f32 {
        try state.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const layer_count = try checkedModelLayerCount(core);
        const order = OFTB.resonance_order;
        const samples = 1 + (layer_count + order - 1) / order;
        const report = try allocator.alloc(f32, samples);
        errdefer allocator.free(report);
        var working = try state.clone(allocator);
        defer working.deinit();
        var blocked = try allocBlockedLatentTensor(allocator, &working);
        defer blocked.deinit();
        try latentToBlockedTensor(&working, &blocked);
        report[0] = @floatCast(blockedMaxAbs(&blocked));
        var l: usize = 0;
        var seen: usize = 1;
        while (l < layer_count) {
            _ = try blockedMeanLogDetAndApplyForward(core, &blocked, l, l + 1);
            l += 1;
            if (l % order == 0 or l == layer_count) {
                if (seen >= samples) return error.InvalidModelState;
                const m = blockedMaxAbs(&blocked);
                const v: f32 = @floatCast(m);
                if (!std.math.isFinite(v)) return error.NonFinite;
                report[seen] = v;
                seen += 1;
            }
        }
        if (seen != samples) return error.InvalidModelState;
        if (!std.math.isFinite(report[0])) return error.NonFinite;
        return report;
    }

    pub fn save(self: *const RSF, path: []const u8) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const allocator = scratchAllocator();
        core.rwlock.lockShared();
        var snapshot = snapshotModelForSave(allocator, core) catch |err| {
            core.rwlock.unlockShared();
            return err;
        };
        core.rwlock.unlockShared();
        defer snapshot.deinit();
        try writeSnapshotToPath(&snapshot, path, allocator);
    }
    pub fn load(allocator: Allocator, path: []const u8) !RSF {
        return loadWithConfig(allocator, path, null);
    }
    pub fn loadWithConfig(allocator: Allocator, path: []const u8, policy: ?RSFConfig) !RSF {
        if (policy) |p| {
            try validateClipRange(p.clip_min, p.clip_max);
            if (p.max_dim == 0 or p.max_layers == 0) return error.InvalidConfig;
        }
        const file = try core_io.openFilePath(path, .{});
        defer file.close();
        var buffered = std.io.bufferedReader(file.reader());
        const r = buffered.reader();
        var magic: [4]u8 = undefined;
        try r.readNoEof(&magic);
        if (!std.mem.eql(u8, &magic, "RSF0")) return error.BadFileFormat;
        const version = try r.readInt(u32, .little);
        if (version != SAVE_VERSION and version != SAVE_VERSION_LEGACY) return error.UnsupportedVersion;
        const num_layers_u64 = try r.readInt(u64, .little);
        const dim_u64 = try r.readInt(u64, .little);
        if (num_layers_u64 == 0) return error.InvalidLayerCount;
        if (dim_u64 == 0) return error.InvalidDimension;
        const policy_max_dim: usize = if (policy) |p| p.max_dim else (1 << 20);
        const policy_max_layers: usize = if (policy) |p| p.max_layers else (1 << 20);
        if (num_layers_u64 > @as(u64, @intCast(policy_max_layers)) or dim_u64 > @as(u64, @intCast(policy_max_dim))) return error.TooLarge;
        const num_layers = try checkedCastU64ToUsize(num_layers_u64);
        const model_dim = try checkedCastU64ToUsize(dim_u64);
        _ = try checkedMul(model_dim, coupling_width);
        _ = try checkedMul(model_dim, 2);
        var hasher = std.hash.Crc32.init();
        hasher.update("RSF0");
        crcUpdateU32LE(&hasher, version);
        crcUpdateU64LE(&hasher, num_layers_u64);
        crcUpdateU64LE(&hasher, dim_u64);
        const clip_min_bits = try r.readInt(u32, .little);
        const clip_max_bits = try r.readInt(u32, .little);
        const clip_min: f32 = @bitCast(clip_min_bits);
        const clip_max: f32 = @bitCast(clip_max_bits);
        const grad_mean = try readEncodedBool(r);
        try validateClipRange(clip_min, clip_max);
        if (policy) |p| {
            if (clip_min != p.clip_min or clip_max != p.clip_max or grad_mean != p.grad_mean) return error.PolicyMismatch;
        }
        crcUpdateU32LE(&hasher, clip_min_bits);
        crcUpdateU32LE(&hasher, clip_max_bits);
        crcUpdateU8(&hasher, if (grad_mean) @as(u8, 1) else @as(u8, 0));
        const saved_max_dim_u64 = try r.readInt(u64, .little);
        const saved_max_layers_u64 = try r.readInt(u64, .little);
        crcUpdateU64LE(&hasher, saved_max_dim_u64);
        crcUpdateU64LE(&hasher, saved_max_layers_u64);
        if (saved_max_dim_u64 == 0 or saved_max_layers_u64 == 0) return error.InvalidConfig;
        if (saved_max_dim_u64 < dim_u64 or saved_max_layers_u64 < num_layers_u64) return error.InvalidConfig;
        const effective_max_dim: usize = if (policy) |p| p.max_dim else try checkedCastU64ToUsize(saved_max_dim_u64);
        const effective_max_layers: usize = if (policy) |p| p.max_layers else try checkedCastU64ToUsize(saved_max_layers_u64);
        var global_diffusion = false;
        if (version == SAVE_VERSION) {
            global_diffusion = try readEncodedBool(r);
            crcUpdateU8(&hasher, if (global_diffusion) @as(u8, 1) else @as(u8, 0));
        }
        if (policy) |p| {
            if (p.global_diffusion != global_diffusion) return error.PolicyMismatch;
        }
        const loaded_cfg = RSFConfig{
            .clip_min = clip_min,
            .clip_max = clip_max,
            .grad_mean = grad_mean,
            .max_dim = effective_max_dim,
            .max_layers = effective_max_layers,
            .global_diffusion = global_diffusion,
        };
        try validateModelConfigValues(model_dim, num_layers, loaded_cfg);
        const core = try allocator.create(RSFCore);
        errdefer allocator.destroy(core);
        core.* = .{
            .allocator = allocator,
            .dim = model_dim,
            .num_layers = num_layers,
            .layers = try allocator.alloc(LayerCore, num_layers),
            .cfg = loaded_cfg,
            .rwlock = .{},
            .gpu_accel = null,
            .gpu_available = std.atomic.Value(u8).init(0),
            .gpu_weight_version = 0,
            .cpu_weight_version = 1,
            .f16_buf = null,
            .oftb = try initOFTBForConfig(model_dim, loaded_cfg.global_diffusion),
            .layer_applications = std.atomic.Value(usize).init(0),
            .frontier_depth = std.atomic.Value(usize).init(0),
        };
        errdefer {
            if (core.gpu_accel) |*ga| {
                ga.deinit();
                core.gpu_accel = null;
            }
            if (core.f16_buf) |buf| {
                allocator.free(buf);
                core.f16_buf = null;
            }
            core.gpu_available.store(0, .monotonic);
            core.oftb.deinit();
        }
        errdefer allocator.free(core.layers);
        var initialized: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < initialized) : (j += 1) core.layers[j].deinitOwned();
        }
        var i: usize = 0;
        while (i < num_layers) : (i += 1) {
            const layer_clip_min_bits = try r.readInt(u32, .little);
            const layer_clip_max_bits = try r.readInt(u32, .little);
            const layer_clip_min: f32 = @bitCast(layer_clip_min_bits);
            const layer_clip_max: f32 = @bitCast(layer_clip_max_bits);
            const layer_grad_mean = try readEncodedBool(r);
            try validateClipRange(layer_clip_min, layer_clip_max);
            if (layer_clip_min != clip_min or layer_clip_max != clip_max or layer_grad_mean != grad_mean) return error.InvalidConfig;
            crcUpdateU32LE(&hasher, layer_clip_min_bits);
            crcUpdateU32LE(&hasher, layer_clip_max_bits);
            crcUpdateU8(&hasher, if (layer_grad_mean) @as(u8, 1) else @as(u8, 0));
            var s_w_new = try readTensorData(allocator, r, model_dim, coupling_width);
            errdefer s_w_new.deinit();
            var t_w_new = try readTensorData(allocator, r, model_dim, coupling_width);
            errdefer t_w_new.deinit();
            try validateTensor2DShape(&s_w_new, model_dim, coupling_width);
            try validateTensor2DShape(&t_w_new, model_dim, coupling_width);
            try ensureFiniteSlice(s_w_new.data);
            try ensureFiniteSlice(t_w_new.data);
            hashTensorData(&hasher, &s_w_new);
            hashTensorData(&hasher, &t_w_new);
            core.layers[i] = .{
                .s_weight = s_w_new,
                .t_weight = t_w_new,
                .s_weight_grad = null,
                .t_weight_grad = null,
                .dim = model_dim,
                .allocator = allocator,
                .clip_min = layer_clip_min,
                .clip_max = layer_clip_max,
                .grad_mean = layer_grad_mean,
                .gn_block = null,
                .model_id = 0,
                .layer_index = i,
                .rwlock = .{},
            };
            initialized += 1;
        }
        if (version == SAVE_VERSION) {
            const gn_len = try checkedMul(model_dim, 3);
            var g: usize = 0;
            while (g < num_layers) : (g += 1) {
                const blk = try allocator.alloc(f32, gn_len);
                errdefer if (core.layers[g].gn_block == null) allocator.free(blk);
                var k: usize = 0;
                while (k < gn_len) : (k += 1) {
                    const bits = try r.readInt(u32, .little);
                    crcUpdateU32LE(&hasher, bits);
                    const value: f32 = @bitCast(bits);
                    if (!std.math.isFinite(value) or value < 0.0) return error.NonFinite;
                    blk[k] = value;
                }
                core.layers[g].gn_block = blk;
            }
        }
        const stored_crc = try r.readInt(u32, .little);
        if (stored_crc != hasher.final()) return error.ChecksumMismatch;
        var eof_buf: [1]u8 = undefined;
        if ((try r.read(&eof_buf)) != 0) return error.TrailingData;
        try validateModelMetadata(core);
        if (modelGPUCompatible(core)) {
            syncAllLayersGPU(core) catch disableGPU(core);
        }
        const id = try registerModelCore(core);
        assignLayerBindings(core, id);
        return RSF{ .id = id, .ctrl = core };
    }
    pub fn saveLoadRoundtrip(allocator: Allocator, self: *const RSF, path: []const u8, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        try self.save(path);
        var loaded = try RSF.load(allocator, path);
        defer loaded.deinit();
        const id1 = try handleId(self.id);
        const core1 = try acquireModelCore(id1);
        defer releaseModelCore(id1);
        const id2 = try handleId(loaded.id);
        const core2 = try acquireModelCore(id2);
        defer releaseModelCore(id2);
        core1.rwlock.lockShared();
        defer core1.rwlock.unlockShared();
        core2.rwlock.lockShared();
        defer core2.rwlock.unlockShared();
        const layer_count1 = try checkedModelLayerCount(core1);
        const layer_count2 = try checkedModelLayerCount(core2);
        if (core1.dim != core2.dim) return false;
        if (layer_count1 != layer_count2) return false;
        if (core1.cfg.clip_min != core2.cfg.clip_min or core1.cfg.clip_max != core2.cfg.clip_max or core1.cfg.grad_mean != core2.cfg.grad_mean) return false;
        if (core1.cfg.max_dim != core2.cfg.max_dim or core1.cfg.max_layers != core2.cfg.max_layers) return false;
        var i: usize = 0;
        while (i < layer_count1) : (i += 1) {
            if (!(try tensorAllCloseEq(&core1.layers[i].s_weight, &core2.layers[i].s_weight, abs_tol, rel_tol))) return false;
            if (!(try tensorAllCloseEq(&core1.layers[i].t_weight, &core2.layers[i].t_weight, abs_tol, rel_tol))) return false;
            if (core1.layers[i].clip_min != core2.layers[i].clip_min or core1.layers[i].clip_max != core2.layers[i].clip_max or core1.layers[i].grad_mean != core2.layers[i].grad_mean) return false;
        }
        return true;
    }
};
fn crcUpdateU32LE(hasher: *std.hash.Crc32, v: u32) void {
    const le = std.mem.nativeToLittle(u32, v);
    hasher.update(std.mem.asBytes(&le));
}
fn crcUpdateU64LE(hasher: *std.hash.Crc32, v: u64) void {
    const le = std.mem.nativeToLittle(u64, v);
    hasher.update(std.mem.asBytes(&le));
}
fn crcUpdateU8(hasher: *std.hash.Crc32, v: u8) void {
    hasher.update(&.{v});
}
fn writeTensorData(w: anytype, hasher: *std.hash.Crc32, t: *const Tensor) !void {
    try validateTensor2D(t);
    try ensureFiniteSlice(t.data);
    const rows = t.shape.dims[0];
    const cols = t.shape.dims[1];
    try w.writeInt(u64, TENSOR_RANK_TAG, .little);
    crcUpdateU64LE(hasher, TENSOR_RANK_TAG);
    try w.writeInt(u64, @intCast(rows), .little);
    try w.writeInt(u64, @intCast(cols), .little);
    crcUpdateU64LE(hasher, @intCast(rows));
    crcUpdateU64LE(hasher, @intCast(cols));
    for (t.data) |v| {
        const bits = @as(u32, @bitCast(v));
        try w.writeInt(u32, bits, .little);
        crcUpdateU32LE(hasher, bits);
    }
}
fn hashTensorData(hasher: *std.hash.Crc32, t: *const Tensor) void {
    crcUpdateU64LE(hasher, TENSOR_RANK_TAG);
    crcUpdateU64LE(hasher, @intCast(t.shape.dims[0]));
    crcUpdateU64LE(hasher, @intCast(t.shape.dims[1]));
    for (t.data) |v| crcUpdateU32LE(hasher, @as(u32, @bitCast(v)));
}
fn readEncodedBool(r: anytype) !bool {
    const b = try r.readByte();
    return switch (b) {
        0 => false,
        1 => true,
        else => error.BadFileFormat,
    };
}
fn readTensorData(allocator: Allocator, r: anytype, expected_rows: usize, expected_cols: usize) !Tensor {
    if ((try r.readInt(u64, .little)) != TENSOR_RANK_TAG) return error.BadFileFormat;
    const d0_u64 = try r.readInt(u64, .little);
    const d1_u64 = try r.readInt(u64, .little);
    const expected_rows_u64: u64 = @intCast(expected_rows);
    const expected_cols_u64: u64 = @intCast(expected_cols);
    if (d0_u64 != expected_rows_u64 or d1_u64 != expected_cols_u64) return error.ShapeMismatch;
    const expected = try checkedMul(expected_rows, expected_cols);
    var t = try Tensor.init(allocator, &.{ expected_rows, expected_cols });
    errdefer t.deinit();
    var i: usize = 0;
    while (i < expected) : (i += 1) {
        const bits = try r.readInt(u32, .little);
        t.data[i] = @as(f32, @bitCast(bits));
    }
    return t;
}
const TempFile = struct {
    file: std.fs.File,
    tmp_name: []u8,
};
fn hexEncodeLower(dst: []u8, src: []const u8) []u8 {
    const alphabet = "0123456789abcdef";
    var i: usize = 0;
    while (i < src.len) : (i += 1) {
        const hi: usize = @intCast((src[i] >> 4) & 0x0f);
        const lo: usize = @intCast(src[i] & 0x0f);
        dst[i * 2] = alphabet[hi];
        dst[i * 2 + 1] = alphabet[lo];
    }
    return dst[0 .. src.len * 2];
}
fn createUniqueTempFile(dir: *std.fs.Dir, allocator: Allocator, base_name: []const u8) !TempFile {
    var attempt: usize = 0;
    while (attempt < 64) : (attempt += 1) {
        var rnd: [16]u8 = undefined;
        std.crypto.random.bytes(&rnd);
        var hex_buf: [32]u8 = undefined;
        const hex = hexEncodeLower(&hex_buf, &rnd);
        const tmp_name = try std.fmt.allocPrint(allocator, ".{s}.tmp.{s}", .{ base_name, hex });
        errdefer allocator.free(tmp_name);
        const file = dir.createFile(tmp_name, .{ .exclusive = true, .mode = 0o600 }) catch |e| switch (e) {
            error.PathAlreadyExists => {
                allocator.free(tmp_name);
                continue;
            },
            else => return e,
        };
        return .{ .file = file, .tmp_name = tmp_name };
    }
    return error.TempFileCollision;
}
fn syncDirectory(dir: std.fs.Dir) !void {
    if (comptime builtin.os.tag == .windows) return;
    std.posix.fsync(dir.fd) catch |err| switch (err) {
        error.AccessDenied => return,
        else => return err,
    };
}
fn writeSnapshotToPath(snapshot: *const SavedModelSnapshot, path: []const u8, allocator: Allocator) !void {
    if (snapshot.num_layers != snapshot.layers.len) return error.InvalidModelState;
    try validateModelConfigValues(snapshot.dim, snapshot.num_layers, snapshot.cfg);
    if (path.len == 0) return error.InvalidPath;
    const parent_path = if (std.fs.path.dirname(path)) |p| p else ".";
    const base_name = std.fs.path.basename(path);
    if (base_name.len == 0) return error.InvalidPath;
    var parent_dir = if (std.fs.path.isAbsolute(parent_path)) try std.fs.openDirAbsolute(parent_path, .{ .iterate = true }) else try std.fs.cwd().openDir(parent_path, .{ .iterate = true });
    defer parent_dir.close();
    const temp = try createUniqueTempFile(&parent_dir, allocator, base_name);
    defer allocator.free(temp.tmp_name);
    const file = temp.file;
    var file_open = true;
    var tmp_exists = true;
    errdefer {
        if (file_open) file.close();
        if (tmp_exists) parent_dir.deleteFile(temp.tmp_name) catch {};
    }
    var buffered = std.io.bufferedWriter(file.writer());
    const w = buffered.writer();
    var hasher = std.hash.Crc32.init();
    try w.writeAll("RSF0");
    hasher.update("RSF0");
    try w.writeInt(u32, SAVE_VERSION, .little);
    crcUpdateU32LE(&hasher, SAVE_VERSION);
    try w.writeInt(u64, @intCast(snapshot.num_layers), .little);
    crcUpdateU64LE(&hasher, @intCast(snapshot.num_layers));
    try w.writeInt(u64, @intCast(snapshot.dim), .little);
    crcUpdateU64LE(&hasher, @intCast(snapshot.dim));
    const clip_min_bits = @as(u32, @bitCast(snapshot.cfg.clip_min));
    const clip_max_bits = @as(u32, @bitCast(snapshot.cfg.clip_max));
    try w.writeInt(u32, clip_min_bits, .little);
    try w.writeInt(u32, clip_max_bits, .little);
    crcUpdateU32LE(&hasher, clip_min_bits);
    crcUpdateU32LE(&hasher, clip_max_bits);
    const gm_byte: u8 = if (snapshot.cfg.grad_mean) 1 else 0;
    try w.writeByte(gm_byte);
    crcUpdateU8(&hasher, gm_byte);
    try w.writeInt(u64, @intCast(snapshot.cfg.max_dim), .little);
    try w.writeInt(u64, @intCast(snapshot.cfg.max_layers), .little);
    crcUpdateU64LE(&hasher, @intCast(snapshot.cfg.max_dim));
    crcUpdateU64LE(&hasher, @intCast(snapshot.cfg.max_layers));
    const gd_byte: u8 = if (snapshot.cfg.global_diffusion) 1 else 0;
    try w.writeByte(gd_byte);
    crcUpdateU8(&hasher, gd_byte);
    var i: usize = 0;
    while (i < snapshot.layers.len) : (i += 1) {
        const layer = &snapshot.layers[i];
        if (layer.clip_min != snapshot.cfg.clip_min or layer.clip_max != snapshot.cfg.clip_max or layer.grad_mean != snapshot.cfg.grad_mean) return error.InvalidConfig;
        const lmin_bits = @as(u32, @bitCast(layer.clip_min));
        const lmax_bits = @as(u32, @bitCast(layer.clip_max));
        try w.writeInt(u32, lmin_bits, .little);
        try w.writeInt(u32, lmax_bits, .little);
        crcUpdateU32LE(&hasher, lmin_bits);
        crcUpdateU32LE(&hasher, lmax_bits);
        const lgm: u8 = if (layer.grad_mean) 1 else 0;
        try w.writeByte(lgm);
        crcUpdateU8(&hasher, lgm);
        try writeTensorData(w, &hasher, &layer.s_weight);
        try writeTensorData(w, &hasher, &layer.t_weight);
    }
    const gn_len = try checkedMul(snapshot.dim, 3);
    var g: usize = 0;
    while (g < snapshot.layers.len) : (g += 1) {
        const gn = snapshot.layers[g].gn_block;
        if (gn.len < gn_len) return error.InvalidModelState;
        var k: usize = 0;
        while (k < gn_len) : (k += 1) {
            const bits = @as(u32, @bitCast(gn[k]));
            try w.writeInt(u32, bits, .little);
            crcUpdateU32LE(&hasher, bits);
        }
    }
    try w.writeInt(u32, hasher.final(), .little);
    try buffered.flush();
    try file.sync();
    file.close();
    file_open = false;
    try parent_dir.rename(temp.tmp_name, base_name);
    tmp_exists = false;
    try syncDirectory(parent_dir);
}
