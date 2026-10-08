const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("../core/tensor.zig");
const types = @import("../core/types.zig");
const Tensor = tensor.Tensor;

pub const OFTB = struct {
    pub const FRACTAL_SCALE: f32 = 0.7071067811865476;
    pub const FRACTAL_SCALE_SQ: f32 = 0.5000000000000001;
    pub const LOG_DET_JACOBIAN: f32 = 0.0;
    pub const resonance_order: usize = 8;

    dim: usize,
    global_diffusion: bool,
    layout: ?types.RSFDiffusionLayout,

    pub fn init(d: usize) OFTB {
        std.debug.assert(d != 0);
        std.debug.assert(d <= std.math.maxInt(usize) / 2);
        return OFTB{
            .dim = d,
            .global_diffusion = false,
            .layout = null,
        };
    }

    pub fn initDiffusing(d: usize) !OFTB {
        return initWithDiffusion(d, true);
    }

    pub fn initWithDiffusion(d: usize, global_diffusion: bool) !OFTB {
        if (d == 0) return error.InvalidDimension;
        if (d > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        if (!global_diffusion) {
            return OFTB{ .dim = d, .global_diffusion = false, .layout = null };
        }
        const row_len = d * 2;
        const layout = types.rsfDiffusionLayout(row_len) orelse return error.InvalidDiffusionLayout;
        if (!tensor.diffusionLayoutIsApplicable(row_len, layout)) return error.InvalidDiffusionLayout;
        return OFTB{ .dim = d, .global_diffusion = true, .layout = layout };
    }

    pub fn deinit(self: *OFTB) void {
        self.* = undefined;
    }

    pub fn diffusionEnabled(self: OFTB) bool {
        return self.global_diffusion and self.layout != null;
    }

    pub fn diffusionLayout(self: OFTB) ?types.RSFDiffusionLayout {
        return self.layout;
    }

    pub fn rowLen(self: OFTB) usize {
        return self.dim * 2;
    }

    pub fn vectorLen() usize {
        if (comptime builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) {
            return 16;
        }
        return 8;
    }

    pub fn applyRotationSliceInPlace(self: OFTB, data: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (data.len != total) return;
        const half = self.dim;
        const x1 = data[0..half];
        const x2 = data[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = comptime vectorLen();
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const va: @Vector(VLEN, f32) = x1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = x2[i..][0..VLEN].*;
            const vscale: @Vector(VLEN, f32) = @splat(scale);
            x1[i..][0..VLEN].* = (va - vb) * vscale;
            x2[i..][0..VLEN].* = (va + vb) * vscale;
        }
        while (i < half) : (i += 1) {
            const a = x1[i];
            const b = x2[i];
            x1[i] = (a - b) * scale;
            x2[i] = (a + b) * scale;
        }
    }

    pub fn applyRotationAdjointSliceInPlace(self: OFTB, data: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (data.len != total) return;
        const half = self.dim;
        const g1 = data[0..half];
        const g2 = data[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = comptime vectorLen();
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const va: @Vector(VLEN, f32) = g1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = g2[i..][0..VLEN].*;
            const vscale: @Vector(VLEN, f32) = @splat(scale);
            g1[i..][0..VLEN].* = (va + vb) * vscale;
            g2[i..][0..VLEN].* = (vb - va) * vscale;
        }
        while (i < half) : (i += 1) {
            const a = g1[i];
            const b = g2[i];
            g1[i] = (a + b) * scale;
            g2[i] = (b - a) * scale;
        }
    }

    pub fn applyDiffusionSliceInPlace(self: OFTB, data: []f32) !void {
        const layout = self.layout orelse return error.DimensionMismatch;
        if (data.len != self.dim * 2) return error.DimensionMismatch;
        if (!self.diffusionEnabled()) return;
        // f64 accumulation: the f32 butterfly drifts past the 1e-7 involution
        // bound on long rows (measured 8.9e-7 at row_len = 98304).
        try tensor.diffuseRowAccurateInPlace(data, layout);
    }

    pub fn fastWalshHadamardTransformInPlace(data: []f32) !void {
        if (data.len == 0 or (data.len & (data.len - 1)) != 0) return error.InvalidDimension;
        try tensor.hadamardBlockInPlace(data);
    }

    pub fn mixRadixBlocksInPlace(self: OFTB, data: []f32) !void {
        const layout = self.layout orelse return;
        if (data.len != layout.row_len) return error.DimensionMismatch;
        // Q_1 = I: the radix mix is a no-op for r = 1.
        if (layout.radix == 1) return;
        try tensor.mixRadixBlocksInPlace(data, layout);
    }

    pub fn diffuseSliceInPlace(self: OFTB, data: []f32) !void {
        if (!self.diffusionEnabled()) return;
        const layout = self.layout.?;
        if (data.len != layout.row_len) return error.DimensionMismatch;
        try tensor.globalDiffuseRowStack(data, layout);
    }

    pub fn logDetContribution(self: OFTB) f32 {
        _ = self;
        return LOG_DET_JACOBIAN;
    }

    pub fn rotationIsIdentityAfterOrder(self: OFTB, data: []f32) !bool {
        if (data.len != self.dim * 2) return error.DimensionMismatch;
        const copy = try std.heap.smp_allocator.dupe(f32, data);
        defer std.heap.smp_allocator.free(copy);
        var check = OFTB{ .dim = self.dim, .global_diffusion = false, .layout = null };
        var i: usize = 0;
        while (i < resonance_order) : (i += 1) check.applyRotationSliceInPlace(copy);
        for (copy, data) |a, b| if (@abs(a - b) > 1.0e-6) return false;
        return true;
    }


    pub fn applyDiffusionRowsInPlace(self: OFTB, rows: []f32, count: usize) void {
        const layout = self.layout orelse return;
        const total = self.dim * 2;
        var r: usize = 0;
        while (r < count) : (r += 1) {
            const base = r * total;
            if (base + total > rows.len) return;
            tensor.globalDiffuseRowUnchecked(rows[base .. base + total], layout);
        }
    }

    pub fn forwardInPlace(self: OFTB, x: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (x.data.len != total) return error.DimensionMismatch;
        try self.forwardSliceInPlace(x.data);
    }

    pub fn forwardSliceInPlace(self: OFTB, data: []f32) !void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (data.len != total) return;
        self.applyRotationSliceInPlace(data);
        try self.diffuseSliceInPlace(data);
    }

    pub fn backwardInPlace(self: OFTB, grad: []f32) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (grad.len != total) return error.DimensionMismatch;
        try self.backwardSliceInPlace(grad);
    }

    pub fn backwardSliceInPlace(self: OFTB, grad: []f32) !void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (grad.len != total) return;
        try self.diffuseSliceInPlace(grad);
        self.applyRotationAdjointSliceInPlace(grad);
    }

    pub fn inverseInPlace(self: OFTB, x: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (x.data.len != total) return error.DimensionMismatch;
        try self.backwardSliceInPlace(x.data);
    }

    pub fn inverseSliceInPlace(self: OFTB, data: []f32) !void {
        try self.backwardSliceInPlace(data);
    }

    pub fn forwardBackwardFusedInPlace(self: OFTB, activation: []f32, grad: []f32) !void {
        try self.forwardSliceInPlace(activation);
        try self.backwardSliceInPlace(grad);
    }

    pub fn symplecticReversalInPlace(self: OFTB, activation: []f32, grad: []f32) !void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (activation.len != total or grad.len != total) return;
        try self.applyDiffusionSliceInPlace(activation);
        try self.applyDiffusionSliceInPlace(grad);
        const half = self.dim;
        const a1 = activation[0..half];
        const a2 = activation[half..][0..half];
        const g1 = grad[0..half];
        const g2 = grad[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = comptime vectorLen();
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const wa: @Vector(VLEN, f32) = a1[i..][0..VLEN].*;
            const wb: @Vector(VLEN, f32) = a2[i..][0..VLEN].*;
            const wscale: @Vector(VLEN, f32) = @splat(scale);
            a1[i..][0..VLEN].* = (wa + wb) * wscale;
            a2[i..][0..VLEN].* = (wb - wa) * wscale;
            const va: @Vector(VLEN, f32) = g1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = g2[i..][0..VLEN].*;
            g1[i..][0..VLEN].* = (va + vb) * wscale;
            g2[i..][0..VLEN].* = (vb - va) * wscale;
        }
        while (i < half) : (i += 1) {
            const a = a1[i];
            const b = a2[i];
            a1[i] = (a + b) * scale;
            a2[i] = (b - a) * scale;
            const ga = g1[i];
            const gb = g2[i];
            g1[i] = (ga + gb) * scale;
            g2[i] = (gb - ga) * scale;
        }
    }

    pub fn logDeterminantJacobian(self: OFTB) f32 {
        _ = self;
        return LOG_DET_JACOBIAN;
    }

    pub fn logDeterminantAdjointShift(_: OFTB) f32 {
        return 1.0;
    }

    pub fn isSymplectic(_: OFTB) bool {
        return true;
    }

    pub fn isOrthogonal(_: OFTB) bool {
        return true;
    }

    pub fn diffusionIsInvolution(self: OFTB, data: []const f32, allocator: std.mem.Allocator) !bool {
        if (!self.diffusionEnabled()) return true;
        if (data.len != self.dim * 2) return error.DimensionMismatch;
        const copy = try allocator.dupe(f32, data);
        defer allocator.free(copy);
        try self.diffuseSliceInPlace(copy);
        try self.diffuseSliceInPlace(copy);
        for (copy, data) |a, b| if (@abs(a - b) > 1.0e-7) return false;
        return true;
    }

    pub fn resonanceOrder(_: OFTB) usize {
        return resonance_order;
    }

    pub fn forwardRows(self: OFTB, rows: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (rows.shape.dims.len != 2) return error.DimensionMismatch;
        const total = self.dim * 2;
        if (rows.shape.dims[1] != total) return error.DimensionMismatch;
        if (!rows.shape.isContiguous()) return error.DimensionMismatch;
        const row_count = rows.shape.dims[0];
        var r: usize = 0;
        while (r < row_count) : (r += 1) {
            try self.forwardSliceInPlace(rows.data[r * total ..][0..total]);
        }
    }

    pub fn inverseRows(self: OFTB, rows: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (rows.shape.dims.len != 2) return error.DimensionMismatch;
        const total = self.dim * 2;
        if (rows.shape.dims[1] != total) return error.DimensionMismatch;
        if (!rows.shape.isContiguous()) return error.DimensionMismatch;
        const row_count = rows.shape.dims[0];
        var r: usize = 0;
        while (r < row_count) : (r += 1) {
            try self.backwardSliceInPlace(rows.data[r * total ..][0..total]);
        }
    }
};

pub fn mixForward(oftb: OFTB, x: *Tensor) !void {
    try oftb.forwardInPlace(x);
}

pub fn mixBackward(oftb: OFTB, grad: []f32) !void {
    try oftb.backwardInPlace(grad);
}

pub fn mixInverse(oftb: OFTB, x: *Tensor) !void {
    try oftb.inverseInPlace(x);
}

comptime {
    _ = OFTB;
}
