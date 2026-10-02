const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const core_io = @import("io.zig");

pub const LearnedEmbedding = struct {
    weight: Tensor,
    grad: Tensor,
    velocity: Tensor,
    vocab_size: usize,
    dim: usize,
    allocator: Allocator,
    fisher: ?Tensor = null,

    pub fn init(allocator: Allocator, v_size: usize, d: usize, seed: u64) !LearnedEmbedding {
        var w = try Tensor.init(allocator, &.{ v_size, d });
        errdefer w.deinit();
        var g = try Tensor.init(allocator, &.{ v_size, d });
        errdefer g.deinit();
        var v = try Tensor.init(allocator, &.{ v_size, d });
        errdefer v.deinit();
        @memset(g.data, 0.0);
        @memset(v.data, 0.0);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        var i: usize = 0;
        while (i < w.data.len) : (i += 1) {
            w.data[i] = (random.float(f32) - 0.5) * 0.02;
        }
        return LearnedEmbedding{
            .weight = w,
            .grad = g,
            .velocity = v,
            .vocab_size = v_size,
            .dim = d,
            .allocator = allocator,
        };
    }

    pub fn initWithWeights(allocator: Allocator, v_size: usize, d: usize, weights: []const f32) !LearnedEmbedding {
        if (v_size == 0 or d == 0) return error.InvalidDimensions;
        const expected = std.math.mul(usize, v_size, d) catch return error.InvalidDimensions;
        if (weights.len != expected) return error.InvalidDimensions;

        var w = try Tensor.init(allocator, &.{ v_size, d });
        errdefer w.deinit();
        var g = try Tensor.init(allocator, &.{ v_size, d });
        errdefer g.deinit();
        var v = try Tensor.init(allocator, &.{ v_size, d });
        errdefer v.deinit();

        for (weights, 0..) |value, index| {
            if (!std.math.isFinite(value)) return error.InvalidWeight;
            w.data[index] = value;
        }
        @memset(g.data, 0.0);
        @memset(v.data, 0.0);

        return LearnedEmbedding{
            .weight = w,
            .grad = g,
            .velocity = v,
            .vocab_size = v_size,
            .dim = d,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LearnedEmbedding) void {
        self.weight.deinit();
        self.grad.deinit();
        self.velocity.deinit();
        if (self.fisher) |*f| f.deinit();
        self.fisher = null;
    }

    pub fn forward(self: *LearnedEmbedding, allocator: Allocator, tokens: []const u32, max_seq_len: usize) !Tensor {
        const seq_len = @min(tokens.len, if (max_seq_len > 0) max_seq_len else tokens.len);
        if (seq_len == 0) return error.EmptyTokens;
        const out = try Tensor.init(allocator, &.{ seq_len, self.dim });
        @memset(out.data, 0.0);
        var r: usize = 0;
        while (r < seq_len) : (r += 1) {
            const t = @min(@as(usize, tokens[r]), self.vocab_size - 1);
            var c: usize = 0;
            while (c < self.dim) : (c += 1) {
                const w_idx = t * self.dim + c;
                const out_idx = r * self.dim + c;
                if (w_idx < self.weight.data.len and out_idx < out.data.len) {
                    out.data[out_idx] = self.weight.data[w_idx];
                }
            }
        }
        return out;
    }

    pub fn lookupRows(self: *const LearnedEmbedding, tokens: []const u32, dst: []f32) void {
        var r: usize = 0;
        while (r < tokens.len) : (r += 1) {
            const t = @min(@as(usize, tokens[r]), self.vocab_size - 1);
            const src = self.weight.data[t * self.dim ..][0..self.dim];
            const target = dst[r * self.dim ..][0..self.dim];
            @memcpy(target, src);
        }
    }

    pub fn backward(self: *LearnedEmbedding, tokens: []const u32, out_grad: []const f32, max_seq_len: usize) void {
        const seq_len = @min(tokens.len, if (max_seq_len > 0) max_seq_len else tokens.len);
        if (seq_len == 0) return;
        var r: usize = 0;
        while (r < seq_len) : (r += 1) {
            const t = @min(@as(usize, tokens[r]), self.vocab_size - 1);
            var c: usize = 0;
            while (c < self.dim) : (c += 1) {
                const g_idx = t * self.dim + c;
                const grad_idx = r * self.dim + c;
                if (g_idx < self.grad.data.len and grad_idx < out_grad.len) {
                    self.grad.data[g_idx] += out_grad[grad_idx];
                }
            }
        }
    }

    pub fn backwardAccumulate(self: *LearnedEmbedding, tokens: []const u32, lengths: []const u32, out_grad: []const f32, padded_seq_len: usize, batch_size: usize) void {
        const rows = @min(tokens.len / padded_seq_len, batch_size);
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            const valid: usize = if (r < lengths.len) @min(@as(usize, lengths[r]), padded_seq_len) else padded_seq_len;
            var s: usize = 0;
            while (s < valid) : (s += 1) {
                const t = @min(@as(usize, tokens[r * padded_seq_len + s]), self.vocab_size - 1);
                const src_off = (r * padded_seq_len + s) * self.dim;
                const dst_off = t * self.dim;
                var c: usize = 0;
                while (c + 8 <= self.dim) : (c += 8) {
                    const g: @Vector(8, f32) = self.grad.data[dst_off + c ..][0..8].*;
                    const u: @Vector(8, f32) = out_grad[src_off + c ..][0..8].*;
                    self.grad.data[dst_off + c ..][0..8].* = g + u;
                }
                while (c < self.dim) : (c += 1) {
                    self.grad.data[dst_off + c] += out_grad[src_off + c];
                }
            }
        }
    }

    pub fn zeroGrad(self: *LearnedEmbedding) void {
        @memset(self.grad.data, 0.0);
    }

    pub fn applyGradients(self: *LearnedEmbedding, lr: f32, momentum: f32) void {
        var i: usize = 0;
        const n = self.weight.data.len;
        while (i + 8 <= n) : (i += 8) {
            const v: @Vector(8, f32) = self.velocity.data[i..][0..8].*;
            const g: @Vector(8, f32) = self.grad.data[i..][0..8].*;
            const m: @Vector(8, f32) = @splat(momentum);
            const new_v = v * m + g;
            self.velocity.data[i..][0..8].* = new_v;
            const w: @Vector(8, f32) = self.weight.data[i..][0..8].*;
            const l: @Vector(8, f32) = @splat(lr);
            self.weight.data[i..][0..8].* = w - l * new_v;
        }
        while (i < n) : (i += 1) {
            self.velocity.data[i] = momentum * self.velocity.data[i] + self.grad.data[i];
            self.weight.data[i] -= lr * self.velocity.data[i];
        }
    }

    pub fn ensureFisherState(self: *LearnedEmbedding) !void {
        if (self.fisher != null) return;
        const f = try Tensor.init(self.allocator, &.{ self.vocab_size, self.dim });
        @memset(f.data, 0.0);
        self.fisher = f;
    }

    pub fn applyGradientsSFD(self: *LearnedEmbedding, lr: f32, momentum_beta: f32, fisher_gamma: f32, epsilon: f32) !void {
        try self.ensureFisherState();
        if (self.fisher) |*fisher_tensor| {
            const n = self.weight.data.len;
            var i: usize = 0;
            while (i + 8 <= n) : (i += 8) {
                const g: @Vector(8, f32) = self.grad.data[i..][0..8].*;
                const v: @Vector(8, f32) = self.velocity.data[i..][0..8].*;
                const beta: @Vector(8, f32) = @splat(momentum_beta);
                const new_v = v * beta + g;
                self.velocity.data[i..][0..8].* = new_v;
                const f_old: @Vector(8, f32) = fisher_tensor.data[i..][0..8].*;
                const gamma: @Vector(8, f32) = @splat(fisher_gamma);
                const one_minus_gamma: @Vector(8, f32) = @splat(1.0 - fisher_gamma);
                const new_f = f_old * gamma + g * g * one_minus_gamma;
                fisher_tensor.data[i..][0..8].* = new_f;
                const eps: @Vector(8, f32) = @splat(epsilon);
                const denom = @sqrt(new_f) + eps;
                const w: @Vector(8, f32) = self.weight.data[i..][0..8].*;
                const l: @Vector(8, f32) = @splat(lr);
                const updated = w - l * (new_v / denom);
                const clamp_bound: @Vector(8, f32) = @splat(65504.0);
                const neg_bound: @Vector(8, f32) = @splat(-65504.0);
                self.weight.data[i..][0..8].* = @min(@max(updated, neg_bound), clamp_bound);
            }
            while (i < n) : (i += 1) {
                const g = self.grad.data[i];
                const new_v = momentum_beta * self.velocity.data[i] + g;
                self.velocity.data[i] = new_v;
                const new_f = fisher_gamma * fisher_tensor.data[i] + (1.0 - fisher_gamma) * g * g;
                fisher_tensor.data[i] = new_f;
                const updated = self.weight.data[i] - lr * new_v / (@sqrt(new_f) + epsilon);
                self.weight.data[i] = std.math.clamp(updated, @as(f32, -65504.0), @as(f32, 65504.0));
            }
        }
    }

    pub fn spectralNormMax(self: *const LearnedEmbedding, iterations: usize) !f32 {
        const rows = self.vocab_size;
        const cols = self.dim;
        if (rows == 0 or cols == 0) return 0.0;
        const w = self.weight.data;
        const v_buf = try self.allocator.alloc(f32, cols);
        defer self.allocator.free(v_buf);
        const u_buf = try self.allocator.alloc(f32, rows);
        defer self.allocator.free(u_buf);
        var it: usize = 0;
        var c: usize = 0;
        while (c < cols) : (c += 1) v_buf[c] = 1.0 / @sqrt(@as(f32, @floatFromInt(cols)));
        var sigma: f32 = 0.0;
        while (it < iterations) : (it += 1) {
            var r: usize = 0;
            while (r < rows) : (r += 1) {
                var acc: f32 = 0.0;
                c = 0;
                while (c < cols) : (c += 1) acc += w[r * cols + c] * v_buf[c];
                u_buf[r] = acc;
            }
            var u_norm: f32 = 0.0;
            r = 0;
            while (r < rows) : (r += 1) u_norm += u_buf[r] * u_buf[r];
            u_norm = @sqrt(u_norm);
            if (u_norm < 1e-12) return 0.0;
            r = 0;
            while (r < rows) : (r += 1) u_buf[r] /= u_norm;
            c = 0;
            while (c < cols) : (c += 1) {
                var acc: f32 = 0.0;
                r = 0;
                while (r < rows) : (r += 1) acc += w[r * cols + c] * u_buf[r];
                v_buf[c] = acc;
            }
            var v_norm: f32 = 0.0;
            c = 0;
            while (c < cols) : (c += 1) v_norm += v_buf[c] * v_buf[c];
            v_norm = @sqrt(v_norm);
            if (v_norm < 1e-12) return 0.0;
            sigma = v_norm;
            c = 0;
            while (c < cols) : (c += 1) v_buf[c] /= v_norm;
        }
        return sigma;
    }

    pub fn spectralNormalize(self: *LearnedEmbedding, target_sigma: f32, iterations: usize) !void {
        const sigma = try self.spectralNormMax(iterations);
        if (sigma <= target_sigma or sigma < 1e-12) return;
        const scale = target_sigma / sigma;
        var i: usize = 0;
        const n = self.weight.data.len;
        while (i + 8 <= n) : (i += 8) {
            const w: @Vector(8, f32) = self.weight.data[i..][0..8].*;
            const s: @Vector(8, f32) = @splat(scale);
            self.weight.data[i..][0..8].* = w * s;
        }
        while (i < n) : (i += 1) self.weight.data[i] *= scale;
    }

    pub fn paramCount(self: *const LearnedEmbedding) usize {
        return self.vocab_size * self.dim;
    }

    pub fn flattenParams(self: *const LearnedEmbedding, dst: []f32) void {
        const count = @min(dst.len, self.weight.data.len);
        @memcpy(dst[0..count], self.weight.data[0..count]);
    }

    pub fn flattenGrads(self: *const LearnedEmbedding, dst: []f32) void {
        const count = @min(dst.len, self.grad.data.len);
        @memcpy(dst[0..count], self.grad.data[0..count]);
    }

    pub fn flattenVelocity(self: *const LearnedEmbedding, dst: []f32) void {
        const count = @min(dst.len, self.velocity.data.len);
        @memcpy(dst[0..count], self.velocity.data[0..count]);
    }

    pub fn flattenFisher(self: *const LearnedEmbedding, dst: []f32) void {
        if (self.fisher) |f| {
            const count = @min(dst.len, f.data.len);
            @memcpy(dst[0..count], f.data[0..count]);
        } else {
            @memset(dst, 0.0);
        }
    }

    pub fn scatterParams(self: *LearnedEmbedding, src: []const f32) void {
        const count = @min(src.len, self.weight.data.len);
        @memcpy(self.weight.data[0..count], src[0..count]);
    }

    pub fn scatterVelocity(self: *LearnedEmbedding, src: []const f32) void {
        const count = @min(src.len, self.velocity.data.len);
        @memcpy(self.velocity.data[0..count], src[0..count]);
    }

    pub fn scatterFisher(self: *LearnedEmbedding, src: []const f32) !void {
        try self.ensureFisherState();
        const count = @min(src.len, self.fisher.?.data.len);
        @memcpy(self.fisher.?.data[0..count], src[0..count]);
    }

    pub fn save(self: *const LearnedEmbedding, path: []const u8) !void {
        const file = try core_io.createFilePath(path, .{});
        defer file.close();
        var buf_writer = std.io.bufferedWriter(file.writer());
        const writer = buf_writer.writer();
        try writer.writeInt(u32, 0x4A454D42, .little);
        try writer.writeInt(u32, 1, .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.vocab_size)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.dim)), .little);
        for (self.weight.data) |w| {
            try writer.writeInt(u32, @as(u32, @bitCast(w)), .little);
        }
        try buf_writer.flush();
    }

    pub fn saveWithState(self: *const LearnedEmbedding, path: []const u8) !void {
        var tmp_buf: [4096]u8 = undefined;
        if (path.len + 4 > tmp_buf.len) return error.PathTooLong;
        @memcpy(tmp_buf[0..path.len], path);
        @memcpy(tmp_buf[path.len..][0..4], ".tmp");
        const tmp_path = tmp_buf[0 .. path.len + 4];
        const file = try core_io.createFilePath(tmp_path, .{});
        defer file.close();
        var buf_writer = std.io.bufferedWriter(file.writer());
        const writer = buf_writer.writer();
        try writer.writeInt(u32, 0x4A454D42, .little);
        try writer.writeInt(u32, 2, .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.vocab_size)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.dim)), .little);
        var crc = std.hash.Crc32.init();
        for (self.weight.data) |w| {
            const bits: u32 = @bitCast(w);
            try writer.writeInt(u32, bits, .little);
            var le: [4]u8 = undefined;
            std.mem.writeInt(u32, &le, bits, .little);
            crc.update(&le);
        }
        for (self.velocity.data) |v| {
            const bits: u32 = @bitCast(v);
            try writer.writeInt(u32, bits, .little);
            var le: [4]u8 = undefined;
            std.mem.writeInt(u32, &le, bits, .little);
            crc.update(&le);
        }
        if (self.fisher) |f| {
            try writer.writeInt(u32, 1, .little);
            for (f.data) |fv| {
                const bits: u32 = @bitCast(fv);
                try writer.writeInt(u32, bits, .little);
                var le: [4]u8 = undefined;
                std.mem.writeInt(u32, &le, bits, .little);
                crc.update(&le);
            }
        } else {
            try writer.writeInt(u32, 0, .little);
        }
        try writer.writeInt(u32, crc.final(), .little);
        try buf_writer.flush();
        file.sync() catch {};
        try core_io.renameFilePath(tmp_path, path);
    }

    pub fn load(allocator: Allocator, path: []const u8) !LearnedEmbedding {
        const file = core_io.openFilePath(path, .{}) catch return error.FileNotFound;
        defer file.close();
        var buf_reader = std.io.bufferedReader(file.reader());
        const reader = buf_reader.reader();
        const magic = try reader.readInt(u32, .little);
        if (magic != 0x4A454D42) return error.InvalidFormat;
        const version = try reader.readInt(u32, .little);
        if (version != 1 and version != 2) return error.IncompatibleVersion;
        const v_size = @as(usize, @intCast(try reader.readInt(u64, .little)));
        const d = @as(usize, @intCast(try reader.readInt(u64, .little)));
        var w = try Tensor.init(allocator, &.{ v_size, d });
        errdefer w.deinit();
        var g = try Tensor.init(allocator, &.{ v_size, d });
        errdefer g.deinit();
        var v = try Tensor.init(allocator, &.{ v_size, d });
        errdefer v.deinit();
        @memset(g.data, 0.0);
        @memset(v.data, 0.0);
        var crc = std.hash.Crc32.init();
        var i: usize = 0;
        while (i < w.data.len) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            w.data[i] = @bitCast(bits);
            var le: [4]u8 = undefined;
            std.mem.writeInt(u32, &le, bits, .little);
            crc.update(&le);
        }
        var fisher_state: ?Tensor = null;
        errdefer if (fisher_state) |*f| f.deinit();
        if (version == 2) {
            i = 0;
            while (i < v.data.len) : (i += 1) {
                const bits = try reader.readInt(u32, .little);
                v.data[i] = @bitCast(bits);
                var le: [4]u8 = undefined;
                std.mem.writeInt(u32, &le, bits, .little);
                crc.update(&le);
            }
            const has_fisher = try reader.readInt(u32, .little);
            if (has_fisher == 1) {
                var f = try Tensor.init(allocator, &.{ v_size, d });
                errdefer f.deinit();
                i = 0;
                while (i < f.data.len) : (i += 1) {
                    const bits = try reader.readInt(u32, .little);
                    f.data[i] = @bitCast(bits);
                    var le: [4]u8 = undefined;
                    std.mem.writeInt(u32, &le, bits, .little);
                    crc.update(&le);
                }
                fisher_state = f;
            }
            const expected_crc = try reader.readInt(u32, .little);
            if (crc.final() != expected_crc) return error.ChecksumMismatch;
        }
        return LearnedEmbedding{
            .weight = w,
            .grad = g,
            .velocity = v,
            .vocab_size = v_size,
            .dim = d,
            .allocator = allocator,
            .fisher = fisher_state,
        };
    }
};
