const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");
const core_tensor = @import("../../core/tensor.zig");
const core_memory = @import("../../core/memory.zig");
const types = @import("../../core/types.zig");

pub const gpu_enabled: bool = @import("build_options").gpu_acceleration;

pub const rsf_coupling_width: usize = 2;
pub const rsf_fisher_width: usize = 3;
pub const rsf_bias_column: usize = 1;
pub const pinned_alignment: usize = 64;
pub const rsf_default_clip_min: f16 = -5.0;
pub const rsf_default_clip_max: f16 = 5.0;
pub const rsf_scratch_initial_capacity: usize = 256;
pub const graph_default_chunk_size: usize = 4096;

pub const AcceleratorMode = enum {
    cpu_aligned,
    cuda_host,
    none,
};

pub const AccelError = error{
    FutharkConfigFailed,
    FutharkContextFailed,
    FutharkSyncFailed,
    FutharkArrayNewFailed,
    FutharkValuesFailed,
    FutharkForwardFailed,
    FutharkInverseFailed,
    FutharkTrainingStepFailed,
    FutharkScaleWeightsFailed,
    FutharkShapeFailed,
    FutharkComputeLossFailed,
    FutharkBackwardFailed,
    FutharkSFDUpdateFailed,
    FutharkProjectionFailed,
    FutharkNormalizationFailed,
    CudaHostAllocFailed,
    CudaFreeFailed,
    NullPointer,
    InvalidDimensions,
    InvalidDataLength,
    InvalidHyperparameter,
    InvalidClipRange,
    InvalidToken,
    InvalidSequenceLength,
    AllocationFailed,
    PartialRowCleanup,
    Overflow,
    UninitializedContext,
    ContextMismatch,
    InvalidResourceState,
    OptimizerStepOverflow,
    InvalidAlignment,
    InvalidCausalMask,
};

pub const BackendDiagnostics = struct {
    code: ?AccelError = null,
    message: ?[]const u8 = null,
    allocator: ?std.mem.Allocator = null,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        if (self.message) |message| {
            if (self.allocator) |allocator| allocator.free(message);
            self.message = null;
        }
        self.allocator = null;
        self.code = null;
    }

    pub fn text(self: *const Self) ?[]const u8 {
        return self.message;
    }

    pub fn errorCode(self: *const Self) ?AccelError {
        return self.code;
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.code == null and self.message == null;
    }
};

fn checkedByteCount(count: usize, comptime T: type) AccelError!usize {
    if (count == 0) return AccelError.InvalidDimensions;
    return std.math.mul(usize, count, @sizeOf(T)) catch AccelError.Overflow;
}

fn checkedElementCount2(first: usize, second: usize) AccelError!usize {
    if (first == 0 or second == 0) return AccelError.InvalidDimensions;
    return std.math.mul(usize, first, second) catch AccelError.Overflow;
}

fn checkedElementCount3(first: usize, second: usize, third: usize) AccelError!usize {
    if (first == 0 or second == 0 or third == 0) return AccelError.InvalidDimensions;
    const partial = std.math.mul(usize, first, second) catch return AccelError.Overflow;
    return std.math.mul(usize, partial, third) catch AccelError.Overflow;
}

fn checkedDimension(value: usize) AccelError!i64 {
    if (value == 0) return AccelError.InvalidDimensions;
    if (value > @as(usize, @intCast(std.math.maxInt(i64)))) return AccelError.Overflow;
    return @as(i64, @intCast(value));
}

fn checkedSigned(value: usize) AccelError!i64 {
    if (value > @as(usize, @intCast(std.math.maxInt(i64)))) return AccelError.Overflow;
    return @as(i64, @intCast(value));
}

fn allocHost(comptime T: type, allocator: std.mem.Allocator, count: usize) AccelError![]T {
    if (count == 0) return AccelError.InvalidDimensions;
    _ = try checkedByteCount(count, T);
    return allocator.alloc(T, count) catch AccelError.AllocationFailed;
}

fn dupeHost(comptime T: type, allocator: std.mem.Allocator, source: []const T) AccelError![]T {
    if (source.len == 0) return AccelError.InvalidDimensions;
    const copy = try allocHost(T, allocator, source.len);
    @memcpy(copy, source);
    return copy;
}

fn allFiniteF32(values: []const f32) bool {
    for (values) |value| {
        if (!std.math.isFinite(value)) return false;
    }
    return true;
}

fn allFiniteNonNegativeF32(values: []const f32) bool {
    for (values) |value| {
        if (!std.math.isFinite(value) or value < 0.0) return false;
    }
    return true;
}

fn allFiniteFisherBlocks(values: []const f32) bool {
    if (values.len % 3 != 0) return false;
    var index: usize = 0;
    while (index < values.len) : (index += 3) {
        if (!std.math.isFinite(values[index]) or values[index] < 0.0) return false;
        if (!std.math.isFinite(values[index + 1])) return false;
        if (!std.math.isFinite(values[index + 2]) or values[index + 2] < 0.0) return false;
    }
    return true;
}

fn allFiniteF16(values: []const f16) bool {
    for (values) |value| {
        const widened: f32 = @floatCast(value);
        if (!std.math.isFinite(widened)) return false;
    }
    return true;
}

fn allocateContextId() u64 {
    context_id_mutex.lock();
    defer context_id_mutex.unlock();
    const id = next_context_id;
    next_context_id +%= 1;
    return id;
}

var context_id_mutex: std.Thread.Mutex = .{};
var next_context_id: u64 = 1;

pub const ContextOwner = struct {
    allocator: std.mem.Allocator,
    cfg: ?*futhark.struct_futhark_context_config,
    ctx: ?*futhark.struct_futhark_context,
    mutex: ContextMutex = .init,
    refs_mutex: std.Thread.Mutex = .{},
    refs: u32 = 1,
    id: u64 = 0,
    alive: bool = false,
    diagnostics: BackendDiagnostics = .{},

    const Self = @This();

    fn create(allocator: std.mem.Allocator) AccelError!*Self {
        const self = allocator.create(Self) catch return AccelError.AllocationFailed;
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .cfg = null,
            .ctx = null,
            .mutex = .init,
            .refs_mutex = .{},
            .refs = 1,
            .id = allocateContextId(),
            .alive = false,
            .diagnostics = .{},
        };

        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return AccelError.FutharkConfigFailed;
        self.cfg = cfg;
        errdefer {
            futhark.futhark_context_config_free(cfg);
            self.cfg = null;
        }

        if (comptime gpu_enabled) {
            const cache_file: ?[*:0]const u8 = if (std.posix.getenv("JAIDE_FUTHARK_CACHE")) |cache_path|
                @as([*:0]const u8, @ptrCast(cache_path.ptr))
            else
                null;
            futhark.configureGpuContext(cfg, cache_file) catch {
                return AccelError.FutharkConfigFailed;
            };
        }

        const ctx = futhark.futhark_context_new(cfg);
        if (ctx == null) return AccelError.FutharkContextFailed;
        self.ctx = ctx;
        errdefer {
            futhark.futhark_context_free(ctx);
            self.ctx = null;
        }

        if (futhark.futhark_context_sync(ctx) != 0) {
            return self.recordFailureUnlocked(AccelError.FutharkSyncFailed);
        }

        self.alive = true;
        return self;
    }

    fn destroy(self: *Self) void {
        const allocator = self.allocator;
        if (self.ctx) |handle| {
            _ = futhark.futhark_context_clear_caches(handle);
            futhark.futhark_context_free(handle);
            self.ctx = null;
        }
        if (self.cfg) |cfg| {
            futhark.futhark_context_config_free(cfg);
            self.cfg = null;
        }
        self.alive = false;
        self.diagnostics.deinit();
        allocator.destroy(self);
    }

    fn retain(self: *Self) void {
        self.refs_mutex.lock();
        defer self.refs_mutex.unlock();
        if (self.refs == 0) return;
        self.refs += 1;
    }

    fn release(self: *Self) void {
        self.refs_mutex.lock();
        var last = false;
        if (self.refs != 0) {
            self.refs -= 1;
            last = self.refs == 0;
        }
        self.refs_mutex.unlock();
        if (last) self.destroy();
    }

    fn lock(self: *Self) void {
        self.retain();
        self.mutex.lock();
    }

    fn unlock(self: *Self) void {
        self.mutex.unlock();
        self.release();
    }

    fn checkLive(self: *Self) AccelError!void {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.checkLiveUnlocked();
    }

    fn checkLiveUnlocked(self: *Self) AccelError!void {
        if (!self.alive or self.ctx == null) return AccelError.UninitializedContext;
    }

    fn requireHandleUnlocked(self: *Self) AccelError!*futhark.struct_futhark_context {
        if (!self.alive) return AccelError.UninitializedContext;
        const handle = self.ctx orelse return AccelError.UninitializedContext;
        return handle;
    }

    fn recordFailureUnlocked(self: *Self, failure: AccelError) AccelError {
        self.diagnostics.deinit();
        self.diagnostics.code = failure;
        if (self.ctx) |handle| {
            const raw = futhark.futhark_context_get_error(handle);
            const maybe: ?[*:0]const u8 = @as(?[*:0]const u8, raw);
            if (maybe) |pointer| {
                const text: []const u8 = std.mem.span(pointer);
                if (self.allocator.dupe(u8, text)) |copy| {
                    self.diagnostics.message = copy;
                    self.diagnostics.allocator = self.allocator;
                } else |_| {}
            }
        }
        return failure;
    }

    fn syncContextUnlocked(self: *Self) AccelError!void {
        const handle = try self.requireHandleUnlocked();
        if (futhark.futhark_context_sync(handle) != 0) {
            return self.recordFailureUnlocked(AccelError.FutharkSyncFailed);
        }
    }

    fn takeDiagnosticsUnlocked(self: *Self, allocator: std.mem.Allocator) AccelError!BackendDiagnostics {
        var result = BackendDiagnostics{};
        errdefer result.deinit();
        result.code = self.diagnostics.code;
        if (self.diagnostics.message) |message| {
            const copy = try dupeHost(u8, allocator, message);
            result.message = copy;
            result.allocator = allocator;
        }
        return result;
    }
};

pub const ContextMutex = std.Thread.Mutex.Recursive;

var detached_context_mutex: ContextMutex = .init;

pub const FutharkContext = struct {
    owner: ?*ContextOwner,
    mutex: *ContextMutex,

    const Self = @This();

    pub fn init() AccelError!Self {
        return initWithAllocator(std.heap.page_allocator);
    }

    pub fn initWithAllocator(allocator: std.mem.Allocator) AccelError!Self {
        const owner = try ContextOwner.create(allocator);
        return Self{ .owner = owner, .mutex = &owner.mutex };
    }

    pub fn bind(owner: ?*ContextOwner) Self {
        return Self{
            .owner = owner,
            .mutex = if (owner) |live| &live.mutex else &detached_context_mutex,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.owner) |owner| {
            owner.release();
            self.owner = null;
        }
        self.mutex = &detached_context_mutex;
    }

    pub fn isValid(self: *const Self) bool {
        const owner = self.owner orelse return false;
        owner.lock();
        defer owner.unlock();
        return owner.alive and owner.ctx != null;
    }

    pub fn contextId(self: *const Self) AccelError!u64 {
        const owner = self.owner orelse return AccelError.UninitializedContext;
        owner.lock();
        defer owner.unlock();
        try owner.checkLiveUnlocked();
        return owner.id;
    }

    pub fn allocatorUsed(self: *const Self) AccelError!std.mem.Allocator {
        const owner = self.owner orelse return AccelError.UninitializedContext;
        return owner.allocator;
    }

    fn requireOwner(self: *const Self) AccelError!*ContextOwner {
        const owner = self.owner orelse return AccelError.UninitializedContext;
        try owner.checkLive();
        return owner;
    }

    fn retainOwner(self: *const Self) AccelError!*ContextOwner {
        const owner = self.owner orelse return AccelError.UninitializedContext;
        owner.lock();
        const live = owner.alive and owner.ctx != null;
        if (live) owner.retain();
        owner.unlock();
        if (!live) return AccelError.UninitializedContext;
        return owner;
    }

    fn ownerPointer(self: *const Self) ?*ContextOwner {
        return self.owner;
    }

    pub fn getRawContext(self: *const Self) AccelError!*futhark.struct_futhark_context {
        const owner = try self.requireOwner();
        return owner.ctx orelse AccelError.UninitializedContext;
    }

    pub fn sync(self: *Self) AccelError!void {
        const owner = try self.requireOwner();
        owner.lock();
        defer owner.unlock();
        return owner.syncContextUnlocked();
    }

    pub fn clearCaches(self: *Self) AccelError!void {
        const owner = try self.requireOwner();
        owner.lock();
        defer owner.unlock();
        const handle = try owner.requireHandleUnlocked();
        if (futhark.futhark_context_clear_caches(handle) != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkSyncFailed);
        }
    }

    pub fn lastError(self: *Self) ?AccelError {
        const owner = self.owner orelse return null;
        owner.lock();
        defer owner.unlock();
        return owner.diagnostics.code;
    }

    pub fn lastErrorMessage(self: *Self) ?[]const u8 {
        const owner = self.owner orelse return null;
        owner.lock();
        defer owner.unlock();
        return owner.diagnostics.message;
    }

    pub fn takeDiagnostics(self: *Self, allocator: std.mem.Allocator) AccelError!BackendDiagnostics {
        const owner = try self.requireOwner();
        owner.lock();
        defer owner.unlock();
        return owner.takeDiagnosticsUnlocked(allocator);
    }

    pub fn clearDiagnostics(self: *Self) void {
        const owner = self.owner orelse return;
        owner.lock();
        defer owner.unlock();
        owner.diagnostics.deinit();
    }

    pub fn getDataPointerF32_2D(self: *Self, array: *FutharkArray2DF32) AccelError!DeviceBufferF32 {
        const owner = try self.requireOwner();
        try array.requireSameContext(self);
        owner.lock();
        defer owner.unlock();
        return array.deviceBufferUnlocked();
    }

    pub fn getDataPointerF32_3D(self: *Self, array: *FutharkArray3DF32) AccelError!DeviceBufferF32 {
        const owner = try self.requireOwner();
        try array.requireSameContext(self);
        owner.lock();
        defer owner.unlock();
        return array.deviceBufferUnlocked();
    }
};

pub const DeviceBufferF32 = struct {
    ptr: ?*anyopaque,
    count: usize,
    context_id: u64,
    owner: ?*ContextOwner,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        if (self.owner) |owner| owner.release();
        self.owner = null;
        self.ptr = null;
        self.count = 0;
        self.context_id = 0;
    }

    pub fn rawPointer(self: *const Self) AccelError!*anyopaque {
        const ptr = self.ptr orelse return AccelError.InvalidResourceState;
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        if (!owner.alive or owner.ctx == null) return AccelError.UninitializedContext;
        return ptr;
    }

    pub fn elementCount(self: *const Self) usize {
        return self.count;
    }

    pub fn isValid(self: *const Self) bool {
        if (self.ptr == null or self.count == 0) return false;
        const owner = self.owner orelse return false;
        return owner.alive and owner.ctx != null;
    }

    pub fn contextId(self: *const Self) ?u64 {
        if (self.owner == null) return null;
        return self.context_id;
    }

    pub fn requireSameContext(self: *const Self, ctx: *const FutharkContext) AccelError!void {
        const expected = ctx.ownerPointer() orelse return AccelError.UninitializedContext;
        const actual = self.owner orelse return AccelError.InvalidResourceState;
        if (expected != actual) return AccelError.ContextMismatch;
        try expected.checkLive();
    }
};

pub const PinnedMemory = struct {
    ptr: ?*anyopaque,
    size: usize,
    alignment: usize,
    mode: AcceleratorMode,
    fallback_slice: ?[]align(pinned_alignment) u8,

    const Self = @This();

    pub fn alloc(size: usize) AccelError!Self {
        if (size == 0) {
            return Self{
                .ptr = null,
                .size = 0,
                .alignment = pinned_alignment,
                .mode = .none,
                .fallback_slice = null,
            };
        }
        _ = try checkedByteCount(size, u8);
        if (comptime gpu_enabled) {
            var ptr: ?*anyopaque = null;
            const status = cuda.cudaHostAlloc(&ptr, size, cuda.cudaHostAllocDefault);
            if (status != cuda.cudaSuccess) return AccelError.CudaHostAllocFailed;
            if (ptr == null) return AccelError.CudaHostAllocFailed;
            return Self{
                .ptr = ptr,
                .size = size,
                .alignment = pinned_alignment,
                .mode = .cuda_host,
                .fallback_slice = null,
            };
        }
        const slice = std.heap.page_allocator.alignedAlloc(u8, pinned_alignment, size) catch return AccelError.AllocationFailed;
        if (slice.len != size) return AccelError.AllocationFailed;
        return Self{
            .ptr = @ptrCast(slice.ptr),
            .size = size,
            .alignment = pinned_alignment,
            .mode = .cpu_aligned,
            .fallback_slice = slice,
        };
    }

    pub fn free(self: *Self) AccelError!void {
        switch (self.mode) {
            .none => {
                self.reset();
                return;
            },
            .cpu_aligned => {
                if (self.fallback_slice) |slice| {
                    std.heap.page_allocator.free(slice);
                    self.fallback_slice = null;
                }
                self.reset();
                return;
            },
            .cuda_host => {
                if (comptime !gpu_enabled) {
                    self.reset();
                    return AccelError.InvalidResourceState;
                }
                const ptr = self.ptr orelse {
                    self.reset();
                    return;
                };
                self.reset();
                const status = cuda.cudaFreeHost(ptr);
                if (status != cuda.cudaSuccess) return AccelError.CudaFreeFailed;
                return;
            },
        }
    }

    pub fn asSlice(self: *Self, comptime T: type) AccelError![]T {
        if (@sizeOf(T) == 0) return AccelError.InvalidDimensions;
        if (self.mode == .none or self.ptr == null or self.size == 0) return AccelError.InvalidResourceState;
        if (self.size % @sizeOf(T) != 0) return AccelError.InvalidDataLength;
        if (self.alignment % @alignOf(T) != 0) return AccelError.InvalidAlignment;
        const address = @intFromPtr(self.ptr.?);
        if (address % @alignOf(T) != 0) return AccelError.InvalidAlignment;
        const count = self.size / @sizeOf(T);
        const typed: [*]T = @ptrCast(@alignCast(self.ptr.?));
        return typed[0..count];
    }

    pub fn isLive(self: *const Self) bool {
        return self.mode != .none and self.ptr != null and self.size > 0;
    }

    pub fn modeKind(self: *const Self) AcceleratorMode {
        return self.mode;
    }

    fn reset(self: *Self) void {
        self.ptr = null;
        self.size = 0;
        self.mode = .none;
        self.fallback_slice = null;
    }
};

fn validateShapeF16_2DUnlocked(owner: *ContextOwner, arr: *futhark.struct_futhark_f16_2d, rows: usize, cols: usize) AccelError!void {
    const has_shape = comptime @hasDecl(futhark, "futhark_shape_f16_2d");
    if (!has_shape) return;
    const handle = try owner.requireHandleUnlocked();
    const shape: ?[*:0]const i64 = @as(?[*:0]const i64, futhark.futhark_shape_f16_2d(handle, arr));
    const pointer = shape orelse return AccelError.FutharkShapeFailed;
    const expected_rows = try checkedDimension(rows);
    const expected_cols = try checkedDimension(cols);
    if (pointer[0] != expected_rows or pointer[1] != expected_cols) return AccelError.FutharkShapeFailed;
}

fn validateShapeF16_3DUnlocked(owner: *ContextOwner, arr: *futhark.struct_futhark_f16_3d, d0: usize, d1: usize, d2: usize) AccelError!void {
    const has_shape = comptime @hasDecl(futhark, "futhark_shape_f16_3d");
    if (!has_shape) return;
    const handle = try owner.requireHandleUnlocked();
    const shape: ?[*:0]const i64 = @as(?[*:0]const i64, futhark.futhark_shape_f16_3d(handle, arr));
    const pointer = shape orelse return AccelError.FutharkShapeFailed;
    const expected_d0 = try checkedDimension(d0);
    const expected_d1 = try checkedDimension(d1);
    const expected_d2 = try checkedDimension(d2);
    if (pointer[0] != expected_d0 or pointer[1] != expected_d1 or pointer[2] != expected_d2) return AccelError.FutharkShapeFailed;
}

fn validateShapeF32_1DUnlocked(owner: *ContextOwner, arr: *futhark.struct_futhark_f32_1d, len: usize) AccelError!void {
    const has_shape = comptime @hasDecl(futhark, "futhark_shape_f32_1d");
    if (!has_shape) return;
    const handle = try owner.requireHandleUnlocked();
    const shape: ?[*:0]const i64 = @as(?[*:0]const i64, futhark.futhark_shape_f32_1d(handle, arr));
    const pointer = shape orelse return AccelError.FutharkShapeFailed;
    const expected_len = try checkedDimension(len);
    if (pointer[0] != expected_len) return AccelError.FutharkShapeFailed;
}

fn validateShapeF32_2DUnlocked(owner: *ContextOwner, arr: *futhark.struct_futhark_f32_2d, rows: usize, cols: usize) AccelError!void {
    const has_shape = comptime @hasDecl(futhark, "futhark_shape_f32_2d");
    if (!has_shape) return;
    const handle = try owner.requireHandleUnlocked();
    const shape: ?[*:0]const i64 = @as(?[*:0]const i64, futhark.futhark_shape_f32_2d(handle, arr));
    const pointer = shape orelse return AccelError.FutharkShapeFailed;
    const expected_rows = try checkedDimension(rows);
    const expected_cols = try checkedDimension(cols);
    if (pointer[0] != expected_rows or pointer[1] != expected_cols) return AccelError.FutharkShapeFailed;
}

fn validateShapeF32_3DUnlocked(owner: *ContextOwner, arr: *futhark.struct_futhark_f32_3d, d0: usize, d1: usize, d2: usize) AccelError!void {
    const has_shape = comptime @hasDecl(futhark, "futhark_shape_f32_3d");
    if (!has_shape) return;
    const handle = try owner.requireHandleUnlocked();
    const shape: ?[*:0]const i64 = @as(?[*:0]const i64, futhark.futhark_shape_f32_3d(handle, arr));
    const pointer = shape orelse return AccelError.FutharkShapeFailed;
    const expected_d0 = try checkedDimension(d0);
    const expected_d1 = try checkedDimension(d1);
    const expected_d2 = try checkedDimension(d2);
    if (pointer[0] != expected_d0 or pointer[1] != expected_d1 or pointer[2] != expected_d2) return AccelError.FutharkShapeFailed;
}

pub const FutharkArray1DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_1d,
    len: usize,
    owner: ?*ContextOwner,
    owned: bool,

    const Self = @This();

    fn adoptUnlocked(owner: *ContextOwner, handle: *futhark.struct_futhark_f32_1d, len: usize) Self {
        owner.retain();
        return Self{ .arr = handle, .len = len, .owner = owner, .owned = true };
    }

    fn borrowUnlocked(owner: *ContextOwner, handle: *futhark.struct_futhark_f32_1d, len: usize) Self {
        owner.retain();
        return Self{ .arr = handle, .len = len, .owner = owner, .owned = false };
    }

    pub fn newFromSlice(ctx: *FutharkContext, data: []const f32) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newFromSliceUnlocked(owner, data);
    }

    pub fn newFromSliceUnlocked(owner: *ContextOwner, data: []const f32) AccelError!Self {
        const handle = try owner.requireHandleUnlocked();
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(handle, data.ptr, try checkedDimension(data.len));
        if (arr == null) return owner.recordFailureUnlocked(AccelError.FutharkArrayNewFailed);
        return adoptUnlocked(owner, arr.?, data.len);
    }

    pub fn newZeros(ctx: *FutharkContext, len: usize, allocator: std.mem.Allocator) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newZerosUnlocked(owner, allocator, len);
    }

    pub fn newZerosUnlocked(owner: *ContextOwner, allocator: std.mem.Allocator, len: usize) AccelError!Self {
        if (len == 0) return AccelError.InvalidDimensions;
        const zeros = try allocHost(f32, allocator, len);
        defer allocator.free(zeros);
        @memset(zeros, 0.0);
        return newFromSliceUnlocked(owner, zeros);
    }

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return fromTensorUnlocked(owner, tensor);
    }

    pub fn fromTensorUnlocked(owner: *ContextOwner, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (tensor.shape.dims.len != 1) return AccelError.InvalidDimensions;
        const len = tensor.shape.dims[0];
        if (len == 0) return AccelError.InvalidDimensions;
        const expected = try checkedElementCount2(len, 1);
        if (tensor.data.len != expected) return AccelError.InvalidDataLength;
        return newFromSliceUnlocked(owner, tensor.data);
    }

    pub fn deinit(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        owner.lock();
        self.deinitUnlocked();
        owner.unlock();
    }

    pub fn deinitUnlocked(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        if (self.arr) |arr| {
            if (self.owned) {
                if (owner.requireHandleUnlocked()) |handle| {
                    _ = futhark.futhark_free_f32_1d(handle, arr);
                } else |_| {}
            }
        }
        self.clear();
        owner.release();
    }

    pub fn free(self: *Self, ctx: *FutharkContext) AccelError!void {
        try self.requireSameContext(ctx);
        self.deinit();
    }

    pub fn valuesSlice(self: *const Self, allocator: std.mem.Allocator) AccelError![]f32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.valuesSliceUnlocked(allocator);
    }

    pub fn valuesSliceUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError![]f32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = try allocHost(f32, allocator, self.len);
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f32_1d(handle, arr, buf.ptr) != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkValuesFailed);
        }
        try owner.syncContextUnlocked();
        return buf;
    }

    pub fn toTensor(self: *const Self, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.toTensorUnlocked(allocator);
    }

    pub fn toTensorUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (self.len == 0) return AccelError.InvalidDimensions;
        const values = try self.valuesSliceUnlocked(allocator);
        defer allocator.free(values);
        const shape = [_]usize{self.len};
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        errdefer tensor.deinit();
        if (tensor.data.len != values.len) return AccelError.InvalidDataLength;
        @memcpy(tensor.data, values);
        return tensor;
    }

    pub fn deviceBuffer(self: *const Self) AccelError!DeviceBufferF32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.deviceBufferUnlocked();
    }

    pub fn deviceBufferUnlocked(self: *const Self) AccelError!DeviceBufferF32 {
        const has_raw = comptime @hasDecl(futhark, "futhark_values_raw_f32_1d");
        if (!has_raw) return AccelError.FutharkValuesFailed;
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const raw: ?*anyopaque = @as(?*anyopaque, futhark.futhark_values_raw_f32_1d(handle, arr));
        if (raw == null) return owner.recordFailureUnlocked(AccelError.NullPointer);
        owner.retain();
        return DeviceBufferF32{ .ptr = raw.?, .count = self.len, .context_id = owner.id, .owner = owner };
    }

    pub fn isLive(self: *const Self) bool {
        if (self.arr == null or self.len == 0) return false;
        const owner = self.owner orelse return false;
        return owner.alive and owner.ctx != null;
    }

    pub fn contextId(self: *const Self) ?u64 {
        const owner = self.owner orelse return null;
        return owner.id;
    }

    pub fn requireSameContext(self: *const Self, ctx: *const FutharkContext) AccelError!void {
        const expected = ctx.ownerPointer() orelse return AccelError.UninitializedContext;
        const actual = self.owner orelse return AccelError.InvalidResourceState;
        if (expected != actual) return AccelError.ContextMismatch;
        try expected.checkLive();
    }

    pub fn requireLiveUnlocked(self: *const Self) AccelError!void {
        if (!self.isLive()) return AccelError.InvalidResourceState;
    }

    fn clear(self: *Self) void {
        self.arr = null;
        self.len = 0;
        self.owner = null;
        self.owned = false;
    }
};

pub const FutharkArray1DI64 = struct {
    arr: ?*futhark.struct_futhark_i64_1d,
    len: usize,
    owner: ?*ContextOwner,
    owned: bool,

    const Self = @This();

    fn adoptUnlocked(owner: *ContextOwner, handle: *futhark.struct_futhark_i64_1d, len: usize) Self {
        owner.retain();
        return Self{ .arr = handle, .len = len, .owner = owner, .owned = true };
    }

    pub fn newFromSlice(ctx: *FutharkContext, data: []const i64) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newFromSliceUnlocked(owner, data);
    }

    pub fn newFromSliceUnlocked(owner: *ContextOwner, data: []const i64) AccelError!Self {
        const handle = try owner.requireHandleUnlocked();
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_i64_1d(handle, data.ptr, try checkedDimension(data.len));
        if (arr == null) return owner.recordFailureUnlocked(AccelError.FutharkArrayNewFailed);
        return adoptUnlocked(owner, arr.?, data.len);
    }

    pub fn newZeros(ctx: *FutharkContext, len: usize, allocator: std.mem.Allocator) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newZerosUnlocked(owner, allocator, len);
    }

    pub fn newZerosUnlocked(owner: *ContextOwner, allocator: std.mem.Allocator, len: usize) AccelError!Self {
        if (len == 0) return AccelError.InvalidDimensions;
        const zeros = try allocHost(i64, allocator, len);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        return newFromSliceUnlocked(owner, zeros);
    }

    pub fn deinit(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        owner.lock();
        self.deinitUnlocked();
        owner.unlock();
    }

    pub fn deinitUnlocked(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        if (self.arr) |arr| {
            if (self.owned) {
                if (owner.requireHandleUnlocked()) |handle| {
                    _ = futhark.futhark_free_i64_1d(handle, arr);
                } else |_| {}
            }
        }
        self.clear();
        owner.release();
    }

    pub fn free(self: *Self, ctx: *FutharkContext) AccelError!void {
        try self.requireSameContext(ctx);
        self.deinit();
    }

    pub fn valuesSlice(self: *const Self, allocator: std.mem.Allocator) AccelError![]i64 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.valuesSliceUnlocked(allocator);
    }

    pub fn valuesSliceUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError![]i64 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = try allocHost(i64, allocator, self.len);
        errdefer allocator.free(buf);
        if (futhark.futhark_values_i64_1d(handle, arr, buf.ptr) != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkValuesFailed);
        }
        try owner.syncContextUnlocked();
        return buf;
    }

    pub fn isLive(self: *const Self) bool {
        if (self.arr == null or self.len == 0) return false;
        const owner = self.owner orelse return false;
        return owner.alive and owner.ctx != null;
    }

    pub fn contextId(self: *const Self) ?u64 {
        const owner = self.owner orelse return null;
        return owner.id;
    }

    pub fn requireSameContext(self: *const Self, ctx: *const FutharkContext) AccelError!void {
        const expected = ctx.ownerPointer() orelse return AccelError.UninitializedContext;
        const actual = self.owner orelse return AccelError.InvalidResourceState;
        if (expected != actual) return AccelError.ContextMismatch;
        try expected.checkLive();
    }

    pub fn requireLiveUnlocked(self: *const Self) AccelError!void {
        if (!self.isLive()) return AccelError.InvalidResourceState;
    }

    fn clear(self: *Self) void {
        self.arr = null;
        self.len = 0;
        self.owner = null;
        self.owned = false;
    }
};

pub const FutharkArray1DU64 = struct {
    arr: ?*futhark.struct_futhark_u64_1d,
    len: usize,
    owner: ?*ContextOwner,
    owned: bool,

    const Self = @This();

    fn adoptUnlocked(owner: *ContextOwner, handle: *futhark.struct_futhark_u64_1d, len: usize) Self {
        owner.retain();
        return Self{ .arr = handle, .len = len, .owner = owner, .owned = true };
    }

    pub fn newFromSlice(ctx: *FutharkContext, data: []const u64) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newFromSliceUnlocked(owner, data);
    }

    pub fn newFromSliceUnlocked(owner: *ContextOwner, data: []const u64) AccelError!Self {
        const handle = try owner.requireHandleUnlocked();
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_u64_1d(handle, data.ptr, try checkedDimension(data.len));
        if (arr == null) return owner.recordFailureUnlocked(AccelError.FutharkArrayNewFailed);
        return adoptUnlocked(owner, arr.?, data.len);
    }

    pub fn newZeros(ctx: *FutharkContext, len: usize, allocator: std.mem.Allocator) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newZerosUnlocked(owner, allocator, len);
    }

    pub fn newZerosUnlocked(owner: *ContextOwner, allocator: std.mem.Allocator, len: usize) AccelError!Self {
        if (len == 0) return AccelError.InvalidDimensions;
        const zeros = try allocHost(u64, allocator, len);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        return newFromSliceUnlocked(owner, zeros);
    }

    pub fn deinit(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        owner.lock();
        self.deinitUnlocked();
        owner.unlock();
    }

    pub fn deinitUnlocked(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        if (self.arr) |arr| {
            if (self.owned) {
                if (owner.requireHandleUnlocked()) |handle| {
                    _ = futhark.futhark_free_u64_1d(handle, arr);
                } else |_| {}
            }
        }
        self.clear();
        owner.release();
    }

    pub fn free(self: *Self, ctx: *FutharkContext) AccelError!void {
        try self.requireSameContext(ctx);
        self.deinit();
    }

    pub fn valuesSlice(self: *const Self, allocator: std.mem.Allocator) AccelError![]u64 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.valuesSliceUnlocked(allocator);
    }

    pub fn valuesSliceUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError![]u64 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = try allocHost(u64, allocator, self.len);
        errdefer allocator.free(buf);
        if (futhark.futhark_values_u64_1d(handle, arr, buf.ptr) != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkValuesFailed);
        }
        try owner.syncContextUnlocked();
        return buf;
    }

    pub fn isLive(self: *const Self) bool {
        if (self.arr == null or self.len == 0) return false;
        const owner = self.owner orelse return false;
        return owner.alive and owner.ctx != null;
    }

    pub fn contextId(self: *const Self) ?u64 {
        const owner = self.owner orelse return null;
        return owner.id;
    }

    pub fn requireSameContext(self: *const Self, ctx: *const FutharkContext) AccelError!void {
        const expected = ctx.ownerPointer() orelse return AccelError.UninitializedContext;
        const actual = self.owner orelse return AccelError.InvalidResourceState;
        if (expected != actual) return AccelError.ContextMismatch;
        try expected.checkLive();
    }

    pub fn requireLiveUnlocked(self: *const Self) AccelError!void {
        if (!self.isLive()) return AccelError.InvalidResourceState;
    }

    fn clear(self: *Self) void {
        self.arr = null;
        self.len = 0;
        self.owner = null;
        self.owned = false;
    }
};

pub const FutharkArray2DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_2d,
    rows: usize,
    cols: usize,
    owner: ?*ContextOwner,
    owned: bool,

    const Self = @This();

    fn adoptUnlocked(owner: *ContextOwner, handle: *futhark.struct_futhark_f16_2d, rows: usize, cols: usize) Self {
        owner.retain();
        return Self{ .arr = handle, .rows = rows, .cols = cols, .owner = owner, .owned = true };
    }

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, rows: usize, cols: usize) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newFromFlatUnlocked(owner, flat_data, rows, cols);
    }

    pub fn newFromFlatUnlocked(owner: *ContextOwner, flat_data: []const f16, rows: usize, cols: usize) AccelError!Self {
        const handle = try owner.requireHandleUnlocked();
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount2(rows, cols);
        if (flat_data.len != total) return AccelError.InvalidDataLength;
        const arr = futhark.futhark_new_f16_2d(
            handle,
            @ptrCast(flat_data.ptr),
            try checkedDimension(rows),
            try checkedDimension(cols),
        );
        if (arr == null) return owner.recordFailureUnlocked(AccelError.FutharkArrayNewFailed);
        return adoptUnlocked(owner, arr.?, rows, cols);
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newZerosUnlocked(owner, allocator, rows, cols);
    }

    pub fn newZerosUnlocked(owner: *ContextOwner, allocator: std.mem.Allocator, rows: usize, cols: usize) AccelError!Self {
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount2(rows, cols);
        const zeros = try allocHost(f16, allocator, total);
        defer allocator.free(zeros);
        @memset(zeros, 0.0);
        return newFromFlatUnlocked(owner, zeros, rows, cols);
    }

    pub fn deinit(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        owner.lock();
        self.deinitUnlocked();
        owner.unlock();
    }

    pub fn deinitUnlocked(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        if (self.arr) |arr| {
            if (self.owned) {
                if (owner.requireHandleUnlocked()) |handle| {
                    _ = futhark.futhark_free_f16_2d(handle, arr);
                } else |_| {}
            }
        }
        self.clear();
        owner.release();
    }

    pub fn free(self: *Self, ctx: *FutharkContext) AccelError!void {
        try self.requireSameContext(ctx);
        self.deinit();
    }

    pub fn valuesFlat(self: *const Self, allocator: std.mem.Allocator) AccelError![]f16 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.valuesFlatUnlocked(allocator);
    }

    pub fn valuesFlatUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError![]f16 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        const total = try checkedElementCount2(self.rows, self.cols);
        const buf = try allocHost(f16, allocator, total);
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_2d(handle, arr, @ptrCast(buf.ptr)) != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkValuesFailed);
        }
        try owner.syncContextUnlocked();
        return buf;
    }

    pub fn isLive(self: *const Self) bool {
        if (self.arr == null or self.rows == 0 or self.cols == 0) return false;
        const owner = self.owner orelse return false;
        return owner.alive and owner.ctx != null;
    }

    pub fn contextId(self: *const Self) ?u64 {
        const owner = self.owner orelse return null;
        return owner.id;
    }

    pub fn requireSameContext(self: *const Self, ctx: *const FutharkContext) AccelError!void {
        const expected = ctx.ownerPointer() orelse return AccelError.UninitializedContext;
        const actual = self.owner orelse return AccelError.InvalidResourceState;
        if (expected != actual) return AccelError.ContextMismatch;
        try expected.checkLive();
    }

    pub fn requireLiveUnlocked(self: *const Self) AccelError!void {
        if (!self.isLive()) return AccelError.InvalidResourceState;
    }

    fn clear(self: *Self) void {
        self.arr = null;
        self.rows = 0;
        self.cols = 0;
        self.owner = null;
        self.owned = false;
    }
};

pub const FutharkArray3DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,
    owner: ?*ContextOwner,
    owned: bool,

    const Self = @This();

    fn adoptUnlocked(owner: *ContextOwner, handle: *futhark.struct_futhark_f16_3d, d0: usize, d1: usize, d2: usize) Self {
        owner.retain();
        return Self{ .arr = handle, .dim0 = d0, .dim1 = d1, .dim2 = d2, .owner = owner, .owned = true };
    }

    pub fn newFromFlat(ctx: *FutharkContext, flat: []const f16, d0: usize, d1: usize, d2: usize) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newFromFlatUnlocked(owner, flat, d0, d1, d2);
    }

    pub fn newFromFlatUnlocked(owner: *ContextOwner, flat: []const f16, d0: usize, d1: usize, d2: usize) AccelError!Self {
        const handle = try owner.requireHandleUnlocked();
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount3(d0, d1, d2);
        if (flat.len != total) return AccelError.InvalidDataLength;
        const arr = futhark.futhark_new_f16_3d(
            handle,
            @ptrCast(flat.ptr),
            try checkedDimension(d0),
            try checkedDimension(d1),
            try checkedDimension(d2),
        );
        if (arr == null) return owner.recordFailureUnlocked(AccelError.FutharkArrayNewFailed);
        return adoptUnlocked(owner, arr.?, d0, d1, d2);
    }

    pub fn newZeros(ctx: *FutharkContext, d0: usize, d1: usize, d2: usize, allocator: std.mem.Allocator) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newZerosUnlocked(owner, allocator, d0, d1, d2);
    }

    pub fn newZerosUnlocked(owner: *ContextOwner, allocator: std.mem.Allocator, d0: usize, d1: usize, d2: usize) AccelError!Self {
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount3(d0, d1, d2);
        const zeros = try allocHost(f16, allocator, total);
        defer allocator.free(zeros);
        @memset(zeros, 0.0);
        return newFromFlatUnlocked(owner, zeros, d0, d1, d2);
    }

    pub fn deinit(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        owner.lock();
        self.deinitUnlocked();
        owner.unlock();
    }

    pub fn deinitUnlocked(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        if (self.arr) |arr| {
            if (self.owned) {
                if (owner.requireHandleUnlocked()) |handle| {
                    _ = futhark.futhark_free_f16_3d(handle, arr);
                } else |_| {}
            }
        }
        self.clear();
        owner.release();
    }

    pub fn free(self: *Self, ctx: *FutharkContext) AccelError!void {
        try self.requireSameContext(ctx);
        self.deinit();
    }

    pub fn valuesFlat(self: *const Self, allocator: std.mem.Allocator) AccelError![]f16 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.valuesFlatUnlocked(allocator);
    }

    pub fn valuesFlatUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError![]f16 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        const total = try checkedElementCount3(self.dim0, self.dim1, self.dim2);
        const buf = try allocHost(f16, allocator, total);
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_3d(handle, arr, @ptrCast(buf.ptr)) != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkValuesFailed);
        }
        try owner.syncContextUnlocked();
        return buf;
    }

    pub fn isLive(self: *const Self) bool {
        if (self.arr == null or self.dim0 == 0 or self.dim1 == 0 or self.dim2 == 0) return false;
        const owner = self.owner orelse return false;
        return owner.alive and owner.ctx != null;
    }

    pub fn contextId(self: *const Self) ?u64 {
        const owner = self.owner orelse return null;
        return owner.id;
    }

    pub fn requireSameContext(self: *const Self, ctx: *const FutharkContext) AccelError!void {
        const expected = ctx.ownerPointer() orelse return AccelError.UninitializedContext;
        const actual = self.owner orelse return AccelError.InvalidResourceState;
        if (expected != actual) return AccelError.ContextMismatch;
        try expected.checkLive();
    }

    pub fn requireLiveUnlocked(self: *const Self) AccelError!void {
        if (!self.isLive()) return AccelError.InvalidResourceState;
    }

    fn clear(self: *Self) void {
        self.arr = null;
        self.dim0 = 0;
        self.dim1 = 0;
        self.dim2 = 0;
        self.owner = null;
        self.owned = false;
    }
};

pub const FutharkArray2DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_2d,
    rows: usize,
    cols: usize,
    owner: ?*ContextOwner,
    owned: bool,

    const Self = @This();

    fn adoptUnlocked(owner: *ContextOwner, handle: *futhark.struct_futhark_f32_2d, rows: usize, cols: usize) Self {
        owner.retain();
        return Self{ .arr = handle, .rows = rows, .cols = cols, .owner = owner, .owned = true };
    }

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, rows: usize, cols: usize) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newFromFlatUnlocked(owner, data, rows, cols);
    }

    pub fn newFromFlatUnlocked(owner: *ContextOwner, data: []const f32, rows: usize, cols: usize) AccelError!Self {
        const handle = try owner.requireHandleUnlocked();
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount2(rows, cols);
        if (data.len != total) return AccelError.InvalidDataLength;
        const arr = futhark.futhark_new_f32_2d(
            handle,
            data.ptr,
            try checkedDimension(rows),
            try checkedDimension(cols),
        );
        if (arr == null) return owner.recordFailureUnlocked(AccelError.FutharkArrayNewFailed);
        return adoptUnlocked(owner, arr.?, rows, cols);
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newZerosUnlocked(owner, allocator, rows, cols);
    }

    pub fn newZerosUnlocked(owner: *ContextOwner, allocator: std.mem.Allocator, rows: usize, cols: usize) AccelError!Self {
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount2(rows, cols);
        const zeros = try allocHost(f32, allocator, total);
        defer allocator.free(zeros);
        @memset(zeros, 0.0);
        return newFromFlatUnlocked(owner, zeros, rows, cols);
    }

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return fromTensorUnlocked(owner, tensor);
    }

    pub fn fromTensorUnlocked(owner: *ContextOwner, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (tensor.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = tensor.shape.dims[0];
        const cols = tensor.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const expected = try checkedElementCount2(rows, cols);
        if (tensor.data.len != expected) return AccelError.InvalidDataLength;
        return newFromFlatUnlocked(owner, tensor.data, rows, cols);
    }

    pub fn deinit(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        owner.lock();
        self.deinitUnlocked();
        owner.unlock();
    }

    pub fn deinitUnlocked(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        if (self.arr) |arr| {
            if (self.owned) {
                if (owner.requireHandleUnlocked()) |handle| {
                    _ = futhark.futhark_free_f32_2d(handle, arr);
                } else |_| {}
            }
        }
        self.clear();
        owner.release();
    }

    pub fn free(self: *Self, ctx: *FutharkContext) AccelError!void {
        try self.requireSameContext(ctx);
        self.deinit();
    }

    pub fn valuesFlat(self: *const Self, allocator: std.mem.Allocator) AccelError![]f32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.valuesFlatUnlocked(allocator);
    }

    pub fn valuesFlatUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError![]f32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        const total = try checkedElementCount2(self.rows, self.cols);
        const buf = try allocHost(f32, allocator, total);
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f32_2d(handle, arr, buf.ptr) != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkValuesFailed);
        }
        try owner.syncContextUnlocked();
        return buf;
    }

    pub fn toTensor(self: *const Self, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.toTensorUnlocked(allocator);
    }

    pub fn toTensorUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        const values = try self.valuesFlatUnlocked(allocator);
        defer allocator.free(values);
        const shape = [_]usize{ self.rows, self.cols };
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        errdefer tensor.deinit();
        if (tensor.data.len != values.len) return AccelError.InvalidDataLength;
        @memcpy(tensor.data, values);
        return tensor;
    }

    pub fn deviceBuffer(self: *const Self) AccelError!DeviceBufferF32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.deviceBufferUnlocked();
    }

    pub fn deviceBufferUnlocked(self: *const Self) AccelError!DeviceBufferF32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        const total = try checkedElementCount2(self.rows, self.cols);
        const raw: ?*anyopaque = @as(?*anyopaque, futhark.futhark_values_raw_f32_2d(handle, arr));
        if (raw == null) return owner.recordFailureUnlocked(AccelError.NullPointer);
        owner.retain();
        return DeviceBufferF32{ .ptr = raw.?, .count = total, .context_id = owner.id, .owner = owner };
    }

    pub fn isLive(self: *const Self) bool {
        if (self.arr == null or self.rows == 0 or self.cols == 0) return false;
        const owner = self.owner orelse return false;
        return owner.alive and owner.ctx != null;
    }

    pub fn contextId(self: *const Self) ?u64 {
        const owner = self.owner orelse return null;
        return owner.id;
    }

    pub fn requireSameContext(self: *const Self, ctx: *const FutharkContext) AccelError!void {
        const expected = ctx.ownerPointer() orelse return AccelError.UninitializedContext;
        const actual = self.owner orelse return AccelError.InvalidResourceState;
        if (expected != actual) return AccelError.ContextMismatch;
        try expected.checkLive();
    }

    pub fn requireLiveUnlocked(self: *const Self) AccelError!void {
        if (!self.isLive()) return AccelError.InvalidResourceState;
    }

    fn clear(self: *Self) void {
        self.arr = null;
        self.rows = 0;
        self.cols = 0;
        self.owner = null;
        self.owned = false;
    }
};

pub const FutharkArray3DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,
    owner: ?*ContextOwner,
    owned: bool,

    const Self = @This();

    fn adoptUnlocked(owner: *ContextOwner, handle: *futhark.struct_futhark_f32_3d, d0: usize, d1: usize, d2: usize) Self {
        owner.retain();
        return Self{ .arr = handle, .dim0 = d0, .dim1 = d1, .dim2 = d2, .owner = owner, .owned = true };
    }

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, d0: usize, d1: usize, d2: usize) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newFromFlatUnlocked(owner, data, d0, d1, d2);
    }

    pub fn newFromFlatUnlocked(owner: *ContextOwner, data: []const f32, d0: usize, d1: usize, d2: usize) AccelError!Self {
        const handle = try owner.requireHandleUnlocked();
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount3(d0, d1, d2);
        if (data.len != total) return AccelError.InvalidDataLength;
        const arr = futhark.futhark_new_f32_3d(
            handle,
            data.ptr,
            try checkedDimension(d0),
            try checkedDimension(d1),
            try checkedDimension(d2),
        );
        if (arr == null) return owner.recordFailureUnlocked(AccelError.FutharkArrayNewFailed);
        return adoptUnlocked(owner, arr.?, d0, d1, d2);
    }

    pub fn newZeros(ctx: *FutharkContext, d0: usize, d1: usize, d2: usize, allocator: std.mem.Allocator) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return newZerosUnlocked(owner, allocator, d0, d1, d2);
    }

    pub fn newZerosUnlocked(owner: *ContextOwner, allocator: std.mem.Allocator, d0: usize, d1: usize, d2: usize) AccelError!Self {
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount3(d0, d1, d2);
        const zeros = try allocHost(f32, allocator, total);
        defer allocator.free(zeros);
        @memset(zeros, 0.0);
        return newFromFlatUnlocked(owner, zeros, d0, d1, d2);
    }

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        const owner = try ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        return fromTensorUnlocked(owner, tensor);
    }

    pub fn fromTensorUnlocked(owner: *ContextOwner, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (tensor.shape.dims.len != 3) return AccelError.InvalidDimensions;
        const d0 = tensor.shape.dims[0];
        const d1 = tensor.shape.dims[1];
        const d2 = tensor.shape.dims[2];
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        const expected = try checkedElementCount3(d0, d1, d2);
        if (tensor.data.len != expected) return AccelError.InvalidDataLength;
        return newFromFlatUnlocked(owner, tensor.data, d0, d1, d2);
    }

    pub fn deinit(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        owner.lock();
        self.deinitUnlocked();
        owner.unlock();
    }

    pub fn deinitUnlocked(self: *Self) void {
        const owner = self.owner orelse {
            self.clear();
            return;
        };
        if (self.arr) |arr| {
            if (self.owned) {
                if (owner.requireHandleUnlocked()) |handle| {
                    _ = futhark.futhark_free_f32_3d(handle, arr);
                } else |_| {}
            }
        }
        self.clear();
        owner.release();
    }

    pub fn free(self: *Self, ctx: *FutharkContext) AccelError!void {
        try self.requireSameContext(ctx);
        self.deinit();
    }

    pub fn valuesFlat(self: *const Self, allocator: std.mem.Allocator) AccelError![]f32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.valuesFlatUnlocked(allocator);
    }

    pub fn valuesFlatUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError![]f32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        const total = try checkedElementCount3(self.dim0, self.dim1, self.dim2);
        const buf = try allocHost(f32, allocator, total);
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f32_3d(handle, arr, buf.ptr) != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkValuesFailed);
        }
        try owner.syncContextUnlocked();
        return buf;
    }

    pub fn toTensor(self: *const Self, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.toTensorUnlocked(allocator);
    }

    pub fn toTensorUnlocked(self: *const Self, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        const values = try self.valuesFlatUnlocked(allocator);
        defer allocator.free(values);
        const shape = [_]usize{ self.dim0, self.dim1, self.dim2 };
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        errdefer tensor.deinit();
        if (tensor.data.len != values.len) return AccelError.InvalidDataLength;
        @memcpy(tensor.data, values);
        return tensor;
    }

    pub fn deviceBuffer(self: *const Self) AccelError!DeviceBufferF32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        return self.deviceBufferUnlocked();
    }

    pub fn deviceBufferUnlocked(self: *const Self) AccelError!DeviceBufferF32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const arr = self.arr orelse return AccelError.InvalidResourceState;
        const total = try checkedElementCount3(self.dim0, self.dim1, self.dim2);
        const raw: ?*anyopaque = @as(?*anyopaque, futhark.futhark_values_raw_f32_3d(handle, arr));
        if (raw == null) return owner.recordFailureUnlocked(AccelError.NullPointer);
        owner.retain();
        return DeviceBufferF32{ .ptr = raw.?, .count = total, .context_id = owner.id, .owner = owner };
    }

    pub fn isLive(self: *const Self) bool {
        if (self.arr == null or self.dim0 == 0 or self.dim1 == 0 or self.dim2 == 0) return false;
        const owner = self.owner orelse return false;
        return owner.alive and owner.ctx != null;
    }

    pub fn contextId(self: *const Self) ?u64 {
        const owner = self.owner orelse return null;
        return owner.id;
    }

    pub fn requireSameContext(self: *const Self, ctx: *const FutharkContext) AccelError!void {
        const expected = ctx.ownerPointer() orelse return AccelError.UninitializedContext;
        const actual = self.owner orelse return AccelError.InvalidResourceState;
        if (expected != actual) return AccelError.ContextMismatch;
        try expected.checkLive();
    }

    pub fn requireLiveUnlocked(self: *const Self) AccelError!void {
        if (!self.isLive()) return AccelError.InvalidResourceState;
    }

    fn clear(self: *Self) void {
        self.arr = null;
        self.dim0 = 0;
        self.dim1 = 0;
        self.dim2 = 0;
        self.owner = null;
        self.owned = false;
    }
};

fn convertMasterHandleToF16_2DUnlocked(
    owner: *ContextOwner,
    master_arr: *futhark.struct_futhark_f32_2d,
    rows: usize,
    cols: usize,
) AccelError!FutharkArray2DF16 {
    const handle = try owner.requireHandleUnlocked();
    if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
    var out: ?*futhark.struct_futhark_f16_2d = null;
    const rc = futhark.futhark_entry_master_weights_to_f16_2d(handle, &out, master_arr);
    if (rc != 0 or out == null) {
        if (out) |produced| _ = futhark.futhark_free_f16_2d(handle, produced);
        return owner.recordFailureUnlocked(AccelError.FutharkScaleWeightsFailed);
    }
    try validateShapeF16_2DUnlocked(owner, out.?, rows, cols);
    return FutharkArray2DF16.adoptUnlocked(owner, out.?, rows, cols);
}

fn convertMasterToF16_2DUnlocked(owner: *ContextOwner, master: *const FutharkArray2DF32) AccelError!FutharkArray2DF16 {
    try master.requireLiveUnlocked();
    const master_arr = master.arr orelse return AccelError.InvalidResourceState;
    return convertMasterHandleToF16_2DUnlocked(owner, master_arr, master.rows, master.cols);
}

fn convertMasterToF16_2D(ctx: *FutharkContext, master: *const FutharkArray2DF32) AccelError!FutharkArray2DF16 {
    const owner = try ctx.requireOwner();
    try master.requireSameContext(ctx);
    owner.lock();
    defer owner.unlock();
    return convertMasterToF16_2DUnlocked(owner, master);
}

fn convertMasterHandleToF16_3DUnlocked(
    owner: *ContextOwner,
    master_arr: *futhark.struct_futhark_f32_3d,
    d0: usize,
    d1: usize,
    d2: usize,
) AccelError!FutharkArray3DF16 {
    const handle = try owner.requireHandleUnlocked();
    if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
    var out: ?*futhark.struct_futhark_f16_3d = null;
    const rc = futhark.futhark_entry_master_weights_to_f16_3d(handle, &out, master_arr);
    if (rc != 0 or out == null) {
        if (out) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
        return owner.recordFailureUnlocked(AccelError.FutharkScaleWeightsFailed);
    }
    try validateShapeF16_3DUnlocked(owner, out.?, d0, d1, d2);
    return FutharkArray3DF16.adoptUnlocked(owner, out.?, d0, d1, d2);
}

fn convertMasterToF16_3DUnlocked(owner: *ContextOwner, master: *const FutharkArray3DF32) AccelError!FutharkArray3DF16 {
    try master.requireLiveUnlocked();
    const master_arr = master.arr orelse return AccelError.InvalidResourceState;
    return convertMasterHandleToF16_3DUnlocked(owner, master_arr, master.dim0, master.dim1, master.dim2);
}

fn convertMasterToF16_3D(ctx: *FutharkContext, master: *const FutharkArray3DF32) AccelError!FutharkArray3DF16 {
    const owner = try ctx.requireOwner();
    try master.requireSameContext(ctx);
    owner.lock();
    defer owner.unlock();
    return convertMasterToF16_3DUnlocked(owner, master);
}

pub const rsf_initialization_stddev: f32 = 0.25;

var seed_state_mutex: std.Thread.Mutex = .{};
var seed_state_counter: u64 = 0;

pub fn defaultSeed() u64 {
    seed_state_mutex.lock();
    defer seed_state_mutex.unlock();
    seed_state_counter +%= 1;
    const stamp: i64 = @truncate(std.time.nanoTimestamp());
    const mixed: u64 = @bitCast(stamp);
    const thread_id: u64 = @intCast(std.Thread.getCurrentId());
    return mixed ^ (seed_state_counter *% 0x9E3779B97F4A7C15) ^ (thread_id *% 0xD1B54A32D192ED03);
}

pub fn deriveSeed(model_dim: usize, num_layers: usize) u64 {
    const base: u64 = 0x4A41494445204E4F;
    const dim_part: u64 = @as(u64, @intCast(model_dim)) *% 0xBF58476D1CE4E5B9;
    const layer_part: u64 = @as(u64, @intCast(num_layers)) *% 0x94D049BB133111EB;
    return base ^ dim_part ^ layer_part;
}

pub const RSFOptimizerState = struct {
    master_weights_s: []f32 = &.{},
    master_weights_t: []f32 = &.{},
    momentum_s: []f32 = &.{},
    momentum_t: []f32 = &.{},
    fisher_blocks_s: []f32 = &.{},
    fisher_blocks_t: []f32 = &.{},
    fisher_s: []f32 = &.{},
    fisher_t: []f32 = &.{},
    step: u64 = 0,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn empty(allocator: std.mem.Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        if (self.master_weights_s.len > 0) {
            self.allocator.free(self.master_weights_s);
            self.master_weights_s = &.{};
        }
        if (self.master_weights_t.len > 0) {
            self.allocator.free(self.master_weights_t);
            self.master_weights_t = &.{};
        }
        if (self.momentum_s.len > 0) {
            self.allocator.free(self.momentum_s);
            self.momentum_s = &.{};
        }
        if (self.momentum_t.len > 0) {
            self.allocator.free(self.momentum_t);
            self.momentum_t = &.{};
        }
        if (self.fisher_blocks_s.len > 0) {
            self.allocator.free(self.fisher_blocks_s);
            self.fisher_blocks_s = &.{};
            self.fisher_s = &.{};
        } else if (self.fisher_s.len > 0) {
            self.allocator.free(self.fisher_s);
            self.fisher_s = &.{};
        }
        if (self.fisher_blocks_t.len > 0) {
            self.allocator.free(self.fisher_blocks_t);
            self.fisher_blocks_t = &.{};
            self.fisher_t = &.{};
        } else if (self.fisher_t.len > 0) {
            self.allocator.free(self.fisher_t);
            self.fisher_t = &.{};
        }
        self.step = 0;
    }

    pub fn clone(self: *const Self) AccelError!Self {
        var out = Self.empty(self.allocator);
        var committed = false;
        defer if (!committed) out.deinit();
        if (self.master_weights_s.len > 0) out.master_weights_s = try dupeHost(f32, self.allocator, self.master_weights_s);
        if (self.master_weights_t.len > 0) out.master_weights_t = try dupeHost(f32, self.allocator, self.master_weights_t);
        if (self.momentum_s.len > 0) out.momentum_s = try dupeHost(f32, self.allocator, self.momentum_s);
        if (self.momentum_t.len > 0) out.momentum_t = try dupeHost(f32, self.allocator, self.momentum_t);
        if (self.fisher_blocks_s.len > 0) {
            out.fisher_blocks_s = try dupeHost(f32, self.allocator, self.fisher_blocks_s);
            out.fisher_s = out.fisher_blocks_s;
        } else if (self.fisher_s.len > 0) {
            out.fisher_s = try dupeHost(f32, self.allocator, self.fisher_s);
        }
        if (self.fisher_blocks_t.len > 0) {
            out.fisher_blocks_t = try dupeHost(f32, self.allocator, self.fisher_blocks_t);
            out.fisher_t = out.fisher_blocks_t;
        } else if (self.fisher_t.len > 0) {
            out.fisher_t = try dupeHost(f32, self.allocator, self.fisher_t);
        }
        out.step = self.step;
        committed = true;
        return out;
    }
};

pub fn promoteDiagonalFisher(allocator: std.mem.Allocator, fisher_s: []const f32, fisher_t: []const f32) AccelError!struct { blocks_s: []f32, blocks_t: []f32 } {
    if (fisher_s.len != fisher_t.len) return AccelError.InvalidDataLength;
    if (fisher_s.len % 2 != 0) return AccelError.InvalidDataLength;
    const pairs = fisher_s.len / 2;
    const block_len = std.math.mul(usize, pairs, 3) catch return AccelError.Overflow;
    const blocks_s = try allocHost(f32, allocator, block_len);
    errdefer allocator.free(blocks_s);
    const blocks_t = try allocHost(f32, allocator, block_len);
    errdefer allocator.free(blocks_t);
    var pair: usize = 0;
    while (pair < pairs) : (pair += 1) {
        const src = pair * 2;
        const dst = pair * 3;
        blocks_s[dst] = fisher_s[src];
        blocks_s[dst + 1] = 0.0;
        blocks_s[dst + 2] = fisher_s[src + 1];
        blocks_t[dst] = fisher_t[src];
        blocks_t[dst + 1] = 0.0;
        blocks_t[dst + 2] = fisher_t[src + 1];
    }
    return .{ .blocks_s = blocks_s, .blocks_t = blocks_t };
}

pub const EmbeddingOptimizerState = struct {
    master_weights: []f32 = &.{},
    momentum: []f32 = &.{},
    fisher: []f32 = &.{},
    step: u64 = 0,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn empty(allocator: std.mem.Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        if (self.master_weights.len > 0) {
            self.allocator.free(self.master_weights);
            self.master_weights = &.{};
        }
        if (self.momentum.len > 0) {
            self.allocator.free(self.momentum);
            self.momentum = &.{};
        }
        if (self.fisher.len > 0) {
            self.allocator.free(self.fisher);
            self.fisher = &.{};
        }
        self.step = 0;
    }
};

pub const FusedStepScalars = struct {
    loss: f32,
    reconstruction_loss: f32,
    logdet_mean: f32,
};

pub const MidpointStepScalars = struct {
    collision_loss: f32,
    logdet_forward_mean: f32,
    logdet_backward_mean: f32,
};

pub const FusedStepState = enum {
    pending,
    finalized,
    failed,
    deinitialized,
};

fn freeFusedTupleUnlocked(owner: *ContextOwner, tuple: *futhark.struct_futhark_opaque_tup6_fused_stack_gradients) void {
    if (owner.requireHandleUnlocked()) |handle| {
        _ = futhark.futhark_free_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(handle, tuple);
    } else |_| {}
}

fn freeStackSfdTupleUnlocked(owner: *ContextOwner, tuple: *futhark.struct_futhark_opaque_tup3_stack_sfd) void {
    if (owner.requireHandleUnlocked()) |handle| {
        _ = futhark.futhark_free_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32(handle, tuple);
    } else |_| {}
}

fn freeStackSpectralTupleUnlocked(owner: *ContextOwner, tuple: *futhark.struct_futhark_opaque_tup3_stack_spectral) void {
    if (owner.requireHandleUnlocked()) |handle| {
        _ = futhark.futhark_free_opaque_tup3_arr3d_f32_f32_f32(handle, tuple);
    } else |_| {}
}

pub const FusedStepResult = struct {
    owner: ?*ContextOwner,
    stack_gradient_s: FutharkArray3DF32,
    stack_gradient_t: FutharkArray3DF32,
    input_delta: FutharkArray3DF16,
    pending: ?*futhark.struct_futhark_opaque_tup6_fused_stack_gradients,
    state: FusedStepState,
    scalars: FusedStepScalars,
    midpoint: ?MidpointStepScalars,
    map_midpoint: bool,
    failure: ?AccelError,

    const Self = @This();

    fn initOwned(
        owner: *ContextOwner,
        gradient_s: *futhark.struct_futhark_f32_3d,
        gradient_t: *futhark.struct_futhark_f32_3d,
        delta: *futhark.struct_futhark_f16_3d,
        tuple: *futhark.struct_futhark_opaque_tup6_fused_stack_gradients,
        layers: usize,
        half: usize,
        columns: usize,
        batch: usize,
        time: usize,
        features: usize,
    ) Self {
        owner.retain();
        return Self{
            .owner = owner,
            .stack_gradient_s = FutharkArray3DF32.adoptUnlocked(owner, gradient_s, layers, half, columns),
            .stack_gradient_t = FutharkArray3DF32.adoptUnlocked(owner, gradient_t, layers, half, columns),
            .input_delta = FutharkArray3DF16.adoptUnlocked(owner, delta, batch, time, features),
            .pending = tuple,
            .state = .pending,
            .scalars = .{ .loss = 0.0, .reconstruction_loss = 0.0, .logdet_mean = 0.0 },
            .midpoint = null,
            .map_midpoint = false,
            .failure = null,
        };
    }

    pub fn finalize(self: *Self) AccelError!FusedStepScalars {
        if (self.owner) |owner| {
            owner.lock();
            defer owner.unlock();
            return self.finalizeUnlocked();
        }
        return self.finalizeWithoutContext();
    }

    fn finalizeWithoutContext(self: *Self) AccelError!FusedStepScalars {
        switch (self.state) {
            .finalized => return self.scalars,
            .failed => return self.failure orelse AccelError.FutharkTrainingStepFailed,
            .deinitialized => return AccelError.InvalidResourceState,
            .pending => return AccelError.InvalidResourceState,
        }
    }

    pub fn finalizeUnlocked(self: *Self) AccelError!FusedStepScalars {
        switch (self.state) {
            .finalized => return self.scalars,
            .failed => return self.failure orelse AccelError.FutharkTrainingStepFailed,
            .deinitialized => return AccelError.InvalidResourceState,
            .pending => {},
        }
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        const handle = try owner.requireHandleUnlocked();
        const tuple = self.pending orelse return AccelError.InvalidResourceState;
        var loss: f32 = 0.0;
        var reconstruction: f32 = 0.0;
        var logdet: f32 = 0.0;
        const p3 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_3(handle, &loss, tuple);
        const p4 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_4(handle, &reconstruction, tuple);
        const p5 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_5(handle, &logdet, tuple);
        const sync_status = owner.syncContextUnlocked();
        freeFusedTupleUnlocked(owner, tuple);
        self.pending = null;
        if (p3 != 0 or p4 != 0 or p5 != 0) {
            self.state = .failed;
            self.failure = AccelError.FutharkTrainingStepFailed;
            return self.failure.?;
        }
        if (sync_status) |_| {} else |sync_error| {
            self.state = .failed;
            self.failure = sync_error;
            return sync_error;
        }
        if (!std.math.isFinite(loss) or !std.math.isFinite(reconstruction) or !std.math.isFinite(logdet)) {
            self.state = .failed;
            self.failure = AccelError.FutharkTrainingStepFailed;
            return self.failure.?;
        }
        if (self.map_midpoint) {
            self.midpoint = .{
                .collision_loss = loss,
                .logdet_forward_mean = reconstruction,
                .logdet_backward_mean = logdet,
            };
            self.scalars = .{
                .loss = loss,
                .reconstruction_loss = loss,
                .logdet_mean = reconstruction + logdet,
            };
        } else {
            self.scalars = .{ .loss = loss, .reconstruction_loss = reconstruction, .logdet_mean = logdet };
        }
        self.state = .finalized;
        return self.scalars;
    }

    pub fn deinit(self: *Self) void {
        const owner = self.owner orelse {
            self.reset();
            return;
        };
        owner.lock();
        self.deinitUnlocked();
        owner.unlock();
    }

    pub fn deinitUnlocked(self: *Self) void {
        const owner = self.owner orelse {
            self.reset();
            return;
        };
        if (self.state != .deinitialized) {
            self.stack_gradient_t.deinitUnlocked();
            self.stack_gradient_s.deinitUnlocked();
            self.input_delta.deinitUnlocked();
            if (self.pending) |tuple| {
                freeFusedTupleUnlocked(owner, tuple);
                self.pending = null;
            }
        }
        self.reset();
        owner.release();
    }

    pub fn gradientDeviceBuffers(self: *Self) AccelError![2]DeviceBufferF32 {
        const owner = self.owner orelse return AccelError.InvalidResourceState;
        owner.lock();
        defer owner.unlock();
        if (self.state == .deinitialized) return AccelError.InvalidResourceState;
        var first = try self.stack_gradient_s.deviceBufferUnlocked();
        const second = self.stack_gradient_t.deviceBufferUnlocked() catch |err| {
            first.deinit();
            return err;
        };
        return .{ first, second };
    }

    pub fn gradientsLive(self: *const Self) bool {
        if (self.state == .deinitialized) return false;
        return self.stack_gradient_s.isLive() and self.stack_gradient_t.isLive();
    }

    pub fn inputDeltaLive(self: *const Self) bool {
        if (self.state == .deinitialized) return false;
        return self.input_delta.isLive();
    }

    pub fn stateKind(self: *const Self) FusedStepState {
        return self.state;
    }

    pub fn failureCode(self: *const Self) ?AccelError {
        return self.failure;
    }

    pub fn contextId(self: *const Self) ?u64 {
        const owner = self.owner orelse return null;
        return owner.id;
    }

    fn reset(self: *Self) void {
        self.pending = null;
        self.state = .deinitialized;
        self.owner = null;
        self.map_midpoint = false;
        self.midpoint = null;
        self.failure = null;
    }
};

const NormalizedStack = struct {
    array: FutharkArray3DF32,
    before: f32,
    after: f32,
};

fn checkStackArray3D(array: anytype, d0: usize, d1: usize, d2: usize, context_id: u64) AccelError!void {
    const value = array orelse return AccelError.InvalidResourceState;
    if (!value.isLive()) return AccelError.InvalidResourceState;
    const observed = value.contextId() orelse return AccelError.InvalidResourceState;
    if (observed != context_id) return AccelError.ContextMismatch;
    if (value.dim0 != d0 or value.dim1 != d1 or value.dim2 != d2) return AccelError.InvalidDimensions;
}

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    owner: ?*ContextOwner,
    allocator: std.mem.Allocator,
    model_dim: usize,
    num_layers: usize,
    clip_min: f16,
    clip_max: f16,
    diffusion_enabled: bool,
    diffusion_radix: i64,
    diffusion_block: i64,
    diffusion_stages: i64,
    initialized: bool,
    optimizer_step: u64,
    last_spectral_before_s: f32,
    last_spectral_after_s: f32,
    last_spectral_before_t: f32,
    last_spectral_after_t: f32,
    scratch_lengths_buf: []i64,
    scratch_lengths_cap: usize,
    stack_weights_s: ?FutharkArray3DF16,
    stack_weights_t: ?FutharkArray3DF16,
    stack_master_weights_s: ?FutharkArray3DF32,
    stack_master_weights_t: ?FutharkArray3DF32,
    stack_momentum_s: ?FutharkArray3DF32,
    stack_momentum_t: ?FutharkArray3DF32,
    stack_fisher_s: ?FutharkArray3DF32,
    stack_fisher_t: ?FutharkArray3DF32,
    cached_mask: ?*futhark.struct_futhark_u8_2d,
    cached_mask_id: usize,
    cached_mask_seq: usize,
    cached_mask_nnz: usize,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        return initMultiLayer(model_dim, 1, std.heap.page_allocator);
    }

    pub fn initMultiLayer(model_dim: usize, num_layers: usize, allocator: std.mem.Allocator) AccelError!Self {
        return initMultiLayerWithDepthScale(model_dim, num_layers, allocator, true);
    }

    pub fn initMultiLayerWithDepthScale(
        model_dim: usize,
        num_layers: usize,
        allocator: std.mem.Allocator,
        depth_compensation: bool,
    ) AccelError!Self {
        return initMultiLayerWithSeed(model_dim, num_layers, allocator, depth_compensation, deriveSeed(model_dim, num_layers));
    }

    pub fn initMultiLayerWithSeed(
        model_dim: usize,
        num_layers: usize,
        allocator: std.mem.Allocator,
        depth_compensation: bool,
        seed: u64,
    ) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;
        if (model_dim % 2 != 0) return AccelError.InvalidDimensions;
        if (num_layers == 0) return AccelError.InvalidDimensions;
        if (rsf_bias_column >= rsf_coupling_width) return AccelError.InvalidDimensions;
        const half = model_dim / 2;
        const columns = rsf_coupling_width;
        _ = try checkedDimension(model_dim);
        _ = try checkedDimension(num_layers);
        _ = try checkedDimension(half);
        _ = try checkedDimension(columns);
        const per_layer = try checkedElementCount2(half, columns);
        const stack_count = try checkedElementCount2(num_layers, per_layer);
        _ = try checkedByteCount(stack_count, f32);
        _ = try checkedByteCount(stack_count, f16);

        var ctx = try FutharkContext.initWithAllocator(allocator);
        errdefer ctx.deinit();
        const owner = try ctx.retainOwner();
        errdefer owner.release();

        const depth_scale: f32 = if (depth_compensation)
            1.0 / @sqrt(@as(f32, @floatFromInt(num_layers)))
        else
            1.0;
        const init_stddev: f32 = depth_scale * rsf_initialization_stddev;

        const master_s_data = try allocHost(f32, allocator, stack_count);
        defer allocator.free(master_s_data);
        const master_t_data = try allocHost(f32, allocator, stack_count);
        defer allocator.free(master_t_data);

        var layer_index: usize = 0;
        while (layer_index < num_layers) : (layer_index += 1) {
            const layer_seed = seed +% (@as(u64, @intCast(layer_index)) *% 0x9E3779B97F4A7C15) +% (@as(u64, @intCast(layer_index)) << 32);
            var rng = std.Random.DefaultPrng.init(layer_seed);
            const random = rng.random();
            const base = layer_index * per_layer;
            var index: usize = 0;
            while (index < per_layer) : (index += 1) {
                master_s_data[base + index] = random.floatNorm(f32) * init_stddev;
                master_t_data[base + index] = random.floatNorm(f32) * init_stddev;
            }
            var row: usize = 0;
            while (row < half) : (row += 1) {
                const bias_index = base + row * columns + rsf_bias_column;
                master_s_data[bias_index] = 0.0;
                master_t_data[bias_index] = 0.0;
            }
        }

        var stack_master_s = try FutharkArray3DF32.newFromFlat(&ctx, master_s_data, num_layers, half, columns);
        errdefer stack_master_s.deinit();
        var stack_master_t = try FutharkArray3DF32.newFromFlat(&ctx, master_t_data, num_layers, half, columns);
        errdefer stack_master_t.deinit();
        var stack_shadow_s = try convertMasterToF16_3D(&ctx, &stack_master_s);
        errdefer stack_shadow_s.deinit();
        var stack_shadow_t = try convertMasterToF16_3D(&ctx, &stack_master_t);
        errdefer stack_shadow_t.deinit();
        var momentum_s = try FutharkArray3DF32.newZeros(&ctx, num_layers, half, columns, allocator);
        errdefer momentum_s.deinit();
        var momentum_t = try FutharkArray3DF32.newZeros(&ctx, num_layers, half, columns, allocator);
        errdefer momentum_t.deinit();
        var fisher_s = try FutharkArray3DF32.newZeros(&ctx, num_layers, half, rsf_fisher_width, allocator);
        errdefer fisher_s.deinit();
        var fisher_t = try FutharkArray3DF32.newZeros(&ctx, num_layers, half, rsf_fisher_width, allocator);
        errdefer fisher_t.deinit();

        var diffusion_enabled = false;
        var diffusion_radix: i64 = 1;
        var diffusion_block: i64 = try checkedSigned(model_dim);
        var diffusion_stages: i64 = 0;
        if (types.rsfDiffusionLayout(model_dim)) |layout| {
            diffusion_enabled = true;
            diffusion_radix = try checkedSigned(layout.radix);
            diffusion_block = try checkedSigned(layout.block);
            diffusion_stages = try checkedSigned(layout.stages);
        }

        var accelerator = Self{
            .ctx = ctx,
            .owner = owner,
            .allocator = allocator,
            .model_dim = model_dim,
            .num_layers = num_layers,
            .clip_min = rsf_default_clip_min,
            .clip_max = rsf_default_clip_max,
            .diffusion_enabled = diffusion_enabled,
            .diffusion_radix = diffusion_radix,
            .diffusion_block = diffusion_block,
            .diffusion_stages = diffusion_stages,
            .initialized = true,
            .optimizer_step = 0,
            .last_spectral_before_s = 0.0,
            .last_spectral_after_s = 0.0,
            .last_spectral_before_t = 0.0,
            .last_spectral_after_t = 0.0,
            .scratch_lengths_buf = &.{},
            .scratch_lengths_cap = 0,
            .stack_weights_s = stack_shadow_s,
            .stack_weights_t = stack_shadow_t,
            .stack_master_weights_s = stack_master_s,
            .stack_master_weights_t = stack_master_t,
            .stack_momentum_s = momentum_s,
            .stack_momentum_t = momentum_t,
            .stack_fisher_s = fisher_s,
            .stack_fisher_t = fisher_t,
            .cached_mask = null,
            .cached_mask_id = 0,
            .cached_mask_seq = 0,
            .cached_mask_nnz = 0,
        };
        try accelerator.checkInvariants();
        return accelerator;
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        self.freeStackArrays();
        if (self.scratch_lengths_buf.len > 0) {
            self.allocator.free(self.scratch_lengths_buf);
        }
        self.scratch_lengths_buf = &.{};
        self.scratch_lengths_cap = 0;
        if (self.owner) |owner| {
            owner.release();
            self.owner = null;
        }
        self.ctx.deinit();
        self.initialized = false;
    }

    fn freeStackArrays(self: *Self) void {
        if (self.stack_fisher_t) |*array| array.deinit();
        if (self.stack_fisher_s) |*array| array.deinit();
        if (self.stack_momentum_t) |*array| array.deinit();
        if (self.stack_momentum_s) |*array| array.deinit();
        if (self.stack_master_weights_t) |*array| array.deinit();
        if (self.stack_master_weights_s) |*array| array.deinit();
        if (self.stack_weights_t) |*array| array.deinit();
        if (self.stack_weights_s) |*array| array.deinit();
        self.stack_fisher_t = null;
        self.stack_fisher_s = null;
        self.stack_momentum_t = null;
        self.stack_momentum_s = null;
        self.stack_master_weights_t = null;
        self.stack_master_weights_s = null;
        self.stack_weights_t = null;
        self.stack_weights_s = null;
        self.releaseCachedMask();
    }

    fn releaseCachedMask(self: *Self) void {
        const arr = self.cached_mask orelse return;
        if (self.owner) |owner| {
            owner.lock();
            defer owner.unlock();
            if (owner.requireHandleUnlocked()) |handle| {
                _ = futhark.futhark_free_u8_2d(handle, arr);
            } else |_| {}
        }
        self.cached_mask = null;
        self.cached_mask_id = 0;
        self.cached_mask_seq = 0;
        self.cached_mask_nnz = 0;
    }

    fn releaseCachedMaskUnlocked(self: *Self, owner: *ContextOwner) void {
        if (self.cached_mask) |arr| {
            if (owner.requireHandleUnlocked()) |handle| {
                _ = futhark.futhark_free_u8_2d(handle, arr);
            } else |_| {}
        }
        self.cached_mask = null;
        self.cached_mask_id = 0;
        self.cached_mask_seq = 0;
        self.cached_mask_nnz = 0;
    }

    fn uploadMaskUnlocked(self: *Self, owner: *ContextOwner, mask: *const types.RSFSequenceMask) AccelError!*futhark.struct_futhark_u8_2d {
        const id = @intFromPtr(mask);
        if (self.cached_mask) |existing| {
            if (self.cached_mask_id == id and self.cached_mask_seq == mask.seq_len and self.cached_mask_nnz == mask.nnz) {
                return existing;
            }
        }
        self.releaseCachedMaskUnlocked(owner);
        const bytes = mask.toBytes(self.allocator) catch return AccelError.AllocationFailed;
        defer self.allocator.free(bytes);
        try self.validateMaskBytes(bytes, mask.seq_len);
        const handle = try owner.requireHandleUnlocked();
        const seq_i64 = try checkedDimension(mask.seq_len);
        const mask_arr = futhark.futhark_new_u8_2d(handle, bytes.ptr, seq_i64, seq_i64) orelse return owner.recordFailureUnlocked(AccelError.FutharkArrayNewFailed);
        self.cached_mask = mask_arr;
        self.cached_mask_id = id;
        self.cached_mask_seq = mask.seq_len;
        self.cached_mask_nnz = mask.nnz;
        return mask_arr;
    }

    pub fn numLayers(self: *const Self) usize {
        return self.num_layers;
    }

    pub fn modelDim(self: *const Self) usize {
        return self.model_dim;
    }

    pub fn optimizerStep(self: *const Self) u64 {
        return self.optimizer_step;
    }

    pub fn clipRange(self: *const Self) [2]f16 {
        return .{ self.clip_min, self.clip_max };
    }

    pub fn spectralBefore(self: *const Self) f32 {
        return @max(self.last_spectral_before_s, self.last_spectral_before_t);
    }

    pub fn spectralAfter(self: *const Self) f32 {
        return @max(self.last_spectral_after_s, self.last_spectral_after_t);
    }

    pub fn isInitialized(self: *const Self) bool {
        return self.initialized;
    }

    fn requireOwnerLive(self: *const Self) AccelError!*ContextOwner {
        if (!self.initialized) return AccelError.UninitializedContext;
        const owner = self.owner orelse return AccelError.UninitializedContext;
        try owner.checkLive();
        return owner;
    }

    fn requireOwnerLiveUnlocked(self: *const Self) AccelError!*ContextOwner {
        if (!self.initialized) return AccelError.UninitializedContext;
        const owner = self.owner orelse return AccelError.UninitializedContext;
        try owner.checkLiveUnlocked();
        return owner;
    }

    pub fn checkInvariants(self: *const Self) AccelError!void {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        return self.checkInvariantsWithOwnerUnlocked(owner);
    }

    fn checkInvariantsUnlocked(self: *const Self) AccelError!void {
        const owner = try self.requireOwnerLiveUnlocked();
        return self.checkInvariantsWithOwnerUnlocked(owner);
    }

    fn checkInvariantsWithOwnerUnlocked(self: *const Self, owner: *ContextOwner) AccelError!void {
        if (!self.initialized) return AccelError.UninitializedContext;
        if (self.owner != owner) return AccelError.ContextMismatch;
        if (!owner.alive or owner.ctx == null) return AccelError.UninitializedContext;
        if (self.model_dim == 0 or self.model_dim % 2 != 0) return AccelError.InvalidDimensions;
        if (self.num_layers == 0) return AccelError.InvalidDimensions;
        if (self.scratch_lengths_buf.len != self.scratch_lengths_cap) return AccelError.InvalidResourceState;
        const minimum: f32 = @floatCast(self.clip_min);
        const maximum: f32 = @floatCast(self.clip_max);
        if (!std.math.isFinite(minimum) or !std.math.isFinite(maximum)) return AccelError.InvalidClipRange;
        if (minimum >= maximum) return AccelError.InvalidClipRange;
        const half = self.model_dim / 2;
        const columns = rsf_coupling_width;
        try checkStackArray3D(self.stack_weights_s, self.num_layers, half, columns, owner.id);
        try checkStackArray3D(self.stack_weights_t, self.num_layers, half, columns, owner.id);
        try checkStackArray3D(self.stack_master_weights_s, self.num_layers, half, columns, owner.id);
        try checkStackArray3D(self.stack_master_weights_t, self.num_layers, half, columns, owner.id);
        try checkStackArray3D(self.stack_momentum_s, self.num_layers, half, columns, owner.id);
        try checkStackArray3D(self.stack_momentum_t, self.num_layers, half, columns, owner.id);
        try checkStackArray3D(self.stack_fisher_s, self.num_layers, half, rsf_fisher_width, owner.id);
        try checkStackArray3D(self.stack_fisher_t, self.num_layers, half, rsf_fisher_width, owner.id);
    }

    fn ensureScratchLengthsUnlocked(self: *Self, count: usize) AccelError![]i64 {
        if (count == 0) return AccelError.InvalidDimensions;
        if (count > self.scratch_lengths_cap) {
            var capacity = self.scratch_lengths_cap;
            if (capacity == 0) capacity = rsf_scratch_initial_capacity;
            while (capacity < count) {
                capacity = std.math.mul(usize, capacity, 2) catch return AccelError.Overflow;
            }
            const buffer = try allocHost(i64, self.allocator, capacity);
            if (self.scratch_lengths_buf.len > 0) {
                @memcpy(buffer[0..self.scratch_lengths_buf.len], self.scratch_lengths_buf);
                self.allocator.free(self.scratch_lengths_buf);
            }
            self.scratch_lengths_buf = buffer;
            self.scratch_lengths_cap = capacity;
        }
        return self.scratch_lengths_buf[0..count];
    }

    pub fn forward(self: *Self, input: *FutharkArray2DF16) AccelError!FutharkArray2DF16 {
        const owner = try self.requireOwnerLive();
        try input.requireSameContext(&self.ctx);
        if (input.rows == 0 or input.cols == 0) return AccelError.InvalidDimensions;
        if (input.cols != self.model_dim) return AccelError.InvalidDimensions;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        try input.requireLiveUnlocked();
        if (input.rows == 0 or input.cols == 0 or input.cols != self.model_dim) return AccelError.InvalidDimensions;

        const half = self.model_dim / 2;
        const columns = rsf_coupling_width;
        const per_layer = try checkedElementCount2(half, columns);
        const stack_count = try checkedElementCount2(self.num_layers, per_layer);
        const weights_s = try (self.stack_weights_s orelse return AccelError.InvalidResourceState).valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(weights_s);
        const weights_t = try (self.stack_weights_t orelse return AccelError.InvalidResourceState).valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(weights_t);
        if (weights_s.len != stack_count or weights_t.len != stack_count) return AccelError.InvalidDimensions;

        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        var produced: ?FutharkArray2DF16 = null;
        errdefer if (produced) |*current| current.deinitUnlocked();
        var layer_index: usize = 0;
        while (layer_index < self.num_layers) : (layer_index += 1) {
            const start = layer_index * per_layer;
            var layer_s = try FutharkArray2DF16.newFromFlatUnlocked(owner, weights_s[start .. start + per_layer], half, columns);
            defer layer_s.deinitUnlocked();
            var layer_t = try FutharkArray2DF16.newFromFlatUnlocked(owner, weights_t[start .. start + per_layer], half, columns);
            defer layer_t.deinitUnlocked();
            const source: *const FutharkArray2DF16 = if (produced) |*current| current else input;
            const next = try rsfForwardLayerUnlocked(owner, source, &layer_s, &layer_t, clip_min_bits, clip_max_bits, self.diffusion_enabled, self.diffusion_radix, self.diffusion_block, self.diffusion_stages);
            if (produced) |*previous| previous.deinitUnlocked();
            produced = next;
        }
        const result = produced orelse return AccelError.InvalidResourceState;
        produced = null;
        return result;
    }

    pub fn stackForward(self: *Self, inputs: *FutharkArray3DF16) AccelError!FutharkArray3DF16 {
        const owner = try self.requireOwnerLive();
        try inputs.requireSameContext(&self.ctx);
        if (inputs.dim0 == 0 or inputs.dim1 == 0 or inputs.dim2 == 0) return AccelError.InvalidDimensions;
        if (inputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        try inputs.requireLiveUnlocked();
        if (inputs.dim0 == 0 or inputs.dim1 == 0 or inputs.dim2 == 0 or inputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;

        const weights_s = self.stack_weights_s orelse return AccelError.InvalidResourceState;
        const weights_t = self.stack_weights_t orelse return AccelError.InvalidResourceState;
        try weights_s.requireLiveUnlocked();
        try weights_t.requireLiveUnlocked();
        const handle = try owner.requireHandleUnlocked();
        var out: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_rsf_stack_forward(
            handle,
            &out,
            inputs.arr,
            weights_s.arr,
            weights_t.arr,
            @bitCast(self.clip_min),
            @bitCast(self.clip_max),
            self.diffusion_enabled,
            self.diffusion_radix,
            self.diffusion_block,
            self.diffusion_stages,
        );
        if (rc != 0 or out == null) {
            if (out) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkForwardFailed);
        }
        try validateShapeF16_3DUnlocked(owner, out.?, inputs.dim0, inputs.dim1, inputs.dim2);
        return FutharkArray3DF16.adoptUnlocked(owner, out.?, inputs.dim0, inputs.dim1, inputs.dim2);
    }

    pub fn stackInverse(self: *Self, outputs: *FutharkArray3DF16) AccelError!FutharkArray3DF16 {
        const owner = try self.requireOwnerLive();
        try outputs.requireSameContext(&self.ctx);
        if (outputs.dim0 == 0 or outputs.dim1 == 0 or outputs.dim2 == 0) return AccelError.InvalidDimensions;
        if (outputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        try outputs.requireLiveUnlocked();
        if (outputs.dim0 == 0 or outputs.dim1 == 0 or outputs.dim2 == 0 or outputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;

        const weights_s = self.stack_weights_s orelse return AccelError.InvalidResourceState;
        const weights_t = self.stack_weights_t orelse return AccelError.InvalidResourceState;
        try weights_s.requireLiveUnlocked();
        try weights_t.requireLiveUnlocked();
        const handle = try owner.requireHandleUnlocked();
        var out: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_rsf_stack_inverse(
            handle,
            &out,
            outputs.arr,
            weights_s.arr,
            weights_t.arr,
            @bitCast(self.clip_min),
            @bitCast(self.clip_max),
            self.diffusion_enabled,
            self.diffusion_radix,
            self.diffusion_block,
            self.diffusion_stages,
        );
        if (rc != 0 or out == null) {
            if (out) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkInverseFailed);
        }
        try validateShapeF16_3DUnlocked(owner, out.?, outputs.dim0, outputs.dim1, outputs.dim2);
        return FutharkArray3DF16.adoptUnlocked(owner, out.?, outputs.dim0, outputs.dim1, outputs.dim2);
    }

    pub fn fusedTrainingStep(
        self: *Self,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        sequence_lengths: []const usize,
        grad_mean: bool,
        gradient_scale: f32,
        reconstruction_alpha: f32,
        forward_scale: f32,
        logdet_weight: f32,
    ) AccelError!FusedStepResult {
        const owner = try self.requireOwnerLive();
        try inputs.requireSameContext(&self.ctx);
        try targets.requireSameContext(&self.ctx);
        if (!std.math.isFinite(gradient_scale) or gradient_scale < 0.0 or gradient_scale > 1.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(reconstruction_alpha) or reconstruction_alpha < 0.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(forward_scale) or forward_scale < 0.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(logdet_weight) or logdet_weight < 0.0) return AccelError.InvalidHyperparameter;
        if (inputs.dim0 == 0 or inputs.dim1 == 0 or inputs.dim2 == 0) return AccelError.InvalidDimensions;
        if (inputs.dim0 != targets.dim0 or inputs.dim1 != targets.dim1 or inputs.dim2 != targets.dim2) return AccelError.InvalidDimensions;
        if (inputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        if (sequence_lengths.len != inputs.dim0) return AccelError.InvalidDimensions;

        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        try inputs.requireLiveUnlocked();
        try targets.requireLiveUnlocked();
        if (inputs.dim0 == 0 or inputs.dim1 == 0 or inputs.dim2 == 0) return AccelError.InvalidDimensions;
        if (inputs.dim0 != targets.dim0 or inputs.dim1 != targets.dim1 or inputs.dim2 != targets.dim2) return AccelError.InvalidDimensions;
        if (inputs.dim2 != self.model_dim or sequence_lengths.len != inputs.dim0) return AccelError.InvalidDimensions;

        const lengths_i64 = try self.ensureScratchLengthsUnlocked(sequence_lengths.len);
        for (sequence_lengths, lengths_i64) |length, *target| {
            if (length == 0) return AccelError.InvalidSequenceLength;
            if (length > inputs.dim1) return AccelError.InvalidSequenceLength;
            target.* = try checkedSigned(length);
        }
        var lengths_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, lengths_i64);
        defer lengths_array.deinitUnlocked();

        const weights_s = self.stack_weights_s orelse return AccelError.InvalidResourceState;
        const weights_t = self.stack_weights_t orelse return AccelError.InvalidResourceState;
        try weights_s.requireLiveUnlocked();
        try weights_t.requireLiveUnlocked();
        const handle = try owner.requireHandleUnlocked();
        const clip_min_f32: f32 = @floatCast(self.clip_min);
        const clip_max_f32: f32 = @floatCast(self.clip_max);

        var final_outputs: ?*futhark.struct_futhark_f16_3d = null;
        const forward_rc = futhark.futhark_entry_rsf_stack_forward(
            handle,
            &final_outputs,
            inputs.arr,
            weights_s.arr,
            weights_t.arr,
            @bitCast(self.clip_min),
            @bitCast(self.clip_max),
            self.diffusion_enabled,
            self.diffusion_radix,
            self.diffusion_block,
            self.diffusion_stages,
        );
        if (forward_rc != 0 or final_outputs == null) {
            if (final_outputs) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkForwardFailed);
        }
        defer _ = futhark.futhark_free_f16_3d(handle, final_outputs.?);

        var tuple: ?*futhark.struct_futhark_opaque_tup6_fused_stack_gradients = null;
        const backward_rc = futhark.futhark_entry_rsf_stack_backward_gradients_fused(
            handle,
            &tuple,
            final_outputs.?,
            targets.arr,
            inputs.arr,
            lengths_array.arr,
            weights_s.arr,
            weights_t.arr,
            grad_mean,
            gradient_scale,
            clip_min_f32,
            clip_max_f32,
            reconstruction_alpha,
            forward_scale,
            logdet_weight,
            self.diffusion_enabled,
            self.diffusion_radix,
            self.diffusion_block,
            self.diffusion_stages,
        );
        if (backward_rc != 0 or tuple == null) {
            if (tuple) |produced| freeFusedTupleUnlocked(owner, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkTrainingStepFailed);
        }
        const pending_tuple = tuple.?;

        var gradient_s: ?*futhark.struct_futhark_f32_3d = null;
        var gradient_t: ?*futhark.struct_futhark_f32_3d = null;
        var delta: ?*futhark.struct_futhark_f16_3d = null;
        const p0 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_0(handle, &gradient_s, pending_tuple);
        const p1 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_1(handle, &gradient_t, pending_tuple);
        const p2 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_2(handle, &delta, pending_tuple);
        if (p0 != 0 or p1 != 0 or p2 != 0 or gradient_s == null or gradient_t == null or delta == null) {
            if (gradient_s) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
            if (gradient_t) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
            if (delta) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
            freeFusedTupleUnlocked(owner, pending_tuple);
            return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
        }
        const half = self.model_dim / 2;
        const columns = rsf_coupling_width;
        const shapes_valid = blk: {
            validateShapeF32_3DUnlocked(owner, gradient_s.?, self.num_layers, half, columns) catch break :blk false;
            validateShapeF32_3DUnlocked(owner, gradient_t.?, self.num_layers, half, columns) catch break :blk false;
            validateShapeF16_3DUnlocked(owner, delta.?, inputs.dim0, inputs.dim1, inputs.dim2) catch break :blk false;
            break :blk true;
        };
        if (!shapes_valid) {
            _ = futhark.futhark_free_f32_3d(handle, gradient_s.?);
            _ = futhark.futhark_free_f32_3d(handle, gradient_t.?);
            _ = futhark.futhark_free_f16_3d(handle, delta.?);
            freeFusedTupleUnlocked(owner, pending_tuple);
            return AccelError.FutharkShapeFailed;
        }
        return FusedStepResult.initOwned(
            owner,
            gradient_s.?,
            gradient_t.?,
            delta.?,
            pending_tuple,
            self.num_layers,
            half,
            columns,
            inputs.dim0,
            inputs.dim1,
            inputs.dim2,
        );
    }

    pub fn applyStackGradientsSFD(
        self: *Self,
        gradient_s: *FutharkArray3DF32,
        gradient_t: *FutharkArray3DF32,
        learning_rate: f32,
        momentum_beta: f32,
        fisher_gamma: f32,
        epsilon: f32,
        trust_ratio: f32,
        weight_floor: f32,
    ) AccelError!void {
        const owner = try self.requireOwnerLive();
        try gradient_s.requireSameContext(&self.ctx);
        try gradient_t.requireSameContext(&self.ctx);
        if (!validateSFDHyperparameters(learning_rate, momentum_beta, fisher_gamma, epsilon, trust_ratio, weight_floor)) {
            return AccelError.InvalidHyperparameter;
        }
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        try gradient_s.requireLiveUnlocked();
        try gradient_t.requireLiveUnlocked();
        const half = self.model_dim / 2;
        const columns = rsf_coupling_width;
        if (gradient_s.dim0 != self.num_layers or gradient_s.dim1 != half or gradient_s.dim2 != columns) return AccelError.InvalidDimensions;
        if (gradient_t.dim0 != self.num_layers or gradient_t.dim1 != half or gradient_t.dim2 != columns) return AccelError.InvalidDimensions;
        if (self.optimizer_step >= @as(u64, @intCast(std.math.maxInt(i64)))) return AccelError.OptimizerStepOverflow;
        const next_step: i64 = try checkedSigned(self.optimizer_step + 1);

        const master_s = self.stack_master_weights_s orelse return AccelError.InvalidResourceState;
        const master_t = self.stack_master_weights_t orelse return AccelError.InvalidResourceState;
        const momentum_s = self.stack_momentum_s orelse return AccelError.InvalidResourceState;
        const momentum_t = self.stack_momentum_t orelse return AccelError.InvalidResourceState;
        const fisher_s = self.stack_fisher_s orelse return AccelError.InvalidResourceState;
        const fisher_t = self.stack_fisher_t orelse return AccelError.InvalidResourceState;
        try master_s.requireLiveUnlocked();
        try master_t.requireLiveUnlocked();
        try momentum_s.requireLiveUnlocked();
        try momentum_t.requireLiveUnlocked();
        try fisher_s.requireLiveUnlocked();
        try fisher_t.requireLiveUnlocked();
        if (fisher_s.dim2 != rsf_fisher_width or fisher_t.dim2 != rsf_fisher_width) return AccelError.InvalidDimensions;
        const handle = try owner.requireHandleUnlocked();

        var tuple_s: ?*futhark.struct_futhark_opaque_tup3_stack_sfd = null;
        const rc_s = futhark.futhark_entry_stack_update_sfd_block2x2_master(
            handle,
            &tuple_s,
            master_s.arr,
            gradient_s.arr,
            momentum_s.arr,
            fisher_s.arr,
            learning_rate,
            momentum_beta,
            fisher_gamma,
            next_step,
            epsilon,
            trust_ratio,
            weight_floor,
        );
        if (rc_s != 0 or tuple_s == null) {
            if (tuple_s) |produced| freeStackSfdTupleUnlocked(owner, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkSFDUpdateFailed);
        }
        var tuple_t: ?*futhark.struct_futhark_opaque_tup3_stack_sfd = null;
        const rc_t = futhark.futhark_entry_stack_update_sfd_block2x2_master(
            handle,
            &tuple_t,
            master_t.arr,
            gradient_t.arr,
            momentum_t.arr,
            fisher_t.arr,
            learning_rate,
            momentum_beta,
            fisher_gamma,
            next_step,
            epsilon,
            trust_ratio,
            weight_floor,
        );
        if (rc_t != 0 or tuple_t == null) {
            if (tuple_t) |produced| freeStackSfdTupleUnlocked(owner, produced);
            freeStackSfdTupleUnlocked(owner, tuple_s.?);
            return owner.recordFailureUnlocked(AccelError.FutharkSFDUpdateFailed);
        }
        defer freeStackSfdTupleUnlocked(owner, tuple_s.?);
        defer freeStackSfdTupleUnlocked(owner, tuple_t.?);

        var new_master_s: ?*futhark.struct_futhark_f32_3d = null;
        var new_momentum_s: ?*futhark.struct_futhark_f32_3d = null;
        var new_fisher_s: ?*futhark.struct_futhark_f32_3d = null;
        var new_master_t: ?*futhark.struct_futhark_f32_3d = null;
        var new_momentum_t: ?*futhark.struct_futhark_f32_3d = null;
        var new_fisher_t: ?*futhark.struct_futhark_f32_3d = null;
        defer {
            if (new_master_s) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
        }
        defer {
            if (new_momentum_s) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
        }
        defer {
            if (new_fisher_s) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
        }
        defer {
            if (new_master_t) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
        }
        defer {
            if (new_momentum_t) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
        }
        defer {
            if (new_fisher_t) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
        }

        const p0 = futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_0(handle, &new_master_s, tuple_s.?);
        const p1 = futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_1(handle, &new_momentum_s, tuple_s.?);
        const p2 = futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_2(handle, &new_fisher_s, tuple_s.?);
        const p3 = futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_0(handle, &new_master_t, tuple_t.?);
        const p4 = futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_1(handle, &new_momentum_t, tuple_t.?);
        const p5 = futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_2(handle, &new_fisher_t, tuple_t.?);
        if (p0 != 0 or p1 != 0 or p2 != 0 or p3 != 0 or p4 != 0 or p5 != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
        }
        if (new_master_s == null or new_momentum_s == null or new_fisher_s == null or
            new_master_t == null or new_momentum_t == null or new_fisher_t == null)
        {
            return AccelError.FutharkProjectionFailed;
        }
        const shapes_valid = blk: {
            validateShapeF32_3DUnlocked(owner, new_master_s.?, self.num_layers, half, columns) catch break :blk false;
            validateShapeF32_3DUnlocked(owner, new_momentum_s.?, self.num_layers, half, columns) catch break :blk false;
            validateShapeF32_3DUnlocked(owner, new_fisher_s.?, self.num_layers, half, rsf_fisher_width) catch break :blk false;
            validateShapeF32_3DUnlocked(owner, new_master_t.?, self.num_layers, half, columns) catch break :blk false;
            validateShapeF32_3DUnlocked(owner, new_momentum_t.?, self.num_layers, half, columns) catch break :blk false;
            validateShapeF32_3DUnlocked(owner, new_fisher_t.?, self.num_layers, half, rsf_fisher_width) catch break :blk false;
            break :blk true;
        };
        if (!shapes_valid) return AccelError.FutharkShapeFailed;

        var shadow_s = try convertMasterHandleToF16_3DUnlocked(owner, new_master_s.?, self.num_layers, half, columns);
        var committed_shadow_s = false;
        defer if (!committed_shadow_s) shadow_s.deinitUnlocked();
        var shadow_t = try convertMasterHandleToF16_3DUnlocked(owner, new_master_t.?, self.num_layers, half, columns);
        var committed_shadow_t = false;
        defer if (!committed_shadow_t) shadow_t.deinitUnlocked();

        var old_weights_s = self.stack_weights_s orelse return AccelError.InvalidResourceState;
        var old_weights_t = self.stack_weights_t orelse return AccelError.InvalidResourceState;
        var old_master_s = self.stack_master_weights_s orelse return AccelError.InvalidResourceState;
        var old_master_t = self.stack_master_weights_t orelse return AccelError.InvalidResourceState;
        var old_momentum_s = self.stack_momentum_s orelse return AccelError.InvalidResourceState;
        var old_momentum_t = self.stack_momentum_t orelse return AccelError.InvalidResourceState;
        var old_fisher_s = self.stack_fisher_s orelse return AccelError.InvalidResourceState;
        var old_fisher_t = self.stack_fisher_t orelse return AccelError.InvalidResourceState;

        self.stack_weights_s = shadow_s;
        self.stack_weights_t = shadow_t;
        self.stack_master_weights_s = FutharkArray3DF32.adoptUnlocked(owner, new_master_s.?, self.num_layers, half, columns);
        self.stack_master_weights_t = FutharkArray3DF32.adoptUnlocked(owner, new_master_t.?, self.num_layers, half, columns);
        self.stack_momentum_s = FutharkArray3DF32.adoptUnlocked(owner, new_momentum_s.?, self.num_layers, half, columns);
        self.stack_momentum_t = FutharkArray3DF32.adoptUnlocked(owner, new_momentum_t.?, self.num_layers, half, columns);
        self.stack_fisher_s = FutharkArray3DF32.adoptUnlocked(owner, new_fisher_s.?, self.num_layers, half, rsf_fisher_width);
        self.stack_fisher_t = FutharkArray3DF32.adoptUnlocked(owner, new_fisher_t.?, self.num_layers, half, rsf_fisher_width);
        committed_shadow_s = true;
        committed_shadow_t = true;

        new_master_s = null;
        new_momentum_s = null;
        new_fisher_s = null;
        new_master_t = null;
        new_momentum_t = null;
        new_fisher_t = null;

        old_weights_s.deinitUnlocked();
        old_weights_t.deinitUnlocked();
        old_master_s.deinitUnlocked();
        old_master_t.deinitUnlocked();
        old_momentum_s.deinitUnlocked();
        old_momentum_t.deinitUnlocked();
        old_fisher_s.deinitUnlocked();
        old_fisher_t.deinitUnlocked();
        self.optimizer_step += 1;
        try self.checkInvariantsUnlocked();
    }

    pub fn spectralNormalizeLayers(self: *Self, target: f32) AccelError!void {
        const owner = try self.requireOwnerLive();
        if (!std.math.isFinite(target) or target <= 0.0) return AccelError.InvalidHyperparameter;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();

        const master_s = self.stack_master_weights_s orelse return AccelError.InvalidResourceState;
        const master_t = self.stack_master_weights_t orelse return AccelError.InvalidResourceState;
        var normalized_s = try normalizeMasterStackUnlocked(owner, &master_s, target);
        var committed_s = false;
        defer if (!committed_s) normalized_s.array.deinitUnlocked();
        var normalized_t = try normalizeMasterStackUnlocked(owner, &master_t, target);
        var committed_t = false;
        defer if (!committed_t) normalized_t.array.deinitUnlocked();

        var shadow_s = try convertMasterToF16_3DUnlocked(owner, &normalized_s.array);
        var committed_shadow_s = false;
        defer if (!committed_shadow_s) shadow_s.deinitUnlocked();
        var shadow_t = try convertMasterToF16_3DUnlocked(owner, &normalized_t.array);
        var committed_shadow_t = false;
        defer if (!committed_shadow_t) shadow_t.deinitUnlocked();

        var old_master_s = self.stack_master_weights_s orelse return AccelError.InvalidResourceState;
        var old_master_t = self.stack_master_weights_t orelse return AccelError.InvalidResourceState;
        var old_weights_s = self.stack_weights_s orelse return AccelError.InvalidResourceState;
        var old_weights_t = self.stack_weights_t orelse return AccelError.InvalidResourceState;

        self.stack_master_weights_s = normalized_s.array;
        self.stack_master_weights_t = normalized_t.array;
        self.stack_weights_s = shadow_s;
        self.stack_weights_t = shadow_t;
        committed_s = true;
        committed_t = true;
        committed_shadow_s = true;
        committed_shadow_t = true;

        old_master_s.deinitUnlocked();
        old_master_t.deinitUnlocked();
        old_weights_s.deinitUnlocked();
        old_weights_t.deinitUnlocked();

        self.last_spectral_before_s = normalized_s.before;
        self.last_spectral_after_s = normalized_s.after;
        self.last_spectral_before_t = normalized_t.before;
        self.last_spectral_after_t = normalized_t.after;
        try self.checkInvariantsUnlocked();
    }

    pub fn setLayerWeightsS(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        return self.replaceLayerWeights(true, layer_idx, data, rows, cols);
    }

    pub fn setLayerWeightsT(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        return self.replaceLayerWeights(false, layer_idx, data, rows, cols);
    }

    fn replaceLayerWeights(self: *Self, is_s: bool, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const owner = try self.requireOwnerLive();
        if (self.model_dim == 0 or self.model_dim % 2 != 0) return AccelError.InvalidDimensions;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (rows != self.model_dim / 2 or cols != rsf_coupling_width) return AccelError.InvalidDimensions;
        const expected = try checkedElementCount2(rows, cols);
        if (data.len != expected) return AccelError.InvalidDataLength;
        if (!allFiniteF16(data)) return AccelError.InvalidHyperparameter;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        if (layer_idx >= self.num_layers) return AccelError.InvalidDimensions;
        const half = self.model_dim / 2;
        const columns = rsf_coupling_width;
        const per_layer = try checkedElementCount2(half, columns);
        const stack_count = try checkedElementCount2(self.num_layers, per_layer);

        const master_field: *?FutharkArray3DF32 = if (is_s) &self.stack_master_weights_s else &self.stack_master_weights_t;
        const momentum_field: *?FutharkArray3DF32 = if (is_s) &self.stack_momentum_s else &self.stack_momentum_t;
        const fisher_field: *?FutharkArray3DF32 = if (is_s) &self.stack_fisher_s else &self.stack_fisher_t;
        const shadow_field: *?FutharkArray3DF16 = if (is_s) &self.stack_weights_s else &self.stack_weights_t;

        const master_current = master_field.* orelse return AccelError.InvalidResourceState;
        const momentum_current = momentum_field.* orelse return AccelError.InvalidResourceState;
        const fisher_current = fisher_field.* orelse return AccelError.InvalidResourceState;
        const master_host = try master_current.valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(master_host);
        const momentum_host = try momentum_current.valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(momentum_host);
        const fisher_host = try fisher_current.valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(fisher_host);
        const fisher_per_layer = try checkedElementCount2(half, rsf_fisher_width);
        const fisher_stack = try checkedElementCount2(self.num_layers, fisher_per_layer);
        if (master_host.len != stack_count or momentum_host.len != stack_count or fisher_host.len != fisher_stack) {
            return AccelError.InvalidDimensions;
        }

        const start = layer_idx * per_layer;
        var index: usize = 0;
        while (index < per_layer) : (index += 1) {
            master_host[start + index] = @floatCast(data[index]);
            momentum_host[start + index] = 0.0;
        }
        const fisher_start = layer_idx * fisher_per_layer;
        var fisher_index: usize = 0;
        while (fisher_index < fisher_per_layer) : (fisher_index += 1) {
            fisher_host[fisher_start + fisher_index] = 0.0;
        }

        var new_master = try FutharkArray3DF32.newFromFlatUnlocked(owner, master_host, self.num_layers, half, columns);
        var committed_master = false;
        defer if (!committed_master) new_master.deinitUnlocked();
        var new_momentum = try FutharkArray3DF32.newFromFlatUnlocked(owner, momentum_host, self.num_layers, half, columns);
        var committed_momentum = false;
        defer if (!committed_momentum) new_momentum.deinitUnlocked();
        var new_fisher = try FutharkArray3DF32.newFromFlatUnlocked(owner, fisher_host, self.num_layers, half, rsf_fisher_width);
        var committed_fisher = false;
        defer if (!committed_fisher) new_fisher.deinitUnlocked();
        var new_shadow = try convertMasterToF16_3DUnlocked(owner, &new_master);
        var committed_shadow = false;
        defer if (!committed_shadow) new_shadow.deinitUnlocked();

        var old_master = master_field.* orelse return AccelError.InvalidResourceState;
        var old_momentum = momentum_field.* orelse return AccelError.InvalidResourceState;
        var old_fisher = fisher_field.* orelse return AccelError.InvalidResourceState;
        var old_shadow = shadow_field.* orelse return AccelError.InvalidResourceState;

        master_field.* = new_master;
        momentum_field.* = new_momentum;
        fisher_field.* = new_fisher;
        shadow_field.* = new_shadow;
        committed_master = true;
        committed_momentum = true;
        committed_fisher = true;
        committed_shadow = true;

        old_master.deinitUnlocked();
        old_momentum.deinitUnlocked();
        old_fisher.deinitUnlocked();
        old_shadow.deinitUnlocked();
        try self.checkInvariantsUnlocked();
    }

    pub fn readOptimizerState(self: *Self, allocator: std.mem.Allocator) AccelError!RSFOptimizerState {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        var state = RSFOptimizerState.empty(allocator);
        var committed = false;
        defer if (!committed) state.deinit();
        const master_s = self.stack_master_weights_s orelse return AccelError.InvalidResourceState;
        const master_t = self.stack_master_weights_t orelse return AccelError.InvalidResourceState;
        const momentum_s = self.stack_momentum_s orelse return AccelError.InvalidResourceState;
        const momentum_t = self.stack_momentum_t orelse return AccelError.InvalidResourceState;
        const fisher_s = self.stack_fisher_s orelse return AccelError.InvalidResourceState;
        const fisher_t = self.stack_fisher_t orelse return AccelError.InvalidResourceState;
        state.master_weights_s = try master_s.valuesFlatUnlocked(allocator);
        state.master_weights_t = try master_t.valuesFlatUnlocked(allocator);
        state.momentum_s = try momentum_s.valuesFlatUnlocked(allocator);
        state.momentum_t = try momentum_t.valuesFlatUnlocked(allocator);
        state.fisher_blocks_s = try fisher_s.valuesFlatUnlocked(allocator);
        state.fisher_blocks_t = try fisher_t.valuesFlatUnlocked(allocator);
        state.fisher_s = state.fisher_blocks_s;
        state.fisher_t = state.fisher_blocks_t;
        state.step = self.optimizer_step;
        committed = true;
        return state;
    }

    pub fn getOptimizerState(self: *Self, allocator: std.mem.Allocator) AccelError!RSFOptimizerState {
        return self.readOptimizerState(allocator);
    }

    pub fn setOptimizerState(
        self: *Self,
        master_weights_s: []const f32,
        master_weights_t: []const f32,
        momentum_s: []const f32,
        momentum_t: []const f32,
        fisher_s: []const f32,
        fisher_t: []const f32,
        step: u64,
    ) AccelError!void {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        const half = self.model_dim / 2;
        const columns = rsf_coupling_width;
        const per_layer = try checkedElementCount2(half, columns);
        const total = try checkedElementCount2(self.num_layers, per_layer);
        const fisher_total = try checkedElementCount2(self.num_layers, try checkedElementCount2(half, rsf_fisher_width));
        if (master_weights_s.len != total or master_weights_t.len != total or
            momentum_s.len != total or momentum_t.len != total or
            fisher_s.len != fisher_total or fisher_t.len != fisher_total) return AccelError.InvalidDataLength;
        if (!allFiniteF32(master_weights_s) or !allFiniteF32(master_weights_t)) return AccelError.InvalidHyperparameter;
        if (!allFiniteF32(momentum_s) or !allFiniteF32(momentum_t)) return AccelError.InvalidHyperparameter;
        if (!allFiniteFisherBlocks(fisher_s) or !allFiniteFisherBlocks(fisher_t)) return AccelError.InvalidHyperparameter;

        var new_master_s = try FutharkArray3DF32.newFromFlatUnlocked(owner, master_weights_s, self.num_layers, half, columns);
        var committed_master_s = false;
        defer if (!committed_master_s) new_master_s.deinitUnlocked();
        var new_master_t = try FutharkArray3DF32.newFromFlatUnlocked(owner, master_weights_t, self.num_layers, half, columns);
        var committed_master_t = false;
        defer if (!committed_master_t) new_master_t.deinitUnlocked();
        var new_momentum_s = try FutharkArray3DF32.newFromFlatUnlocked(owner, momentum_s, self.num_layers, half, columns);
        var committed_momentum_s = false;
        defer if (!committed_momentum_s) new_momentum_s.deinitUnlocked();
        var new_momentum_t = try FutharkArray3DF32.newFromFlatUnlocked(owner, momentum_t, self.num_layers, half, columns);
        var committed_momentum_t = false;
        defer if (!committed_momentum_t) new_momentum_t.deinitUnlocked();
        var new_fisher_s = try FutharkArray3DF32.newFromFlatUnlocked(owner, fisher_s, self.num_layers, half, rsf_fisher_width);
        var committed_fisher_s = false;
        defer if (!committed_fisher_s) new_fisher_s.deinitUnlocked();
        var new_fisher_t = try FutharkArray3DF32.newFromFlatUnlocked(owner, fisher_t, self.num_layers, half, rsf_fisher_width);
        var committed_fisher_t = false;
        defer if (!committed_fisher_t) new_fisher_t.deinitUnlocked();
        var new_shadow_s = try convertMasterToF16_3DUnlocked(owner, &new_master_s);
        var committed_shadow_s = false;
        defer if (!committed_shadow_s) new_shadow_s.deinitUnlocked();
        var new_shadow_t = try convertMasterToF16_3DUnlocked(owner, &new_master_t);
        var committed_shadow_t = false;
        defer if (!committed_shadow_t) new_shadow_t.deinitUnlocked();

        var old_weights_s = self.stack_weights_s orelse return AccelError.InvalidResourceState;
        var old_weights_t = self.stack_weights_t orelse return AccelError.InvalidResourceState;
        var old_master_s = self.stack_master_weights_s orelse return AccelError.InvalidResourceState;
        var old_master_t = self.stack_master_weights_t orelse return AccelError.InvalidResourceState;
        var old_momentum_s = self.stack_momentum_s orelse return AccelError.InvalidResourceState;
        var old_momentum_t = self.stack_momentum_t orelse return AccelError.InvalidResourceState;
        var old_fisher_s = self.stack_fisher_s orelse return AccelError.InvalidResourceState;
        var old_fisher_t = self.stack_fisher_t orelse return AccelError.InvalidResourceState;

        self.stack_weights_s = new_shadow_s;
        self.stack_weights_t = new_shadow_t;
        self.stack_master_weights_s = new_master_s;
        self.stack_master_weights_t = new_master_t;
        self.stack_momentum_s = new_momentum_s;
        self.stack_momentum_t = new_momentum_t;
        self.stack_fisher_s = new_fisher_s;
        self.stack_fisher_t = new_fisher_t;
        self.optimizer_step = step;
        committed_master_s = true;
        committed_master_t = true;
        committed_momentum_s = true;
        committed_momentum_t = true;
        committed_fisher_s = true;
        committed_fisher_t = true;
        committed_shadow_s = true;
        committed_shadow_t = true;

        old_weights_s.deinitUnlocked();
        old_weights_t.deinitUnlocked();
        old_master_s.deinitUnlocked();
        old_master_t.deinitUnlocked();
        old_momentum_s.deinitUnlocked();
        old_momentum_t.deinitUnlocked();
        old_fisher_s.deinitUnlocked();
        old_fisher_t.deinitUnlocked();
        try self.checkInvariantsUnlocked();
    }

    pub fn setClipRange(self: *Self, clip_min_val: f16, clip_max_val: f16) AccelError!void {
        if (!self.initialized) return AccelError.UninitializedContext;
        const minimum: f32 = @floatCast(clip_min_val);
        const maximum: f32 = @floatCast(clip_max_val);
        if (!std.math.isFinite(minimum) or !std.math.isFinite(maximum)) return AccelError.InvalidClipRange;
        if (minimum >= maximum or minimum < -20.0 or maximum > 20.0) return AccelError.InvalidClipRange;
        self.clip_min = clip_min_val;
        self.clip_max = clip_max_val;
        try self.checkInvariants();
    }

    pub fn sync(self: *Self) AccelError!void {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        return owner.syncContextUnlocked();
    }

    pub fn fusedMidpointTrainingStep(
        self: *Self,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        sequence_lengths: []const usize,
        grad_mean: bool,
        gradient_scale: f32,
        logdet_weight: f32,
    ) AccelError!FusedStepResult {
        const owner = try self.requireOwnerLive();
        try inputs.requireSameContext(&self.ctx);
        try targets.requireSameContext(&self.ctx);
        if (!std.math.isFinite(gradient_scale) or gradient_scale < 0.0 or gradient_scale > 1.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(logdet_weight) or logdet_weight < 0.0) return AccelError.InvalidHyperparameter;
        if (inputs.dim0 == 0 or inputs.dim1 == 0 or inputs.dim2 == 0) return AccelError.InvalidDimensions;
        if (inputs.dim0 != targets.dim0 or inputs.dim1 != targets.dim1 or inputs.dim2 != targets.dim2) return AccelError.InvalidDimensions;
        if (inputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        if (sequence_lengths.len != inputs.dim0) return AccelError.InvalidDimensions;

        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        try inputs.requireLiveUnlocked();
        try targets.requireLiveUnlocked();

        const lengths_i64 = try self.ensureScratchLengthsUnlocked(sequence_lengths.len);
        for (sequence_lengths, lengths_i64) |length, *target| {
            if (length == 0) return AccelError.InvalidSequenceLength;
            if (length > inputs.dim1) return AccelError.InvalidSequenceLength;
            target.* = try checkedSigned(length);
        }
        var lengths_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, lengths_i64);
        defer lengths_array.deinitUnlocked();

        const weights_s = self.stack_weights_s orelse return AccelError.InvalidResourceState;
        const weights_t = self.stack_weights_t orelse return AccelError.InvalidResourceState;
        try weights_s.requireLiveUnlocked();
        try weights_t.requireLiveUnlocked();
        const handle = try owner.requireHandleUnlocked();
        const clip_min_f32: f32 = @floatCast(self.clip_min);
        const clip_max_f32: f32 = @floatCast(self.clip_max);

        var tuple: ?*futhark.struct_futhark_opaque_tup6_fused_stack_gradients = null;
        const rc = futhark.futhark_entry_rsf_stack_midpoint_fused(
            handle,
            &tuple,
            inputs.arr,
            targets.arr,
            lengths_array.arr,
            weights_s.arr,
            weights_t.arr,
            clip_min_f32,
            clip_max_f32,
            logdet_weight,
            self.diffusion_enabled,
            grad_mean,
            gradient_scale,
            self.diffusion_radix,
            self.diffusion_block,
            self.diffusion_stages,
        );
        if (rc != 0 or tuple == null) {
            if (tuple) |produced| freeFusedTupleUnlocked(owner, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkTrainingStepFailed);
        }
        const pending_tuple = tuple.?;

        var gradient_s: ?*futhark.struct_futhark_f32_3d = null;
        var gradient_t: ?*futhark.struct_futhark_f32_3d = null;
        var delta: ?*futhark.struct_futhark_f16_3d = null;
        const p0 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_0(handle, &gradient_s, pending_tuple);
        const p1 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_1(handle, &gradient_t, pending_tuple);
        const p2 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_2(handle, &delta, pending_tuple);
        if (p0 != 0 or p1 != 0 or p2 != 0 or gradient_s == null or gradient_t == null or delta == null) {
            if (gradient_s) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
            if (gradient_t) |produced| _ = futhark.futhark_free_f32_3d(handle, produced);
            if (delta) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
            freeFusedTupleUnlocked(owner, pending_tuple);
            return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
        }
        const half = self.model_dim / 2;
        const columns = rsf_coupling_width;
        const shapes_valid = blk: {
            validateShapeF32_3DUnlocked(owner, gradient_s.?, self.num_layers, half, columns) catch break :blk false;
            validateShapeF32_3DUnlocked(owner, gradient_t.?, self.num_layers, half, columns) catch break :blk false;
            validateShapeF16_3DUnlocked(owner, delta.?, inputs.dim0, inputs.dim1, inputs.dim2) catch break :blk false;
            break :blk true;
        };
        if (!shapes_valid) {
            _ = futhark.futhark_free_f32_3d(handle, gradient_s.?);
            _ = futhark.futhark_free_f32_3d(handle, gradient_t.?);
            _ = futhark.futhark_free_f16_3d(handle, delta.?);
            freeFusedTupleUnlocked(owner, pending_tuple);
            return AccelError.FutharkShapeFailed;
        }
        var result = FusedStepResult.initOwned(
            owner,
            gradient_s.?,
            gradient_t.?,
            delta.?,
            pending_tuple,
            self.num_layers,
            half,
            columns,
            inputs.dim0,
            inputs.dim1,
            inputs.dim2,
        );
        result.map_midpoint = true;
        return result;
    }

    fn validateMaskBytes(bytes: []const u8, seq_len: usize) AccelError!void {
        const expected = std.math.mul(usize, seq_len, seq_len) catch return AccelError.Overflow;
        if (bytes.len != expected) return AccelError.InvalidCausalMask;
        var i: usize = 0;
        while (i < seq_len) : (i += 1) {
            var j: usize = 0;
            while (j < seq_len) : (j += 1) {
                const value = bytes[i * seq_len + j];
                if (value != 0 and value != 1) return AccelError.InvalidCausalMask;
                if (value == 1 and j >= i) return AccelError.InvalidCausalMask;
            }
        }
    }

    fn layerWeights2DUnlocked(self: *Self, owner: *ContextOwner, stack: *const FutharkArray3DF16, layer: usize) AccelError!FutharkArray2DF16 {
        const half = self.model_dim / 2;
        const columns = rsf_coupling_width;
        const per_layer = try checkedElementCount2(half, columns);
        const host = try stack.valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(host);
        const start = layer * per_layer;
        if (start >= host.len or host.len - start < per_layer) return AccelError.InvalidDimensions;
        return FutharkArray2DF16.newFromFlatUnlocked(owner, host[start .. start + per_layer], half, columns);
    }

    pub fn causalSequenceForward(self: *Self, state: *FutharkArray3DF16, mask: *const types.RSFSequenceMask, layer: usize) AccelError!void {
        const owner = try self.requireOwnerLive();
        try state.requireSameContext(&self.ctx);
        if (layer >= self.num_layers) return AccelError.InvalidDimensions;
        if (state.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        if (mask.seq_len != state.dim1) return AccelError.InvalidDimensions;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        try state.requireLiveUnlocked();
        const mask_arr = try self.uploadMaskUnlocked(owner, mask);
        const handle = try owner.requireHandleUnlocked();
        const weights_s_stack = self.stack_weights_s orelse return AccelError.InvalidResourceState;
        const weights_t_stack = self.stack_weights_t orelse return AccelError.InvalidResourceState;
        try weights_s_stack.requireLiveUnlocked();
        try weights_t_stack.requireLiveUnlocked();
        var layer_s = try self.layerWeights2DUnlocked(owner, &weights_s_stack, layer);
        defer layer_s.deinitUnlocked();
        var layer_t = try self.layerWeights2DUnlocked(owner, &weights_t_stack, layer);
        defer layer_t.deinitUnlocked();
        var out: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_rsf_causal_bitmask_forward(
            handle,
            &out,
            state.arr,
            mask_arr,
            layer_s.arr,
            layer_t.arr,
            @floatCast(self.clip_min),
            @floatCast(self.clip_max),
            self.diffusion_enabled,
            self.diffusion_radix,
            self.diffusion_block,
            self.diffusion_stages,
        );
        if (rc != 0 or out == null) {
            if (out) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkForwardFailed);
        }
        validateShapeF16_3DUnlocked(owner, out.?, state.dim0, state.dim1, state.dim2) catch |err| {
            _ = futhark.futhark_free_f16_3d(handle, out.?);
            return err;
        };
        var produced = FutharkArray3DF16.adoptUnlocked(owner, out.?, state.dim0, state.dim1, state.dim2);
        var previous = state.*;
        state.* = produced;
        previous.deinitUnlocked();
    }

    pub fn causalSequenceInverse(self: *Self, state: *FutharkArray3DF16, mask: *const types.RSFSequenceMask, layer: usize) AccelError!void {
        const owner = try self.requireOwnerLive();
        try state.requireSameContext(&self.ctx);
        if (layer >= self.num_layers) return AccelError.InvalidDimensions;
        if (state.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        if (mask.seq_len != state.dim1) return AccelError.InvalidDimensions;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsUnlocked();
        try state.requireLiveUnlocked();
        const mask_arr = try self.uploadMaskUnlocked(owner, mask);
        const handle = try owner.requireHandleUnlocked();
        const weights_s_stack = self.stack_weights_s orelse return AccelError.InvalidResourceState;
        const weights_t_stack = self.stack_weights_t orelse return AccelError.InvalidResourceState;
        try weights_s_stack.requireLiveUnlocked();
        try weights_t_stack.requireLiveUnlocked();
        var layer_s = try self.layerWeights2DUnlocked(owner, &weights_s_stack, layer);
        defer layer_s.deinitUnlocked();
        var layer_t = try self.layerWeights2DUnlocked(owner, &weights_t_stack, layer);
        defer layer_t.deinitUnlocked();
        var out: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_rsf_causal_bitmask_inverse(
            handle,
            &out,
            state.arr,
            mask_arr,
            layer_s.arr,
            layer_t.arr,
            @floatCast(self.clip_min),
            @floatCast(self.clip_max),
            self.diffusion_enabled,
            self.diffusion_radix,
            self.diffusion_block,
            self.diffusion_stages,
        );
        if (rc != 0 or out == null) {
            if (out) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkInverseFailed);
        }
        validateShapeF16_3DUnlocked(owner, out.?, state.dim0, state.dim1, state.dim2) catch |err| {
            _ = futhark.futhark_free_f16_3d(handle, out.?);
            return err;
        };
        var produced = FutharkArray3DF16.adoptUnlocked(owner, out.?, state.dim0, state.dim1, state.dim2);
        var previous = state.*;
        state.* = produced;
        previous.deinitUnlocked();
    }

    pub fn forwardFromTensor(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (input.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = input.shape.dims[0];
        const cols = input.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const expected = try checkedElementCount2(rows, cols);
        if (input.data.len != expected) return AccelError.InvalidDataLength;
        if (cols != self.model_dim) return AccelError.InvalidDimensions;

        const f16_data = try allocHost(f16, allocator, expected);
        defer allocator.free(f16_data);
        for (input.data, f16_data) |value, *target| {
            target.* = @floatCast(value);
        }
        var device_input = try FutharkArray2DF16.newFromFlat(&self.ctx, f16_data, rows, cols);
        defer device_input.deinit();
        var output = try self.forward(&device_input);
        defer output.deinit();
        const values = try output.valuesFlat(allocator);
        defer allocator.free(values);
        if (values.len != expected) return AccelError.InvalidDataLength;
        const shape = [_]usize{ output.rows, output.cols };
        var result = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        errdefer result.deinit();
        if (result.data.len != values.len) return AccelError.InvalidDataLength;
        for (values, result.data) |value, *target| {
            target.* = @floatCast(value);
        }
        return result;
    }
};

fn rsfForwardLayerUnlocked(
    owner: *ContextOwner,
    input: *const FutharkArray2DF16,
    weights_s: *const FutharkArray2DF16,
    weights_t: *const FutharkArray2DF16,
    clip_min_bits: u16,
    clip_max_bits: u16,
    diffusion: bool,
    radix: i64,
    block: i64,
    stages: i64,
) AccelError!FutharkArray2DF16 {
    const handle = try owner.requireHandleUnlocked();
    try input.requireLiveUnlocked();
    try weights_s.requireLiveUnlocked();
    try weights_t.requireLiveUnlocked();
    if (weights_s.rows != weights_t.rows or weights_s.cols != weights_t.cols) return AccelError.InvalidDimensions;
    if (input.cols != weights_s.rows * 2) return AccelError.InvalidDimensions;
    var out: ?*futhark.struct_futhark_f16_2d = null;
    const rc = futhark.futhark_entry_rsf_forward(
        handle,
        &out,
        input.arr,
        weights_s.arr,
        weights_t.arr,
        clip_min_bits,
        clip_max_bits,
        diffusion,
        radix,
        block,
        stages,
    );
    if (rc != 0 or out == null) {
        if (out) |produced| _ = futhark.futhark_free_f16_2d(handle, produced);
        return owner.recordFailureUnlocked(AccelError.FutharkForwardFailed);
    }
    try validateShapeF16_2DUnlocked(owner, out.?, input.rows, input.cols);
    return FutharkArray2DF16.adoptUnlocked(owner, out.?, input.rows, input.cols);
}

fn normalizeMasterStackUnlocked(
    owner: *ContextOwner,
    master: *const FutharkArray3DF32,
    target: f32,
) AccelError!NormalizedStack {
    const handle = try owner.requireHandleUnlocked();
    try master.requireLiveUnlocked();
    const master_arr = master.arr orelse return AccelError.InvalidResourceState;
    var tuple: ?*futhark.struct_futhark_opaque_tup3_stack_spectral = null;
    const rc = futhark.futhark_entry_stack_spectral_normalize_exact(handle, &tuple, master_arr, target);
    if (rc != 0 or tuple == null) {
        if (tuple) |produced| freeStackSpectralTupleUnlocked(owner, produced);
        return owner.recordFailureUnlocked(AccelError.FutharkNormalizationFailed);
    }
    const pending = tuple.?;
    var array: ?*futhark.struct_futhark_f32_3d = null;
    var before: f32 = 0.0;
    var after: f32 = 0.0;
    const p0 = futhark.futhark_project_opaque_tup3_arr3d_f32_f32_f32_0(handle, &array, pending);
    const p1 = futhark.futhark_project_opaque_tup3_arr3d_f32_f32_f32_1(handle, &before, pending);
    const p2 = futhark.futhark_project_opaque_tup3_arr3d_f32_f32_f32_2(handle, &after, pending);
    const sync_status = owner.syncContextUnlocked();
    freeStackSpectralTupleUnlocked(owner, pending);
    const array_handle = array orelse return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
    if (p0 != 0 or p1 != 0 or p2 != 0) {
        _ = futhark.futhark_free_f32_3d(handle, array_handle);
        return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
    }
    if (sync_status) |_| {} else |sync_error| {
        _ = futhark.futhark_free_f32_3d(handle, array_handle);
        return owner.recordFailureUnlocked(sync_error);
    }
    if (!std.math.isFinite(before) or !std.math.isFinite(after)) {
        _ = futhark.futhark_free_f32_3d(handle, array_handle);
        return owner.recordFailureUnlocked(AccelError.FutharkNormalizationFailed);
    }
    const shapes_valid = blk: {
        validateShapeF32_3DUnlocked(owner, array_handle, master.dim0, master.dim1, master.dim2) catch break :blk false;
        break :blk true;
    };
    if (!shapes_valid) {
        _ = futhark.futhark_free_f32_3d(handle, array_handle);
        return AccelError.FutharkShapeFailed;
    }
    return NormalizedStack{
        .array = FutharkArray3DF32.adoptUnlocked(owner, array_handle, master.dim0, master.dim1, master.dim2),
        .before = before,
        .after = after,
    };
}

fn validateSFDHyperparameters(
    learning_rate: f32,
    momentum_beta: f32,
    fisher_gamma: f32,
    epsilon: f32,
    trust_ratio: f32,
    weight_floor: f32,
) bool {
    if (!std.math.isFinite(learning_rate) or learning_rate < 0.0) return false;
    if (!std.math.isFinite(momentum_beta) or momentum_beta < 0.0 or momentum_beta >= 1.0) return false;
    if (!std.math.isFinite(fisher_gamma) or fisher_gamma < 0.0 or fisher_gamma >= 1.0) return false;
    if (!std.math.isFinite(epsilon) or epsilon <= 0.0) return false;
    if (!std.math.isFinite(trust_ratio) or trust_ratio <= 0.0 or trust_ratio > 1.0) return false;
    if (!std.math.isFinite(weight_floor) or weight_floor <= 0.0) return false;
    return true;
}

pub const embedding_scratch_initial_tokens: usize = 4096;
pub const embedding_scratch_initial_lengths: usize = 256;
pub const embedding_scratch_initial_positions: usize = 1024;

pub const EmbeddingAccelerator = struct {
    owner: ?*ContextOwner,
    allocator: std.mem.Allocator,
    weight: FutharkArray2DF16,
    master_weight: FutharkArray2DF32,
    grad_weight: FutharkArray2DF32,
    momentum_state: FutharkArray2DF32,
    fisher_state: FutharkArray2DF32,
    vocab_size: usize,
    dim: usize,
    initialized: bool,
    optimizer_step: u64,
    scratch_token_buf: []i64,
    scratch_token_cap: usize,
    scratch_lengths_buf: []i64,
    scratch_lengths_cap: usize,
    scratch_positions_buf: []i64,
    scratch_positions_cap: usize,
    last_spectral_before: f32,
    last_spectral_after: f32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, ctx: *FutharkContext, vocab_size: usize, dim: usize, seed: u64) AccelError!Self {
        _ = try ctx.requireOwner();
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount2(vocab_size, dim);
        _ = try checkedByteCount(total, f32);
        _ = try checkedByteCount(total, f16);
        const master_values = try allocHost(f32, allocator, total);
        defer allocator.free(master_values);
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        for (master_values) |*value| {
            value.* = (random.float(f32) - 0.5) * 0.02;
        }
        return createFromMaster(ctx, allocator, vocab_size, dim, master_values);
    }

    pub fn initWithWeights(
        ctx: *FutharkContext,
        allocator: std.mem.Allocator,
        vocab_size: usize,
        dim: usize,
        weight_f16: []const f16,
    ) AccelError!Self {
        _ = try ctx.requireOwner();
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount2(vocab_size, dim);
        _ = try checkedByteCount(total, f32);
        if (weight_f16.len != total) return AccelError.InvalidDataLength;
        if (!allFiniteF16(weight_f16)) return AccelError.InvalidHyperparameter;
        const master_values = try allocHost(f32, allocator, total);
        defer allocator.free(master_values);
        for (weight_f16, master_values) |value, *target| {
            target.* = @floatCast(value);
        }
        return createFromMaster(ctx, allocator, vocab_size, dim, master_values);
    }

    pub fn initWithMasterWeights(
        ctx: *FutharkContext,
        allocator: std.mem.Allocator,
        vocab_size: usize,
        dim: usize,
        master_values: []const f32,
    ) AccelError!Self {
        _ = try ctx.requireOwner();
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;
        return createFromMaster(ctx, allocator, vocab_size, dim, master_values);
    }

    fn createFromMaster(
        ctx: *FutharkContext,
        allocator: std.mem.Allocator,
        vocab_size: usize,
        dim: usize,
        master_values: []const f32,
    ) AccelError!Self {
        const total = try checkedElementCount2(vocab_size, dim);
        if (master_values.len != total) return AccelError.InvalidDataLength;
        if (!allFiniteF32(master_values)) return AccelError.InvalidHyperparameter;
        _ = try checkedDimension(vocab_size);
        _ = try checkedDimension(dim);

        const owner = try ctx.retainOwner();
        errdefer owner.release();

        var master_weight = try FutharkArray2DF32.newFromFlat(ctx, master_values, vocab_size, dim);
        errdefer master_weight.deinit();
        var weight = try convertMasterToF16_2D(ctx, &master_weight);
        errdefer weight.deinit();
        var grad_weight = try FutharkArray2DF32.newZeros(ctx, vocab_size, dim, allocator);
        errdefer grad_weight.deinit();
        var momentum_state = try FutharkArray2DF32.newZeros(ctx, vocab_size, dim, allocator);
        errdefer momentum_state.deinit();
        var fisher_state = try FutharkArray2DF32.newZeros(ctx, vocab_size, dim, allocator);
        errdefer fisher_state.deinit();

        var embedding = Self{
            .owner = owner,
            .allocator = allocator,
            .weight = weight,
            .master_weight = master_weight,
            .grad_weight = grad_weight,
            .momentum_state = momentum_state,
            .fisher_state = fisher_state,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
            .optimizer_step = 0,
            .scratch_token_buf = &.{},
            .scratch_token_cap = 0,
            .scratch_lengths_buf = &.{},
            .scratch_lengths_cap = 0,
            .scratch_positions_buf = &.{},
            .scratch_positions_cap = 0,
            .last_spectral_before = 0.0,
            .last_spectral_after = 0.0,
        };
        try embedding.checkInvariants();
        return embedding;
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        self.fisher_state.deinit();
        self.momentum_state.deinit();
        self.grad_weight.deinit();
        self.master_weight.deinit();
        self.weight.deinit();
        if (self.scratch_positions_buf.len > 0) self.allocator.free(self.scratch_positions_buf);
        if (self.scratch_lengths_buf.len > 0) self.allocator.free(self.scratch_lengths_buf);
        if (self.scratch_token_buf.len > 0) self.allocator.free(self.scratch_token_buf);
        self.scratch_positions_buf = &.{};
        self.scratch_positions_cap = 0;
        self.scratch_lengths_buf = &.{};
        self.scratch_lengths_cap = 0;
        self.scratch_token_buf = &.{};
        self.scratch_token_cap = 0;
        if (self.owner) |owner| {
            owner.release();
            self.owner = null;
        }
        self.initialized = false;
    }

    pub fn context(self: *const Self) FutharkContext {
        return FutharkContext.bind(self.owner);
    }

    pub fn isInitialized(self: *const Self) bool {
        return self.initialized;
    }

    pub fn optimizerStep(self: *const Self) u64 {
        return self.optimizer_step;
    }

    pub fn dimensions(self: *const Self) [2]usize {
        return .{ self.vocab_size, self.dim };
    }

    fn requireOwnerLive(self: *const Self) AccelError!*ContextOwner {
        if (!self.initialized) return AccelError.UninitializedContext;
        const owner = self.owner orelse return AccelError.UninitializedContext;
        try owner.checkLive();
        return owner;
    }

    fn requireOwnerLiveUnlocked(self: *const Self) AccelError!*ContextOwner {
        if (!self.initialized) return AccelError.UninitializedContext;
        const owner = self.owner orelse return AccelError.UninitializedContext;
        try owner.checkLiveUnlocked();
        return owner;
    }

    pub fn checkInvariants(self: *const Self) AccelError!void {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        return self.checkInvariantsWithOwnerUnlocked(owner);
    }

    fn checkInvariantsUnlocked(self: *const Self) AccelError!void {
        const owner = try self.requireOwnerLiveUnlocked();
        return self.checkInvariantsWithOwnerUnlocked(owner);
    }

    fn checkInvariantsWithOwnerUnlocked(self: *const Self, owner: *ContextOwner) AccelError!void {
        if (!self.initialized) return AccelError.UninitializedContext;
        if (self.owner != owner) return AccelError.ContextMismatch;
        if (!owner.alive or owner.ctx == null) return AccelError.UninitializedContext;
        if (self.vocab_size == 0 or self.dim == 0) return AccelError.InvalidDimensions;
        if (self.scratch_token_buf.len != self.scratch_token_cap) return AccelError.InvalidResourceState;
        if (self.scratch_lengths_buf.len != self.scratch_lengths_cap) return AccelError.InvalidResourceState;
        if (self.scratch_positions_buf.len != self.scratch_positions_cap) return AccelError.InvalidResourceState;
        try checkEmbeddingArray(self.weight, self.vocab_size, self.dim, owner.id);
        try checkEmbeddingArray(self.master_weight, self.vocab_size, self.dim, owner.id);
        try checkEmbeddingArray(self.grad_weight, self.vocab_size, self.dim, owner.id);
        try checkEmbeddingArray(self.momentum_state, self.vocab_size, self.dim, owner.id);
        try checkEmbeddingArray(self.fisher_state, self.vocab_size, self.dim, owner.id);
    }

    fn ensureScratchTokensUnlocked(self: *Self, count: usize) AccelError![]i64 {
        if (count == 0) return AccelError.InvalidDimensions;
        if (count > self.scratch_token_cap) {
            var capacity = self.scratch_token_cap;
            if (capacity == 0) capacity = embedding_scratch_initial_tokens;
            while (capacity < count) {
                capacity = std.math.mul(usize, capacity, 2) catch return AccelError.Overflow;
            }
            const buffer = try allocHost(i64, self.allocator, capacity);
            if (self.scratch_token_buf.len > 0) {
                @memcpy(buffer[0..self.scratch_token_buf.len], self.scratch_token_buf);
                self.allocator.free(self.scratch_token_buf);
            }
            self.scratch_token_buf = buffer;
            self.scratch_token_cap = capacity;
        }
        return self.scratch_token_buf[0..count];
    }

    fn ensureScratchLengthsUnlocked(self: *Self, count: usize) AccelError![]i64 {
        if (count == 0) return AccelError.InvalidDimensions;
        if (count > self.scratch_lengths_cap) {
            var capacity = self.scratch_lengths_cap;
            if (capacity == 0) capacity = embedding_scratch_initial_lengths;
            while (capacity < count) {
                capacity = std.math.mul(usize, capacity, 2) catch return AccelError.Overflow;
            }
            const buffer = try allocHost(i64, self.allocator, capacity);
            if (self.scratch_lengths_buf.len > 0) {
                @memcpy(buffer[0..self.scratch_lengths_buf.len], self.scratch_lengths_buf);
                self.allocator.free(self.scratch_lengths_buf);
            }
            self.scratch_lengths_buf = buffer;
            self.scratch_lengths_cap = capacity;
        }
        return self.scratch_lengths_buf[0..count];
    }

    fn ensureScratchPositionsUnlocked(self: *Self, count: usize) AccelError![]i64 {
        if (count == 0) return AccelError.InvalidDimensions;
        if (count > self.scratch_positions_cap) {
            var capacity = self.scratch_positions_cap;
            if (capacity == 0) capacity = embedding_scratch_initial_positions;
            while (capacity < count) {
                capacity = std.math.mul(usize, capacity, 2) catch return AccelError.Overflow;
            }
            const buffer = try allocHost(i64, self.allocator, capacity);
            if (self.scratch_positions_buf.len > 0) {
                @memcpy(buffer[0..self.scratch_positions_buf.len], self.scratch_positions_buf);
                self.allocator.free(self.scratch_positions_buf);
            }
            self.scratch_positions_buf = buffer;
            self.scratch_positions_cap = capacity;
        }
        return self.scratch_positions_buf[0..count];
    }

    fn convertTokensUnlocked(self: *Self, tokens: []const u32) AccelError![]i64 {
        const target = try self.ensureScratchTokensUnlocked(tokens.len);
        for (tokens, target) |token, *slot| {
            if (@as(usize, token) >= self.vocab_size) return AccelError.InvalidToken;
            slot.* = try checkedSigned(@as(usize, token));
        }
        return target;
    }

    fn convertLengthsUnlocked(self: *Self, sequence_lengths: []const usize, sequence_length: usize) AccelError![]i64 {
        const target = try self.ensureScratchLengthsUnlocked(sequence_lengths.len);
        for (sequence_lengths, target) |length, *slot| {
            if (length == 0) return AccelError.InvalidSequenceLength;
            if (length > sequence_length) return AccelError.InvalidSequenceLength;
            slot.* = try checkedSigned(length);
        }
        return target;
    }

    fn convertPositionsUnlocked(self: *Self, sequence_length: usize) AccelError![]i64 {
        const target = try self.ensureScratchPositionsUnlocked(sequence_length);
        for (target, 0..) |*slot, index| {
            slot.* = try checkedSigned(index);
        }
        return target;
    }

    pub fn forwardPadded(
        self: *Self,
        tokens: []const u32,
        sequence_lengths: []const usize,
        sequence_length: usize,
    ) AccelError!FutharkArray3DF16 {
        const owner = try self.requireOwnerLive();
        if (sequence_lengths.len == 0 or sequence_length == 0) return AccelError.InvalidDimensions;
        const batch = sequence_lengths.len;
        const expected_tokens = try checkedElementCount2(batch, sequence_length);
        if (tokens.len != expected_tokens) return AccelError.InvalidDataLength;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);

        const token_i64s = try self.convertTokensUnlocked(tokens);
        const lengths_i64 = try self.convertLengthsUnlocked(sequence_lengths, sequence_length);
        const positions_i64 = try self.convertPositionsUnlocked(sequence_length);

        var token_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, token_i64s);
        defer token_array.deinitUnlocked();
        var length_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, lengths_i64);
        defer length_array.deinitUnlocked();
        var position_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, positions_i64);
        defer position_array.deinitUnlocked();

        const handle = try owner.requireHandleUnlocked();
        var output: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_embedding_forward_padded(
            handle,
            &output,
            token_array.arr,
            length_array.arr,
            position_array.arr,
            self.weight.arr,
        );
        if (rc != 0 or output == null) {
            if (output) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkForwardFailed);
        }
        try validateShapeF16_3DUnlocked(owner, output.?, batch, sequence_length, self.dim);
        return FutharkArray3DF16.adoptUnlocked(owner, output.?, batch, sequence_length, self.dim);
    }

    pub fn backwardPaddedAccumulate(
        self: *Self,
        tokens: []const u32,
        sequence_lengths: []const usize,
        gradient_output: *FutharkArray3DF16,
    ) AccelError!void {
        const owner = try self.requireOwnerLive();
        const handle_context = self.context();
        try gradient_output.requireSameContext(&handle_context);
        if (gradient_output.dim0 == 0 or gradient_output.dim1 == 0 or gradient_output.dim2 == 0) return AccelError.InvalidDimensions;
        if (gradient_output.dim2 != self.dim) return AccelError.InvalidDimensions;
        if (sequence_lengths.len != gradient_output.dim0) return AccelError.InvalidDimensions;
        const expected_tokens = try checkedElementCount2(gradient_output.dim0, gradient_output.dim1);
        if (tokens.len != expected_tokens) return AccelError.InvalidDataLength;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        try gradient_output.requireLiveUnlocked();

        const token_i64s = try self.convertTokensUnlocked(tokens);
        const lengths_i64 = try self.convertLengthsUnlocked(sequence_lengths, gradient_output.dim1);

        var token_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, token_i64s);
        defer token_array.deinitUnlocked();
        var length_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, lengths_i64);
        defer length_array.deinitUnlocked();

        const handle = try owner.requireHandleUnlocked();
        var new_gradient: ?*futhark.struct_futhark_f32_2d = null;
        const rc = futhark.futhark_entry_embedding_backward_padded(
            handle,
            &new_gradient,
            token_array.arr,
            length_array.arr,
            gradient_output.arr,
            self.grad_weight.arr,
        );
        if (rc != 0 or new_gradient == null) {
            if (new_gradient) |produced| _ = futhark.futhark_free_f32_2d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkBackwardFailed);
        }
        try validateShapeF32_2DUnlocked(owner, new_gradient.?, self.vocab_size, self.dim);
        var old_gradient = self.grad_weight;
        self.grad_weight = FutharkArray2DF32.adoptUnlocked(owner, new_gradient.?, self.vocab_size, self.dim);
        old_gradient.deinitUnlocked();
        try self.checkInvariantsWithOwnerUnlocked(owner);
    }

    pub fn zeroGrad(self: *Self) AccelError!void {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const replacement = try FutharkArray2DF32.newZerosUnlocked(owner, self.allocator, self.vocab_size, self.dim);
        var old_gradient = self.grad_weight;
        self.grad_weight = replacement;
        old_gradient.deinitUnlocked();
        try self.checkInvariantsWithOwnerUnlocked(owner);
    }

    pub fn readGradient(self: *Self, allocator: std.mem.Allocator) AccelError![]f32 {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const total = try checkedElementCount2(self.vocab_size, self.dim);
        const values = try self.grad_weight.valuesFlatUnlocked(allocator);
        if (values.len != total) {
            allocator.free(values);
            return AccelError.InvalidDataLength;
        }
        return values;
    }

    pub fn getGradientDeviceBuffer(self: *Self) AccelError!DeviceBufferF32 {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        return self.grad_weight.deviceBufferUnlocked();
    }

    pub fn getGradientDevicePtrF32(self: *Self) AccelError!DeviceBufferF32 {
        return self.getGradientDeviceBuffer();
    }

    pub fn clipGradient(self: *Self, clip_norm: f32) AccelError!void {
        const owner = try self.requireOwnerLive();
        if (!std.math.isFinite(clip_norm) or clip_norm <= 0.0) return AccelError.InvalidHyperparameter;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const handle = try owner.requireHandleUnlocked();
        var clipped: ?*futhark.struct_futhark_f32_2d = null;
        const rc = futhark.futhark_entry_clip_matrix_global_norm_f32(handle, &clipped, self.grad_weight.arr, clip_norm);
        if (rc != 0 or clipped == null) {
            if (clipped) |produced| _ = futhark.futhark_free_f32_2d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkScaleWeightsFailed);
        }
        try validateShapeF32_2DUnlocked(owner, clipped.?, self.vocab_size, self.dim);
        var old_gradient = self.grad_weight;
        self.grad_weight = FutharkArray2DF32.adoptUnlocked(owner, clipped.?, self.vocab_size, self.dim);
        old_gradient.deinitUnlocked();
        try self.checkInvariantsWithOwnerUnlocked(owner);
    }

    pub fn scaleGradient(self: *Self, scale_factor: f32) AccelError!void {
        const owner = try self.requireOwnerLive();
        if (!std.math.isFinite(scale_factor) or scale_factor < 0.0) return AccelError.InvalidHyperparameter;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const handle = try owner.requireHandleUnlocked();
        var scaled: ?*futhark.struct_futhark_f32_2d = null;
        const rc = futhark.futhark_entry_scale_matrix_f32(handle, &scaled, self.grad_weight.arr, scale_factor);
        if (rc != 0 or scaled == null) {
            if (scaled) |produced| _ = futhark.futhark_free_f32_2d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkScaleWeightsFailed);
        }
        try validateShapeF32_2DUnlocked(owner, scaled.?, self.vocab_size, self.dim);
        var old_gradient = self.grad_weight;
        self.grad_weight = FutharkArray2DF32.adoptUnlocked(owner, scaled.?, self.vocab_size, self.dim);
        old_gradient.deinitUnlocked();
        try self.checkInvariantsWithOwnerUnlocked(owner);
    }

    pub fn applyGradientsSFD(
        self: *Self,
        learning_rate: f32,
        momentum_beta: f32,
        fisher_gamma: f32,
        epsilon: f32,
        trust_ratio: f32,
        weight_floor: f32,
    ) AccelError!void {
        const owner = try self.requireOwnerLive();
        if (!validateSFDHyperparameters(learning_rate, momentum_beta, fisher_gamma, epsilon, trust_ratio, weight_floor)) {
            return AccelError.InvalidHyperparameter;
        }
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        if (self.optimizer_step >= @as(u64, @intCast(std.math.maxInt(i64)))) return AccelError.OptimizerStepOverflow;
        const next_step: i64 = try checkedSigned(self.optimizer_step + 1);
        const handle = try owner.requireHandleUnlocked();

        var tuple: ?*futhark.struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32 = null;
        const rc = futhark.futhark_entry_embedding_update_sfd_master(
            handle,
            &tuple,
            self.master_weight.arr,
            self.grad_weight.arr,
            self.momentum_state.arr,
            self.fisher_state.arr,
            learning_rate,
            momentum_beta,
            fisher_gamma,
            next_step,
            epsilon,
            trust_ratio,
            weight_floor,
        );
        if (rc != 0 or tuple == null) {
            if (tuple) |produced| _ = futhark.futhark_free_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkSFDUpdateFailed);
        }
        const pending = tuple.?;
        var new_master: ?*futhark.struct_futhark_f32_2d = null;
        var new_momentum: ?*futhark.struct_futhark_f32_2d = null;
        var new_fisher: ?*futhark.struct_futhark_f32_2d = null;
        const p0 = futhark.futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_0(handle, &new_master, pending);
        const p1 = futhark.futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_1(handle, &new_momentum, pending);
        const p2 = futhark.futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_2(handle, &new_fisher, pending);
        _ = futhark.futhark_free_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32(handle, pending);
        if (p0 != 0 or p1 != 0 or p2 != 0 or new_master == null or new_momentum == null or new_fisher == null) {
            if (new_master) |produced| _ = futhark.futhark_free_f32_2d(handle, produced);
            if (new_momentum) |produced| _ = futhark.futhark_free_f32_2d(handle, produced);
            if (new_fisher) |produced| _ = futhark.futhark_free_f32_2d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
        }
        try validateShapeF32_2DUnlocked(owner, new_master.?, self.vocab_size, self.dim);
        try validateShapeF32_2DUnlocked(owner, new_momentum.?, self.vocab_size, self.dim);
        try validateShapeF32_2DUnlocked(owner, new_fisher.?, self.vocab_size, self.dim);

        var new_shadow = try convertMasterHandleToF16_2DUnlocked(owner, new_master.?, self.vocab_size, self.dim);
        var committed_shadow = false;
        defer if (!committed_shadow) new_shadow.deinitUnlocked();
        var new_grad = try FutharkArray2DF32.newZerosUnlocked(owner, self.allocator, self.vocab_size, self.dim);
        var committed_grad = false;
        defer if (!committed_grad) new_grad.deinitUnlocked();

        var old_weight = self.weight;
        var old_master = self.master_weight;
        var old_momentum = self.momentum_state;
        var old_fisher = self.fisher_state;
        var old_grad = self.grad_weight;

        self.weight = new_shadow;
        self.master_weight = FutharkArray2DF32.adoptUnlocked(owner, new_master.?, self.vocab_size, self.dim);
        self.momentum_state = FutharkArray2DF32.adoptUnlocked(owner, new_momentum.?, self.vocab_size, self.dim);
        self.fisher_state = FutharkArray2DF32.adoptUnlocked(owner, new_fisher.?, self.vocab_size, self.dim);
        self.grad_weight = new_grad;
        committed_shadow = true;
        committed_grad = true;

        old_weight.deinitUnlocked();
        old_master.deinitUnlocked();
        old_momentum.deinitUnlocked();
        old_fisher.deinitUnlocked();
        old_grad.deinitUnlocked();
        self.optimizer_step += 1;
        try self.checkInvariantsWithOwnerUnlocked(owner);
    }

    pub fn applyUpdateFusedSFD(
        self: *Self,
        learning_rate: f32,
        momentum_beta: f32,
        fisher_gamma: f32,
        epsilon: f32,
        trust_ratio: f32,
        weight_floor: f32,
    ) AccelError!void {
        return self.applyGradientsSFD(learning_rate, momentum_beta, fisher_gamma, epsilon, trust_ratio, weight_floor);
    }

    pub fn readOptimizerState(self: *Self, allocator: std.mem.Allocator) AccelError!EmbeddingOptimizerState {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        var state = EmbeddingOptimizerState.empty(allocator);
        var committed = false;
        defer if (!committed) state.deinit();
        state.master_weights = try self.master_weight.valuesFlatUnlocked(allocator);
        state.momentum = try self.momentum_state.valuesFlatUnlocked(allocator);
        state.fisher = try self.fisher_state.valuesFlatUnlocked(allocator);
        state.step = self.optimizer_step;
        committed = true;
        return state;
    }

    pub fn setOptimizerState(self: *Self, master_weights: []const f32, momentum: []const f32, fisher: []const f32, step: u64) AccelError!void {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const total = try checkedElementCount2(self.vocab_size, self.dim);
        if (master_weights.len != total or momentum.len != total or fisher.len != total) return AccelError.InvalidDataLength;
        if (!allFiniteF32(master_weights) or !allFiniteF32(momentum) or !allFiniteNonNegativeF32(fisher)) return AccelError.InvalidHyperparameter;

        var new_master = try FutharkArray2DF32.newFromFlatUnlocked(owner, master_weights, self.vocab_size, self.dim);
        var committed_master = false;
        defer if (!committed_master) new_master.deinitUnlocked();
        var new_momentum = try FutharkArray2DF32.newFromFlatUnlocked(owner, momentum, self.vocab_size, self.dim);
        var committed_momentum = false;
        defer if (!committed_momentum) new_momentum.deinitUnlocked();
        var new_fisher = try FutharkArray2DF32.newFromFlatUnlocked(owner, fisher, self.vocab_size, self.dim);
        var committed_fisher = false;
        defer if (!committed_fisher) new_fisher.deinitUnlocked();
        var new_shadow = try convertMasterToF16_2DUnlocked(owner, &new_master);
        var committed_shadow = false;
        defer if (!committed_shadow) new_shadow.deinitUnlocked();

        var old_weight = self.weight;
        var old_master = self.master_weight;
        var old_momentum = self.momentum_state;
        var old_fisher = self.fisher_state;

        self.weight = new_shadow;
        self.master_weight = new_master;
        self.momentum_state = new_momentum;
        self.fisher_state = new_fisher;
        self.optimizer_step = step;
        committed_master = true;
        committed_momentum = true;
        committed_fisher = true;
        committed_shadow = true;

        old_weight.deinitUnlocked();
        old_master.deinitUnlocked();
        old_momentum.deinitUnlocked();
        old_fisher.deinitUnlocked();
        try self.checkInvariantsWithOwnerUnlocked(owner);
    }

    pub fn sourceSumSquares(self: *Self) AccelError!f32 {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const handle = try owner.requireHandleUnlocked();
        var total: f32 = 0.0;
        const rc = futhark.futhark_entry_embedding_sum_squares(handle, &total, self.weight.arr);
        if (rc != 0) return owner.recordFailureUnlocked(AccelError.FutharkComputeLossFailed);
        try owner.syncContextUnlocked();
        if (!std.math.isFinite(total)) return owner.recordFailureUnlocked(AccelError.FutharkComputeLossFailed);
        return total;
    }

    pub fn sourceRootMeanSquare(self: *Self) AccelError!f32 {
        const total = try self.sourceSumSquares();
        const count = try checkedElementCount2(self.vocab_size, self.dim);
        const mean = total / @as(f32, @floatFromInt(count));
        if (!std.math.isFinite(mean) or mean <= 0.0) return 0.0;
        return @sqrt(mean);
    }

    pub fn spectralNormalize(
        self: *Self,
        u: *FutharkArray1DF32,
        v: *FutharkArray1DF32,
        power_iters: usize,
        target: f32,
    ) AccelError!void {
        const owner = try self.requireOwnerLive();
        const handle_context = self.context();
        try u.requireSameContext(&handle_context);
        try v.requireSameContext(&handle_context);
        if (power_iters == 0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(target) or target <= 0.0) return AccelError.InvalidHyperparameter;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        try u.requireLiveUnlocked();
        try v.requireLiveUnlocked();
        if (u.len != self.vocab_size) return AccelError.InvalidDimensions;
        if (v.len != self.dim) return AccelError.InvalidDimensions;
        const handle = try owner.requireHandleUnlocked();
        const iteration_count = try checkedDimension(power_iters);

        var tuple: ?*futhark.struct_futhark_opaque_tup5_embedding_spectral = null;
        const rc = futhark.futhark_entry_embedding_spectral_normalize(
            handle,
            &tuple,
            self.master_weight.arr,
            u.arr,
            v.arr,
            iteration_count,
            target,
        );
        if (rc != 0 or tuple == null) {
            if (tuple) |produced| _ = futhark.futhark_free_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkNormalizationFailed);
        }
        const pending = tuple.?;
        var new_master: ?*futhark.struct_futhark_f32_2d = null;
        var new_u: ?*futhark.struct_futhark_f32_1d = null;
        var new_v: ?*futhark.struct_futhark_f32_1d = null;
        var sigma_before: f32 = 0.0;
        var sigma_after: f32 = 0.0;
        const p0 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_0(handle, &new_master, pending);
        const p1 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_1(handle, &new_u, pending);
        const p2 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_2(handle, &new_v, pending);
        const p3 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_3(handle, &sigma_before, pending);
        const p4 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_4(handle, &sigma_after, pending);
        const sync_status = owner.syncContextUnlocked();
        _ = futhark.futhark_free_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32(handle, pending);
        const master_handle = new_master orelse return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
        const u_handle = new_u orelse {
            _ = futhark.futhark_free_f32_2d(handle, master_handle);
            return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
        };
        const v_handle = new_v orelse {
            _ = futhark.futhark_free_f32_2d(handle, master_handle);
            _ = futhark.futhark_free_f32_1d(handle, u_handle);
            return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
        };
        if (p0 != 0 or p1 != 0 or p2 != 0 or p3 != 0 or p4 != 0) {
            _ = futhark.futhark_free_f32_2d(handle, master_handle);
            _ = futhark.futhark_free_f32_1d(handle, u_handle);
            _ = futhark.futhark_free_f32_1d(handle, v_handle);
            return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
        }
        if (sync_status) |_| {} else |sync_error| {
            _ = futhark.futhark_free_f32_2d(handle, master_handle);
            _ = futhark.futhark_free_f32_1d(handle, u_handle);
            _ = futhark.futhark_free_f32_1d(handle, v_handle);
            return owner.recordFailureUnlocked(sync_error);
        }
        if (!std.math.isFinite(sigma_before) or !std.math.isFinite(sigma_after)) {
            _ = futhark.futhark_free_f32_2d(handle, master_handle);
            _ = futhark.futhark_free_f32_1d(handle, u_handle);
            _ = futhark.futhark_free_f32_1d(handle, v_handle);
            return owner.recordFailureUnlocked(AccelError.FutharkNormalizationFailed);
        }
        const shapes_valid = blk: {
            validateShapeF32_2DUnlocked(owner, master_handle, self.vocab_size, self.dim) catch break :blk false;
            validateShapeF32_1DUnlocked(owner, u_handle, self.vocab_size) catch break :blk false;
            validateShapeF32_1DUnlocked(owner, v_handle, self.dim) catch break :blk false;
            break :blk true;
        };
        if (!shapes_valid) {
            _ = futhark.futhark_free_f32_2d(handle, master_handle);
            _ = futhark.futhark_free_f32_1d(handle, u_handle);
            _ = futhark.futhark_free_f32_1d(handle, v_handle);
            return AccelError.FutharkShapeFailed;
        }

        var new_shadow = try convertMasterHandleToF16_2DUnlocked(owner, master_handle, self.vocab_size, self.dim);
        var committed_shadow = false;
        defer if (!committed_shadow) new_shadow.deinitUnlocked();

        var old_master = self.master_weight;
        var old_weight = self.weight;
        var old_u = u.*;
        var old_v = v.*;

        self.master_weight = FutharkArray2DF32.adoptUnlocked(owner, master_handle, self.vocab_size, self.dim);
        self.weight = new_shadow;
        u.* = FutharkArray1DF32.adoptUnlocked(owner, u_handle, self.vocab_size);
        v.* = FutharkArray1DF32.adoptUnlocked(owner, v_handle, self.dim);
        committed_shadow = true;

        old_master.deinitUnlocked();
        old_weight.deinitUnlocked();
        old_u.deinitUnlocked();
        old_v.deinitUnlocked();

        self.last_spectral_before = sigma_before;
        self.last_spectral_after = sigma_after;
        try self.checkInvariantsWithOwnerUnlocked(owner);
    }

    pub fn cloneDevice(self: *Self) AccelError!Self {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const master = try self.master_weight.valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(master);
        const grad = try self.grad_weight.valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(grad);
        const momentum = try self.momentum_state.valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(momentum);
        const fisher = try self.fisher_state.valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(fisher);
        const step = self.optimizer_step;
        const allocator = self.allocator;
        const vocab_size = self.vocab_size;
        const dim = self.dim;
        owner.unlock();
        defer owner.lock();
        var clone_context = self.context();
        var clone = try createFromMaster(&clone_context, allocator, vocab_size, dim, master);
        errdefer clone.deinit();
        try clone.adoptAccumulatorState(grad, momentum, fisher, step);
        return clone;
    }

    pub fn cloneParameters(self: *Self) AccelError!Self {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const master = try self.master_weight.valuesFlatUnlocked(self.allocator);
        defer self.allocator.free(master);
        const allocator = self.allocator;
        const vocab_size = self.vocab_size;
        const dim = self.dim;
        owner.unlock();
        defer owner.lock();
        var clone_context = self.context();
        return createFromMaster(&clone_context, allocator, vocab_size, dim, master);
    }

    fn adoptAccumulatorState(
        self: *Self,
        grad_host: []const f32,
        momentum_host: []const f32,
        fisher_host: []const f32,
        step: u64,
    ) AccelError!void {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const total = try checkedElementCount2(self.vocab_size, self.dim);
        if (grad_host.len != total or momentum_host.len != total or fisher_host.len != total) return AccelError.InvalidDataLength;
        if (!allFiniteF32(grad_host) or !allFiniteF32(momentum_host) or !allFiniteNonNegativeF32(fisher_host)) {
            return AccelError.InvalidHyperparameter;
        }
        var new_grad = try FutharkArray2DF32.newFromFlatUnlocked(owner, grad_host, self.vocab_size, self.dim);
        var committed_grad = false;
        defer if (!committed_grad) new_grad.deinitUnlocked();
        var new_momentum = try FutharkArray2DF32.newFromFlatUnlocked(owner, momentum_host, self.vocab_size, self.dim);
        var committed_momentum = false;
        defer if (!committed_momentum) new_momentum.deinitUnlocked();
        var new_fisher = try FutharkArray2DF32.newFromFlatUnlocked(owner, fisher_host, self.vocab_size, self.dim);
        var committed_fisher = false;
        defer if (!committed_fisher) new_fisher.deinitUnlocked();

        var old_grad = self.grad_weight;
        var old_momentum = self.momentum_state;
        var old_fisher = self.fisher_state;
        self.grad_weight = new_grad;
        self.momentum_state = new_momentum;
        self.fisher_state = new_fisher;
        self.optimizer_step = step;
        committed_grad = true;
        committed_momentum = true;
        committed_fisher = true;

        old_grad.deinitUnlocked();
        old_momentum.deinitUnlocked();
        old_fisher.deinitUnlocked();
        try self.checkInvariantsWithOwnerUnlocked(owner);
    }
};

fn checkEmbeddingArray(array: anytype, rows: usize, cols: usize, context_id: u64) AccelError!void {
    if (!array.isLive()) return AccelError.InvalidResourceState;
    const observed = array.contextId() orelse return AccelError.InvalidResourceState;
    if (observed != context_id) return AccelError.ContextMismatch;
    if (array.rows != rows or array.cols != cols) return AccelError.InvalidDimensions;
}

pub const FrozenEmbedding = struct {
    owner: ?*ContextOwner,
    allocator: std.mem.Allocator,
    weight: FutharkArray2DF16,
    vocab_size: usize,
    dim: usize,
    initialized: bool,
    scratch_token_buf: []i64,
    scratch_token_cap: usize,
    scratch_lengths_buf: []i64,
    scratch_lengths_cap: usize,
    scratch_positions_buf: []i64,
    scratch_positions_cap: usize,

    const Self = @This();

    pub fn initFromTrainableMaster(src: *EmbeddingAccelerator) AccelError!Self {
        const owner = try src.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try src.checkInvariantsWithOwnerUnlocked(owner);
        var weight = try convertMasterToF16_2DUnlocked(owner, &src.master_weight);
        errdefer weight.deinitUnlocked();
        var frozen = Self{
            .owner = owner,
            .allocator = src.allocator,
            .weight = weight,
            .vocab_size = src.vocab_size,
            .dim = src.dim,
            .initialized = true,
            .scratch_token_buf = &.{},
            .scratch_token_cap = 0,
            .scratch_lengths_buf = &.{},
            .scratch_lengths_cap = 0,
            .scratch_positions_buf = &.{},
            .scratch_positions_cap = 0,
        };
        try frozen.checkInvariantsWithOwnerUnlocked(owner);
        return frozen;
    }

    pub fn initFromMasterWeights(
        ctx: *FutharkContext,
        allocator: std.mem.Allocator,
        vocab_size: usize,
        dim: usize,
        master_values: []const f32,
    ) AccelError!Self {
        _ = try ctx.requireOwner();
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;
        const total = try checkedElementCount2(vocab_size, dim);
        if (master_values.len != total) return AccelError.InvalidDataLength;
        if (!allFiniteF32(master_values)) return AccelError.InvalidHyperparameter;
        const owner = try ctx.retainOwner();
        errdefer owner.release();
        var master = try FutharkArray2DF32.newFromFlat(ctx, master_values, vocab_size, dim);
        defer master.deinit();
        var weight = try convertMasterToF16_2D(ctx, &master);
        errdefer weight.deinit();
        var frozen = Self{
            .owner = owner,
            .allocator = allocator,
            .weight = weight,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
            .scratch_token_buf = &.{},
            .scratch_token_cap = 0,
            .scratch_lengths_buf = &.{},
            .scratch_lengths_cap = 0,
            .scratch_positions_buf = &.{},
            .scratch_positions_cap = 0,
        };
        try frozen.checkInvariants();
        return frozen;
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        self.weight.deinit();
        if (self.scratch_positions_buf.len > 0) self.allocator.free(self.scratch_positions_buf);
        if (self.scratch_lengths_buf.len > 0) self.allocator.free(self.scratch_lengths_buf);
        if (self.scratch_token_buf.len > 0) self.allocator.free(self.scratch_token_buf);
        self.scratch_positions_buf = &.{};
        self.scratch_positions_cap = 0;
        self.scratch_lengths_buf = &.{};
        self.scratch_lengths_cap = 0;
        self.scratch_token_buf = &.{};
        self.scratch_token_cap = 0;
        if (self.owner) |owner| {
            owner.release();
            self.owner = null;
        }
        self.initialized = false;
    }

    pub fn context(self: *const Self) FutharkContext {
        return FutharkContext.bind(self.owner);
    }

    fn requireOwnerLive(self: *const Self) AccelError!*ContextOwner {
        if (!self.initialized) return AccelError.UninitializedContext;
        const owner = self.owner orelse return AccelError.UninitializedContext;
        try owner.checkLive();
        return owner;
    }

    fn requireOwnerLiveUnlocked(self: *const Self) AccelError!*ContextOwner {
        if (!self.initialized) return AccelError.UninitializedContext;
        const owner = self.owner orelse return AccelError.UninitializedContext;
        try owner.checkLiveUnlocked();
        return owner;
    }

    pub fn checkInvariants(self: *const Self) AccelError!void {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        return self.checkInvariantsWithOwnerUnlocked(owner);
    }

    fn checkInvariantsWithOwnerUnlocked(self: *const Self, owner: *ContextOwner) AccelError!void {
        if (!self.initialized) return AccelError.UninitializedContext;
        if (self.owner != owner) return AccelError.ContextMismatch;
        if (!owner.alive or owner.ctx == null) return AccelError.UninitializedContext;
        if (self.vocab_size == 0 or self.dim == 0) return AccelError.InvalidDimensions;
        if (self.scratch_token_buf.len != self.scratch_token_cap) return AccelError.InvalidResourceState;
        if (self.scratch_lengths_buf.len != self.scratch_lengths_cap) return AccelError.InvalidResourceState;
        if (self.scratch_positions_buf.len != self.scratch_positions_cap) return AccelError.InvalidResourceState;
        try checkEmbeddingArray(self.weight, self.vocab_size, self.dim, owner.id);
    }

    fn ensureScratchTokensUnlocked(self: *Self, count: usize) AccelError![]i64 {
        if (count == 0) return AccelError.InvalidDimensions;
        if (count > self.scratch_token_cap) {
            var capacity = self.scratch_token_cap;
            if (capacity == 0) capacity = embedding_scratch_initial_tokens;
            while (capacity < count) {
                capacity = std.math.mul(usize, capacity, 2) catch return AccelError.Overflow;
            }
            const buffer = try allocHost(i64, self.allocator, capacity);
            if (self.scratch_token_buf.len > 0) {
                @memcpy(buffer[0..self.scratch_token_buf.len], self.scratch_token_buf);
                self.allocator.free(self.scratch_token_buf);
            }
            self.scratch_token_buf = buffer;
            self.scratch_token_cap = capacity;
        }
        return self.scratch_token_buf[0..count];
    }

    fn ensureScratchLengthsUnlocked(self: *Self, count: usize) AccelError![]i64 {
        if (count == 0) return AccelError.InvalidDimensions;
        if (count > self.scratch_lengths_cap) {
            var capacity = self.scratch_lengths_cap;
            if (capacity == 0) capacity = embedding_scratch_initial_lengths;
            while (capacity < count) {
                capacity = std.math.mul(usize, capacity, 2) catch return AccelError.Overflow;
            }
            const buffer = try allocHost(i64, self.allocator, capacity);
            if (self.scratch_lengths_buf.len > 0) {
                @memcpy(buffer[0..self.scratch_lengths_buf.len], self.scratch_lengths_buf);
                self.allocator.free(self.scratch_lengths_buf);
            }
            self.scratch_lengths_buf = buffer;
            self.scratch_lengths_cap = capacity;
        }
        return self.scratch_lengths_buf[0..count];
    }

    fn ensureScratchPositionsUnlocked(self: *Self, count: usize) AccelError![]i64 {
        if (count == 0) return AccelError.InvalidDimensions;
        if (count > self.scratch_positions_cap) {
            var capacity = self.scratch_positions_cap;
            if (capacity == 0) capacity = embedding_scratch_initial_positions;
            while (capacity < count) {
                capacity = std.math.mul(usize, capacity, 2) catch return AccelError.Overflow;
            }
            const buffer = try allocHost(i64, self.allocator, capacity);
            if (self.scratch_positions_buf.len > 0) {
                @memcpy(buffer[0..self.scratch_positions_buf.len], self.scratch_positions_buf);
                self.allocator.free(self.scratch_positions_buf);
            }
            self.scratch_positions_buf = buffer;
            self.scratch_positions_cap = capacity;
        }
        return self.scratch_positions_buf[0..count];
    }

    pub fn exportAsF32(self: *Self, allocator: std.mem.Allocator) AccelError![]f32 {
        const owner = try self.requireOwnerLive();
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);
        const total = try checkedElementCount2(self.vocab_size, self.dim);
        const half = try self.weight.valuesFlatUnlocked(allocator);
        defer allocator.free(half);
        if (half.len != total) return AccelError.InvalidDataLength;
        const out = try allocHost(f32, allocator, total);
        for (half, out) |value, *target| {
            target.* = @floatCast(value);
        }
        return out;
    }

    pub fn forwardPadded(
        self: *Self,
        tokens: []const u32,
        sequence_lengths: []const usize,
        sequence_length: usize,
    ) AccelError!FutharkArray3DF16 {
        const owner = try self.requireOwnerLive();
        if (sequence_lengths.len == 0 or sequence_length == 0) return AccelError.InvalidDimensions;
        const batch = sequence_lengths.len;
        const expected_tokens = try checkedElementCount2(batch, sequence_length);
        if (tokens.len != expected_tokens) return AccelError.InvalidDataLength;
        owner.lock();
        defer owner.unlock();
        try self.checkInvariantsWithOwnerUnlocked(owner);

        const token_i64s = try self.ensureScratchTokensUnlocked(tokens.len);
        for (tokens, token_i64s) |token, *slot| {
            if (@as(usize, token) >= self.vocab_size) return AccelError.InvalidToken;
            slot.* = try checkedSigned(@as(usize, token));
        }
        const lengths_i64 = try self.ensureScratchLengthsUnlocked(batch);
        for (sequence_lengths, lengths_i64) |length, *slot| {
            if (length == 0) return AccelError.InvalidSequenceLength;
            if (length > sequence_length) return AccelError.InvalidSequenceLength;
            slot.* = try checkedSigned(length);
        }
        const positions_i64 = try self.ensureScratchPositionsUnlocked(sequence_length);
        for (positions_i64, 0..) |*slot, index| {
            slot.* = try checkedSigned(index);
        }

        var token_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, token_i64s);
        defer token_array.deinitUnlocked();
        var length_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, lengths_i64);
        defer length_array.deinitUnlocked();
        var position_array = try FutharkArray1DI64.newFromSliceUnlocked(owner, positions_i64);
        defer position_array.deinitUnlocked();

        const handle = try owner.requireHandleUnlocked();
        var output: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_embedding_forward_padded(
            handle,
            &output,
            token_array.arr,
            length_array.arr,
            position_array.arr,
            self.weight.arr,
        );
        if (rc != 0 or output == null) {
            if (output) |produced| _ = futhark.futhark_free_f16_3d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkForwardFailed);
        }
        try validateShapeF16_3DUnlocked(owner, output.?, batch, sequence_length, self.dim);
        return FutharkArray3DF16.adoptUnlocked(owner, output.?, batch, sequence_length, self.dim);
    }
};

pub const GPUOps = struct {
    ctx: FutharkContext,

    const Self = @This();

    pub fn init() AccelError!Self {
        return initWithAllocator(std.heap.page_allocator);
    }

    pub fn initWithAllocator(allocator: std.mem.Allocator) AccelError!Self {
        return Self{ .ctx = try FutharkContext.initWithAllocator(allocator) };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
    }

    pub fn sync(self: *Self) AccelError!void {
        return self.ctx.sync();
    }

    pub fn matmul(self: *Self, a: *const core_tensor.Tensor, b: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (a.shape.dims.len != 2 or b.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = a.shape.dims[0];
        const inner = a.shape.dims[1];
        const cols = b.shape.dims[1];
        if (rows == 0 or inner == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (b.shape.dims[0] != inner) return AccelError.InvalidDimensions;
        if (a.data.len != try checkedElementCount2(rows, inner)) return AccelError.InvalidDataLength;
        if (b.data.len != try checkedElementCount2(inner, cols)) return AccelError.InvalidDataLength;

        var fa = try FutharkArray2DF32.fromTensor(&self.ctx, a);
        defer fa.deinit();
        var fb = try FutharkArray2DF32.fromTensor(&self.ctx, b);
        defer fb.deinit();

        const owner = try self.ctx.requireOwner();
        owner.lock();
        defer owner.unlock();
        const handle = try owner.requireHandleUnlocked();
        var out_arr: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_matmul(handle, &out_arr, fa.arr, fb.arr) != 0 or out_arr == null) {
            if (out_arr) |produced| _ = futhark.futhark_free_f32_2d(handle, produced);
            return owner.recordFailureUnlocked(AccelError.FutharkForwardFailed);
        }
        var result = FutharkArray2DF32.adoptUnlocked(owner, out_arr.?, rows, cols);
        defer result.deinitUnlocked();
        try validateShapeF32_2DUnlocked(owner, out_arr.?, rows, cols);
        return result.toTensorUnlocked(allocator);
    }
};

pub const GraphBatchEncodeResult = struct {
    hashes: []u64,
    re_a: []f32,
    im_a: []f32,
    re_b: []f32,
    im_b: []f32,
    edge_srcs: []i64,
    edge_tgts: []i64,
    node_count: usize,
    edge_count: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.hashes);
        self.allocator.free(self.re_a);
        self.allocator.free(self.im_a);
        self.allocator.free(self.re_b);
        self.allocator.free(self.im_b);
        self.allocator.free(self.edge_srcs);
        self.allocator.free(self.edge_tgts);
        self.hashes = &.{};
        self.re_a = &.{};
        self.im_a = &.{};
        self.re_b = &.{};
        self.im_b = &.{};
        self.edge_srcs = &.{};
        self.edge_tgts = &.{};
        self.node_count = 0;
        self.edge_count = 0;
    }
};

fn graphChunkSize() usize {
    if (std.posix.getenv("JAIDE_GRAPH_CHUNK_SIZE")) |raw| {
        const parsed = std.fmt.parseInt(usize, raw, 10) catch return graph_default_chunk_size;
        if (parsed == 0) return graph_default_chunk_size;
        return parsed;
    }
    return graph_default_chunk_size;
}

pub fn batchEncodeGraph(
    ctx: *FutharkContext,
    hashes: []const u64,
    seed: u64,
    allocator: std.mem.Allocator,
) AccelError!GraphBatchEncodeResult {
    const owner = try ctx.requireOwner();
    if (hashes.len == 0) return AccelError.InvalidDimensions;
    owner.lock();
    defer owner.unlock();

    var acc_hashes = std.ArrayList(u64).init(allocator);
    errdefer acc_hashes.deinit();
    var acc_re_a = std.ArrayList(f32).init(allocator);
    errdefer acc_re_a.deinit();
    var acc_im_a = std.ArrayList(f32).init(allocator);
    errdefer acc_im_a.deinit();
    var acc_re_b = std.ArrayList(f32).init(allocator);
    errdefer acc_re_b.deinit();
    var acc_im_b = std.ArrayList(f32).init(allocator);
    errdefer acc_im_b.deinit();
    var acc_edge_srcs = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_srcs.deinit();
    var acc_edge_tgts = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_tgts.deinit();

    acc_hashes.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_re_a.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_im_a.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_re_b.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_im_b.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    const edge_capacity = std.math.mul(usize, hashes.len, 3) catch return AccelError.Overflow;
    acc_edge_srcs.ensureTotalCapacity(edge_capacity) catch return AccelError.AllocationFailed;
    acc_edge_tgts.ensureTotalCapacity(edge_capacity) catch return AccelError.AllocationFailed;

    const chunk_size = graphChunkSize();
    var offset: usize = 0;
    while (offset < hashes.len) {
        const remaining = hashes.len - offset;
        const take = if (remaining < chunk_size) remaining else chunk_size;
        const chunk_end = offset + take;
        const chunk = hashes[offset..chunk_end];
        const chunk_n = chunk.len;
        const chunk_ne = std.math.mul(usize, chunk_n, 3) catch return AccelError.Overflow;

        var in_chunk = try FutharkArray1DU64.newFromSliceUnlocked(owner, chunk);
        defer in_chunk.deinitUnlocked();

        var out_tup: ?*futhark.struct_futhark_opaque_tup7_graph_encode = null;
        defer if (out_tup) |produced| {
            if (owner.requireHandleUnlocked()) |handle| {
                _ = futhark.futhark_free_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64(handle, produced);
            } else |_| {}
        };

        var out_ids: ?*futhark.struct_futhark_u64_1d = null;
        var out_re_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_re_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_edge_srcs: ?*futhark.struct_futhark_i64_1d = null;
        var out_edge_tgts: ?*futhark.struct_futhark_i64_1d = null;

        defer if (out_ids) |produced| {
            if (owner.requireHandleUnlocked()) |handle| _ = futhark.futhark_free_u64_1d(handle, produced) else |_| {}
        };
        defer if (out_re_a) |produced| {
            if (owner.requireHandleUnlocked()) |handle| _ = futhark.futhark_free_f32_1d(handle, produced) else |_| {}
        };
        defer if (out_im_a) |produced| {
            if (owner.requireHandleUnlocked()) |handle| _ = futhark.futhark_free_f32_1d(handle, produced) else |_| {}
        };
        defer if (out_re_b) |produced| {
            if (owner.requireHandleUnlocked()) |handle| _ = futhark.futhark_free_f32_1d(handle, produced) else |_| {}
        };
        defer if (out_im_b) |produced| {
            if (owner.requireHandleUnlocked()) |handle| _ = futhark.futhark_free_f32_1d(handle, produced) else |_| {}
        };
        defer if (out_edge_srcs) |produced| {
            if (owner.requireHandleUnlocked()) |handle| _ = futhark.futhark_free_i64_1d(handle, produced) else |_| {}
        };
        defer if (out_edge_tgts) |produced| {
            if (owner.requireHandleUnlocked()) |handle| _ = futhark.futhark_free_i64_1d(handle, produced) else |_| {}
        };

        const handle = try owner.requireHandleUnlocked();
        const rc = futhark.futhark_entry_graph_batch_encode(handle, &out_tup, in_chunk.arr, seed);
        if (rc != 0) return owner.recordFailureUnlocked(AccelError.FutharkForwardFailed);
        const tup = out_tup orelse return owner.recordFailureUnlocked(AccelError.NullPointer);

        const proj0 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_0(handle, &out_ids, tup);
        const proj1 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_1(handle, &out_re_a, tup);
        const proj2 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_2(handle, &out_im_a, tup);
        const proj3 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_3(handle, &out_re_b, tup);
        const proj4 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_4(handle, &out_im_b, tup);
        const proj5 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_5(handle, &out_edge_srcs, tup);
        const proj6 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_6(handle, &out_edge_tgts, tup);
        if (proj0 != 0 or proj1 != 0 or proj2 != 0 or proj3 != 0 or proj4 != 0 or proj5 != 0 or proj6 != 0) {
            return owner.recordFailureUnlocked(AccelError.FutharkProjectionFailed);
        }
        const ids_array = out_ids orelse return owner.recordFailureUnlocked(AccelError.NullPointer);
        const re_a_array = out_re_a orelse return owner.recordFailureUnlocked(AccelError.NullPointer);
        const im_a_array = out_im_a orelse return owner.recordFailureUnlocked(AccelError.NullPointer);
        const re_b_array = out_re_b orelse return owner.recordFailureUnlocked(AccelError.NullPointer);
        const im_b_array = out_im_b orelse return owner.recordFailureUnlocked(AccelError.NullPointer);
        const edge_src_array = out_edge_srcs orelse return owner.recordFailureUnlocked(AccelError.NullPointer);
        const edge_tgt_array = out_edge_tgts orelse return owner.recordFailureUnlocked(AccelError.NullPointer);

        const ids_buf = try allocHost(u64, allocator, chunk_n);
        defer allocator.free(ids_buf);
        const re_a_buf = try allocHost(f32, allocator, chunk_n);
        defer allocator.free(re_a_buf);
        const im_a_buf = try allocHost(f32, allocator, chunk_n);
        defer allocator.free(im_a_buf);
        const re_b_buf = try allocHost(f32, allocator, chunk_n);
        defer allocator.free(re_b_buf);
        const im_b_buf = try allocHost(f32, allocator, chunk_n);
        defer allocator.free(im_b_buf);
        const edge_src_buf = try allocHost(i64, allocator, chunk_ne);
        defer allocator.free(edge_src_buf);
        const edge_tgt_buf = try allocHost(i64, allocator, chunk_ne);
        defer allocator.free(edge_tgt_buf);

        if (futhark.futhark_values_u64_1d(handle, ids_array, ids_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(handle, re_a_array, re_a_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(handle, im_a_array, im_a_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(handle, re_b_array, re_b_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(handle, im_b_array, im_b_buf.ptr) != 0 or
            futhark.futhark_values_i64_1d(handle, edge_src_array, edge_src_buf.ptr) != 0 or
            futhark.futhark_values_i64_1d(handle, edge_tgt_array, edge_tgt_buf.ptr) != 0)
        {
            return owner.recordFailureUnlocked(AccelError.FutharkValuesFailed);
        }
        try owner.syncContextUnlocked();

        acc_hashes.appendSlice(ids_buf) catch return AccelError.AllocationFailed;
        acc_re_a.appendSlice(re_a_buf) catch return AccelError.AllocationFailed;
        acc_im_a.appendSlice(im_a_buf) catch return AccelError.AllocationFailed;
        acc_re_b.appendSlice(re_b_buf) catch return AccelError.AllocationFailed;
        acc_im_b.appendSlice(im_b_buf) catch return AccelError.AllocationFailed;
        const index_offset: i64 = try checkedSigned(offset);
        for (edge_src_buf) |value| {
            acc_edge_srcs.append(if (value >= 0) value + index_offset else value) catch return AccelError.AllocationFailed;
        }
        for (edge_tgt_buf) |value| {
            acc_edge_tgts.append(if (value >= 0) value + index_offset else value) catch return AccelError.AllocationFailed;
        }

        offset = chunk_end;
    }

    const total_n = acc_hashes.items.len;
    const total_ne = acc_edge_srcs.items.len;

    const owned_hashes = acc_hashes.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_hashes);
    const owned_re_a = acc_re_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_a);
    const owned_im_a = acc_im_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_a);
    const owned_re_b = acc_re_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_b);
    const owned_im_b = acc_im_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_b);
    const owned_edge_srcs = acc_edge_srcs.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_edge_srcs);
    const owned_edge_tgts = acc_edge_tgts.toOwnedSlice() catch return AccelError.AllocationFailed;

    return GraphBatchEncodeResult{
        .hashes = owned_hashes,
        .re_a = owned_re_a,
        .im_a = owned_im_a,
        .re_b = owned_re_b,
        .im_b = owned_im_b,
        .edge_srcs = owned_edge_srcs,
        .edge_tgts = owned_edge_tgts,
        .node_count = total_n,
        .edge_count = total_ne,
        .allocator = allocator,
    };
}
