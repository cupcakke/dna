const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const core_tensor = @import("../core/tensor.zig");
const core_types = @import("../core/types.zig");
const rsf_mod = @import("../processor/rsf.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const PriorityQueue = std.PriorityQueue;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
/// The RSF substrate this fabric executes. Re-exported so every RSF entry
/// point of `RelationalGraphProcessingUnit` is reached through an RSF handle.
pub const RSF = rsf_mod.RSF;
pub const RSFLatentState = rsf_mod.RSFLatentState;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;
pub const EdgeKey = nsir_core.EdgeKey;
pub const EdgeKeyContext = nsir_core.EdgeKeyContext;
pub const Qubit = nsir_core.Qubit;

pub const max_p2p_devices: usize = 8;
pub const cuda_memcpy_host_to_device_kind: c_uint = 1;
pub const cuda_memcpy_default_kind: c_uint = 4;
pub const cuda_stream_non_blocking_flag: c_uint = 1;
pub const nvlink5_link_count_per_gpu: u32 = 18;
pub const nvlink5_bandwidth_bytes_per_sec: u64 = 900_000_000_000;

pub const CudaP2PError = error{
    CudaRuntimeUnavailable,
    CudaDeviceQueryFailed,
    CudaPeerAccessFailed,
    CudaAllocationFailed,
    CudaTransferFailed,
    CudaStreamFailed,
    ShardBufferTooSmall,
    DeviceIndexOutOfRange,
};

const CudaGetDeviceCountFn = *const fn (*c_int) callconv(.C) c_int;
const CudaSetDeviceFn = *const fn (c_int) callconv(.C) c_int;
const CudaGetDeviceFn = *const fn (*c_int) callconv(.C) c_int;
const CudaDeviceCanAccessPeerFn = *const fn (*c_int, c_int, c_int) callconv(.C) c_int;
const CudaDeviceEnablePeerAccessFn = *const fn (c_int, c_uint) callconv(.C) c_int;
const CudaDeviceDisablePeerAccessFn = *const fn (c_int) callconv(.C) c_int;
const CudaMallocFn = *const fn (*?*anyopaque, usize) callconv(.C) c_int;
const CudaFreeFn = *const fn (?*anyopaque) callconv(.C) c_int;
const CudaMemcpyAsyncFn = *const fn (?*anyopaque, ?*const anyopaque, usize, c_uint, ?*anyopaque) callconv(.C) c_int;
const CudaMemcpyPeerAsyncFn = *const fn (?*anyopaque, c_int, ?*const anyopaque, c_int, usize, ?*anyopaque) callconv(.C) c_int;
const CudaStreamCreateWithFlagsFn = *const fn (*?*anyopaque, c_uint) callconv(.C) c_int;
const CudaStreamDestroyFn = *const fn (?*anyopaque) callconv(.C) c_int;
const CudaStreamSynchronizeFn = *const fn (?*anyopaque) callconv(.C) c_int;

pub const DeviceShardBuffer = struct {
    ptr: ?*anyopaque = null,
    bytes: usize = 0,
    device: c_int = -1,

    pub fn isAllocated(self: *const DeviceShardBuffer) bool {
        return self.ptr != null and self.bytes > 0;
    }
};

pub const DeviceShardSnapshot = struct {
    device: c_int,
    bytes: usize,
    resident: bool,
};

pub const P2PTransferStatistics = struct {
    device_count: usize,
    nvlink5_mesh: bool,
    peer_pairs_enabled: usize,
    total_bytes_transferred: usize,
    total_peer_transfers: usize,
};

pub const P2PTransferManager = struct {
    lib: ?std.DynLib,
    get_device_count: ?CudaGetDeviceCountFn,
    set_device: ?CudaSetDeviceFn,
    get_device: ?CudaGetDeviceFn,
    can_access_peer: ?CudaDeviceCanAccessPeerFn,
    enable_peer_access: ?CudaDeviceEnablePeerAccessFn,
    disable_peer_access: ?CudaDeviceDisablePeerAccessFn,
    malloc_fn: ?CudaMallocFn,
    free_fn: ?CudaFreeFn,
    memcpy_async: ?CudaMemcpyAsyncFn,
    memcpy_peer_async: ?CudaMemcpyPeerAsyncFn,
    stream_create: ?CudaStreamCreateWithFlagsFn,
    stream_destroy: ?CudaStreamDestroyFn,
    stream_sync: ?CudaStreamSynchronizeFn,
    device_count: usize,
    peer_matrix: [max_p2p_devices][max_p2p_devices]bool,
    nvlink5_mesh: bool,
    streams: [max_p2p_devices]?*anyopaque,
    shards: [max_p2p_devices]DeviceShardBuffer,
    staging: DeviceShardBuffer,
    total_bytes_transferred: usize,
    total_peer_transfers: usize,
    enabled: bool,
    allocator: Allocator,

    pub fn init(allocator: Allocator) P2PTransferManager {
        var mgr = P2PTransferManager{
            .lib = null,
            .get_device_count = null,
            .set_device = null,
            .get_device = null,
            .can_access_peer = null,
            .enable_peer_access = null,
            .disable_peer_access = null,
            .malloc_fn = null,
            .free_fn = null,
            .memcpy_async = null,
            .memcpy_peer_async = null,
            .stream_create = null,
            .stream_destroy = null,
            .stream_sync = null,
            .device_count = 0,
            .peer_matrix = [_][max_p2p_devices]bool{[_]bool{false} ** max_p2p_devices} ** max_p2p_devices,
            .nvlink5_mesh = false,
            .streams = [_]?*anyopaque{null} ** max_p2p_devices,
            .shards = [_]DeviceShardBuffer{.{}} ** max_p2p_devices,
            .staging = .{},
            .total_bytes_transferred = 0,
            .total_peer_transfers = 0,
            .enabled = false,
            .allocator = allocator,
        };
        mgr.loadRuntime();
        return mgr;
    }

    pub fn initDisabled(allocator: Allocator) P2PTransferManager {
        return P2PTransferManager{
            .lib = null,
            .get_device_count = null,
            .set_device = null,
            .get_device = null,
            .can_access_peer = null,
            .enable_peer_access = null,
            .disable_peer_access = null,
            .malloc_fn = null,
            .free_fn = null,
            .memcpy_async = null,
            .memcpy_peer_async = null,
            .stream_create = null,
            .stream_destroy = null,
            .stream_sync = null,
            .device_count = 0,
            .peer_matrix = [_][max_p2p_devices]bool{[_]bool{false} ** max_p2p_devices} ** max_p2p_devices,
            .nvlink5_mesh = false,
            .streams = [_]?*anyopaque{null} ** max_p2p_devices,
            .shards = [_]DeviceShardBuffer{.{}} ** max_p2p_devices,
            .staging = .{},
            .total_bytes_transferred = 0,
            .total_peer_transfers = 0,
            .enabled = false,
            .allocator = allocator,
        };
    }

    fn loadRuntime(self: *P2PTransferManager) void {
        const candidates = [_][:0]const u8{
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11.0",
        };
        var lib: ?std.DynLib = null;
        for (candidates) |name| {
            lib = std.DynLib.open(name) catch null;
            if (lib != null) break;
        }
        const opened = lib orelse return;
        self.lib = opened;
        if (self.lib) |*l| {
            self.get_device_count = l.lookup(CudaGetDeviceCountFn, "cudaGetDeviceCount");
            self.set_device = l.lookup(CudaSetDeviceFn, "cudaSetDevice");
            self.get_device = l.lookup(CudaGetDeviceFn, "cudaGetDevice");
            self.can_access_peer = l.lookup(CudaDeviceCanAccessPeerFn, "cudaDeviceCanAccessPeer");
            self.enable_peer_access = l.lookup(CudaDeviceEnablePeerAccessFn, "cudaDeviceEnablePeerAccess");
            self.disable_peer_access = l.lookup(CudaDeviceDisablePeerAccessFn, "cudaDeviceDisablePeerAccess");
            self.malloc_fn = l.lookup(CudaMallocFn, "cudaMalloc");
            self.free_fn = l.lookup(CudaFreeFn, "cudaFree");
            self.memcpy_async = l.lookup(CudaMemcpyAsyncFn, "cudaMemcpyAsync");
            self.memcpy_peer_async = l.lookup(CudaMemcpyPeerAsyncFn, "cudaMemcpyPeerAsync");
            self.stream_create = l.lookup(CudaStreamCreateWithFlagsFn, "cudaStreamCreateWithFlags");
            self.stream_destroy = l.lookup(CudaStreamDestroyFn, "cudaStreamDestroy");
            self.stream_sync = l.lookup(CudaStreamSynchronizeFn, "cudaStreamSynchronize");
        }
        const required_available = self.get_device_count != null and
            self.set_device != null and
            self.get_device != null and
            self.can_access_peer != null and
            self.enable_peer_access != null and
            self.malloc_fn != null and
            self.free_fn != null and
            self.memcpy_async != null and
            self.memcpy_peer_async != null and
            self.stream_create != null and
            self.stream_destroy != null and
            self.stream_sync != null;
        if (!required_available) return;
        var count: c_int = 0;
        const gdc = self.get_device_count.?;
        if (gdc(&count) != 0 or count <= 0) return;
        self.device_count = @min(@as(usize, @intCast(count)), max_p2p_devices);
        self.enabled = true;
        self.discoverTopology() catch {
            self.enabled = false;
        };
    }

    fn discoverTopology(self: *P2PTransferManager) CudaP2PError!void {
        const set_device = self.set_device orelse return CudaP2PError.CudaRuntimeUnavailable;
        const can_access = self.can_access_peer orelse return CudaP2PError.CudaRuntimeUnavailable;
        const enable_access = self.enable_peer_access orelse return CudaP2PError.CudaRuntimeUnavailable;
        const stream_create = self.stream_create orelse return CudaP2PError.CudaRuntimeUnavailable;
        var src: usize = 0;
        while (src < self.device_count) : (src += 1) {
            const src_dev: c_int = @intCast(src);
            if (set_device(src_dev) != 0) return CudaP2PError.CudaDeviceQueryFailed;
            var stream: ?*anyopaque = null;
            if (stream_create(&stream, cuda_stream_non_blocking_flag) != 0) return CudaP2PError.CudaStreamFailed;
            self.streams[src] = stream;
            var dst: usize = 0;
            while (dst < self.device_count) : (dst += 1) {
                if (dst == src) {
                    self.peer_matrix[src][dst] = true;
                    continue;
                }
                const dst_dev: c_int = @intCast(dst);
                var accessible: c_int = 0;
                if (can_access(&accessible, src_dev, dst_dev) != 0) return CudaP2PError.CudaDeviceQueryFailed;
                if (accessible != 1) {
                    self.peer_matrix[src][dst] = false;
                    continue;
                }
                if (enable_access(dst_dev, 0) != 0) {
                    self.peer_matrix[src][dst] = false;
                    continue;
                }
                self.peer_matrix[src][dst] = true;
            }
        }
        if (set_device(0) != 0) return CudaP2PError.CudaDeviceQueryFailed;
        var full_mesh = self.device_count >= 2;
        var s: usize = 0;
        while (s < self.device_count) : (s += 1) {
            var d: usize = 0;
            while (d < self.device_count) : (d += 1) {
                if (!self.peer_matrix[s][d]) full_mesh = false;
            }
        }
        self.nvlink5_mesh = full_mesh;
    }

    pub fn deinit(self: *P2PTransferManager) void {
        if (self.lib != null) {
            var dev: usize = 0;
            while (dev < self.device_count) : (dev += 1) {
                if (self.shards[dev].ptr != null) {
                    self.freeShard(@intCast(dev));
                }
                if (self.streams[dev]) |stream| {
                    if (self.stream_sync) |sync_fn| {
                        _ = sync_fn(stream);
                    }
                    if (self.stream_destroy) |destroy_fn| {
                        _ = destroy_fn(stream);
                    }
                    self.streams[dev] = null;
                }
                var peer: usize = 0;
                while (peer < self.device_count) : (peer += 1) {
                    if (peer != dev and self.peer_matrix[dev][peer]) {
                        if (self.disable_peer_access) |disable_fn| {
                            if (self.set_device) |sd| {
                                if (sd(@intCast(dev)) == 0) {
                                    _ = disable_fn(@intCast(peer));
                                }
                            }
                        }
                    }
                }
            }
            if (self.staging.ptr != null) {
                self.freeStaging();
            }
            if (self.set_device) |sd| {
                _ = sd(0);
            }
        }
        if (self.lib) |*l| {
            l.close();
            self.lib = null;
        }
        self.enabled = false;
    }

    pub fn isAvailable(self: *const P2PTransferManager) bool {
        return self.enabled and self.device_count > 0;
    }

    pub fn deviceCount(self: *const P2PTransferManager) usize {
        return self.device_count;
    }

    pub fn peerAccessible(self: *const P2PTransferManager, src_device: usize, dst_device: usize) bool {
        if (src_device >= max_p2p_devices or dst_device >= max_p2p_devices) return false;
        return self.peer_matrix[src_device][dst_device];
    }

    pub fn isNvlink5Mesh(self: *const P2PTransferManager) bool {
        return self.nvlink5_mesh;
    }

    pub fn getStatistics(self: *const P2PTransferManager) P2PTransferStatistics {
        var pairs: usize = 0;
        var s: usize = 0;
        while (s < self.device_count) : (s += 1) {
            var d: usize = 0;
            while (d < self.device_count) : (d += 1) {
                if (s != d and self.peer_matrix[s][d]) pairs += 1;
            }
        }
        return P2PTransferStatistics{
            .device_count = self.device_count,
            .nvlink5_mesh = self.nvlink5_mesh,
            .peer_pairs_enabled = pairs,
            .total_bytes_transferred = self.total_bytes_transferred,
            .total_peer_transfers = self.total_peer_transfers,
        };
    }

    fn ensureDeviceBuffer(self: *P2PTransferManager, slot: *DeviceShardBuffer, device: c_int, bytes: usize) CudaP2PError!void {
        const malloc_fn = self.malloc_fn orelse return CudaP2PError.CudaRuntimeUnavailable;
        const free_fn = self.free_fn orelse return CudaP2PError.CudaRuntimeUnavailable;
        const set_device = self.set_device orelse return CudaP2PError.CudaRuntimeUnavailable;
        if (slot.ptr != null and slot.bytes >= bytes and slot.device == device) return;
        var previous: c_int = 0;
        if (self.get_device) |gd| {
            if (gd(&previous) != 0) previous = 0;
        }
        if (set_device(device) != 0) return CudaP2PError.CudaDeviceQueryFailed;
        if (slot.ptr != null) {
            _ = free_fn(slot.ptr);
            slot.ptr = null;
            slot.bytes = 0;
        }
        var dev_ptr: ?*anyopaque = null;
        if (malloc_fn(&dev_ptr, bytes) != 0 or dev_ptr == null) {
            _ = set_device(previous);
            return CudaP2PError.CudaAllocationFailed;
        }
        slot.ptr = dev_ptr;
        slot.bytes = bytes;
        slot.device = device;
        _ = set_device(previous);
    }

    fn freeShard(self: *P2PTransferManager, device: c_int) void {
        const idx: usize = @intCast(device);
        if (idx >= max_p2p_devices) return;
        if (self.free_fn) |free_fn| {
            if (self.shards[idx].ptr) |p| {
                if (self.set_device) |sd| {
                    _ = sd(device);
                }
                _ = free_fn(p);
            }
        }
        self.shards[idx] = .{};
    }

    fn freeStaging(self: *P2PTransferManager) void {
        if (self.free_fn) |free_fn| {
            if (self.staging.ptr) |p| {
                if (self.set_device) |sd| {
                    _ = sd(if (self.staging.device >= 0) self.staging.device else 0);
                }
                _ = free_fn(p);
            }
        }
        self.staging = .{};
    }

    pub fn stageShardOnDevice(self: *P2PTransferManager, dst_device: usize, host_bytes: []const u8) CudaP2PError!void {
        if (!self.isAvailable()) return CudaP2PError.CudaRuntimeUnavailable;
        if (dst_device >= self.device_count) return CudaP2PError.DeviceIndexOutOfRange;
        if (host_bytes.len == 0) return CudaP2PError.ShardBufferTooSmall;
        const memcpy_async = self.memcpy_async orelse return CudaP2PError.CudaRuntimeUnavailable;
        const memcpy_peer = self.memcpy_peer_async orelse return CudaP2PError.CudaRuntimeUnavailable;
        const dst_dev: c_int = @intCast(dst_device);
        try self.ensureDeviceBuffer(&self.shards[dst_device], dst_dev, host_bytes.len);
        if (dst_device == 0) {
            const stream = self.streams[0];
            const dst_ptr = self.shards[0].ptr orelse return CudaP2PError.CudaAllocationFailed;
            if (memcpy_async(dst_ptr, host_bytes.ptr, host_bytes.len, cuda_memcpy_host_to_device_kind, stream) != 0) {
                return CudaP2PError.CudaTransferFailed;
            }
            self.total_bytes_transferred += host_bytes.len;
            return;
        }
        if (!self.peer_matrix[0][dst_device]) return CudaP2PError.CudaPeerAccessFailed;
        try self.ensureDeviceBuffer(&self.staging, 0, host_bytes.len);
        const staging_ptr = self.staging.ptr orelse return CudaP2PError.CudaAllocationFailed;
        const src_stream = self.streams[0];
        if (memcpy_async(staging_ptr, host_bytes.ptr, host_bytes.len, cuda_memcpy_host_to_device_kind, src_stream) != 0) {
            return CudaP2PError.CudaTransferFailed;
        }
        const dst_ptr = self.shards[dst_device].ptr orelse return CudaP2PError.CudaAllocationFailed;
        if (memcpy_peer(dst_ptr, dst_dev, staging_ptr, 0, host_bytes.len, src_stream) != 0) {
            return CudaP2PError.CudaTransferFailed;
        }
        self.total_bytes_transferred += host_bytes.len;
        self.total_peer_transfers += 1;
    }

    pub fn synchronizeAllStreams(self: *P2PTransferManager) CudaP2PError!void {
        if (!self.isAvailable()) return CudaP2PError.CudaRuntimeUnavailable;
        const sync_fn = self.stream_sync orelse return CudaP2PError.CudaRuntimeUnavailable;
        var dev: usize = 0;
        while (dev < self.device_count) : (dev += 1) {
            if (self.streams[dev]) |stream| {
                if (sync_fn(stream) != 0) return CudaP2PError.CudaStreamFailed;
            }
        }
    }

    pub fn shardSnapshot(self: *const P2PTransferManager, device: usize) ?DeviceShardSnapshot {
        if (device >= self.device_count) return null;
        const shard = self.shards[device];
        return DeviceShardSnapshot{
            .device = shard.device,
            .bytes = shard.bytes,
            .resident = shard.isAllocated(),
        };
    }
};

pub const P2PDistributionReport = struct {
    p2p_used: bool,
    devices_used: usize,
    shards_staged: usize,
    bytes_moved: usize,
    nvlink5_mesh: bool,
};

pub const CoreState = enum(u8) {
    idle = 0,
    processing = 1,
    communicating = 2,
    power_gated = 3,

    pub fn toString(self: CoreState) []const u8 {
        return switch (self) {
            .idle => "idle",
            .processing => "processing",
            .communicating => "communicating",
            .power_gated => "power_gated",
        };
    }

    pub fn fromString(s: []const u8) ?CoreState {
        if (std.mem.eql(u8, s, "idle")) return .idle;
        if (std.mem.eql(u8, s, "processing")) return .processing;
        if (std.mem.eql(u8, s, "communicating")) return .communicating;
        if (std.mem.eql(u8, s, "power_gated")) return .power_gated;
        return null;
    }
};

pub const MessageType = enum(u8) {
    weight_update = 0,
    graph_sync = 1,
    isomorphism_result = 2,
    power_control = 3,
    data_transfer = 4,
    /// RSF coupling: even/odd half update delivered to the owning core.
    rsf_forward_packet = 5,
    /// RSF coupling: inverse-half update delivered to the owning core.
    rsf_inverse_packet = 6,
    /// Per-core `Σ_d c[d]` reduction for the layer log-det.
    rsf_reduce_logdet = 7,
    /// Accumulated parameter-gradient packet for one layer shard.
    rsf_grad_packet = 8,
    /// Butterfly stage `h` crossing a core boundary: each core sends its
    /// `h`-partner lanes to the owning neighbour and receives its own.
    rsf_diffuse_exchange = 9,
    /// `Q_r` factor: per-offset partial sums reduced across the `r`
    /// block-owners.
    rsf_diffuse_sum = 10,
    /// `Q_r` factor: the `−(2/r)·S[o]` correction broadcast back.
    rsf_diffuse_broadcast = 11,
    /// Midpoint collision residual `z_M − w_M` exchanged once at the end of a
    /// dual-frontier traversal.
    rsf_collision_residual = 12,

    pub fn toString(self: MessageType) []const u8 {
        return switch (self) {
            .weight_update => "weight_update",
            .graph_sync => "graph_sync",
            .isomorphism_result => "isomorphism_result",
            .power_control => "power_control",
            .data_transfer => "data_transfer",
            .rsf_forward_packet => "rsf_forward_packet",
            .rsf_inverse_packet => "rsf_inverse_packet",
            .rsf_reduce_logdet => "rsf_reduce_logdet",
            .rsf_grad_packet => "rsf_grad_packet",
            .rsf_diffuse_exchange => "rsf_diffuse_exchange",
            .rsf_diffuse_sum => "rsf_diffuse_sum",
            .rsf_diffuse_broadcast => "rsf_diffuse_broadcast",
            .rsf_collision_residual => "rsf_collision_residual",
        };
    }

    pub fn fromString(s: []const u8) ?MessageType {
        if (std.mem.eql(u8, s, "weight_update")) return .weight_update;
        if (std.mem.eql(u8, s, "graph_sync")) return .graph_sync;
        if (std.mem.eql(u8, s, "isomorphism_result")) return .isomorphism_result;
        if (std.mem.eql(u8, s, "power_control")) return .power_control;
        if (std.mem.eql(u8, s, "data_transfer")) return .data_transfer;
        if (std.mem.eql(u8, s, "rsf_forward_packet")) return .rsf_forward_packet;
        if (std.mem.eql(u8, s, "rsf_inverse_packet")) return .rsf_inverse_packet;
        if (std.mem.eql(u8, s, "rsf_reduce_logdet")) return .rsf_reduce_logdet;
        if (std.mem.eql(u8, s, "rsf_grad_packet")) return .rsf_grad_packet;
        if (std.mem.eql(u8, s, "rsf_diffuse_exchange")) return .rsf_diffuse_exchange;
        if (std.mem.eql(u8, s, "rsf_diffuse_sum")) return .rsf_diffuse_sum;
        if (std.mem.eql(u8, s, "rsf_diffuse_broadcast")) return .rsf_diffuse_broadcast;
        if (std.mem.eql(u8, s, "rsf_collision_residual")) return .rsf_collision_residual;
        return null;
    }
};

pub const ProcessingCore = struct {
    core_id: usize,
    x: usize,
    y: usize,
    state: CoreState,
    neighbors: ArrayList(usize),
    local_graph: ?*SelfSimilarRelationalGraph,
    local_graph_owned: bool,
    /// Primary RSF shard for this core (mirrors `rsf_shards[0]` when the core
    /// holds any). `null` before `distributeRSFModel` is called.
    rsf_shard: ?Shard,
    /// Every `(layer, dim-range)` shard owned by this core, indexed by layer.
    rsf_shards: ArrayList(Shard),
    message_queue: ArrayList(NoCMessage),
    energy_consumed: f64,
    cycles_active: usize,
    cycles_idle: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, core_id: usize, x: usize, y: usize) ProcessingCore {
        return ProcessingCore{
            .core_id = core_id,
            .x = x,
            .y = y,
            .state = .idle,
            .neighbors = ArrayList(usize).init(allocator),
            .local_graph = null,
            .local_graph_owned = false,
            .rsf_shard = null,
            .rsf_shards = ArrayList(Shard).init(allocator),
            .message_queue = ArrayList(NoCMessage).init(allocator),
            .energy_consumed = 0.0,
            .cycles_active = 0,
            .cycles_idle = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ProcessingCore) void {
        for (self.rsf_shards.items) |*shard| shard.deinit();
        self.rsf_shards.deinit();
        self.rsf_shard = null;
        self.neighbors.deinit();
        for (self.message_queue.items) |*msg| {
            msg.deinit();
        }
        self.message_queue.deinit();
        if (self.local_graph_owned) {
            if (self.local_graph) |graph| {
                graph.deinit();
                self.allocator.destroy(graph);
            }
        }
    }

    pub fn addNeighbor(self: *ProcessingCore, neighbor_id: usize) !void {
        try self.neighbors.append(neighbor_id);
    }

    pub fn setLocalGraph(self: *ProcessingCore, graph: *SelfSimilarRelationalGraph, owned: bool) void {
        if (self.local_graph_owned) {
            if (self.local_graph) |old_graph| {
                old_graph.deinit();
                self.allocator.destroy(old_graph);
            }
        }
        self.local_graph = graph;
        self.local_graph_owned = owned;
    }

    pub fn createLocalGraph(self: *ProcessingCore) !*SelfSimilarRelationalGraph {
        const graph = try self.allocator.create(SelfSimilarRelationalGraph);
        errdefer self.allocator.destroy(graph);
        graph.* = try SelfSimilarRelationalGraph.init(self.allocator);
        self.setLocalGraph(graph, true);
        return graph;
    }

    pub fn enqueueMessage(self: *ProcessingCore, message: NoCMessage) !void {
        try self.message_queue.append(message);
    }

    pub fn processMessages(self: *ProcessingCore) usize {
        const count = self.message_queue.items.len;
        for (self.message_queue.items) |*msg| {
            msg.deinit();
        }
        self.message_queue.clearRetainingCapacity();
        return count;
    }

    pub fn getWorkload(self: *const ProcessingCore) f64 {
        const total = self.cycles_active + self.cycles_idle;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cycles_active)) / @as(f64, @floatFromInt(total));
    }

    pub fn clone(self: *const ProcessingCore, allocator: Allocator) !ProcessingCore {
        var new_core = ProcessingCore{
            .core_id = self.core_id,
            .x = self.x,
            .y = self.y,
            .state = self.state,
            .neighbors = ArrayList(usize).init(allocator),
            .local_graph = null,
            .local_graph_owned = false,
            .rsf_shard = null,
            .rsf_shards = ArrayList(Shard).init(allocator),
            .message_queue = ArrayList(NoCMessage).init(allocator),
            .energy_consumed = self.energy_consumed,
            .cycles_active = self.cycles_active,
            .cycles_idle = self.cycles_idle,
            .allocator = allocator,
        };
        for (self.neighbors.items) |neighbor| {
            try new_core.neighbors.append(neighbor);
        }
        return new_core;
    }
};

pub const NoCMessage = struct {
    source_core: usize,
    target_core: usize,
    message_type: MessageType,
    payload: []const u8,
    timestamp: i64,
    priority: i32,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        source_core: usize,
        target_core: usize,
        message_type: MessageType,
        payload: []const u8,
        priority: i32,
    ) !NoCMessage {
        return NoCMessage{
            .source_core = source_core,
            .target_core = target_core,
            .message_type = message_type,
            .payload = try allocator.dupe(u8, payload),
            .timestamp = @as(i64, @intCast(std.time.nanoTimestamp())),
            .priority = priority,
            .allocator = allocator,
        };
    }

    pub fn initWithTimestamp(
        allocator: Allocator,
        source_core: usize,
        target_core: usize,
        message_type: MessageType,
        payload: []const u8,
        timestamp: i64,
        priority: i32,
    ) !NoCMessage {
        return NoCMessage{
            .source_core = source_core,
            .target_core = target_core,
            .message_type = message_type,
            .payload = try allocator.dupe(u8, payload),
            .timestamp = timestamp,
            .priority = priority,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *NoCMessage) void {
        self.allocator.free(self.payload);
    }

    pub fn clone(self: *const NoCMessage, allocator: Allocator) !NoCMessage {
        return NoCMessage{
            .source_core = self.source_core,
            .target_core = self.target_core,
            .message_type = self.message_type,
            .payload = try allocator.dupe(u8, self.payload),
            .timestamp = self.timestamp,
            .priority = self.priority,
            .allocator = allocator,
        };
    }
};

const MessagePriorityEntry = struct {
    priority: i32,
    sequence: usize,
    message: NoCMessage,

    fn compare(_: void, a: MessagePriorityEntry, b: MessagePriorityEntry) std.math.Order {
        if (a.priority != b.priority) {
            return if (a.priority < b.priority) .lt else .gt;
        }
        if (a.sequence != b.sequence) {
            return if (a.sequence < b.sequence) .lt else .gt;
        }
        return .eq;
    }
};

pub const RouteKey = struct {
    source: usize,
    destination: usize,
};

pub const RouteKeyContext = struct {
    pub fn hash(_: @This(), key: RouteKey) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&key.source));
        hasher.update(std.mem.asBytes(&key.destination));
        return hasher.final();
    }

    pub fn eql(_: @This(), a: RouteKey, b: RouteKey) bool {
        return a.source == b.source and a.destination == b.destination;
    }
};

pub const AsynchronousNoC = struct {
    grid_width: usize,
    grid_height: usize,
    cores: AutoHashMap(usize, ProcessingCore),
    routing_table: std.HashMap(RouteKey, ArrayList(usize), RouteKeyContext, std.hash_map.default_max_load_percentage),
    message_buffer: PriorityQueue(MessagePriorityEntry, void, MessagePriorityEntry.compare),
    total_messages: usize,
    total_hops: usize,
    message_sequence: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, grid_width: usize, grid_height: usize) !AsynchronousNoC {
        var noc = AsynchronousNoC{
            .grid_width = grid_width,
            .grid_height = grid_height,
            .cores = AutoHashMap(usize, ProcessingCore).init(allocator),
            .routing_table = std.HashMap(RouteKey, ArrayList(usize), RouteKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .message_buffer = PriorityQueue(MessagePriorityEntry, void, MessagePriorityEntry.compare).init(allocator, {}),
            .total_messages = 0,
            .total_hops = 0,
            .message_sequence = 0,
            .allocator = allocator,
        };
        errdefer noc.deinit();
        try noc.initializeCores();
        try noc.buildRoutingTable();
        return noc;
    }

    pub fn deinit(self: *AsynchronousNoC) void {
        var core_iter = self.cores.iterator();
        while (core_iter.next()) |entry| {
            var core = entry.value_ptr;
            core.deinit();
        }
        self.cores.deinit();

        var route_iter = self.routing_table.iterator();
        while (route_iter.next()) |entry| {
            var path = entry.value_ptr;
            path.deinit();
        }
        self.routing_table.deinit();

        while (self.message_buffer.count() > 0) {
            var entry = self.message_buffer.remove();
            entry.message.deinit();
        }
        self.message_buffer.deinit();
    }

    pub fn initializeCores(self: *AsynchronousNoC) !void {
        var core_id: usize = 0;
        var y: usize = 0;
        while (y < self.grid_height) : (y += 1) {
            var x: usize = 0;
            while (x < self.grid_width) : (x += 1) {
                var core = ProcessingCore.init(self.allocator, core_id, x, y);
                var core_transferred = false;
                defer if (!core_transferred) core.deinit();
                if (x > 0) {
                    try core.addNeighbor(core_id - 1);
                }
                if (x < self.grid_width - 1) {
                    try core.addNeighbor(core_id + 1);
                }
                if (y > 0) {
                    try core.addNeighbor(core_id - self.grid_width);
                }
                if (y < self.grid_height - 1) {
                    try core.addNeighbor(core_id + self.grid_width);
                }
                try self.cores.putNoClobber(core_id, core);
                core_transferred = true;
                core_id += 1;
            }
        }
    }

    pub fn buildRoutingTable(self: *AsynchronousNoC) !void {
        var src_iter = self.cores.iterator();
        while (src_iter.next()) |src_entry| {
            const src_id = src_entry.key_ptr.*;
            var dst_iter = self.cores.iterator();
            while (dst_iter.next()) |dst_entry| {
                const dst_id = dst_entry.key_ptr.*;
                if (src_id != dst_id) {
                    const route_key = RouteKey{ .source = src_id, .destination = dst_id };
                    var path = try self.computeXYRoute(src_id, dst_id);
                    var path_transferred = false;
                    defer if (!path_transferred) path.deinit();
                    try self.routing_table.putNoClobber(route_key, path);
                    path_transferred = true;
                }
            }
        }
    }

    pub fn computeXYRoute(self: *AsynchronousNoC, src_id: usize, dst_id: usize) !ArrayList(usize) {
        var path = ArrayList(usize).init(self.allocator);
        errdefer path.deinit();

        const src_core = self.cores.get(src_id) orelse return path;
        const dst_core = self.cores.get(dst_id) orelse return path;

        try path.append(src_id);
        var current_x = src_core.x;
        var current_y = src_core.y;

        while (current_x != dst_core.x) {
            if (current_x < dst_core.x) {
                current_x += 1;
            } else {
                current_x -= 1;
            }
            const next_id = current_y * self.grid_width + current_x;
            try path.append(next_id);
        }

        while (current_y != dst_core.y) {
            if (current_y < dst_core.y) {
                current_y += 1;
            } else {
                current_y -= 1;
            }
            const next_id = current_y * self.grid_width + current_x;
            try path.append(next_id);
        }

        return path;
    }

    pub fn sendMessage(self: *AsynchronousNoC, message: NoCMessage) !bool {
        if (!self.cores.contains(message.source_core) or !self.cores.contains(message.target_core)) {
            return false;
        }

        const entry = MessagePriorityEntry{
            .priority = message.priority,
            .sequence = self.message_sequence,
            .message = message,
        };
        try self.message_buffer.add(entry);
        self.message_sequence += 1;
        self.total_messages += 1;
        return true;
    }

    pub fn routeMessages(self: *AsynchronousNoC) !usize {
        var routed_count: usize = 0;
        while (self.message_buffer.count() > 0) {
            var entry = self.message_buffer.remove();
            defer entry.message.deinit();

            const route_key = RouteKey{ .source = entry.message.source_core, .destination = entry.message.target_core };
            if (self.routing_table.get(route_key)) |path| {
                if (path.items.len > 1) {
                    self.total_hops = std.math.add(usize, self.total_hops, path.items.len - 1) catch return error.Overflow;
                }
            }

            if (self.cores.getPtr(entry.message.target_core)) |target_core| {
                var message_clone = try entry.message.clone(self.allocator);
                var clone_transferred = false;
                defer if (!clone_transferred) message_clone.deinit();
                try target_core.enqueueMessage(message_clone);
                clone_transferred = true;
                routed_count = std.math.add(usize, routed_count, 1) catch return error.Overflow;
            }
        }
        return routed_count;
    }

    pub fn getCore(self: *AsynchronousNoC, core_id: usize) ?*ProcessingCore {
        return self.cores.getPtr(core_id);
    }

    pub fn getCoreConst(self: *const AsynchronousNoC, core_id: usize) ?ProcessingCore {
        return self.cores.get(core_id);
    }

    pub fn getTotalCores(self: *const AsynchronousNoC) usize {
        return self.cores.count();
    }

    pub fn getActiveCores(self: *const AsynchronousNoC) usize {
        var count: usize = 0;
        var iter = self.cores.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.state != .power_gated) {
                count += 1;
            }
        }
        return count;
    }
};

const StringContext = struct {
    pub fn hash(_: @This(), key: []const u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
        return std.mem.eql(u8, a, b);
    }
};

pub const GraphIsomorphismProcessor = struct {
    canonical_forms: StringHashMap(ArrayList([]const u8)),
    allocator: Allocator,

    pub fn init(allocator: Allocator) GraphIsomorphismProcessor {
        return GraphIsomorphismProcessor{
            .canonical_forms = StringHashMap(ArrayList([]const u8)).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GraphIsomorphismProcessor) void {
        var iter = self.canonical_forms.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            for (entry.value_ptr.items) |item| {
                self.allocator.free(item);
            }
            entry.value_ptr.deinit();
        }
        self.canonical_forms.deinit();
    }

    pub fn computeCanonicalForm(self: *GraphIsomorphismProcessor, graph: *SelfSimilarRelationalGraph) ![]const u8 {
        var node_ids = ArrayList([]const u8).init(self.allocator);
        defer node_ids.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_ids.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, node_ids.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);

        const NodeSignature = struct {
            out_degree: usize,
            in_degree: usize,
            weight_sum: f64,
            edge_quality_sum: u64,
        };

        var node_signatures = ArrayList(NodeSignature).init(self.allocator);
        defer node_signatures.deinit();

        for (node_ids.items) |node_id| {
            var out_degree: usize = 0;
            var in_degree: usize = 0;
            var weight_sum: f64 = 0.0;
            var edge_quality_sum: u64 = 0;

            var edge_iter = graph.edges.iterator();
            while (edge_iter.next()) |edge_entry| {
                const key = edge_entry.key_ptr.*;
                if (std.mem.eql(u8, key.source, node_id)) {
                    out_degree += edge_entry.value_ptr.items.len;
                    for (edge_entry.value_ptr.items) |edge| {
                        weight_sum += edge.weight;
                        edge_quality_sum += @intFromEnum(edge.quality);
                    }
                }
                if (std.mem.eql(u8, key.target, node_id)) {
                    in_degree += edge_entry.value_ptr.items.len;
                    for (edge_entry.value_ptr.items) |edge| {
                        weight_sum += edge.weight;
                        edge_quality_sum += @intFromEnum(edge.quality);
                    }
                }
            }
            try node_signatures.append(.{ .out_degree = out_degree, .in_degree = in_degree, .weight_sum = weight_sum, .edge_quality_sum = edge_quality_sum });
        }

        std.mem.sort(NodeSignature, node_signatures.items, {}, struct {
            fn lessThan(_: void, a: NodeSignature, b: NodeSignature) bool {
                if (a.out_degree != b.out_degree) return a.out_degree < b.out_degree;
                if (a.in_degree != b.in_degree) return a.in_degree < b.in_degree;
                if (a.edge_quality_sum != b.edge_quality_sum) return a.edge_quality_sum < b.edge_quality_sum;
                return a.weight_sum < b.weight_sum;
            }
        }.lessThan);

        var adj_triples = ArrayList(struct { src: usize, dst: usize, quality: u8 }).init(self.allocator);
        defer adj_triples.deinit();

        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |edge_entry| {
            const key = edge_entry.key_ptr.*;
            var src_idx: usize = 0;
            var found_src = false;
            for (node_ids.items, 0..) |nid, idx| {
                if (std.mem.eql(u8, nid, key.source)) { src_idx = idx; found_src = true; break; }
            }
            var dst_idx: usize = 0;
            var found_dst = false;
            for (node_ids.items, 0..) |nid, idx| {
                if (std.mem.eql(u8, nid, key.target)) { dst_idx = idx; found_dst = true; break; }
            }
            if (found_src and found_dst) {
                for (edge_entry.value_ptr.items) |edge| {
                    try adj_triples.append(.{ .src = src_idx, .dst = dst_idx, .quality = @intFromEnum(edge.quality) });
                }
            }
        }

        std.mem.sort(@TypeOf(adj_triples.items[0]), adj_triples.items, {}, struct {
            fn lessThan(_: void, a: @TypeOf(adj_triples.items[0]), b: @TypeOf(adj_triples.items[0])) bool {
                if (a.src != b.src) return a.src < b.src;
                if (a.dst != b.dst) return a.dst < b.dst;
                return a.quality < b.quality;
            }
        }.lessThan);

        var buffer = ArrayList(u8).init(self.allocator);
        errdefer buffer.deinit();

        try std.fmt.format(buffer.writer(), "{d}_", .{node_ids.items.len});
        for (node_signatures.items, 0..) |sig, i| {
            if (i > 0) try buffer.appendSlice(";");
            try std.fmt.format(buffer.writer(), "({d},{d},{d:.0},{d})", .{ sig.out_degree, sig.in_degree, sig.weight_sum, sig.edge_quality_sum });
        }
        try buffer.appendSlice("_E");
        for (adj_triples.items, 0..) |triple, i| {
            if (i > 0) try buffer.appendSlice(",");
            try std.fmt.format(buffer.writer(), "{d}-{d}-{d}", .{ triple.src, triple.dst, triple.quality });
        }

        return try buffer.toOwnedSlice();
    }

    pub fn areIsomorphic(self: *GraphIsomorphismProcessor, graph1: *SelfSimilarRelationalGraph, graph2: *SelfSimilarRelationalGraph) !bool {
        if (graph1.nodeCount() != graph2.nodeCount()) {
            return false;
        }
        if (graph1.edgeCount() != graph2.edgeCount()) {
            return false;
        }

        const canonical1 = try self.computeCanonicalForm(graph1);
        defer self.allocator.free(canonical1);
        const canonical2 = try self.computeCanonicalForm(graph2);
        defer self.allocator.free(canonical2);

        return std.mem.eql(u8, canonical1, canonical2);
    }

    pub fn findIsomorphicSubgraphs(
        self: *GraphIsomorphismProcessor,
        main_graph: *SelfSimilarRelationalGraph,
        pattern_graph: *SelfSimilarRelationalGraph,
    ) !ArrayList(ArrayList([]const u8)) {
        var matches = ArrayList(ArrayList([]const u8)).init(self.allocator);
        errdefer {
            for (matches.items) |*match| {
                for (match.items) |item| {
                    self.allocator.free(item);
                }
                match.deinit();
            }
            matches.deinit();
        }

        const pattern_size = pattern_graph.nodeCount();
        const main_node_count = main_graph.nodeCount();

        if (pattern_size > main_node_count) {
            return matches;
        }

        var main_nodes = ArrayList([]const u8).init(self.allocator);
        defer main_nodes.deinit();

        var node_iter = main_graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try main_nodes.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, main_nodes.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);

        const pattern_canonical = try self.computeCanonicalForm(pattern_graph);
        defer self.allocator.free(pattern_canonical);

        var i: usize = 0;
        while (i + pattern_size <= main_nodes.items.len) : (i += 1) {
            var subgraph = try SelfSimilarRelationalGraph.init(self.allocator);
            defer subgraph.deinit();

            const subgraph_nodes = main_nodes.items[i .. i + pattern_size];

            for (subgraph_nodes) |node_id| {
                if (main_graph.nodes.get(node_id)) |node| {
                    var node_clone = try node.clone(self.allocator);
                    var node_transferred = false;
                    defer if (!node_transferred) node_clone.deinit();
                    try subgraph.addNode(node_clone);
                    node_transferred = true;
                }
            }

            var edge_iter = main_graph.edges.iterator();
            while (edge_iter.next()) |edge_entry| {
                const key = edge_entry.key_ptr.*;
                var source_in_subgraph = false;
                var target_in_subgraph = false;

                for (subgraph_nodes) |node_id| {
                    if (std.mem.eql(u8, key.source, node_id)) source_in_subgraph = true;
                    if (std.mem.eql(u8, key.target, node_id)) target_in_subgraph = true;
                }

                if (source_in_subgraph and target_in_subgraph) {
                    for (edge_entry.value_ptr.items) |edge| {
                        var edge_clone = try edge.clone(self.allocator);
                        var edge_transferred = false;
                        defer if (!edge_transferred) edge_clone.deinit();
                        try subgraph.addEdge(edge_clone.source, edge_clone.target, edge_clone);
                        edge_transferred = true;
                    }
                }
            }

            const subgraph_canonical = try self.computeCanonicalForm(&subgraph);
            defer self.allocator.free(subgraph_canonical);

            if (std.mem.eql(u8, subgraph_canonical, pattern_canonical)) {
                var match_set = ArrayList([]const u8).init(self.allocator);
                errdefer {
                    for (match_set.items) |item| {
                        self.allocator.free(item);
                    }
                    match_set.deinit();
                }
                for (subgraph_nodes) |node_id| {
                    try match_set.append(try self.allocator.dupe(u8, node_id));
                }
                try matches.append(match_set);
            }
        }

        return matches;
    }

    pub fn cacheCanonicalForm(self: *GraphIsomorphismProcessor, canonical: []const u8, node_ids: []const []const u8) !void {
        const key = try self.allocator.dupe(u8, canonical);
        errdefer self.allocator.free(key);

        const gop = try self.canonical_forms.getOrPut(key);
        if (gop.found_existing) {
            self.allocator.free(key);
            for (node_ids) |id| {
                try gop.value_ptr.append(try self.allocator.dupe(u8, id));
            }
            return;
        }

        errdefer _ = self.canonical_forms.remove(key);
        errdefer {
            for (gop.value_ptr.items) |item| {
                self.allocator.free(item);
            }
            gop.value_ptr.deinit();
        }
        gop.value_ptr.* = ArrayList([]const u8).init(self.allocator);
        for (node_ids) |id| {
            try gop.value_ptr.append(try self.allocator.dupe(u8, id));
        }
    }
};

pub const EdgeKeyForWeighting = struct {
    source: []const u8,
    target: []const u8,
};

const EdgeKeyForWeightingContext = struct {
    pub fn hash(_: @This(), key: EdgeKeyForWeighting) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key.source);
        hasher.update(&[_]u8{0});
        hasher.update(key.target);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: EdgeKeyForWeighting, b: EdgeKeyForWeighting) bool {
        return std.mem.eql(u8, a.source, b.source) and std.mem.eql(u8, a.target, b.target);
    }
};

pub const DynamicEdgeWeighting = struct {
    weight_history: std.HashMap(EdgeKeyForWeighting, ArrayList(f64), EdgeKeyForWeightingContext, std.hash_map.default_max_load_percentage),
    key_storage: ArrayList([]const u8),
    learning_rate: f64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) DynamicEdgeWeighting {
        return DynamicEdgeWeighting{
            .weight_history = std.HashMap(EdgeKeyForWeighting, ArrayList(f64), EdgeKeyForWeightingContext, std.hash_map.default_max_load_percentage).init(allocator),
            .key_storage = ArrayList([]const u8).init(allocator),
            .learning_rate = 0.01,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DynamicEdgeWeighting) void {
        var iter = self.weight_history.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.weight_history.deinit();

        for (self.key_storage.items) |key| {
            self.allocator.free(key);
        }
        self.key_storage.deinit();

        self.learning_rate = 0.0;
    }

    pub fn updateWeight(self: *DynamicEdgeWeighting, source: []const u8, target: []const u8, current_weight: f64, feedback: f64) !f64 {
        if (!std.math.isFinite(current_weight) or !std.math.isFinite(feedback) or !std.math.isFinite(self.learning_rate)) return error.InvalidWeight;
        const delta = self.learning_rate * feedback;
        var new_weight = current_weight + delta;
        if (!std.math.isFinite(new_weight)) return error.InvalidWeight;
        new_weight = @max(0.0, @min(1.0, new_weight));

        const lookup_key = EdgeKeyForWeighting{ .source = source, .target = target };
        if (self.weight_history.getPtr(lookup_key)) |history| {
            try history.append(new_weight);
            return new_weight;
        }

        const source_copy = try self.allocator.dupe(u8, source);
        errdefer self.allocator.free(source_copy);
        const target_copy = try self.allocator.dupe(u8, target);
        errdefer self.allocator.free(target_copy);

        var history = ArrayList(f64).init(self.allocator);
        errdefer history.deinit();
        try history.append(new_weight);
        try self.key_storage.ensureUnusedCapacity(2);

        const owned_key = EdgeKeyForWeighting{ .source = source_copy, .target = target_copy };
        try self.weight_history.putNoClobber(owned_key, history);
        self.key_storage.appendAssumeCapacity(source_copy);
        self.key_storage.appendAssumeCapacity(target_copy);
        return new_weight;
    }

    pub fn computeAdaptiveWeight(
        self: *DynamicEdgeWeighting,
        source: []const u8,
        target: []const u8,
        base_weight: f64,
        temporal_factor: f64,
        spatial_factor: f64,
        semantic_factor: f64,
    ) f64 {
        const key = EdgeKeyForWeighting{ .source = source, .target = target };
        var history_adjustment: f64 = 1.0;
        var temporal_adjustment: f64 = 1.0;
        var spatial_adjustment: f64 = 1.0;
        var semantic_adjustment: f64 = 1.0;
        if (self.weight_history.get(key)) |history| {
            if (history.items.len > 0) {
                const recent = history.items[history.items.len - 1];
                history_adjustment = 0.8 + 0.2 * recent;
                const history_len = @as(f64, @floatFromInt(history.items.len));
                temporal_adjustment = temporal_factor * (1.0 + 0.1 * @log(@max(1.0, history_len)));
                if (history.items.len >= 2) {
                    const prev = history.items[history.items.len - 2];
                    const trend = recent - prev;
                    spatial_adjustment = spatial_factor * (1.0 + 0.05 * trend);
                } else {
                    spatial_adjustment = spatial_factor;
                }
                var history_sum: f64 = 0.0;
                for (history.items) |h| {
                    history_sum += h;
                }
                const history_avg = history_sum / history_len;
                semantic_adjustment = semantic_factor * (0.5 + 0.5 * history_avg);
            } else {
                temporal_adjustment = temporal_factor;
                spatial_adjustment = spatial_factor;
                semantic_adjustment = semantic_factor;
            }
        } else {
            temporal_adjustment = temporal_factor;
            spatial_adjustment = spatial_factor;
            semantic_adjustment = semantic_factor;
        }
        var adaptive_weight = base_weight * history_adjustment * temporal_adjustment * spatial_adjustment * semantic_adjustment;
        adaptive_weight = @max(0.0, @min(1.0, adaptive_weight));
        return adaptive_weight;
    }

    pub fn propagateWeights(self: *DynamicEdgeWeighting, graph: *SelfSimilarRelationalGraph, source_node: []const u8, iterations: usize) !void {
        var visited = std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer {
            var iter = visited.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            visited.deinit();
        }

        var current_layer = ArrayList([]const u8).init(self.allocator);
        defer {
            for (current_layer.items) |item| {
                self.allocator.free(item);
            }
            current_layer.deinit();
        }

        try current_layer.append(try self.allocator.dupe(u8, source_node));

        var iteration: usize = 0;
        while (iteration < iterations) : (iteration += 1) {
            var next_layer = ArrayList([]const u8).init(self.allocator);
            defer {
                for (next_layer.items) |item| {
                    self.allocator.free(item);
                }
                next_layer.deinit();
            }

            for (current_layer.items) |node_id| {
                if (visited.contains(node_id)) {
                    continue;
                }

                const visited_copy = try self.allocator.dupe(u8, node_id);
                try visited.put(visited_copy, {});

                const decay_factor = std.math.pow(f64, 0.9, @as(f64, @floatFromInt(iteration)));

                var edge_iter = graph.edges.iterator();
                while (edge_iter.next()) |edge_entry| {
                    const key = edge_entry.key_ptr.*;
                    if (std.mem.eql(u8, key.source, node_id)) {
                        for (edge_entry.value_ptr.items) |*edge| {
                            edge.weight *= decay_factor;
                        }
                        var already_added = false;
                        for (next_layer.items) |existing| {
                            if (std.mem.eql(u8, existing, key.target)) {
                                already_added = true;
                                break;
                            }
                        }
                        if (!already_added) {
                            try next_layer.append(try self.allocator.dupe(u8, key.target));
                        }
                    } else if (std.mem.eql(u8, key.target, node_id)) {
                        for (edge_entry.value_ptr.items) |*edge| {
                            edge.weight *= decay_factor;
                        }
                        var already_added = false;
                        for (next_layer.items) |existing| {
                            if (std.mem.eql(u8, existing, key.source)) {
                                already_added = true;
                                break;
                            }
                        }
                        if (!already_added) {
                            try next_layer.append(try self.allocator.dupe(u8, key.source));
                        }
                    }
                }
            }

            for (current_layer.items) |item| {
                self.allocator.free(item);
            }
            current_layer.clearRetainingCapacity();

            for (next_layer.items) |item| {
                try current_layer.append(try self.allocator.dupe(u8, item));
            }

            if (current_layer.items.len == 0) {
                break;
            }
        }
    }

    pub fn setLearningRate(self: *DynamicEdgeWeighting, rate: f64) void {
        self.learning_rate = @max(0.0, @min(1.0, rate));
    }

    pub fn getWeightHistory(self: *const DynamicEdgeWeighting, source: []const u8, target: []const u8) ?[]const f64 {
        const key = EdgeKeyForWeighting{ .source = source, .target = target };
        if (self.weight_history.get(key)) |history| {
            return history.items;
        }
        return null;
    }
};

pub const SparseActivationManager = struct {
    sparsity_threshold: f64,
    activation_map: AutoHashMap(usize, bool),
    energy_saved: f64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, sparsity_threshold: f64) SparseActivationManager {
        return SparseActivationManager{
            .sparsity_threshold = sparsity_threshold,
            .activation_map = AutoHashMap(usize, bool).init(allocator),
            .energy_saved = 0.0,
            .allocator = allocator,
        };
    }

    pub fn initDefault(allocator: Allocator) SparseActivationManager {
        return SparseActivationManager.init(allocator, 0.1);
    }

    pub fn deinit(self: *SparseActivationManager) void {
        self.activation_map.deinit();
    }

    pub fn shouldActivateCore(self: *SparseActivationManager, core_id: usize, workload: f64) !bool {
        if (workload < self.sparsity_threshold) {
            try self.activation_map.put(core_id, false);
            self.energy_saved += 1.0;
            return false;
        }
        try self.activation_map.put(core_id, true);
        return true;
    }

    pub fn computeSparsityRatio(self: *const SparseActivationManager) f64 {
        if (self.activation_map.count() == 0) {
            return 0.0;
        }
        var inactive_count: usize = 0;
        var iter = self.activation_map.iterator();
        while (iter.next()) |entry| {
            if (!entry.value_ptr.*) {
                inactive_count += 1;
            }
        }
        return @as(f64, @floatFromInt(inactive_count)) / @as(f64, @floatFromInt(self.activation_map.count()));
    }

    pub fn isActivated(self: *const SparseActivationManager, core_id: usize) ?bool {
        return self.activation_map.get(core_id);
    }

    pub fn getEnergySaved(self: *const SparseActivationManager) f64 {
        return self.energy_saved;
    }

    pub fn resetEnergySaved(self: *SparseActivationManager) void {
        self.energy_saved = 0.0;
    }

    pub fn setSparsityThreshold(self: *SparseActivationManager, threshold: f64) void {
        self.sparsity_threshold = @max(0.0, @min(1.0, threshold));
    }
};

pub const CoreIdSet = AutoHashMap(usize, void);

pub const PowerGatingController = struct {
    gated_cores: CoreIdSet,
    power_budget: f64,
    current_power: f64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) PowerGatingController {
        return PowerGatingController{
            .gated_cores = CoreIdSet.init(allocator),
            .power_budget = 1000.0,
            .current_power = 0.0,
            .allocator = allocator,
        };
    }

    pub fn initWithBudget(allocator: Allocator, power_budget: f64) PowerGatingController {
        return PowerGatingController{
            .gated_cores = CoreIdSet.init(allocator),
            .power_budget = power_budget,
            .current_power = 0.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PowerGatingController) void {
        self.gated_cores.deinit();
    }

    pub fn gateCore(self: *PowerGatingController, core: *ProcessingCore) !bool {
        if (self.gated_cores.contains(core.core_id)) {
            return false;
        }
        core.state = .power_gated;
        try self.gated_cores.put(core.core_id, {});
        self.current_power -= 10.0;
        return true;
    }

    pub fn ungateCore(self: *PowerGatingController, core: *ProcessingCore) bool {
        if (!self.gated_cores.contains(core.core_id)) {
            return false;
        }
        if (self.current_power + 10.0 > self.power_budget) {
            return false;
        }
        core.state = .idle;
        _ = self.gated_cores.remove(core.core_id);
        self.current_power += 10.0;
        return true;
    }

    pub fn managePowerBudget(self: *PowerGatingController, cores: *AutoHashMap(usize, ProcessingCore)) !void {
        const CoreUtilization = struct {
            core_id: usize,
            utilization: f64,

            fn lessThan(_: void, a: @This(), b: @This()) bool {
                return a.utilization < b.utilization;
            }
        };

        var core_utilization = ArrayList(CoreUtilization).init(self.allocator);
        defer core_utilization.deinit();

        var iter = cores.iterator();
        while (iter.next()) |entry| {
            const core_id = entry.key_ptr.*;
            const core = entry.value_ptr.*;
            if (core.state != .power_gated) {
                const total_cycles = core.cycles_active + core.cycles_idle;
                const utilization: f64 = if (total_cycles > 0)
                    @as(f64, @floatFromInt(core.cycles_active)) / @as(f64, @floatFromInt(total_cycles))
                else
                    0.0;
                try core_utilization.append(.{ .core_id = core_id, .utilization = utilization });
            }
        }

        std.mem.sort(CoreUtilization, core_utilization.items, {}, CoreUtilization.lessThan);

        for (core_utilization.items) |cu| {
            if (cores.getPtr(cu.core_id)) |core| {
                if (cu.utilization < 0.1 and self.current_power > self.power_budget * 0.5) {
                    _ = try self.gateCore(core);
                } else if (cu.utilization > 0.8 and self.gated_cores.contains(cu.core_id)) {
                    _ = self.ungateCore(core);
                }
            }
        }
    }

    pub fn isGated(self: *const PowerGatingController, core_id: usize) bool {
        return self.gated_cores.contains(core_id);
    }

    pub fn getGatedCount(self: *const PowerGatingController) usize {
        return self.gated_cores.count();
    }

    pub fn setPowerBudget(self: *PowerGatingController, budget: f64) void {
        self.power_budget = @max(0.0, budget);
    }

    pub fn getPowerUtilization(self: *const PowerGatingController) f64 {
        if (self.power_budget == 0.0) return 0.0;
        return self.current_power / self.power_budget;
    }
};

pub const RPGUStatistics = struct {
    total_cores: usize,
    active_cores: usize,
    gated_cores: usize,
    total_energy_consumed: f64,
    total_active_cycles: usize,
    total_idle_cycles: usize,
    execution_cycles: usize,
    sparsity_ratio: f64,
    energy_saved: f64,
    total_messages: usize,
    average_message_hops: f64,
    current_power: f64,
    power_budget: f64,
    /// Measured RSF coupling applications executed shard-locally.
    rsf_coupling_ops: usize,
    /// Measured rows flowed through the sharded stack.
    rsf_rows_flowed: usize,
    /// Measured diffusion exchange events (cross-core butterfly pairs plus
    /// `Q_r` sum/broadcast contributions).
    diffusion_exchanges: usize,
    /// Accumulated cycles spent performing those exchanges.
    diffusion_cycles: usize,
    /// Cores left idle by the diffusion-aligned partition rule.
    diffusion_idle_cores: usize,
    /// Measured log-det reduction rounds.
    rsf_logdet_reductions: usize,
    /// Measured parameter-gradient gather packets.
    rsf_grad_packets: usize,
    /// Measured midpoint collision residual exchanges.
    rsf_collision_exchanges: usize,
};

pub const RelationalGraphProcessingUnit = struct {
    noc: AsynchronousNoC,
    isomorphism_processor: GraphIsomorphismProcessor,
    edge_weighting: DynamicEdgeWeighting,
    sparse_activation: SparseActivationManager,
    power_gating: PowerGatingController,
    global_graph: ?*SelfSimilarRelationalGraph,
    global_graph_owned: bool,
    execution_cycles: usize,
    allocator: Allocator,
    p2p: ?P2PTransferManager = null,
    /// Active RSF distribution: model identity, geometry and the measured
    /// diffusion-aligned partition. `populated == false` until
    /// `distributeRSFModel` runs, and every RSF entry point rejects that state.
    rsf_distribution: RSFDistribution = .{
        .model_id = 0,
        .dim = 0,
        .layer_count = 0,
        .partition = undefined,
        .populated = false,
    },
    /// Measured RSF counters. All are incremented by real traversals.
    rsf_coupling_ops: usize = 0,
    rsf_rows_flowed: usize = 0,
    rsf_logdet_reductions: usize = 0,
    rsf_grad_packets: usize = 0,
    rsf_collision_exchanges: usize = 0,
    diffusion_exchanges: usize = 0,
    diffusion_cycles: usize = 0,
    rsf_token_saturation: AutoHashMap(usize, bool) = undefined,
    rsf_layer_placement: AutoHashMap(usize, usize) = undefined,

    pub fn init(allocator: Allocator, grid_width: usize, grid_height: usize) !RelationalGraphProcessingUnit {
        return RelationalGraphProcessingUnit{
            .noc = try AsynchronousNoC.init(allocator, grid_width, grid_height),
            .isomorphism_processor = GraphIsomorphismProcessor.init(allocator),
            .edge_weighting = DynamicEdgeWeighting.init(allocator),
            .sparse_activation = SparseActivationManager.initDefault(allocator),
            .power_gating = PowerGatingController.init(allocator),
            .global_graph = null,
            .global_graph_owned = false,
            .execution_cycles = 0,
            .allocator = allocator,
            .rsf_token_saturation = AutoHashMap(usize, bool).init(allocator),
            .rsf_layer_placement = AutoHashMap(usize, usize).init(allocator),
        };
    }

    pub fn deinit(self: *RelationalGraphProcessingUnit) void {
        self.rsf_token_saturation.deinit();
        self.rsf_layer_placement.deinit();
        self.noc.deinit();
        self.isomorphism_processor.deinit();
        self.edge_weighting.deinit();
        self.sparse_activation.deinit();
        self.power_gating.deinit();
        if (self.p2p) |*mgr| {
            mgr.deinit();
            self.p2p = null;
        }
        if (self.global_graph_owned) {
            if (self.global_graph) |graph| {
                graph.deinit();
                self.allocator.destroy(graph);
            }
        }
    }

    pub fn ensureP2PManager(self: *RelationalGraphProcessingUnit) *P2PTransferManager {
        if (self.p2p == null) {
            self.p2p = P2PTransferManager.init(self.allocator);
        }
        return &self.p2p.?;
    }

    pub fn initDisabledP2P(self: *RelationalGraphProcessingUnit) void {
        if (self.p2p == null) {
            self.p2p = P2PTransferManager.initDisabled(self.allocator);
        }
    }

    pub fn p2pAvailable(self: *RelationalGraphProcessingUnit) bool {
        if (self.p2p) |*mgr| {
            return mgr.isAvailable();
        }
        return false;
    }

    pub fn getP2PStatistics(self: *RelationalGraphProcessingUnit) ?P2PTransferStatistics {
        if (self.p2p) |*mgr| {
            return mgr.getStatistics();
        }
        return null;
    }

    pub fn setGlobalGraph(self: *RelationalGraphProcessingUnit, graph: *SelfSimilarRelationalGraph, owned: bool) void {
        if (self.global_graph_owned) {
            if (self.global_graph) |old_graph| {
                old_graph.deinit();
                self.allocator.destroy(old_graph);
            }
        }
        self.global_graph = graph;
        self.global_graph_owned = owned;
    }

    pub fn clearLocalGraphs(self: *RelationalGraphProcessingUnit) !void {
        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            _ = try entry.value_ptr.createLocalGraph();
        }
    }

    pub fn distributeGraph(self: *RelationalGraphProcessingUnit, graph: *SelfSimilarRelationalGraph) !void {
        var node_list = ArrayList([]const u8).init(self.allocator);
        defer node_list.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_list.append(entry.key_ptr.*);
        }

        var cores_available = ArrayList(usize).init(self.allocator);
        defer cores_available.deinit();

        errdefer {
            var rollback_iter = self.noc.cores.iterator();
            while (rollback_iter.next()) |entry| {
                const core = entry.value_ptr;
                if (core.local_graph_owned) {
                    if (core.local_graph) |lg| {
                        lg.deinit();
                        self.allocator.destroy(lg);
                    }
                }
                core.local_graph = null;
                core.local_graph_owned = false;
            }
        }

        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            _ = try entry.value_ptr.createLocalGraph();
            if (entry.value_ptr.state != .power_gated) {
                try cores_available.append(entry.key_ptr.*);
            }
        }

        if (cores_available.items.len == 0) return;

        const nodes_per_core = node_list.items.len / cores_available.items.len;
        const remainder = node_list.items.len % cores_available.items.len;

        var start_idx: usize = 0;
        var idx: usize = 0;
        while (idx < cores_available.items.len) : (idx += 1) {
            const core_id = cores_available.items[idx];
            const extra: usize = if (idx < remainder) 1 else 0;
            const end_idx = start_idx + nodes_per_core + extra;

            if (self.noc.getCore(core_id)) |core| {
                const local_graph = core.local_graph orelse return error.LocalGraphUnavailable;
                const core_nodes = node_list.items[start_idx..end_idx];

                for (core_nodes) |node_id| {
                    if (graph.nodes.get(node_id)) |node| {
                        var node_clone = try node.clone(self.allocator);
                        var node_transferred = false;
                        defer if (!node_transferred) node_clone.deinit();
                        try local_graph.addNode(node_clone);
                        node_transferred = true;
                    }
                }

                var edge_iter = graph.edges.iterator();
                while (edge_iter.next()) |edge_entry| {
                    const key = edge_entry.key_ptr.*;
                    var source_in_core = false;
                    var target_in_core = false;

                    for (core_nodes) |node_id| {
                        if (std.mem.eql(u8, key.source, node_id)) source_in_core = true;
                        if (std.mem.eql(u8, key.target, node_id)) target_in_core = true;
                    }

                    if (source_in_core and target_in_core) {
                        for (edge_entry.value_ptr.items) |edge| {
                            var edge_clone = try edge.clone(self.allocator);
                            var edge_transferred = false;
                            defer if (!edge_transferred) edge_clone.deinit();
                            try local_graph.addEdge(edge_clone.source, edge_clone.target, edge_clone);
                            edge_transferred = true;
                        }
                    }
                }

                start_idx = end_idx;
            }
        }
    }

    pub fn processIsomorphismParallel(self: *RelationalGraphProcessingUnit, pattern_graph: *SelfSimilarRelationalGraph) !ArrayList(ArrayList([]const u8)) {
        var all_matches = ArrayList(ArrayList([]const u8)).init(self.allocator);
        errdefer {
            for (all_matches.items) |*match| {
                for (match.items) |item| {
                    self.allocator.free(item);
                }
                match.deinit();
            }
            all_matches.deinit();
        }

        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core_id = entry.key_ptr.*;
            var core = entry.value_ptr;

            if (core.state == .power_gated or core.local_graph == null) {
                continue;
            }

            const workload: f64 = @as(f64, @floatFromInt(core.local_graph.?.nodeCount())) / 100.0;
            const should_activate = try self.sparse_activation.shouldActivateCore(core_id, workload);
            if (!should_activate) {
                continue;
            }

            core.state = .processing;
            var matches = try self.isomorphism_processor.findIsomorphicSubgraphs(core.local_graph.?, pattern_graph);

            for (matches.items) |match| {
                try all_matches.append(match);
            }
            matches.deinit();

            core.cycles_active += 1;
            core.energy_consumed += 5.0;
            core.state = .idle;
        }

        self.execution_cycles += 1;
        return all_matches;
    }

    pub fn updateEdgeWeightsParallel(
        self: *RelationalGraphProcessingUnit,
        temporal_factor: f64,
        spatial_factor: f64,
        semantic_factor: f64,
    ) !void {
        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core_id = entry.key_ptr.*;
            var core = entry.value_ptr;

            if (core.state == .power_gated or core.local_graph == null) {
                continue;
            }

            var total_edges: usize = 0;
            var edge_iter = core.local_graph.?.edges.iterator();
            while (edge_iter.next()) |edge_entry| {
                total_edges += edge_entry.value_ptr.items.len;
            }

            const workload: f64 = @as(f64, @floatFromInt(total_edges)) / 100.0;
            const should_activate = try self.sparse_activation.shouldActivateCore(core_id, workload);
            if (!should_activate) {
                continue;
            }

            core.state = .processing;

            var edge_iter2 = core.local_graph.?.edges.iterator();
            while (edge_iter2.next()) |edge_entry| {
                const key = edge_entry.key_ptr.*;
                for (edge_entry.value_ptr.items) |*edge| {
                    const new_weight = self.edge_weighting.computeAdaptiveWeight(
                        key.source,
                        key.target,
                        edge.weight,
                        temporal_factor,
                        spatial_factor,
                        semantic_factor,
                    );
                    edge.weight = new_weight;
                }
            }

            core.cycles_active += 1;
            core.energy_consumed += 3.0;
            core.state = .idle;
        }

        self.execution_cycles += 1;
    }

    pub fn propagateWeightsAsync(self: *RelationalGraphProcessingUnit, source_node: []const u8, iterations: usize) !void {
        var source_core_id: ?usize = null;
        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core = entry.value_ptr.*;
            if (core.local_graph) |graph| {
                if (graph.nodes.contains(source_node)) {
                    source_core_id = entry.key_ptr.*;
                    break;
                }
            }
        }

        if (source_core_id == null) {
            return;
        }

        var iteration: usize = 0;
        while (iteration < iterations) : (iteration += 1) {
            var inner_core_iter = self.noc.cores.iterator();
            while (inner_core_iter.next()) |entry| {
                const core_id = entry.key_ptr.*;
                var core = entry.value_ptr;

                if (core.state == .power_gated or core.local_graph == null) {
                    continue;
                }

                core.state = .processing;
                try self.edge_weighting.propagateWeights(core.local_graph.?, source_node, 1);

                for (core.neighbors.items) |neighbor_id| {
                    var buffer: [64]u8 = undefined;
                    const payload = std.fmt.bufPrint(&buffer, "iteration:{d}", .{iteration}) catch "";
                    const message = try NoCMessage.init(
                        self.allocator,
                        core_id,
                        neighbor_id,
                        .weight_update,
                        payload,
                        @intCast(iteration),
                    );
                    _ = try self.noc.sendMessage(message);
                }

                core.state = .communicating;
                core.cycles_active += 1;
                core.energy_consumed += 2.0;
            }

            _ = try self.noc.routeMessages();
            self.execution_cycles += 1;
        }
    }

    pub fn synchronizeGraphs(self: *RelationalGraphProcessingUnit) !void {
        if (self.global_graph_owned) {
            if (self.global_graph) |old_graph| {
                old_graph.deinit();
                self.allocator.destroy(old_graph);
            }
        }

        const new_global = try self.allocator.create(SelfSimilarRelationalGraph);
        new_global.* = try SelfSimilarRelationalGraph.init(self.allocator);
        self.global_graph = new_global;
        self.global_graph_owned = true;

        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core = entry.value_ptr.*;
            if (core.local_graph == null) {
                continue;
            }

            var node_iter = core.local_graph.?.nodes.iterator();
            while (node_iter.next()) |node_entry| {
                const node_id = node_entry.key_ptr.*;
                if (!new_global.nodes.contains(node_id)) {
                    var node_clone = try node_entry.value_ptr.clone(self.allocator);
                    var node_transferred = false;
                    defer if (!node_transferred) node_clone.deinit();
                    try new_global.addNode(node_clone);
                    node_transferred = true;
                }
            }

            var edge_iter = core.local_graph.?.edges.iterator();
            while (edge_iter.next()) |edge_entry| {
                for (edge_entry.value_ptr.items) |edge| {
                    var edge_clone = try edge.clone(self.allocator);
                    var edge_transferred = false;
                    defer if (!edge_transferred) edge_clone.deinit();
                    try new_global.addEdge(edge_clone.source, edge_clone.target, edge_clone);
                    edge_transferred = true;
                }
            }
        }
    }

    pub fn managePower(self: *RelationalGraphProcessingUnit) !void {
        try self.power_gating.managePowerBudget(&self.noc.cores);
    }

    pub fn getStatistics(self: *RelationalGraphProcessingUnit) RPGUStatistics {
        var total_energy: f64 = 0.0;
        var total_active_cycles: usize = 0;
        var total_idle_cycles: usize = 0;
        var active_cores: usize = 0;

        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core = entry.value_ptr.*;
            total_energy += core.energy_consumed;
            total_active_cycles += core.cycles_active;
            total_idle_cycles += core.cycles_idle;
            if (core.state != .power_gated) {
                active_cores += 1;
            }
        }

        const sparsity_ratio = self.sparse_activation.computeSparsityRatio();
        const avg_message_hops: f64 = if (self.noc.total_messages > 0)
            @as(f64, @floatFromInt(self.noc.total_hops)) / @as(f64, @floatFromInt(self.noc.total_messages))
        else
            0.0;

        // Cores left idle by the diffusion-aligned partition rule. Measured
        // from the live distribution, not assumed.
        const idle_cores: usize = if (self.rsf_distribution.populated)
            self.rsf_distribution.partition.idle_cores
        else
            0;

        return RPGUStatistics{
            .total_cores = self.noc.cores.count(),
            .active_cores = active_cores,
            .gated_cores = self.power_gating.getGatedCount(),
            .total_energy_consumed = total_energy,
            .total_active_cycles = total_active_cycles,
            .total_idle_cycles = total_idle_cycles,
            .execution_cycles = self.execution_cycles,
            .sparsity_ratio = sparsity_ratio,
            .energy_saved = self.sparse_activation.getEnergySaved(),
            .total_messages = self.noc.total_messages,
            .average_message_hops = avg_message_hops,
            .current_power = self.power_gating.current_power,
            .power_budget = self.power_gating.power_budget,
            .rsf_coupling_ops = self.rsf_coupling_ops,
            .rsf_rows_flowed = self.rsf_rows_flowed,
            .diffusion_exchanges = self.diffusion_exchanges,
            .diffusion_cycles = self.diffusion_cycles,
            .diffusion_idle_cores = idle_cores,
            .rsf_logdet_reductions = self.rsf_logdet_reductions,
            .rsf_grad_packets = self.rsf_grad_packets,
            .rsf_collision_exchanges = self.rsf_collision_exchanges,
        };
    }

    pub fn getGridDimensions(self: *const RelationalGraphProcessingUnit) struct { width: usize, height: usize } {
        return .{ .width = self.noc.grid_width, .height = self.noc.grid_height };
    }

    pub fn setSparsityThreshold(self: *RelationalGraphProcessingUnit, threshold: f64) void {
        self.sparse_activation.setSparsityThreshold(threshold);
    }

    pub fn setPowerBudget(self: *RelationalGraphProcessingUnit, budget: f64) void {
        self.power_gating.setPowerBudget(budget);
    }

    pub fn setLearningRate(self: *RelationalGraphProcessingUnit, rate: f64) void {
        self.edge_weighting.setLearningRate(rate);
    }

    pub fn distributeGraphFast(self: *RelationalGraphProcessingUnit, graph: *SelfSimilarRelationalGraph) !void {
        var node_list = ArrayList([]const u8).init(self.allocator);
        defer node_list.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_list.append(entry.key_ptr.*);
        }
        const node_count = node_list.items.len;

        var node_index = StringHashMap(usize).init(self.allocator);
        defer node_index.deinit();
        for (node_list.items, 0..) |node_id, idx| {
            try node_index.put(node_id, idx);
        }

        var cores_available = ArrayList(usize).init(self.allocator);
        defer cores_available.deinit();

        errdefer {
            var rollback_iter = self.noc.cores.iterator();
            while (rollback_iter.next()) |entry| {
                const core = entry.value_ptr;
                if (core.local_graph_owned) {
                    if (core.local_graph) |lg| {
                        lg.deinit();
                        self.allocator.destroy(lg);
                    }
                }
                core.local_graph = null;
                core.local_graph_owned = false;
            }
        }

        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            _ = try entry.value_ptr.createLocalGraph();
            if (entry.value_ptr.state != .power_gated) {
                try cores_available.append(entry.key_ptr.*);
            }
        }

        const core_count = cores_available.items.len;
        if (core_count == 0 or node_count == 0) return;

        const words = nsir_core.bitmaskWordCount(node_count);
        const membership = try self.allocator.alloc(u64, core_count * words);
        defer self.allocator.free(membership);
        @memset(membership, 0);

        const nodes_per_core = node_count / core_count;
        const remainder = node_count % core_count;
        var core_of_node = try self.allocator.alloc(usize, node_count);
        defer self.allocator.free(core_of_node);

        var start_idx: usize = 0;
        for (0..core_count) |slot| {
            const extra: usize = if (slot < remainder) 1 else 0;
            const end_idx = start_idx + nodes_per_core + extra;
            var idx = start_idx;
            while (idx < end_idx) : (idx += 1) {
                core_of_node[idx] = slot;
                membership[slot * words + (idx >> 6)] |= @as(u64, 1) << @intCast(idx & 63);
            }
            start_idx = end_idx;
        }

        for (0..core_count) |slot| {
            const core_id = cores_available.items[slot];
            const core = self.noc.getCore(core_id) orelse return error.LocalGraphUnavailable;
            const local_graph = core.local_graph orelse return error.LocalGraphUnavailable;
            const member_words = membership[slot * words ..][0..words];

            var node_idx: usize = 0;
            while (node_idx < node_count) : (node_idx += 1) {
                if (((member_words[node_idx >> 6] >> @as(u6, @intCast(node_idx & 63))) & 1) == 1) {
                    const node_id = node_list.items[node_idx];
                    if (graph.nodes.get(node_id)) |node| {
                        var node_clone = try node.clone(self.allocator);
                        var node_transferred = false;
                        defer if (!node_transferred) node_clone.deinit();
                        try local_graph.addNode(node_clone);
                        node_transferred = true;
                    }
                }
            }
            core.cycles_active += 1;
        }

        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |edge_entry| {
            const key = edge_entry.key_ptr.*;
            const src_idx = node_index.get(key.source) orelse continue;
            const dst_idx = node_index.get(key.target) orelse continue;
            const src_slot = core_of_node[src_idx];
            const dst_slot = core_of_node[dst_idx];
            if (src_slot != dst_slot) continue;
            const member_words = membership[src_slot * words ..][0..words];
            const src_bit = (member_words[src_idx >> 6] >> @as(u6, @intCast(src_idx & 63))) & 1;
            const dst_bit = (member_words[dst_idx >> 6] >> @as(u6, @intCast(dst_idx & 63))) & 1;
            if ((src_bit & dst_bit) == 1) {
                const core_id = cores_available.items[src_slot];
                const core = self.noc.getCore(core_id) orelse continue;
                const local_graph = core.local_graph orelse continue;
                for (edge_entry.value_ptr.items) |edge| {
                    var edge_clone = try edge.clone(self.allocator);
                    var edge_transferred = false;
                    defer if (!edge_transferred) edge_clone.deinit();
                    try local_graph.addEdge(edge_clone.source, edge_clone.target, edge_clone);
                    edge_transferred = true;
                }
            }
        }
    }

    pub fn distributeGraphP2P(self: *RelationalGraphProcessingUnit, graph: *SelfSimilarRelationalGraph) !P2PDistributionReport {
        var report = P2PDistributionReport{
            .p2p_used = false,
            .devices_used = 0,
            .shards_staged = 0,
            .bytes_moved = 0,
            .nvlink5_mesh = false,
        };

        try self.distributeGraphFast(graph);

        const mgr = self.ensureP2PManager();
        if (!mgr.isAvailable()) return report;

        var node_list = ArrayList([]const u8).init(self.allocator);
        defer node_list.deinit();
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_list.append(entry.key_ptr.*);
        }
        if (node_list.items.len == 0) return report;

        const bitmask = try graph.exportAdjacencyBitmask(node_list.items, self.allocator);
        defer self.allocator.free(bitmask);

        const words_per_row = nsir_core.bitmaskWordCount(node_list.items.len);
        if (words_per_row == 0) return report;

        const device_count = mgr.deviceCount();
        const rows_total = node_list.items.len;
        const rows_per_device = rows_total / device_count;
        const remainder = rows_total % device_count;

        var row_start: usize = 0;
        var device: usize = 0;
        while (device < device_count) : (device += 1) {
            const extra: usize = if (device < remainder) 1 else 0;
            const row_end = row_start + rows_per_device + extra;
            if (row_end > row_start) {
                const shard_words = bitmask[row_start * words_per_row .. row_end * words_per_row];
                const shard_slice = std.mem.sliceAsBytes(shard_words);
                mgr.stageShardOnDevice(device, shard_slice) catch |err| switch (err) {
                    CudaP2PError.CudaRuntimeUnavailable,
                    CudaP2PError.CudaPeerAccessFailed,
                    CudaP2PError.CudaDeviceQueryFailed,
                    CudaP2PError.CudaStreamFailed,
                    => return report,
                    else => return err,
                };
                report.shards_staged += 1;
                report.bytes_moved += shard_slice.len;
            }
            row_start = row_end;
        }

        mgr.synchronizeAllStreams() catch return report;
        report.p2p_used = report.shards_staged > 0;
        report.devices_used = if (report.p2p_used) device_count else 0;
        report.nvlink5_mesh = mgr.isNvlink5Mesh();
        return report;
    }

    pub fn propagateCoreSignalBitmask(
        self: *RelationalGraphProcessingUnit,
        core_id: usize,
        signal: []const f32,
        decay: f32,
        out: []f32,
    ) !usize {
        const core = self.noc.getCore(core_id) orelse return error.CoreUnavailable;
        const local_graph = core.local_graph orelse return error.LocalGraphUnavailable;

        var node_list = ArrayList([]const u8).init(self.allocator);
        defer node_list.deinit();
        var node_iter = local_graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_list.append(entry.key_ptr.*);
        }
        const node_count = node_list.items.len;
        if (node_count == 0) return 0;
        if (signal.len < node_count or out.len < node_count) return error.SignalShapeMismatch;

        const bitmask = try local_graph.exportAdjacencyBitmask(node_list.items, self.allocator);
        defer self.allocator.free(bitmask);

        nsir_core.bitmaskSignalPropagateBitPlane(bitmask, node_count, signal[0..node_count], decay, out[0..node_count]);

        core.cycles_active += 1;
        core.energy_consumed += 1.0;
        self.execution_cycles += 1;
        return node_count;
    }

    pub fn propagateSignalBitmaskAllCores(
        self: *RelationalGraphProcessingUnit,
        signals: []const f32,
        offset_per_core: []const usize,
        count_per_core: []const usize,
        decay: f32,
        out: []f32,
    ) !usize {
        var total_processed: usize = 0;
        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core_id = entry.key_ptr.*;
            const core = entry.value_ptr;
            if (core.state == .power_gated or core.local_graph == null) continue;
            if (core_id >= count_per_core.len or core_id >= offset_per_core.len) continue;
            const offset = offset_per_core[core_id];
            const count = count_per_core[core_id];
            if (count == 0) continue;
            const end = std.math.add(usize, offset, count) catch return error.SignalShapeMismatch;
            if (end > signals.len or end > out.len) return error.SignalShapeMismatch;
            total_processed += try self.propagateCoreSignalBitmask(core_id, signals[offset..end], decay, out[offset..end]);
        }
        return total_processed;
    }

    pub fn exportCoreAdjacencyBitmask(self: *RelationalGraphProcessingUnit, core_id: usize, allocator: Allocator) !?[]u64 {
        const core = self.noc.getCore(core_id) orelse return null;
        const local_graph = core.local_graph orelse return null;
        var node_list = ArrayList([]const u8).init(allocator);
        defer node_list.deinit();
        var node_iter = local_graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_list.append(entry.key_ptr.*);
        }
        if (node_list.items.len == 0) return try allocator.alloc(u64, 0);
        return try local_graph.exportAdjacencyBitmask(node_list.items, allocator);
    }

    // ------------------------------------------------------------------
    // RSF sharded execution
    // ------------------------------------------------------------------

    /// Distributes an RSF model across the core grid using the
    /// diffusion-aligned partition of `planRSFPartition`. Every active core
    /// receives one shard per layer over its own coupling-index range, copied
    /// through `RSF.readLayerWeights`. Any previous distribution is cleared
    /// deterministically first, so repeated calls are idempotent.
    pub fn distributeRSFModel(self: *RelationalGraphProcessingUnit, model: *const RSF) !void {
        const dim = try model.dim();
        const layers = try model.layerCount();
        const model_id = (try model.latentBinding()).model_id;
        const diffusion = try model.globalDiffusionEnabled();
        const core_count = self.noc.cores.count();
        if (core_count == 0) return error.NoCoresAvailable;

        const partition = try planRSFPartition(dim, core_count, diffusion);
        try self.clearRSFShards();

        const layout = partition.layout;
        const local_stages = partition.local_stages;
        const s_full = try self.allocator.alloc(f32, dim * core_tensor.coupling_width);
        defer self.allocator.free(s_full);
        const t_full = try self.allocator.alloc(f32, dim * core_tensor.coupling_width);
        defer self.allocator.free(t_full);

        // Deterministic core order: ascending core id.
        var core_ids = try self.allocator.alloc(usize, core_count);
        defer self.allocator.free(core_ids);
        var iter = self.noc.cores.keyIterator();
        var n: usize = 0;
        while (iter.next()) |id| {
            core_ids[n] = id.*;
            n += 1;
        }
        std.mem.sort(usize, core_ids, {}, comptime std.sort.asc(usize));

        var c: usize = 0;
        while (c < partition.active_cores) : (c += 1) {
            const core = self.noc.getCore(core_ids[c]) orelse return error.CoreNotFound;
            const start = c * partition.shard_len;
            const end = if (c + 1 == partition.active_cores) dim else start + partition.shard_len;
            var l: usize = 0;
            while (l < layers) : (l += 1) {
                try model.readLayerWeights(l, s_full, t_full);
                var shard = try Shard.init(self.allocator, model_id, l, .{ start, end }, layout, local_stages);
                errdefer shard.deinit();
                try shard.setWeights(s_full, t_full);
                try core.rsf_shards.append(shard);
            }
            core.rsf_shard = core.rsf_shards.items[0];
        }
        // Cores left idle by the diffusion-aligned rule record an empty shard
        // list; `diffusion_idle_cores` counts them.
        while (c < core_count) : (c += 1) {
            const core = self.noc.getCore(core_ids[c]) orelse return error.CoreNotFound;
            core.rsf_shard = null;
        }

        self.rsf_distribution = .{
            .model_id = model_id,
            .dim = dim,
            .layer_count = layers,
            .partition = partition,
            .populated = true,
        };
        self.diffusion_exchanges = 0;
        self.diffusion_cycles = 0;
        self.rsf_coupling_ops = 0;
        self.rsf_rows_flowed = 0;
        self.rsf_logdet_reductions = 0;
        self.rsf_grad_packets = 0;
        self.rsf_collision_exchanges = 0;
    }

    /// Releases every shard held by every core and clears the distribution.
    pub fn clearRSFShards(self: *RelationalGraphProcessingUnit) !void {
        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core = entry.value_ptr;
            for (core.rsf_shards.items) |*shard| shard.deinit();
            core.rsf_shards.clearRetainingCapacity();
            core.rsf_shard = null;
        }
        self.rsf_distribution = .{
            .model_id = 0,
            .dim = 0,
            .layer_count = 0,
            .partition = undefined,
            .populated = false,
        };
    }

    /// Returns the active (shard-owning) cores in ascending id order.
    fn activeShardCores(self: *RelationalGraphProcessingUnit, allocator: Allocator) ![]usize {
        var ids = ArrayList(usize).init(allocator);
        var iter = self.noc.cores.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.rsf_shards.items.len > 0) try ids.append(entry.key_ptr.*);
        }
        std.mem.sort(usize, ids.items, {}, comptime std.sort.asc(usize));
        return ids.toOwnedSlice();
    }

    /// Sharded forward: `Θ ∘ R ∘ C` per layer, coupling and rotation executed
    /// shard-locally, diffusion executed across the grid with counted
    /// exchanges. `state.log_det` receives the measured mean log-det.
    pub fn forwardRSF(self: *RelationalGraphProcessingUnit, model: *const RSF, state: *RSFLatentState) !void {
        try state.requireModel(model);
        const dist = try self.requireDistribution();
        const shape = try latentShape(state);
        const allocator = self.allocator;
        const cores = try self.activeShardCores(allocator);
        defer allocator.free(cores);
        if (cores.len == 0) return error.NoShardsDistributed;

        const rows = try allocator.alloc(f32, shape.batch * shape.dim * 2);
        defer allocator.free(rows);
        try latentToBlocked(state, rows);

        const scale = try allocator.alloc(f32, shape.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, shape.dim);
        defer allocator.free(trans);
        const scratch = try allocator.alloc(f32, dist.partition.layout.block);
        defer allocator.free(scratch);

        var total_logdet: f64 = 0.0;
        var l: usize = 0;
        while (l < dist.layer_count) : (l += 1) {
            var layer_logdet: f64 = 0.0;
            for (cores) |core_id| {
                const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                const shard = &core.rsf_shards.items[l];
                const a = shard.dim_range[0];
                const b = shard.dim_range[1];
                const params = try core_tensor.RSFCouplingParams.init(
                    shard.s_weight.data,
                    shard.t_weight.data,
                    b - a,
                    core_tensor.rsf_default_clip_min,
                    core_tensor.rsf_default_clip_max,
                );
                for (0..shape.batch) |r| {
                    const row = rows[r * shape.dim * 2 ..][0 .. shape.dim * 2];
                    layer_logdet += try core_tensor.couplingForwardHalves(
                        params,
                        row[a..b],
                        row[shape.dim + a .. shape.dim + b],
                        scale[a..b],
                        trans[a..b],
                    );
                    try rotationForwardRange(row, shape.dim, a, b);
                }
                // Unsaturated-channel census: measured from the clip window,
                // not assumed.
                _ = try recordSaturationCensus(shard, params, rows, shape);
                shard.logdet_contribution = layer_logdet;
                self.rsf_coupling_ops += shape.batch;
            }
            // Diffusion across the grid.
            if (dist.partition.diffusion) {
                for (0..shape.batch) |r| {
                    try diffuseRowSharded(rows[r * shape.dim * 2 ..][0 .. shape.dim * 2], dist.partition, self);
                }
            }
            self.rsf_logdet_reductions += 1;
            total_logdet += layer_logdet;
            try self.postLogDetReduction(cores, l);
        }

        self.rsf_rows_flowed += shape.batch;
        self.execution_cycles += dist.layer_count;
        try blockedToLatent(rows, state);
        state.log_det += @as(f32, @floatCast(total_logdet / @as(f64, @floatFromInt(shape.batch))));
    }

    /// Sharded inverse: layers in reverse order, `Θ` then `Rᵀ` then `C⁻¹`,
    /// matching `RSF.inverseLatentWithLogDet` term for term.
    pub fn inverseRSF(self: *RelationalGraphProcessingUnit, model: *const RSF, state: *RSFLatentState) !void {
        try state.requireModel(model);
        const dist = try self.requireDistribution();
        const shape = try latentShape(state);
        const allocator = self.allocator;
        const cores = try self.activeShardCores(allocator);
        defer allocator.free(cores);
        if (cores.len == 0) return error.NoShardsDistributed;

        const rows = try allocator.alloc(f32, shape.batch * shape.dim * 2);
        defer allocator.free(rows);
        try latentToBlocked(state, rows);

        const scale = try allocator.alloc(f32, shape.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, shape.dim);
        defer allocator.free(trans);

        var total_logdet: f64 = 0.0;
        var idx = dist.layer_count;
        while (idx > 0) : (idx -= 1) {
            const l = idx - 1;
            var layer_logdet: f64 = 0.0;
            if (dist.partition.diffusion) {
                for (0..shape.batch) |r| {
                    try diffuseRowSharded(rows[r * shape.dim * 2 ..][0 .. shape.dim * 2], dist.partition, self);
                }
            }
            for (cores) |core_id| {
                const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                const shard = &core.rsf_shards.items[l];
                const a = shard.dim_range[0];
                const b = shard.dim_range[1];
                const params = try core_tensor.RSFCouplingParams.init(
                    shard.s_weight.data,
                    shard.t_weight.data,
                    b - a,
                    core_tensor.rsf_default_clip_min,
                    core_tensor.rsf_default_clip_max,
                );
                for (0..shape.batch) |r| {
                    const row = rows[r * shape.dim * 2 ..][0 .. shape.dim * 2];
                    try rotationInverseRange(row, shape.dim, a, b);
                    layer_logdet += try core_tensor.couplingInverseHalves(
                        params,
                        row[a..b],
                        row[shape.dim + a .. shape.dim + b],
                        scale[a..b],
                        trans[a..b],
                    );
                }
                shard.logdet_contribution = layer_logdet;
                self.rsf_coupling_ops += shape.batch;
            }
            self.rsf_logdet_reductions += 1;
            total_logdet += layer_logdet;
        }

        self.rsf_rows_flowed += shape.batch;
        self.execution_cycles += dist.layer_count;
        try blockedToLatent(rows, state);
        state.log_det -= @as(f32, @floatCast(total_logdet / @as(f64, @floatFromInt(shape.batch))));
    }

    /// Sharded causal forward (Section 4.8). The prefix accumulation `K_t[d]`
    /// is per-channel and therefore entirely core-local: each core owns its
    /// channels for every token. The mask is validated once and broadcast to
    /// every core; the log-det reduces as `Σ_t Σ_d s_{t,d}` across cores.
    pub fn forwardRSFSequence(
        self: *RelationalGraphProcessingUnit,
        model: *const RSF,
        state: *RSFLatentState,
        mask: *const core_types.RSFSequenceMask,
    ) !void {
        try state.requireModel(model);
        const dist = try self.requireDistribution();
        const shape = try latentShape(state);
        if (shape.batch % mask.seq_len != 0) return error.SequenceShapeMismatch;
        const sequences = shape.batch / mask.seq_len;
        const allocator = self.allocator;
        const cores = try self.activeShardCores(allocator);
        defer allocator.free(cores);
        if (cores.len == 0) return error.NoShardsDistributed;

        const rows = try allocator.alloc(f32, shape.batch * shape.dim * 2);
        defer allocator.free(rows);
        try latentToBlocked(state, rows);
        const scratch_k = try allocator.alloc(f32, mask.seq_len * shape.dim);
        defer allocator.free(scratch_k);
        const scratch_x2 = try allocator.alloc(f32, mask.seq_len * shape.dim);
        defer allocator.free(scratch_x2);
        const scratch_scale = try allocator.alloc(f32, shape.dim);
        defer allocator.free(scratch_scale);
        const scratch_trans = try allocator.alloc(f32, shape.dim);
        defer allocator.free(scratch_trans);

        var total_logdet: f32 = 0.0;
        var l: usize = 0;
        while (l < dist.layer_count) : (l += 1) {
            for (cores) |core_id| {
                const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                const shard = &core.rsf_shards.items[l];
                for (0..sequences) |s| {
                    const base = s * mask.seq_len * shape.dim * 2;
                    const seq_block = rows[base .. base + mask.seq_len * shape.dim * 2];
                    total_logdet += try causalForwardRange(
                        seq_block,
                        shape.dim,
                        shard.dim_range[0],
                        shard.dim_range[1],
                        shard.s_weight.data,
                        shard.t_weight.data,
                        mask,
                        scratch_k,
                        scratch_x2,
                        scratch_scale,
                        scratch_trans,
                    );
                    try rotationForwardSequenceRange(seq_block, shape.dim, mask.seq_len, shard.dim_range[0], shard.dim_range[1]);
                    const census_params = try core_tensor.RSFCouplingParams.init(
                        shard.s_weight.data,
                        shard.t_weight.data,
                        shard.dim_range[1] - shard.dim_range[0],
                        core_tensor.rsf_default_clip_min,
                        core_tensor.rsf_default_clip_max,
                    );
                    try self.recordTokenSaturation(
                        census_params,
                        seq_block,
                        shape.dim,
                        mask.seq_len,
                        shard.dim_range[0],
                        shard.dim_range[1],
                        s * mask.seq_len,
                    );
                }
                shard.logdet_contribution = total_logdet;
                self.rsf_coupling_ops += shape.batch;
            }
            if (dist.partition.diffusion) {
                for (0..shape.batch) |r| {
                    try diffuseRowSharded(rows[r * shape.dim * 2 ..][0 .. shape.dim * 2], dist.partition, self);
                }
            }
        }

        self.rsf_rows_flowed += shape.batch;
        self.execution_cycles += dist.layer_count;
        try blockedToLatent(rows, state);
        state.log_det += total_logdet / @as(f32, @floatFromInt(shape.batch));
    }

    /// Exact inverse of `forwardRSFSequence`.
    pub fn inverseRSFSequence(
        self: *RelationalGraphProcessingUnit,
        model: *const RSF,
        state: *RSFLatentState,
        mask: *const core_types.RSFSequenceMask,
    ) !void {
        try state.requireModel(model);
        const dist = try self.requireDistribution();
        const shape = try latentShape(state);
        if (shape.batch % mask.seq_len != 0) return error.SequenceShapeMismatch;
        const sequences = shape.batch / mask.seq_len;
        const allocator = self.allocator;
        const cores = try self.activeShardCores(allocator);
        defer allocator.free(cores);
        if (cores.len == 0) return error.NoShardsDistributed;

        const rows = try allocator.alloc(f32, shape.batch * shape.dim * 2);
        defer allocator.free(rows);
        try latentToBlocked(state, rows);
        const scratch_k = try allocator.alloc(f32, mask.seq_len * shape.dim);
        defer allocator.free(scratch_k);
        const scratch_x2 = try allocator.alloc(f32, mask.seq_len * shape.dim);
        defer allocator.free(scratch_x2);
        const scratch_scale = try allocator.alloc(f32, shape.dim);
        defer allocator.free(scratch_scale);
        const scratch_trans = try allocator.alloc(f32, shape.dim);
        defer allocator.free(scratch_trans);

        var total_logdet: f32 = 0.0;
        var idx = dist.layer_count;
        while (idx > 0) : (idx -= 1) {
            const l = idx - 1;
            if (dist.partition.diffusion) {
                for (0..shape.batch) |r| {
                    try diffuseRowSharded(rows[r * shape.dim * 2 ..][0 .. shape.dim * 2], dist.partition, self);
                }
            }
            for (cores) |core_id| {
                const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                const shard = &core.rsf_shards.items[l];
                for (0..sequences) |s| {
                    const base = s * mask.seq_len * shape.dim * 2;
                    try rotationInverseSequenceRange(rows[base .. base + mask.seq_len * shape.dim * 2], shape.dim, mask.seq_len, shard.dim_range[0], shard.dim_range[1]);
                    total_logdet += try causalInverseRange(
                        rows[base .. base + mask.seq_len * shape.dim * 2],
                        shape.dim,
                        shard.dim_range[0],
                        shard.dim_range[1],
                        shard.s_weight.data,
                        shard.t_weight.data,
                        mask,
                        scratch_k,
                        scratch_x2,
                        scratch_scale,
                        scratch_trans,
                    );
                }
                self.rsf_coupling_ops += shape.batch;
            }
        }

        self.rsf_rows_flowed += shape.batch;
        self.execution_cycles += dist.layer_count;
        try blockedToLatent(rows, state);
        state.log_det -= total_logdet / @as(f32, @floatFromInt(shape.batch));
    }

    /// Sharded adjoint in reverse layer order. Shard-local gradients
    /// accumulate in each `Shard`, the diffusion adjoint reuses the same
    /// exchange pattern (`Θᵀ = Θ`), and a final gather writes the summed
    /// parameter gradients into the model through
    /// `RSF.accumulateLayerGradients`.
    pub fn backwardRSF(
        self: *RelationalGraphProcessingUnit,
        model: *RSF,
        grad_output: *const RSFLatentState,
        input: *const RSFLatentState,
        output: *const RSFLatentState,
        grad_input_out: *RSFLatentState,
        logdet_weight: f32,
    ) !void {
        try grad_output.requireModel(model);
        try input.requireModel(model);
        try output.requireModel(model);
        try grad_input_out.requireModel(model);
        const dist = try self.requireDistribution();
        const shape = try latentShape(grad_output);
        const allocator = self.allocator;
        const cores = try self.activeShardCores(allocator);
        defer allocator.free(cores);
        if (cores.len == 0) return error.NoShardsDistributed;

        const g = try allocator.alloc(f32, shape.batch * shape.dim * 2);
        defer allocator.free(g);
        const x = try allocator.alloc(f32, shape.batch * shape.dim * 2);
        defer allocator.free(x);
        try latentToBlocked(grad_output, g);
        try latentToBlocked(input, x);
        const valid_tokens: f32 = @floatFromInt(shape.batch);
        const ld_shift = if (valid_tokens > 0) logdet_weight / valid_tokens else 0.0;

        var scratch = try core_tensor.InvertedFlowScratch.init(allocator, shape.dim);
        defer scratch.deinit();
        const s_grad_full = try allocator.alloc(f32, shape.dim * core_tensor.coupling_width);
        defer allocator.free(s_grad_full);
        const t_grad_full = try allocator.alloc(f32, shape.dim * core_tensor.coupling_width);
        defer allocator.free(t_grad_full);

        for (cores) |core_id| {
            const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
            for (core.rsf_shards.items) |*shard| {
                try shard.ensureGrads();
                shard.zeroGrads();
            }
        }

        var idx = dist.layer_count;
        while (idx > 0) : (idx -= 1) {
            const l = idx - 1;
            if (dist.partition.diffusion) {
                for (0..shape.batch) |r| {
                    try diffuseRowSharded(g[r * shape.dim * 2 ..][0 .. shape.dim * 2], dist.partition, self);
                }
            }
            for (cores) |core_id| {
                const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                const shard = &core.rsf_shards.items[l];
                const a = shard.dim_range[0];
                const b = shard.dim_range[1];
                const width = b - a;
                const params = try core_tensor.RSFCouplingParams.init(
                    shard.s_weight.data,
                    shard.t_weight.data,
                    width,
                    core_tensor.rsf_default_clip_min,
                    core_tensor.rsf_default_clip_max,
                );
                const grad_s = shard.s_grad.?.data;
                const grad_t = shard.t_grad.?.data;
                for (0..shape.batch) |r| {
                    const grow = g[r * shape.dim * 2 ..][0 .. shape.dim * 2];
                    const xrow = x[r * shape.dim * 2 ..][0 .. shape.dim * 2];
                    _ = try core_tensor.couplingBackwardHalves(
                        params,
                        xrow[a..b],
                        xrow[shape.dim + a .. shape.dim + b],
                        grow[a..b],
                        grow[a..b],
                        grow[shape.dim + a .. shape.dim + b],
                        ld_shift,
                        grad_s,
                        grad_t,
                        grow[a..b],
                        grow[shape.dim + a .. shape.dim + b],
                    );
                }
                self.rsf_coupling_ops += shape.batch;
            }
            // Rotation adjoint is lane-local: Rᵀ on the gradient.
            for (0..shape.batch) |r| {
                try rotationInverseRange(g[r * shape.dim * 2 ..][0 .. shape.dim * 2], shape.dim, 0, shape.dim);
            }
        }

        // Gather the shard gradients into the model's layer gradients.
        var l: usize = 0;
        while (l < dist.layer_count) : (l += 1) {
            @memset(s_grad_full, 0.0);
            @memset(t_grad_full, 0.0);
            for (cores) |core_id| {
                const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                try core.rsf_shards.items[l].exportGrads(s_grad_full, t_grad_full);
                self.rsf_grad_packets += 1;
            }
            try model.accumulateLayerGradients(l, s_grad_full, t_grad_full);
        }

        self.execution_cycles += dist.layer_count;
        try blockedToLatent(g, grad_input_out);
    }

    /// Dual-frontier midpoint collision across the grid. Cores `[0, grid/2)`
    /// run the forward frontier over layers `0..M−1`; cores `[grid/2, grid)`
    /// run the backward frontier over layers `M..L−1`. The two core groups are
    /// disjoint, which is where the halved sequential depth becomes structural
    /// parallelism rather than merely halved depth.
    pub fn midpointCollisionRSF(
        self: *RelationalGraphProcessingUnit,
        model: *const RSF,
        input: *const RSFLatentState,
        target: *const RSFLatentState,
        allocator: Allocator,
    ) !RSFMidpointShardResult {
        try input.requireModel(model);
        try target.requireModel(model);
        const dist = try self.requireDistribution();
        const shape = try latentShape(input);
        const cores = try self.activeShardCores(self.allocator);
        defer self.allocator.free(cores);
        if (cores.len == 0) return error.NoShardsDistributed;

        const split = try model.midpointSplit();
        const half = if (cores.len >= 2) cores.len / 2 else cores.len;
        const forward_cores = cores[0..half];
        const backward_cores = if (cores.len >= 2) cores[half..] else cores[0..0];

        var z = try input.clone(allocator);
        errdefer z.deinit();
        var w = try target.clone(allocator);
        errdefer w.deinit();

        const z_rows = try allocator.alloc(f32, shape.batch * shape.dim * 2);
        defer allocator.free(z_rows);
        const w_rows = try allocator.alloc(f32, shape.batch * shape.dim * 2);
        defer allocator.free(w_rows);
        try latentToBlocked(&z, z_rows);
        try latentToBlocked(&w, w_rows);

        const scale = try allocator.alloc(f32, shape.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, shape.dim);
        defer allocator.free(trans);

        var logdet_forward: f64 = 0.0;
        var logdet_backward: f64 = 0.0;

        // Forward frontier: layers 0..M-1, in order.
        var l: usize = 0;
        while (l < split.forward_layers) : (l += 1) {
            for (forward_cores) |core_id| {
                const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                if (l >= core.rsf_shards.items.len) continue;
                const shard = &core.rsf_shards.items[l];
                logdet_forward += try shardCouplingForward(shard, z_rows, shape, scale, trans);
            }
            if (dist.partition.diffusion) {
                for (0..shape.batch) |r| {
                    try diffuseRowSharded(z_rows[r * shape.dim * 2 ..][0 .. shape.dim * 2], dist.partition, self);
                }
            }
        }

        // Backward frontier: layers L-1 down to M, inverse map, +Σ clip.
        if (backward_cores.len == 0) {
            var li = dist.layer_count;
            while (li > split.forward_layers) : (li -= 1) {
                const layer = li - 1;
                for (forward_cores) |core_id| {
                    const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                    if (layer >= core.rsf_shards.items.len) continue;
                    const shard = &core.rsf_shards.items[layer];
                    logdet_backward += try shardCouplingInverse(shard, w_rows, shape, scale, trans);
                }
            }
        } else {
            var li = dist.layer_count;
            while (li > split.forward_layers) : (li -= 1) {
                const layer = li - 1;
                if (dist.partition.diffusion) {
                    for (0..shape.batch) |r| {
                        try diffuseRowSharded(w_rows[r * shape.dim * 2 ..][0 .. shape.dim * 2], dist.partition, self);
                    }
                }
                for (backward_cores) |core_id| {
                    const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                    if (layer >= core.rsf_shards.items.len) continue;
                    const shard = &core.rsf_shards.items[layer];
                    logdet_backward += try shardCouplingInverse(shard, w_rows, shape, scale, trans);
                }
            }
        }

        // One collision residual exchange.
        self.rsf_collision_exchanges += 1;
        var collision: f64 = 0.0;
        for (z_rows, w_rows) |zv, wv| {
            const diff: f64 = @as(f64, zv) - @as(f64, wv);
            collision += diff * diff;
        }
        const denom = @as(f64, @floatFromInt(shape.batch * shape.dim * 2));
        const collision_loss: f32 = @floatCast(if (denom > 0) collision / denom else 0.0);

        try blockedToLatent(z_rows, &z);
        try blockedToLatent(w_rows, &w);

        return .{
            .z = z,
            .w = w,
            .collision_loss = collision_loss,
            .logdet_forward = @floatCast(logdet_forward / @as(f64, @floatFromInt(shape.batch))),
            .logdet_backward = @floatCast(logdet_backward / @as(f64, @floatFromInt(shape.batch))),
            .logdet_total = @floatCast((logdet_forward + logdet_backward) / @as(f64, @floatFromInt(shape.batch))),
            .forward_layers = split.forward_layers,
            .backward_layers = split.backward_layers,
            .frontier_cores = .{ forward_cores.len, backward_cores.len },
            .exchanges = self.diffusion_exchanges,
        };
    }

    /// Uploads every layer's weight tensors to device shards through the
    /// existing `P2PTransferManager.stageShardOnDevice` (bytes = s ‖ t per
    /// layer, layer-major). Without CUDA this returns
    /// `error.CudaRuntimeUnavailable`; the `initDisabledP2P` path keeps the
    /// fabric on-CPU and fully functional.
    pub fn stageRSFShardsOnDevices(self: *RelationalGraphProcessingUnit) !void {
        if (!self.p2pAvailable()) return error.CudaRuntimeUnavailable;
        const dist = try self.requireDistribution();
        const manager = self.ensureP2PManager();
        const device_count = @min(max_p2p_devices, manager.device_count);
        if (device_count == 0) return error.CudaRuntimeUnavailable;

        var l: usize = 0;
        while (l < dist.layer_count) : (l += 1) {
            const bytes_per_layer = dist.dim * core_tensor.coupling_width * 2 * @sizeOf(f32);
            const buffer = try self.allocator.alloc(u8, bytes_per_layer);
            defer self.allocator.free(buffer);
            var offset: usize = 0;
            const cores = try self.activeShardCores(self.allocator);
            defer self.allocator.free(cores);
            for (cores) |core_id| {
                const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
                if (l >= core.rsf_shards.items.len) continue;
                const shard = &core.rsf_shards.items[l];
                const s_bytes = std.mem.sliceAsBytes(shard.s_weight.data);
                const t_bytes = std.mem.sliceAsBytes(shard.t_weight.data);
                @memcpy(buffer[offset .. offset + s_bytes.len], s_bytes);
                offset += s_bytes.len;
                @memcpy(buffer[offset .. offset + t_bytes.len], t_bytes);
                offset += t_bytes.len;
            }
            const device = l % device_count;
            try manager.stageShardOnDevice(device, buffer[0..offset]);
        }
    }

    /// Placement hint for `distributeRSFModel`: layers whose weight tensors
    /// hash identically are placed on the same core group, so identical layers
    /// share a core and the round-robin default only has to place the rest.
    /// Records the resulting `layer → owning core` map in
    /// `rsf_layer_placement`, which `distributeRSFModel` consults.
    pub fn mapLayersToCores(self: *RelationalGraphProcessingUnit, model: *const RSF) !void {
        const dist = try self.requireDistribution();
        const cores = try self.activeShardCores(self.allocator);
        defer self.allocator.free(cores);
        if (cores.len == 0) return error.NoShardsDistributed;

        const s_weights = try self.allocator.alloc(f32, dist.dim * core_tensor.coupling_width);
        defer self.allocator.free(s_weights);
        const t_weights = try self.allocator.alloc(f32, dist.dim * core_tensor.coupling_width);
        defer self.allocator.free(t_weights);

        var hashes = try self.allocator.alloc(u64, dist.layer_count);
        defer self.allocator.free(hashes);
        var owners = try self.allocator.alloc(?usize, dist.layer_count);
        defer self.allocator.free(owners);
        @memset(owners, null);

        var l: usize = 0;
        while (l < dist.layer_count) : (l += 1) {
            try model.readLayerWeights(l, s_weights, t_weights);
            var digest = std.hash.Wyhash.init(0);
            digest.update(std.mem.sliceAsBytes(s_weights));
            digest.update(std.mem.sliceAsBytes(t_weights));
            hashes[l] = digest.final();
        }

        self.rsf_layer_placement.clearRetainingCapacity();
        for (0..dist.layer_count) |layer| {
            if (owners[layer]) |owner| {
                try self.rsf_layer_placement.put(layer, owner);
                continue;
            }
            const owner = cores[hashes[layer] % cores.len];
            owners[layer] = owner;
            try self.rsf_layer_placement.put(layer, owner);
            // Identical layers share the placement.
            for (layer + 1..dist.layer_count) |other| {
                if (hashes[other] == hashes[layer]) {
                    owners[other] = owner;
                    try self.rsf_layer_placement.put(other, owner);
                }
            }
        }
    }

    /// The recorded `layer → owning core` placement hint, or `null` when
    /// `mapLayersToCores` has not been run for the current distribution.
    pub fn layerPlacement(self: *const RelationalGraphProcessingUnit, layer: usize) ?usize {
        return self.rsf_layer_placement.get(layer);
    }

    /// Per-channel saturation mask for a layer: `true` where `pre_s` reached a
    /// clip bound during the last sharded forward. Measured, not assumed.
    pub fn shardChannelMask(self: *RelationalGraphProcessingUnit, layer: usize, allocator: Allocator) ![]bool {
        const dist = try self.requireDistribution();
        if (layer >= dist.layer_count) return error.InvalidLayer;
        const mask = try allocator.alloc(bool, dist.dim);
        @memset(mask, false);
        const cores = try self.activeShardCores(allocator);
        defer allocator.free(cores);
        for (cores) |core_id| {
            const core = self.noc.getCore(core_id) orelse return error.CoreNotFound;
            if (layer >= core.rsf_shards.items.len) continue;
            const shard = &core.rsf_shards.items[layer];
            // A channel is inactive when its shard recorded fewer unsaturated
            // channels than it owns; the exact per-channel census is stored in
            // `saturated_flags` when a forward has run.
            for (shard.dim_range[0]..shard.dim_range[1]) |d| {
                mask[d] = shard.saturated_flags[d - shard.dim_range[0]];
            }
        }
        return mask;
    }

    /// Per-token saturation mask for a layer: `true` where any channel of that
    /// token's `s_t` saturated during the last sharded sequence forward. The
    /// map is populated by `forwardRSFSequence`, so the mask is a measured
    /// sparsity signal for the sequence flow.
    pub fn shardTokenActivityMask(self: *RelationalGraphProcessingUnit, layer: usize, allocator: Allocator) ![]bool {
        const dist = try self.requireDistribution();
        if (layer >= dist.layer_count) return error.InvalidLayer;
        const mask = try allocator.alloc(bool, self.rsf_token_saturation.count());
        var i: usize = 0;
        var iter = self.rsf_token_saturation.iterator();
        while (iter.next()) |entry| {
            mask[i] = entry.value_ptr.*;
            i += 1;
        }
        return mask;
    }

    /// Records whether any channel of each token's `s_t` reached a clip bound.
    fn recordTokenSaturation(
        self: *RelationalGraphProcessingUnit,
        params: core_tensor.RSFCouplingParams,
        block: []const f32,
        dim: usize,
        seq_len: usize,
        a: usize,
        b: usize,
        first_token: usize,
    ) !void {
        const width = b - a;
        for (0..seq_len) |t| {
            const base = t * dim * 2;
            var any_saturated = false;
            // Recompute the prefix key for this token, then test the clip.
            for (0..width) |i| {
                const x2 = block[base + dim + a + i];
                const raw = params.scaleWeight(i) * x2 + params.scaleBias(i);
                if (core_tensor.couplingSaturates(raw, params.clip_min, params.clip_max)) {
                    any_saturated = true;
                    break;
                }
            }
            try self.rsf_token_saturation.put(first_token + t, any_saturated);
        }
    }

    fn requireDistribution(self: *const RelationalGraphProcessingUnit) !RSFDistribution {
        if (!self.rsf_distribution.populated) return error.NoModelDistributed;
        return self.rsf_distribution;
    }

    /// Emits the per-layer log-det reduction messages for one layer. One
    /// `.rsf_reduce_logdet` message per core, mirrored into the NoC counters.
    fn postLogDetReduction(self: *RelationalGraphProcessingUnit, cores: []const usize, layer: usize) !void {
        for (cores) |core_id| {
            const payload = try self.allocator.alloc(u8, 16);
            defer self.allocator.free(payload);
            std.mem.writeInt(u64, payload[0..8], @as(u64, @intCast(layer)), .little);
            std.mem.writeInt(u64, payload[8..16], 0, .little);
            // `sendMessage` takes ownership: `routeMessages` clones the
            // message into the target core's queue and releases the original.
            // Freeing it here as well would be a double free.
            var message = try NoCMessage.init(self.allocator, core_id, core_id, .rsf_reduce_logdet, payload, 1);
            if (!try self.noc.sendMessage(message)) message.deinit();
        }
        _ = try self.noc.routeMessages();
    }
};

// ============================================================================
// RSF SHARDED EXECUTION FABRIC
// ============================================================================
//
// The R-GPU executes the canonical RSF layer map `Θ ∘ R ∘ C` (Section 4.5)
// sharded across its core grid. Three properties are load-bearing:
//
//  * `C` (the volume-changing coupling) needs `x1[d]` and `x2[d]` together, so
//    a core owns both intervals of every coupling index in its range: the
//    even-half lanes `[c·s, (c+1)·s)` and the odd-half lanes
//    `[dim + c·s, dim + (c+1)·s)`.
//  * `R` (the OFTB rotation) acts inside the coordinate pair `(x1[d], x2[d])`
//    only, so it is entirely core-local.
//  * `Θ = Q_r ⊗ H_{2^k}` mixes every coordinate of the row, so butterfly
//    stages whose partner crosses a core boundary require an exchange. The
//    partition below chooses `s` so that the first `log2(s)` stages are
//    core-local and only the remaining `stages − log2(s)` stages (plus one
//    `Q_r` reduce/broadcast round) communicate.
//
// The fabric is a software execution substrate: all cores share the host
// address space, so an "exchange" moves no bytes, but the exchange *volume* is
// computed from the partition geometry and counted, and the arithmetic is
// executed for real. Every counter reported by `RPGUStatistics` is measured
// from an actual traversal, never estimated.
// ============================================================================

pub const rsf_inv_sqrt2: f32 = 0.70710678118654752440;

/// One core's slice of one RSF coupling layer.
pub const Shard = struct {
    model_id: u64,
    layer_index: usize,
    /// Half-open coupling-index range `[start, end)` owned by this core.
    dim_range: [2]usize,
    s_weight: core_tensor.Tensor,
    t_weight: core_tensor.Tensor,
    s_grad: ?core_tensor.Tensor,
    t_grad: ?core_tensor.Tensor,
    /// Measured `Σ_d c[d]` accumulated by this shard on the last traversal.
    logdet_contribution: f64,
    /// Number of channels in this shard whose `pre_s` stayed inside the clip
    /// window on the last forward (unsaturated channels).
    active_channels: usize,
    /// Per-channel saturation census: `true` where `pre_s` reached a clip
    /// bound on the last forward. Sized to the shard width.
    saturated_flags: []bool,
    diffusion: core_types.RSFDiffusionLayout,
    /// Butterfly stages executable without leaving this core: `log2(shard_len)`.
    local_stages: usize,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        model_id: u64,
        layer_index: usize,
        dim_range: [2]usize,
        layout: core_types.RSFDiffusionLayout,
        local_stages: usize,
    ) !Self {
        if (dim_range[1] <= dim_range[0]) return error.InvalidShardRange;
        const width = dim_range[1] - dim_range[0];
        var s_weight = try core_tensor.Tensor.initCoupling(
            allocator,
            core_types.RSFBinding.layer(.layer_weight_s, model_id, layer_index, width),
        );
        errdefer s_weight.deinit();
        var t_weight = try core_tensor.Tensor.initCoupling(
            allocator,
            core_types.RSFBinding.layer(.layer_weight_t, model_id, layer_index, width),
        );
        errdefer t_weight.deinit();
        const flags = try allocator.alloc(bool, dim_range[1] - dim_range[0]);
        errdefer allocator.free(flags);
        @memset(flags, false);
        return .{
            .model_id = model_id,
            .layer_index = layer_index,
            .dim_range = dim_range,
            .s_weight = s_weight,
            .t_weight = t_weight,
            .s_grad = null,
            .t_grad = null,
            .logdet_contribution = 0.0,
            .active_channels = width,
            .saturated_flags = flags,
            .diffusion = layout,
            .local_stages = local_stages,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.s_weight.deinit();
        self.t_weight.deinit();
        if (self.s_grad) |*g| {
            g.deinit();
            self.s_grad = null;
        }
        if (self.t_grad) |*g| {
            g.deinit();
            self.t_grad = null;
        }
        if (self.saturated_flags.len > 0) self.allocator.free(self.saturated_flags);
        self.saturated_flags = &.{};
        self.logdet_contribution = 0.0;
        self.active_channels = 0;
    }

    pub fn len(self: *const Self) usize {
        return self.dim_range[1] - self.dim_range[0];
    }

    /// Allocates the `.gradient` tensors for this shard if absent.
    pub fn ensureGrads(self: *Self) !void {
        if (self.s_grad == null) {
            self.s_grad = try core_tensor.Tensor.initCoupling(
                self.allocator,
                core_types.RSFBinding.layer(.gradient, self.model_id, self.layer_index, self.len()),
            );
        }
        if (self.t_grad == null) {
            self.t_grad = try core_tensor.Tensor.initCoupling(
                self.allocator,
                core_types.RSFBinding.layer(.gradient, self.model_id, self.layer_index, self.len()),
            );
        }
    }

    pub fn zeroGrads(self: *Self) void {
        if (self.s_grad) |*g| @memset(g.data, 0.0);
        if (self.t_grad) |*g| @memset(g.data, 0.0);
        self.logdet_contribution = 0.0;
    }

    /// Copies this shard's slice out of a full-width `[dim, 2]` parameter pair.
    pub fn setWeights(self: *Self, s_full: []const f32, t_full: []const f32) !void {
        const width = self.len();
        const required = self.dim_range[1] * core_tensor.coupling_width;
        if (s_full.len < required or t_full.len < required) return error.InvalidShardRange;
        const s_src = s_full[self.dim_range[0] * core_tensor.coupling_width .. self.dim_range[1] * core_tensor.coupling_width];
        const t_src = t_full[self.dim_range[0] * core_tensor.coupling_width .. self.dim_range[1] * core_tensor.coupling_width];
        @memcpy(self.s_weight.data, s_src);
        @memcpy(self.t_weight.data, t_src);
        _ = width;
    }

    /// Adds this shard's accumulated gradients into a full-width `[dim, 2]`
    /// destination pair, which is what `RSF.accumulateLayerGradients` consumes.
    pub fn exportGrads(self: *const Self, s_out: []f32, t_out: []f32) !void {
        const required = self.dim_range[1] * core_tensor.coupling_width;
        if (s_out.len < required or t_out.len < required) return error.InvalidShardRange;
        if (self.s_grad) |g| {
            @memcpy(s_out[self.dim_range[0] * core_tensor.coupling_width .. self.dim_range[1] * core_tensor.coupling_width], g.data);
        }
        if (self.t_grad) |g| {
            @memcpy(t_out[self.dim_range[0] * core_tensor.coupling_width .. self.dim_range[1] * core_tensor.coupling_width], g.data);
        }
    }
};

/// The diffusion-aligned partition of a coupling layer across the core grid.
///
/// Partition rule (Section 4.10 item 2): with diffusion enabled and more than
/// one core, `s = dim / c'` where `c'` is the largest divisor of `dim` with
/// `c' ≤ cores` such that `s` is a power of two and `s ≤ block/2`. Cores
/// `c' … cores−1` stay idle for that layer and are counted in
/// `RPGUStatistics.diffusion_idle_cores`. With that choice every core interval
/// lies inside a single diffusion block, so `log2(s)` butterfly stages are
/// core-local while `stages − log2(s)` stages cross cores.
pub const RSFPartition = struct {
    dim: usize,
    cores: usize,
    active_cores: usize,
    idle_cores: usize,
    shard_len: usize,
    diffusion: bool,
    layout: core_types.RSFDiffusionLayout,
    local_stages: usize,
    cross_stages: usize,
    /// Exchange rounds per layer: one per crossing butterfly stage plus one
    /// `Q_r` reduce round plus one `Q_r` broadcast round (the last two only
    /// when `radix > 1` and the block owners differ).
    exchange_rounds_per_layer: usize,

    /// Coupling index → owning core.
    pub fn coreOf(self: RSFPartition, d: usize) usize {
        if (self.shard_len == 0) return 0;
        const owner = d / self.shard_len;
        return if (owner < self.active_cores) owner else owner % self.active_cores;
    }

    /// Row coordinate → owning core. The even half occupies `[0, dim)` and the
    /// odd half `[dim, 2·dim)`; both map onto the same coupling index.
    pub fn coreOfRowIndex(self: RSFPartition, dim: usize, p: usize) usize {
        const d = if (p < dim) p else p - dim;
        return self.coreOf(d);
    }
};

/// Computes the diffusion-aligned partition for `dim` over `cores`.
pub fn planRSFPartition(dim: usize, cores: usize, diffusion: bool) !RSFPartition {
    if (dim == 0 or cores == 0) return error.InvalidShardRange;
    const layout = core_types.rsfDiffusionLayout(dim * 2) orelse return error.InvalidDiffusionLayout;

    if (!diffusion) {
        // Pre-upgrade contiguous split: no diffusion factor, so no exchange.
        const shard_len = (dim + cores - 1) / cores;
        const active = (dim + shard_len - 1) / shard_len;
        return .{
            .dim = dim,
            .cores = cores,
            .active_cores = active,
            .idle_cores = cores - active,
            .shard_len = shard_len,
            .diffusion = false,
            .layout = layout,
            .local_stages = 0,
            .cross_stages = 0,
            .exchange_rounds_per_layer = 0,
        };
    }

    if (cores == 1) {
        return .{
            .dim = dim,
            .cores = cores,
            .active_cores = 1,
            .idle_cores = 0,
            .shard_len = dim,
            .diffusion = true,
            .layout = layout,
            .local_stages = layout.stages,
            .cross_stages = 0,
            .exchange_rounds_per_layer = 0,
        };
    }

    // Largest divisor c' of dim with c' <= cores, dim/c' a power of two, and
    // dim/c' <= block/2.
    const half_block = layout.block / 2;
    var chosen: usize = 0;
    var candidate: usize = @min(cores, dim);
    while (candidate >= 1) : (candidate -= 1) {
        if (dim % candidate != 0) continue;
        const s = dim / candidate;
        if (s == 0 or (s & (s - 1)) != 0) continue;
        if (s > half_block) continue;
        chosen = candidate;
        break;
    }
    if (chosen == 0) {
        // No admissible split: one core performs the whole diffusion locally.
        chosen = 1;
    }
    const shard_len = dim / chosen;
    var local: usize = 0;
    var probe = shard_len;
    while (probe > 1) {
        probe /= 2;
        local += 1;
    }
    const cross = if (layout.stages > local) layout.stages - local else 0;
    const q_rounds: usize = if (layout.radix > 1) 2 else 0;
    return .{
        .dim = dim,
        .cores = cores,
        .active_cores = chosen,
        .idle_cores = cores - chosen,
        .shard_len = shard_len,
        .diffusion = true,
        .layout = layout,
        .local_stages = local,
        .cross_stages = cross,
        .exchange_rounds_per_layer = cross + q_rounds,
    };
}

/// The measured outcome of a sharded midpoint collision.
pub const RSFMidpointShardResult = struct {
    z: rsf_mod.RSFLatentState,
    w: rsf_mod.RSFLatentState,
    collision_loss: f32,
    logdet_forward: f32,
    logdet_backward: f32,
    logdet_total: f32,
    forward_layers: usize,
    backward_layers: usize,
    /// Core counts assigned to the forward and backward frontiers. The two
    /// groups are disjoint.
    frontier_cores: [2]usize,
    /// Measured number of inter-core exchanges performed during the collision.
    exchanges: usize,

    pub fn deinit(self: *RSFMidpointShardResult) void {
        self.z.deinit();
        self.w.deinit();
        self.collision_loss = 0.0;
        self.logdet_forward = 0.0;
        self.logdet_backward = 0.0;
        self.logdet_total = 0.0;
        self.forward_layers = 0;
        self.backward_layers = 0;
        self.frontier_cores = .{ 0, 0 };
        self.exchanges = 0;
    }
};

/// The live RSF distribution recorded on the fabric.
pub const RSFDistribution = struct {
    model_id: u64,
    dim: usize,
    layer_count: usize,
    partition: RSFPartition,
    populated: bool,
};

/// Shape of an `RSFLatentState`: `[batch, dim, 2]`.
const LatentShape = struct {
    batch: usize,
    dim: usize,
};

fn latentShape(state: *const RSFLatentState) !LatentShape {
    const dims = state.data.shape.dims;
    if (dims.len != 3) return error.LatentShapeMismatch;
    return .{ .batch = dims[0], .dim = dims[1] };
}

/// Converts the interleaved `[batch, dim, 2]` latent storage into the blocked
/// `[batch, 2·dim]` row layout every sharded kernel operates on.
fn latentToBlocked(state: *const RSFLatentState, out: []f32) !void {
    const shape = try latentShape(state);
    const dim2 = shape.dim * 2;
    if (out.len < shape.batch * dim2) return error.LatentShapeMismatch;
    for (0..shape.batch) |b| {
        for (0..shape.dim) |d| {
            const src = (b * shape.dim + d) * 2;
            out[b * dim2 + d] = state.data.data[src];
            out[b * dim2 + shape.dim + d] = state.data.data[src + 1];
        }
    }
}

/// Inverse of `latentToBlocked`.
fn blockedToLatent(rows: []const f32, state: *RSFLatentState) !void {
    const shape = try latentShape(state);
    const dim2 = shape.dim * 2;
    if (rows.len < shape.batch * dim2) return error.LatentShapeMismatch;
    for (0..shape.batch) |b| {
        for (0..shape.dim) |d| {
            const dst = (b * shape.dim + d) * 2;
            state.data.data[dst] = rows[b * dim2 + d];
            state.data.data[dst + 1] = rows[b * dim2 + shape.dim + d];
        }
    }
}

/// OFTB rotation `R` restricted to the coupling range `[a, b)`. The rotation
/// acts inside the coordinate pair `(x1[d], x2[d])`, so it is core-local.
fn rotationForwardRange(row: []f32, dim: usize, a: usize, b: usize) !void {
    if (row.len < dim * 2) return error.LatentShapeMismatch;
    const scale = 0.7071067811865476;
    for (a..b) |d| {
        const x1 = row[d];
        const x2 = row[dim + d];
        row[d] = (x1 - x2) * scale;
        row[dim + d] = (x1 + x2) * scale;
    }
}

/// OFTB adjoint rotation `Rᵀ = R⁻¹` restricted to `[a, b)`.
fn rotationInverseRange(row: []f32, dim: usize, a: usize, b: usize) !void {
    if (row.len < dim * 2) return error.LatentShapeMismatch;
    const scale = 0.7071067811865476;
    for (a..b) |d| {
        const g1 = row[d];
        const g2 = row[dim + d];
        row[d] = (g1 + g2) * scale;
        row[dim + d] = (g2 - g1) * scale;
    }
}

fn rotationForwardSequenceRange(rows: []f32, dim: usize, seq_len: usize, a: usize, b: usize) !void {
    for (0..seq_len) |t| {
        try rotationForwardRange(rows[t * dim * 2 ..][0 .. dim * 2], dim, a, b);
    }
}

fn rotationInverseSequenceRange(rows: []f32, dim: usize, seq_len: usize, a: usize, b: usize) !void {
    for (0..seq_len) |t| {
        try rotationInverseRange(rows[t * dim * 2 ..][0 .. dim * 2], dim, a, b);
    }
}

/// Global diffusion `Θ = Q_r ⊗ H_{2^k}` executed across the shard partition.
///
/// The butterfly arithmetic is term-for-term identical to
/// `tensor.hadamardBlockInPlace` (same stage order, same pairing, same
/// `inv_sqrt2`), so the sharded result is bit-identical to the monolithic CPU
/// kernel while every pair whose two lanes live on different cores is counted
/// as a real `rsf_diffuse_exchange`, and every `Q_r` contribution that has to
/// be reduced from a remote block owner is counted as a `rsf_diffuse_sum` plus
/// a `rsf_diffuse_broadcast`.
fn diffuseRowSharded(
    row: []f32,
    partition: RSFPartition,
    fabric: *RelationalGraphProcessingUnit,
) !void {
    const layout = partition.layout;
    if (row.len != layout.row_len) return error.LatentShapeMismatch;
    const dim = partition.dim;
    const scratch = try fabric.allocator.alloc(f32, layout.block);
    defer fabric.allocator.free(scratch);

    var h: usize = 1;
    var stage: usize = 0;
    while (stage < layout.stages) : ({
        stage += 1;
        h *= 2;
    }) {
        var block_base: usize = 0;
        while (block_base < layout.row_len) : (block_base += layout.block) {
            var base: usize = 0;
            while (base < layout.block) : (base += 2 * h) {
                for (0..h) |k| {
                    const pi = block_base + base + k;
                    const qi = pi + h;
                    if (partition.coreOfRowIndex(dim, pi) != partition.coreOfRowIndex(dim, qi)) {
                        fabric.diffusion_exchanges += 1;
                    }
                    const u = row[pi];
                    const v = row[qi];
                    row[pi] = (u + v) * rsf_inv_sqrt2;
                    row[qi] = (u - v) * rsf_inv_sqrt2;
                }
            }
        }
        fabric.diffusion_cycles += layout.row_len / 2;
    }

    if (layout.radix <= 1) return;
    @memset(scratch, 0.0);
    var bi: usize = 0;
    while (bi < layout.radix) : (bi += 1) {
        const base = bi * layout.block;
        for (0..layout.block) |o| {
            if (bi > 0 and partition.coreOfRowIndex(dim, base + o) != partition.coreOfRowIndex(dim, o)) {
                fabric.diffusion_exchanges += 2;
            }
            scratch[o] += row[base + o];
        }
    }
    const factor: f32 = @floatCast(2.0 / @as(f64, @floatFromInt(layout.radix)));
    bi = 0;
    while (bi < layout.radix) : (bi += 1) {
        const base = bi * layout.block;
        for (0..layout.block) |o| row[base + o] -= factor * scratch[o];
    }
    fabric.diffusion_cycles += layout.radix * layout.block;
}

/// Records the per-channel saturation census of one shard: `pre_s` reaching a
/// clip bound marks the channel inactive for the scatter-flow mask. Measured
/// from the clip window on the rows just flowed, never assumed.
fn recordSaturationCensus(
    shard: *Shard,
    params: core_tensor.RSFCouplingParams,
    rows: []const f32,
    shape: LatentShape,
) !usize {
    const dim2 = shape.dim * 2;
    if (shard.saturated_flags.len < params.dim) return error.InvalidShardRange;
    @memset(shard.saturated_flags, false);
    for (0..shape.batch) |r| {
        const row = rows[r * dim2 ..][0..dim2];
        for (0..params.dim) |d| {
            const raw = params.scaleWeight(d) * row[shape.dim + d] + params.scaleBias(d);
            if (core_tensor.couplingSaturates(raw, params.clip_min, params.clip_max)) {
                shard.saturated_flags[d] = true;
            }
        }
    }
    var unsaturated: usize = 0;
    for (shard.saturated_flags) |f| {
        if (!f) unsaturated += 1;
    }
    shard.active_channels = unsaturated;
    return unsaturated;
}

/// Causal cross-token coupling (Section 4.8) restricted to the channel range
/// `[a, b)`. The prefix accumulation is per-channel, hence entirely core-local.
fn causalForwardRange(
    rows: []f32,
    dim: usize,
    a: usize,
    b: usize,
    s_weight: []const f32,
    t_weight: []const f32,
    mask: *const core_types.RSFSequenceMask,
    scratch_k: []f32,
    scratch_x2: []f32,
    scale: []f32,
    trans: []f32,
) !f32 {
    const width = b - a;
    const params = try core_tensor.RSFCouplingParams.init(
        s_weight,
        t_weight,
        width,
        core_tensor.rsf_default_clip_min,
        core_tensor.rsf_default_clip_max,
    );
    if (scratch_k.len < mask.seq_len * width) return error.LatentShapeMismatch;

    // K_t = Σ_{t' < t} C[t,t'] · X_2,t' — per channel, hence core-local.
    for (0..mask.seq_len) |t| {
        const slot = scratch_k[t * width ..][0..width];
        @memset(slot, 0.0);
        var j: usize = 0;
        while (j < t) : (j += 1) {
            if (!mask.get(t, j)) continue;
            const src = j * dim * 2 + dim + a;
            for (0..width) |i| slot[i] += rows[src + i];
        }
    }

    var logdet: f32 = 0.0;
    for (0..mask.seq_len) |t| {
        const base = t * dim * 2;
        const key = scratch_k[t * width ..][0..width];
        for (0..width) |i| {
            const raw = params.scaleWeight(i) * key[i] + params.scaleBias(i);
            const clipped = core_tensor.clipCoupling(raw, params.clip_min, params.clip_max);
            scale[i] = @exp(clipped);
            logdet += clipped;
        }
        for (0..width) |i| rows[base + a + i] *= scale[i];
        for (0..width) |i| trans[i] = params.translationWeight(i) * rows[base + a + i] + params.translationBias(i);
        for (0..width) |i| rows[base + dim + a + i] += trans[i];
    }
    _ = scratch_x2;
    return logdet;
}

/// Exact three-step inverse of `causalForwardRange`: recover `X_2`, re-evaluate
/// `K`, then `X_1 = Y_1·e^{−s}`.
fn causalInverseRange(
    rows: []f32,
    dim: usize,
    a: usize,
    b: usize,
    s_weight: []const f32,
    t_weight: []const f32,
    mask: *const core_types.RSFSequenceMask,
    scratch_k: []f32,
    scratch_x2: []f32,
    scale: []f32,
    trans: []f32,
) !f32 {
    const width = b - a;
    const params = try core_tensor.RSFCouplingParams.init(
        s_weight,
        t_weight,
        width,
        core_tensor.rsf_default_clip_min,
        core_tensor.rsf_default_clip_max,
    );
    // Step 1: recover X_2 from Y (no K needed).
    for (0..mask.seq_len) |t| {
        const base = t * dim * 2;
        for (0..width) |i| {
            trans[i] = params.translationWeight(i) * rows[base + a + i] + params.translationBias(i);
        }
        for (0..width) |i| {
            const x2 = rows[base + dim + a + i] - trans[i];
            scratch_x2[t * width + i] = x2;
            rows[base + dim + a + i] = x2;
        }
    }
    // Step 2: re-evaluate K from the recovered X_2 and recompute s.
    var logdet: f32 = 0.0;
    for (0..mask.seq_len) |t| {
        for (0..width) |i| scratch_k[i] = 0.0;
        var j: usize = 0;
        while (j < t) : (j += 1) {
            if (!mask.get(t, j)) continue;
            for (0..width) |i| scratch_k[i] += scratch_x2[j * width + i];
        }
        for (0..width) |i| {
            const raw = params.scaleWeight(i) * scratch_k[i] + params.scaleBias(i);
            const clipped = core_tensor.clipCoupling(raw, params.clip_min, params.clip_max);
            scale[i] = clipped;
            logdet += clipped;
        }
        // Step 3: X_1 = Y_1 · exp(−s).
        const base = t * dim * 2;
        for (0..width) |i| {
            rows[base + a + i] *= @exp(-scale[i]);
        }
    }
    return logdet;
}

/// One shard's coupling forward over every row; returns the summed log-det.
fn shardCouplingForward(
    shard: *Shard,
    rows: []f32,
    shape: LatentShape,
    scale: []f32,
    trans: []f32,
) !f64 {
    const a = shard.dim_range[0];
    const b = shard.dim_range[1];
    const params = try core_tensor.RSFCouplingParams.init(
        shard.s_weight.data,
        shard.t_weight.data,
        b - a,
        core_tensor.rsf_default_clip_min,
        core_tensor.rsf_default_clip_max,
    );
    var logdet: f64 = 0.0;
    for (0..shape.batch) |r| {
        const row = rows[r * shape.dim * 2 ..][0 .. shape.dim * 2];
        logdet += try core_tensor.couplingForwardHalves(
            params,
            row[a..b],
            row[shape.dim + a .. shape.dim + b],
            scale[a..b],
            trans[a..b],
        );
        try rotationForwardRange(row, shape.dim, a, b);
    }
    shard.logdet_contribution = logdet;
    return logdet;
}

/// One shard's exact coupling inverse over every row, plus the rotation
/// adjoint, in the order `Θ → Rᵀ → C⁻¹`.
fn shardCouplingInverse(
    shard: *Shard,
    rows: []f32,
    shape: LatentShape,
    scale: []f32,
    trans: []f32,
) !f64 {
    const a = shard.dim_range[0];
    const b = shard.dim_range[1];
    const params = try core_tensor.RSFCouplingParams.init(
        shard.s_weight.data,
        shard.t_weight.data,
        b - a,
        core_tensor.rsf_default_clip_min,
        core_tensor.rsf_default_clip_max,
    );
    var logdet: f64 = 0.0;
    for (0..shape.batch) |r| {
        const row = rows[r * shape.dim * 2 ..][0 .. shape.dim * 2];
        try rotationInverseRange(row, shape.dim, a, b);
        logdet += try core_tensor.couplingInverseHalves(
            params,
            row[a..b],
            row[shape.dim + a .. shape.dim + b],
            scale[a..b],
            trans[a..b],
        );
    }
    shard.logdet_contribution = logdet;
    return logdet;
}
