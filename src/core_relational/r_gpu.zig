const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const PriorityQueue = std.PriorityQueue;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
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

    pub fn toString(self: MessageType) []const u8 {
        return switch (self) {
            .weight_update => "weight_update",
            .graph_sync => "graph_sync",
            .isomorphism_result => "isomorphism_result",
            .power_control => "power_control",
            .data_transfer => "data_transfer",
        };
    }

    pub fn fromString(s: []const u8) ?MessageType {
        if (std.mem.eql(u8, s, "weight_update")) return .weight_update;
        if (std.mem.eql(u8, s, "graph_sync")) return .graph_sync;
        if (std.mem.eql(u8, s, "isomorphism_result")) return .isomorphism_result;
        if (std.mem.eql(u8, s, "power_control")) return .power_control;
        if (std.mem.eql(u8, s, "data_transfer")) return .data_transfer;
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
            .message_queue = ArrayList(NoCMessage).init(allocator),
            .energy_consumed = 0.0,
            .cycles_active = 0,
            .cycles_idle = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ProcessingCore) void {
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
        };
    }

    pub fn deinit(self: *RelationalGraphProcessingUnit) void {
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

        nsir_core.bitmaskSignalPropagate(bitmask, node_count, signal[0..node_count], decay, out[0..node_count]);

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
            if (offset + count > signals.len or offset + count > out.len) return error.SignalShapeMismatch;
            total_processed += try self.propagateCoreSignalBitmask(core_id, signals[offset .. offset + count], decay, out[offset .. offset + count]);
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
};
