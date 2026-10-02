const std = @import("std");
const nccl = @import("nccl_bindings.zig");

fn constOpaquePtrFrom(value: anytype) !*const anyopaque {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .pointer => |ptr_info| switch (ptr_info.size) {
            .one => @ptrCast(value),
            .slice => blk: {
                if (value.len == 0) {
                    return error.EmptyBuffer;
                }
                break :blk @ptrCast(&value[0]);
            },
            .many, .c => @compileError("constOpaquePtrFrom: expected single pointer or slice, got unbounded pointer type " ++ @typeName(T)),
        },
        else => @compileError("constOpaquePtrFrom: expected pointer or slice, got " ++ @typeName(T)),
    };
}

fn opaquePtrFrom(value: anytype) !*anyopaque {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .pointer => |ptr_info| switch (ptr_info.size) {
            .one => blk: {
                if (ptr_info.is_const) {
                    @compileError("opaquePtrFrom: expected mutable pointer, got " ++ @typeName(T));
                }
                break :blk @ptrCast(value);
            },
            .slice => blk: {
                if (ptr_info.is_const) {
                    @compileError("opaquePtrFrom: expected mutable slice, got " ++ @typeName(T));
                }
                if (value.len == 0) {
                    return error.EmptyBuffer;
                }
                break :blk @ptrCast(&value[0]);
            },
            .many, .c => @compileError("opaquePtrFrom: expected single pointer or slice, got unbounded pointer type " ++ @typeName(T)),
        },
        else => @compileError("opaquePtrFrom: expected pointer or slice, got " ++ @typeName(T)),
    };
}

fn checkCuda(err: nccl.CudaError, comptime tag: []const u8, fail_error: anyerror) !void {
    if (err != .cudaSuccess) {
        const err_str = nccl.cudaGetErrorString(err);
        std.debug.print("CUDA error [{s}]: {s}\n", .{ tag, err_str });
        return fail_error;
    }
}

fn checkNccl(err: nccl.ncclResult_t, comptime tag: []const u8, fail_error: anyerror) !void {
    if (err != .ncclSuccess) {
        const err_str = nccl.ncclGetErrorString(err);
        std.debug.print("NCCL error [{s}]: {s}\n", .{ tag, err_str });
        return fail_error;
    }
}

fn logCudaFailure(err: nccl.CudaError, comptime tag: []const u8) void {
    if (err != .cudaSuccess) {
        const err_str = nccl.cudaGetErrorString(err);
        std.debug.print("CUDA cleanup error [{s}]: {s}\n", .{ tag, err_str });
    }
}

fn logNcclFailure(err: nccl.ncclResult_t, comptime tag: []const u8) void {
    if (err != .ncclSuccess) {
        const err_str = nccl.ncclGetErrorString(err);
        std.debug.print("NCCL cleanup error [{s}]: {s}\n", .{ tag, err_str });
    }
}

pub const GPUCoordinator = struct {
    world_size: usize,
    rank: usize,
    device_id: i32,
    nccl_comm: ?*nccl.ncclComm,
    cuda_stream: ?*anyopaque,

    pub fn init(world_size: usize, rank: usize, local_rank: usize, nccl_id: nccl.ncclUniqueId) !GPUCoordinator {
        if (world_size == 0) {
            return error.InvalidWorldSize;
        }
        if (rank >= world_size) {
            return error.InvalidRank;
        }
        if (world_size > @as(usize, @intCast(std.math.maxInt(c_int)))) {
            return error.WorldSizeTooLarge;
        }
        if (rank > @as(usize, @intCast(std.math.maxInt(c_int)))) {
            return error.RankTooLarge;
        }

        var device_count: c_int = 0;
        try checkCuda(nccl.cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error.CudaGetDeviceCountFailed);
        if (device_count <= 0) {
            return error.InsufficientGPUs;
        }

        const local_device_count: usize = @intCast(device_count);
        if (local_rank >= local_device_count) {
            return error.LocalRankExceedsDeviceCount;
        }
        const device_id_usize: usize = local_rank;
        if (device_id_usize > @as(usize, @intCast(std.math.maxInt(i32)))) {
            return error.DeviceIdOutOfRange;
        }
        const device_id: i32 = @intCast(device_id_usize);

        try checkCuda(nccl.cudaSetDevice(device_id), "cudaSetDevice", error.CudaSetDeviceFailed);

        // Single-GPU mode: skip ncclCommInitRank entirely when world_size==1.
        // ncclCommInitRank probes the network stack (InfiniBand, shared memory,
        // etc.) even for a single rank, producing warnings and wasting ~0.5 s.
        // For world_size>1 we still initialize NCCL normally.
        var nccl_comm_opt: ?*nccl.ncclComm = null;
        if (world_size > 1) {
            var nccl_comm_local: *nccl.ncclComm = undefined;
            try checkNccl(
                nccl.ncclCommInitRank(&nccl_comm_local, @intCast(world_size), nccl_id, @intCast(rank)),
                "ncclCommInitRank",
                error.NCCLCommInitFailed,
            );
            errdefer logNcclFailure(nccl.ncclCommDestroy(nccl_comm_local), "ncclCommDestroy(init rollback)");
            nccl_comm_opt = nccl_comm_local;
        } else {
            std.debug.print("[GPUCoordinator] world_size=1: single-GPU mode; skipping ncclCommInitRank\n", .{});
        }

        var cuda_stream_local: *anyopaque = undefined;
        try checkCuda(nccl.cudaStreamCreate(&cuda_stream_local), "cudaStreamCreate", error.CudaStreamCreateFailed);
        errdefer logCudaFailure(nccl.cudaStreamDestroy(cuda_stream_local), "cudaStreamDestroy(init rollback)");

        return GPUCoordinator{
            .world_size = world_size,
            .rank = rank,
            .device_id = device_id,
            .nccl_comm = nccl_comm_opt,
            .cuda_stream = cuda_stream_local,
        };
    }

    pub fn deinit(self: *GPUCoordinator) void {
        logCudaFailure(nccl.cudaSetDevice(self.device_id), "cudaSetDevice(deinit)");

        if (self.cuda_stream) |stream| {
            logCudaFailure(nccl.cudaStreamSynchronize(stream), "cudaStreamSynchronize(deinit)");
        }

        if (self.nccl_comm) |comm| {
            logNcclFailure(nccl.ncclCommFinalize(comm), "ncclCommFinalize");
            logNcclFailure(nccl.ncclCommDestroy(comm), "ncclCommDestroy");
            self.nccl_comm = null;
        }

        if (self.cuda_stream) |stream| {
            logCudaFailure(nccl.cudaStreamDestroy(stream), "cudaStreamDestroy");
            self.cuda_stream = null;
        }
    }

    fn setDevice(self: *GPUCoordinator) !void {
        try checkCuda(nccl.cudaSetDevice(self.device_id), "cudaSetDevice", error.CudaSetDeviceFailed);
    }

    fn requireComm(self: *GPUCoordinator) !*nccl.ncclComm {
        return self.nccl_comm orelse return error.CoordinatorNotInitialized;
    }

    fn requireStream(self: *GPUCoordinator) !*anyopaque {
        return self.cuda_stream orelse return error.CoordinatorNotInitialized;
    }

    pub fn allocDeviceMemory(self: *GPUCoordinator, size: usize) !*anyopaque {
        if (size == 0) {
            return error.InvalidAllocationSize;
        }

        try self.setDevice();

        var dev_ptr: ?*anyopaque = null;
        try checkCuda(nccl.cudaMalloc(&dev_ptr, size), "cudaMalloc", error.CudaMallocFailed);
        return dev_ptr orelse return error.CudaMallocFailed;
    }

    pub fn freeDeviceMemory(self: *GPUCoordinator, ptr: ?*anyopaque) void {
        self.setDevice() catch |err| {
            std.debug.print("CUDA cleanup error [cudaSetDevice(freeDeviceMemory)]: {}\n", .{err});
            return;
        };
        if (ptr) |p| {
            logCudaFailure(nccl.cudaFree(p), "cudaFree");
        }
    }

    fn doMemcpy(
        self: *GPUCoordinator,
        dst_ptr: *anyopaque,
        src_ptr: *const anyopaque,
        size: usize,
        kind: c_int,
        comptime tag: []const u8,
    ) !void {
        if (size == 0) {
            return;
        }

        _ = try self.requireStream();
        try self.setDevice();
        try checkCuda(nccl.cudaMemcpy(dst_ptr, src_ptr, size, kind), tag, error.CudaMemcpyFailed);
    }

    pub fn copyHostToDevice(self: *GPUCoordinator, dst: anytype, src: anytype, size: usize) !void {
        if (size == 0) {
            return;
        }

        const dst_ptr = try opaquePtrFrom(dst);
        const src_ptr = try constOpaquePtrFrom(src);
        try self.doMemcpy(dst_ptr, src_ptr, size, nccl.cudaMemcpyKind.cudaMemcpyHostToDevice, "cudaMemcpyHostToDevice");
    }

    pub fn copyDeviceToHost(self: *GPUCoordinator, dst: anytype, src: anytype, size: usize) !void {
        if (size == 0) {
            return;
        }

        const dst_ptr = try opaquePtrFrom(dst);
        const src_ptr = try constOpaquePtrFrom(src);
        try self.doMemcpy(dst_ptr, src_ptr, size, nccl.cudaMemcpyKind.cudaMemcpyDeviceToHost, "cudaMemcpyDeviceToHost");
    }

    fn doAllReduce(
        self: *GPUCoordinator,
        send_buf: *const anyopaque,
        recv_buf: *anyopaque,
        count: usize,
        dtype: nccl.ncclDataType_t,
        op: nccl.ncclRedOp_t,
        comptime tag: []const u8,
    ) !void {
        if (count == 0) {
            return;
        }
        // For world_size==1 the local value is already the global value.
        // Copy send→recv in case the caller passed distinct buffers, then return.
        if (self.world_size == 1) {
            if (send_buf != recv_buf) {
                const elem_bytes: usize = switch (dtype) {
                    .ncclFloat16 => 2,
                    .ncclFloat32 => 4,
                    else => 4,
                };
                try checkCuda(
                    nccl.cudaMemcpy(recv_buf, send_buf, try std.math.mul(usize, count, elem_bytes), nccl.cudaMemcpyKind.cudaMemcpyDeviceToDevice),
                    tag ++ "(single-rank-copy)",
                    error.CudaMemcpyFailed,
                );
            }
            return;
        }

        const comm = try self.requireComm();
        const stream = try self.requireStream();
        try self.setDevice();

        try checkNccl(
            nccl.ncclAllReduce(send_buf, recv_buf, count, dtype, op, comm, stream),
            tag,
            error.NCCLAllReduceFailed,
        );
    }

    pub fn allReduceFloat32(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        try self.doAllReduce(send_buf, recv_buf, count, .ncclFloat32, .ncclSum, "ncclAllReduceFloat32Sum");
    }

    pub fn allReduceFloat32Max(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        try self.doAllReduce(send_buf, recv_buf, count, .ncclFloat32, .ncclMax, "ncclAllReduceFloat32Max");
    }

    pub fn synchronize(self: *GPUCoordinator) !void {
        const stream = try self.requireStream();
        try self.setDevice();

        try checkCuda(nccl.cudaStreamSynchronize(stream), "cudaStreamSynchronize", error.CudaSynchronizeFailed);

        const pending_err = nccl.cudaGetLastError();
        try checkCuda(pending_err, "cudaGetLastError", error.CudaSynchronizeFailed);
    }

    pub fn isRoot(self: *const GPUCoordinator) bool {
        return self.rank == 0;
    }
};
