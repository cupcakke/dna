const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const gpu_enabled: bool = @import("build_options").gpu_acceleration;

pub const TensorCoreBackend = struct {
    handle: cuda.cublasHandle_t = null,
    stream: cuda.cudaStream_t = null,
    enabled: bool = false,

    const Self = @This();

    pub fn init(stream: cuda.cudaStream_t) Self {
        if (comptime !gpu_enabled) {
            return Self{ .handle = null, .stream = null, .enabled = false };
        }
        var handle: cuda.cublasHandle_t = null;
        const rc = cuda.cublasCreate_v2(&handle);
        if (rc != cuda.cublasStatus_t_success or handle == null) {
            return Self{ .handle = null, .stream = null, .enabled = false };
        }
        _ = cuda.cublasSetStream_v2(handle, stream);
        return Self{ .handle = handle, .stream = stream, .enabled = true };
    }

    pub fn deinit(self: *Self) void {
        if (self.handle) |h| {
            _ = cuda.cublasDestroy_v2(h);
            self.handle = null;
        }
        self.enabled = false;
    }

    pub fn isEnabled(self: *const Self) bool {
        return self.enabled and self.handle != null;
    }

    pub fn gemm_f16(
        self: *Self,
        transa: cuda.cublasOperation_t,
        transb: cuda.cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        A: ?*const anyopaque,
        lda: c_int,
        B: ?*const anyopaque,
        ldb: c_int,
        beta: f32,
        C: ?*anyopaque,
        ldc: c_int,
    ) cuda.CudaError!void {
        if (!self.isEnabled()) return cuda.CudaError.CublasInitFailed;
        const status = cuda.cublasGemmEx(
            self.handle,
            transa,
            transb,
            m, n, k,
            @as(?*const anyopaque, @ptrCast(&alpha)),
            A, cuda.CUDA_R_16F, lda,
            B, cuda.CUDA_R_16F, ldb,
            @as(?*const anyopaque, @ptrCast(&beta)),
            C, cuda.CUDA_R_16F, ldc,
            cuda.CUBLAS_COMPUTE_32F,
            cuda.CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        );
        try cuda.checkCublas(status);
    }

    pub fn gemm_strided_batched_f16(
        self: *Self,
        transa: cuda.cublasOperation_t,
        transb: cuda.cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        A: ?*const anyopaque,
        lda: c_int,
        strideA: i64,
        B: ?*const anyopaque,
        ldb: c_int,
        strideB: i64,
        beta: f32,
        C: ?*anyopaque,
        ldc: c_int,
        strideC: i64,
        batchCount: c_int,
    ) cuda.CudaError!void {
        if (!self.isEnabled()) return cuda.CudaError.CublasInitFailed;
        const status = cuda.cublasGemmStridedBatchedEx(
            self.handle,
            transa,
            transb,
            m, n, k,
            @as(?*const anyopaque, @ptrCast(&alpha)),
            A, cuda.CUDA_R_16F, lda, strideA,
            B, cuda.CUDA_R_16F, ldb, strideB,
            @as(?*const anyopaque, @ptrCast(&beta)),
            C, cuda.CUDA_R_16F, ldc, strideC,
            batchCount,
            cuda.CUBLAS_COMPUTE_32F,
            cuda.CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        );
        try cuda.checkCublas(status);
    }

    pub fn sgemm_f32(
        self: *Self,
        transa: cuda.cublasOperation_t,
        transb: cuda.cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        A: ?*const f32,
        lda: c_int,
        B: ?*const f32,
        ldb: c_int,
        beta: f32,
        C: ?*f32,
        ldc: c_int,
    ) cuda.CudaError!void {
        if (!self.isEnabled()) return cuda.CudaError.CublasInitFailed;
        var a = alpha;
        var b = beta;
        const status = cuda.cublasSgemm_v2(
            self.handle,
            transa, transb,
            m, n, k,
            &a, A, lda, B, ldb,
            &b, C, ldc,
        );
        try cuda.checkCublas(status);
    }

    pub fn sgemv_f32(
        self: *Self,
        trans: cuda.cublasOperation_t,
        m: c_int,
        n: c_int,
        alpha: f32,
        A: ?*const f32,
        lda: c_int,
        x: ?*const f32,
        incx: c_int,
        beta: f32,
        y: ?*f32,
        incy: c_int,
    ) cuda.CudaError!void {
        if (!self.isEnabled()) return cuda.CudaError.CublasInitFailed;
        var a = alpha;
        var b = beta;
        const status = cuda.cublasSgemv_v2(self.handle, trans, m, n, &a, A, lda, x, incx, &b, y, incy);
        try cuda.checkCublas(status);
    }

    pub fn snrm2_f32(self: *Self, n: usize, x: ?*const f32, incx: c_int) cuda.CudaError!f32 {
        if (!self.isEnabled()) return cuda.CudaError.CublasInitFailed;
        var result: f32 = 0.0;
        try cuda.checkCublas(cuda.cublasSnrm2_v2(self.handle, @intCast(n), x, incx, &result));
        return result;
    }

    pub fn sscal_f32(self: *Self, n: usize, alpha: f32, x: ?*f32, incx: c_int) cuda.CudaError!void {
        if (!self.isEnabled()) return cuda.CudaError.CublasInitFailed;
        var a = alpha;
        try cuda.checkCublas(cuda.cublasSscal_v2(self.handle, @intCast(n), &a, x, incx));
    }

    pub fn sdot_f32(self: *Self, n: usize, x: ?*const f32, incx: c_int, y: ?*const f32, incy: c_int) cuda.CudaError!f32 {
        if (!self.isEnabled()) return cuda.CudaError.CublasInitFailed;
        var result: f32 = 0.0;
        try cuda.checkCublas(cuda.cublasSdot_v2(self.handle, @intCast(n), x, incx, y, incy, &result));
        return result;
    }
};
