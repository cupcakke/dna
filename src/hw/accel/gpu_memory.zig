const std = @import("std");
const cuda = @import("cuda_bindings.zig");

const gpu_enabled: bool = blk: {
    const opts = @import("build_options");
    if (@hasDecl(opts, "gpu_acceleration")) break :blk opts.gpu_acceleration;
    break :blk false;
};

pub const MemoryError = error{
    Overflow,
    InvalidDimensions,
    AdmissionRejected,
    DeviceQueryFailed,
};

pub const CouplingLayout = enum {
    diagonal,
    dense_affine,
};

pub const EstimateConfig = struct {
    model_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    batch_size: usize,
    max_seq_len: usize,
    world_size: usize = 1,
    stack_only: bool = true,
    frozen_target_f16_only: bool = true,
    include_graph_chunk: bool = false,
    graph_chunk_size: usize = 0,
    spectral_temps: bool = false,
    coupling: CouplingLayout = .diagonal,
    fisher_block: bool = true,
    reserve_bytes: u64 = 2 * 1024 * 1024 * 1024,
    reserve_fraction: f32 = 0.10,
};

pub const LineItem = struct {
    name: []const u8,
    bytes: u64,
    persistent: bool,
};

pub const Estimate = struct {
    items: [48]LineItem = undefined,
    item_count: usize = 0,
    persistent_bytes: u64 = 0,
    transient_bytes: u64 = 0,
    peak_bytes: u64 = 0,
    rsf_param_bytes: u64 = 0,
    embedding_param_bytes: u64 = 0,
    activation_bytes: u64 = 0,
    coupling_columns: usize = 2,
    elements_per_stack: u64 = 0,
    fisher_elements_per_stack: u64 = 0,
    fisher_bytes_s: u64 = 0,
    fisher_bytes_t: u64 = 0,
    fisher_bytes: u64 = 0,

    pub fn add(self: *Estimate, name: []const u8, bytes: u64, persistent: bool) MemoryError!void {
        if (self.item_count >= self.items.len) return MemoryError.Overflow;
        self.items[self.item_count] = .{ .name = name, .bytes = bytes, .persistent = persistent };
        self.item_count += 1;
        if (persistent) {
            self.persistent_bytes = try addU64(self.persistent_bytes, bytes);
        } else {
            self.transient_bytes = try addU64(self.transient_bytes, bytes);
        }
    }

    pub fn finalize(self: *Estimate) MemoryError!void {
        self.peak_bytes = try addU64(self.persistent_bytes, self.transient_bytes);
    }
};

pub fn addU64(a: u64, b: u64) MemoryError!u64 {
    return std.math.add(u64, a, b) catch MemoryError.Overflow;
}

pub fn mulU64(a: u64, b: u64) MemoryError!u64 {
    return std.math.mul(u64, a, b) catch MemoryError.Overflow;
}

pub fn bytesOf(count: u64, elem: u64) MemoryError!u64 {
    return mulU64(count, elem);
}

pub fn couplingColumns(model_dim: usize, layout: CouplingLayout) MemoryError!usize {
    if (model_dim == 0 or model_dim % 2 != 0) return MemoryError.InvalidDimensions;
    const half = model_dim / 2;
    return switch (layout) {
        .diagonal => 2,
        .dense_affine => std.math.add(usize, half, 1) catch return MemoryError.Overflow,
    };
}

pub fn stackElements(model_dim: usize, num_layers: usize, layout: CouplingLayout) MemoryError!u64 {
    if (num_layers == 0) return MemoryError.InvalidDimensions;
    const half: u64 = @intCast(model_dim / 2);
    const cols: u64 = @intCast(try couplingColumns(model_dim, layout));
    const per = try mulU64(half, cols);
    return mulU64(per, @intCast(num_layers));
}

pub fn fisherStackElements(model_dim: usize, num_layers: usize, fisher_block: bool) MemoryError!u64 {
    if (model_dim == 0 or model_dim % 2 != 0) return MemoryError.InvalidDimensions;
    if (num_layers == 0) return MemoryError.InvalidDimensions;
    const half: u64 = @intCast(model_dim / 2);
    const cols: u64 = if (fisher_block) 3 else 2;
    const per = try mulU64(half, cols);
    return mulU64(per, @intCast(num_layers));
}

pub fn estimate(config: EstimateConfig) MemoryError!Estimate {
    if (config.model_dim == 0 or config.model_dim % 2 != 0) return MemoryError.InvalidDimensions;
    if (config.num_layers == 0 or config.vocab_size == 0) return MemoryError.InvalidDimensions;
    if (config.batch_size == 0 or config.max_seq_len == 0) return MemoryError.InvalidDimensions;

    var out = Estimate{};
    out.coupling_columns = try couplingColumns(config.model_dim, config.coupling);
    out.elements_per_stack = try stackElements(config.model_dim, config.num_layers, config.coupling);
    out.fisher_elements_per_stack = try fisherStackElements(config.model_dim, config.num_layers, config.fisher_block);

    const stack_f32 = try bytesOf(out.elements_per_stack, 4);
    const stack_f16 = try bytesOf(out.elements_per_stack, 2);
    const two_f16 = try mulU64(stack_f16, 2);
    const two_f32 = try mulU64(stack_f32, 2);
    const fisher_f32 = try bytesOf(out.fisher_elements_per_stack, 4);
    const two_fisher = try mulU64(fisher_f32, 2);
    out.fisher_bytes_s = fisher_f32;
    out.fisher_bytes_t = fisher_f32;
    out.fisher_bytes = two_fisher;

    try out.add("rsf_fp16_forward_s_t", two_f16, true);
    try out.add("rsf_fp32_master_s_t", two_f32, true);
    try out.add("rsf_fp32_momentum_s_t", two_f32, true);
    try out.add("rsf_fp32_fisher_s_t", two_fisher, true);
    try out.add("rsf_fp32_step_gradients_s_t", two_f32, false);
    try out.add("rsf_fp32_optimizer_replacement", two_f32, false);
    out.rsf_param_bytes = try addU64(try addU64(two_f16, two_f32), try addU64(two_f32, two_fisher));

    if (!config.stack_only) {
        try out.add("rsf_legacy_per_layer_fp16_mirrors", two_f16, true);
        out.rsf_param_bytes = try addU64(out.rsf_param_bytes, two_f16);
    }

    if (config.spectral_temps) {
        try out.add("rsf_spectral_temp", stack_f32, false);
    }

    const vocab: u64 = @intCast(config.vocab_size);
    const dim: u64 = @intCast(config.model_dim);
    const emb_n = try mulU64(vocab, dim);
    const emb_f16 = try bytesOf(emb_n, 2);
    const emb_f32 = try bytesOf(emb_n, 4);
    try out.add("embedding_fp16_forward", emb_f16, true);
    try out.add("embedding_fp32_master", emb_f32, true);
    try out.add("embedding_fp32_grad", emb_f32, true);
    try out.add("embedding_fp32_momentum", emb_f32, true);
    try out.add("embedding_fp32_fisher", emb_f32, true);
    try out.add("embedding_spectral_u", try bytesOf(vocab, 4), true);
    try out.add("embedding_spectral_v", try bytesOf(dim, 4), true);
    out.embedding_param_bytes = try addU64(emb_f16, try mulU64(emb_f32, 4));

    if (config.frozen_target_f16_only) {
        try out.add("frozen_target_fp16", emb_f16, true);
        out.embedding_param_bytes = try addU64(out.embedding_param_bytes, emb_f16);
    } else {
        const clone = try addU64(emb_f16, try mulU64(emb_f32, 3));
        try out.add("frozen_target_trainable_clone", clone, true);
        out.embedding_param_bytes = try addU64(out.embedding_param_bytes, clone);
    }

    const rows = try mulU64(@intCast(config.batch_size), @intCast(config.max_seq_len));
    const act = try bytesOf(try mulU64(rows, dim), 2);
    try out.add("batch_input_embeddings_f16", act, false);
    try out.add("batch_target_embeddings_f16", act, false);
    try out.add("batch_final_outputs_f16", act, false);
    try out.add("batch_input_delta_f16", act, false);
    try out.add("batch_tokens_i64", try bytesOf(rows, 8), false);
    try out.add("batch_lengths_i64", try bytesOf(@intCast(config.batch_size), 8), false);
    out.activation_bytes = try mulU64(act, 4);

    if (config.world_size > 1) {
        try out.add("nccl_rsf_grad_buffers", two_f32, false);
        try out.add("nccl_embedding_grad_buffer", emb_f32, false);
    }

    if (config.include_graph_chunk and config.graph_chunk_size > 0) {
        const n: u64 = @intCast(config.graph_chunk_size);
        const hashes = try bytesOf(n, 8);
        const states = try bytesOf(try mulU64(n, 4), 4);
        const edges = try bytesOf(try mulU64(n, 3), 8);
        try out.add("graph_chunk_hashes", hashes, false);
        try out.add("graph_chunk_qubit_state", states, false);
        try out.add("graph_chunk_edges", try mulU64(edges, 2), false);
    }

    try out.add("futhark_opaque_tuple_overhead", 64 * 1024 * 1024, false);
    try out.finalize();
    return out;
}

pub const DeviceMemory = struct {
    free_bytes: u64,
    total_bytes: u64,
};

pub fn queryDeviceMemory() MemoryError!DeviceMemory {
    if (comptime !gpu_enabled) return MemoryError.DeviceQueryFailed;
    var free_b: usize = 0;
    var total_b: usize = 0;
    const rc = cuda.cudaMemGetInfo(&free_b, &total_b);
    if (rc != cuda.cudaSuccess) return MemoryError.DeviceQueryFailed;
    return .{ .free_bytes = free_b, .total_bytes = total_b };
}

pub const Admission = struct {
    accepted: bool,
    estimate: Estimate,
    free_bytes: u64,
    total_bytes: u64,
    reserve_bytes: u64,
    budget_bytes: u64,
    largest_name: []const u8 = "",
    largest_bytes: u64 = 0,
};

pub fn admit(config: EstimateConfig, device: DeviceMemory) MemoryError!Admission {
    const est = try estimate(config);
    const frac: u64 = @intFromFloat(@as(f64, @floatFromInt(device.total_bytes)) * @as(f64, config.reserve_fraction));
    const reserve = @max(config.reserve_bytes, frac);
    const budget = if (device.free_bytes > reserve) device.free_bytes - reserve else 0;
    var largest_name: []const u8 = "";
    var largest: u64 = 0;
    var i: usize = 0;
    while (i < est.item_count) : (i += 1) {
        if (est.items[i].bytes > largest) {
            largest = est.items[i].bytes;
            largest_name = est.items[i].name;
        }
    }
    return .{
        .accepted = est.peak_bytes <= budget,
        .estimate = est,
        .free_bytes = device.free_bytes,
        .total_bytes = device.total_bytes,
        .reserve_bytes = reserve,
        .budget_bytes = budget,
        .largest_name = largest_name,
        .largest_bytes = largest,
    };
}
