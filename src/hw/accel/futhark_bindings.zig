const std = @import("std");

const _build_gpu_enabled: bool = blk: {
    const opts = @import("build_options");
    if (@hasDecl(opts, "gpu_acceleration")) break :blk opts.gpu_acceleration;
    break :blk false;
};

pub const struct_futhark_context_config = opaque {};
pub const struct_futhark_context = opaque {};
pub const struct_futhark_f16_2d = opaque {};
pub const struct_futhark_f16_3d = opaque {};
pub const struct_futhark_f32_1d = opaque {};
pub const struct_futhark_f32_2d = opaque {};
pub const struct_futhark_f32_3d = opaque {};
pub const struct_futhark_u64_1d = opaque {};
pub const struct_futhark_i64_1d = opaque {};
pub const struct_futhark_u8_2d = opaque {};
pub const struct_futhark_opaque_tup3_stack_sfd_block2x2 = opaque {};
pub const struct_futhark_opaque_tup5_embedding_spectral = opaque {};
pub const struct_futhark_opaque_tup3_stack_spectral = opaque {};
pub const struct_futhark_opaque_tup7_graph_encode = opaque {};
pub const struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32 = opaque {};
pub const struct_futhark_opaque_tup6_fused_stack_gradients = opaque {};
pub const struct_futhark_opaque_tup3_stack_sfd = opaque {};

pub const GpuConfigurationError = error{GpuAccelerationDisabled};

pub const gpu_default_group_size: c_int = 128;
pub const gpu_default_num_groups: c_int = 2048;
pub const gpu_default_tile_size: c_int = 32;
pub const gpu_arch_sm: c_int = 100;

pub extern "c" fn futhark_context_config_set_device(cfg: ?*struct_futhark_context_config, device: [*:0]const u8) void;
pub extern "c" fn futhark_context_config_set_default_group_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_num_groups(cfg: ?*struct_futhark_context_config, num: c_int) void;
pub extern "c" fn futhark_context_config_set_default_tile_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_cache_file(cfg: ?*struct_futhark_context_config, path: [*:0]const u8) void;
pub extern "c" fn futhark_context_config_set_logging(cfg: ?*struct_futhark_context_config, flag: c_int) void;
pub extern "c" fn futhark_context_config_set_debugging(cfg: ?*struct_futhark_context_config, flag: c_int) void;
pub extern "c" fn futhark_context_config_set_profiling(cfg: ?*struct_futhark_context_config, flag: c_int) void;
pub extern "c" fn futhark_context_config_set_unified_memory(cfg: ?*struct_futhark_context_config, flag: c_int) void;

pub fn configureGpuContext(
    cfg: ?*struct_futhark_context_config,
    cache_file: ?[*:0]const u8,
) GpuConfigurationError!void {
    if (comptime _build_gpu_enabled) {
        const log_on = if (std.posix.getenv("JAIDE_FUTHARK_LOGGING")) |v| std.mem.eql(u8, v, "1") else false;
        const dbg_on = if (std.posix.getenv("JAIDE_FUTHARK_DEBUGGING")) |v| std.mem.eql(u8, v, "1") else false;
        const prof_on = if (std.posix.getenv("JAIDE_FUTHARK_PROFILING")) |v| std.mem.eql(u8, v, "1") else false;
        const unified = if (std.posix.getenv("JAIDE_FUTHARK_UNIFIED_MEMORY")) |v| std.mem.eql(u8, v, "1") else false;
        futhark_context_config_set_logging(cfg, if (log_on) 1 else 0);
        futhark_context_config_set_debugging(cfg, if (dbg_on) 1 else 0);
        futhark_context_config_set_profiling(cfg, if (prof_on) 1 else 0);
        futhark_context_config_set_unified_memory(cfg, if (unified) 1 else 0);
        futhark_context_config_set_device(cfg, "");
        futhark_context_config_set_default_group_size(cfg, gpu_default_group_size);
        futhark_context_config_set_default_num_groups(cfg, gpu_default_num_groups);
        futhark_context_config_set_default_tile_size(cfg, gpu_default_tile_size);
        if (cache_file) |path| futhark_context_config_set_cache_file(cfg, path);
        return;
    }
    return error.GpuAccelerationDisabled;
}

pub extern "c" fn futhark_context_config_new() ?*struct_futhark_context_config;
pub extern "c" fn futhark_context_config_free(cfg: ?*struct_futhark_context_config) void;

pub extern "c" fn futhark_context_new(cfg: ?*struct_futhark_context_config) ?*struct_futhark_context;
pub extern "c" fn futhark_context_free(ctx: ?*struct_futhark_context) void;
pub extern "c" fn futhark_context_sync(ctx: ?*struct_futhark_context) c_int;
pub extern "c" fn futhark_context_get_error(ctx: ?*struct_futhark_context) ?[*:0]const u8;
pub extern "c" fn futhark_context_clear_caches(ctx: ?*struct_futhark_context) c_int;

pub extern "c" fn futhark_new_f16_2d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64, dim1: i64) ?*struct_futhark_f16_2d;
pub extern "c" fn futhark_new_f16_3d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f16_3d;

pub extern "c" fn futhark_free_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d) c_int;
pub extern "c" fn futhark_free_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d) c_int;

pub extern "c" fn futhark_values_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d, data: ?[*]u16) c_int;

pub extern "c" fn futhark_new_f32_1d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64) ?*struct_futhark_f32_1d;
pub extern "c" fn futhark_new_f32_2d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64) ?*struct_futhark_f32_2d;
pub extern "c" fn futhark_new_f32_3d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f32_3d;
pub extern "c" fn futhark_new_u64_1d(ctx: ?*struct_futhark_context, data: ?[*]const u64, dim0: i64) ?*struct_futhark_u64_1d;
pub extern "c" fn futhark_new_i64_1d(ctx: ?*struct_futhark_context, data: ?[*]const i64, dim0: i64) ?*struct_futhark_i64_1d;
pub extern "c" fn futhark_new_u8_2d(ctx: ?*struct_futhark_context, data: ?[*]const u8, dim0: i64, dim1: i64) ?*struct_futhark_u8_2d;

pub extern "c" fn futhark_free_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_free_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_free_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d) c_int;
pub extern "c" fn futhark_free_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d) c_int;
pub extern "c" fn futhark_free_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d) c_int;
pub extern "c" fn futhark_free_u8_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u8_2d) c_int;

pub extern "c" fn futhark_values_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d, data: ?[*]u64) c_int;
pub extern "c" fn futhark_values_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d, data: ?[*]i64) c_int;
pub extern "c" fn futhark_values_u8_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u8_2d, data: ?[*]u8) c_int;

pub extern "c" fn futhark_values_raw_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d) ?*anyopaque;
pub extern "c" fn futhark_values_raw_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d) ?*anyopaque;

pub extern "c" fn futhark_entry_matmul(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_2d, a: ?*const struct_futhark_f32_2d, b: ?*const struct_futhark_f32_2d) c_int;

pub extern "c" fn futhark_entry_rsf_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    input: ?*const struct_futhark_f16_2d,
    weights_s: ?*const struct_futhark_f16_2d,
    weights_t: ?*const struct_futhark_f16_2d,
    clip_min: u16,
    clip_max: u16,
    diffusion: bool,
    radix: i64,
    block: i64,
    stages: i64,
) c_int;

pub extern "c" fn futhark_entry_scale_matrix_f32(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    values: ?*const struct_futhark_f32_2d,
    scale_factor: f32,
) c_int;

pub extern "c" fn futhark_entry_clip_matrix_global_norm_f32(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    values: ?*const struct_futhark_f32_2d,
    clip_norm: f32,
) c_int;

pub extern "c" fn futhark_entry_embedding_sum_squares(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    in0_source: ?*const struct_futhark_f16_2d,
) c_int;

pub extern "c" fn futhark_entry_master_weights_to_f16_3d(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    weights: ?*const struct_futhark_f32_3d,
) c_int;

pub extern "c" fn futhark_entry_master_weights_to_f16_2d(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    weights: ?*const struct_futhark_f32_2d,
) c_int;

pub extern "c" fn futhark_entry_embedding_update_sfd_master(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
    master_weight: ?*const struct_futhark_f32_2d,
    grad_weight: ?*const struct_futhark_f32_2d,
    momentum_state: ?*const struct_futhark_f32_2d,
    fisher_state: ?*const struct_futhark_f32_2d,
    learning_rate: f32,
    momentum_beta: f32,
    fisher_gamma: f32,
    optimizer_step: i64,
    epsilon: f32,
    trust_ratio: f32,
    weight_floor: f32,
) c_int;

pub extern "c" fn futhark_free_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    obj: ?*const struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
) c_int;
pub extern "c" fn futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    obj: ?*const struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
) c_int;
pub extern "c" fn futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    obj: ?*const struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
) c_int;

pub extern "c" fn futhark_entry_rsf_stack_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_inputs: ?*const struct_futhark_f16_3d,
    in1_weights_s: ?*const struct_futhark_f16_3d,
    in2_weights_t: ?*const struct_futhark_f16_3d,
    in3_clip_min: u16,
    in4_clip_max: u16,
    diffusion: bool,
    radix: i64,
    block: i64,
    stages: i64,
) c_int;

pub extern "c" fn futhark_entry_rsf_stack_inverse(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_outputs: ?*const struct_futhark_f16_3d,
    in1_weights_s: ?*const struct_futhark_f16_3d,
    in2_weights_t: ?*const struct_futhark_f16_3d,
    in3_clip_min: u16,
    in4_clip_max: u16,
    diffusion: bool,
    radix: i64,
    block: i64,
    stages: i64,
) c_int;

pub extern "c" fn futhark_entry_rsf_stack_backward_gradients_fused(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup6_fused_stack_gradients,
    in0_final_outputs: ?*const struct_futhark_f16_3d,
    in1_targets: ?*const struct_futhark_f16_3d,
    in2_originals: ?*const struct_futhark_f16_3d,
    in3_lengths: ?*const struct_futhark_i64_1d,
    in4_weights_s: ?*const struct_futhark_f16_3d,
    in5_weights_t: ?*const struct_futhark_f16_3d,
    in6_grad_mean: bool,
    in7_gradient_scale: f32,
    in8_clip_min: f32,
    in9_clip_max: f32,
    in10_reconstruction_alpha: f32,
    in11_forward_scale: f32,
    in12_logdet_weight: f32,
    diffusion: bool,
    radix: i64,
    block: i64,
    stages: i64,
) c_int;

pub extern "c" fn futhark_free_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_3(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_4(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_5(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_entry_stack_update_sfd_block2x2_master(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup3_stack_sfd,
    master_weights: ?*const struct_futhark_f32_3d,
    gradients: ?*const struct_futhark_f32_3d,
    momentum_state: ?*const struct_futhark_f32_3d,
    fisher_blocks: ?*const struct_futhark_f32_3d,
    learning_rate: f32,
    momentum_beta: f32,
    fisher_gamma: f32,
    optimizer_step: i64,
    epsilon: f32,
    trust_ratio: f32,
    weight_floor: f32,
) c_int;

pub extern "c" fn futhark_free_opaque_tup3_stack_sfd_block2x2(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup3_stack_sfd_block2x2,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_stack_sfd_block2x2_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_sfd_block2x2,
) c_int;
pub extern "c" fn futhark_project_opaque_tup3_stack_sfd_block2x2_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_sfd_block2x2,
) c_int;
pub extern "c" fn futhark_project_opaque_tup3_stack_sfd_block2x2_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_sfd_block2x2,
) c_int;

pub extern "c" fn futhark_free_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup3_stack_sfd,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_sfd,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_sfd,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_sfd,
) c_int;

pub extern "c" fn futhark_entry_embedding_forward_padded(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    tokens: ?*const struct_futhark_i64_1d,
    lengths: ?*const struct_futhark_i64_1d,
    positions: ?*const struct_futhark_i64_1d,
    weight: ?*const struct_futhark_f16_2d,
) c_int;

pub extern "c" fn futhark_entry_embedding_backward_padded(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    tokens: ?*const struct_futhark_i64_1d,
    lengths: ?*const struct_futhark_i64_1d,
    grad_output: ?*const struct_futhark_f16_3d,
    grad_weight: ?*const struct_futhark_f32_2d,
) c_int;

pub extern "c" fn futhark_entry_stack_spectral_normalize_exact(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup3_stack_spectral,
    weights: ?*const struct_futhark_f32_3d,
    target: f32,
) c_int;

pub extern "c" fn futhark_free_opaque_tup3_arr3d_f32_f32_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup3_stack_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_f32_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_f32_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup3_stack_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_f32_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup3_stack_spectral,
) c_int;

pub extern "c" fn futhark_entry_embedding_spectral_normalize(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup5_embedding_spectral,
    weight: ?*const struct_futhark_f32_2d,
    u: ?*const struct_futhark_f32_1d,
    v: ?*const struct_futhark_f32_1d,
    power_iters: i64,
    target: f32,
) c_int;

pub extern "c" fn futhark_free_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_3(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_4(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_entry_graph_batch_encode(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup7_graph_encode,
    in0: ?*const struct_futhark_u64_1d,
    in1: u64,
) c_int;

pub extern "c" fn futhark_free_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_u64_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_3(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_4(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_5(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_i64_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_6(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_i64_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_entry_rsf_stack_midpoint_fused(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup6_fused_stack_gradients,
    inputs: ?*const struct_futhark_f16_3d,
    targets: ?*const struct_futhark_f16_3d,
    lengths: ?*const struct_futhark_i64_1d,
    weights_s: ?*const struct_futhark_f16_3d,
    weights_t: ?*const struct_futhark_f16_3d,
    clip_min: f32,
    clip_max: f32,
    logdet_weight: f32,
    diffusion: bool,
    grad_mean: bool,
    gradient_scale: f32,
    radix: i64,
    block: i64,
    stages: i64,
) c_int;

pub extern "c" fn futhark_entry_rsf_causal_bitmask_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    x: ?*const struct_futhark_f16_3d,
    bitmask: ?*const struct_futhark_u8_2d,
    weights_s: ?*const struct_futhark_f16_2d,
    weights_t: ?*const struct_futhark_f16_2d,
    clip_min: f32,
    clip_max: f32,
    diffusion: bool,
    radix: i64,
    block: i64,
    stages: i64,
) c_int;

pub extern "c" fn futhark_entry_rsf_causal_bitmask_inverse(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    y: ?*const struct_futhark_f16_3d,
    bitmask: ?*const struct_futhark_u8_2d,
    weights_s: ?*const struct_futhark_f16_2d,
    weights_t: ?*const struct_futhark_f16_2d,
    clip_min: f32,
    clip_max: f32,
    diffusion: bool,
    radix: i64,
    block: i64,
    stages: i64,
) c_int;

pub extern "c" fn futhark_entry_rsf_causal_bitmask_forward_stack(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    x: ?*const struct_futhark_f16_3d,
    bitmask: ?*const struct_futhark_u8_2d,
    weights_s: ?*const struct_futhark_f16_3d,
    weights_t: ?*const struct_futhark_f16_3d,
    clip_min: f32,
    clip_max: f32,
    diffusion: bool,
    radix: i64,
    block: i64,
    stages: i64,
) c_int;

pub extern "c" fn futhark_entry_rsf_causal_bitmask_inverse_stack(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    y: ?*const struct_futhark_f16_3d,
    bitmask: ?*const struct_futhark_u8_2d,
    weights_s: ?*const struct_futhark_f16_3d,
    weights_t: ?*const struct_futhark_f16_3d,
    clip_min: f32,
    clip_max: f32,
    diffusion: bool,
    radix: i64,
    block: i64,
    stages: i64,
) c_int;
