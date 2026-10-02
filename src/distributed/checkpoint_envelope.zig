const std = @import("std");

pub const MAGIC = [8]u8{ 'J', 'A', 'I', 'D', 'E', 'C', 'K', 'P' };
pub const VERSION: u32 = 7;
pub const ENDIAN_MARKER: u32 = 0x01020304;
pub const TRAILER: u32 = 0xDEADBEEF;

pub const Error = error{
    Truncated,
    ChecksumMismatch,
    MagicMismatch,
    VersionMismatch,
    EndiannessMismatch,
};

pub fn validate(data: []const u8, expected_magic: [8]u8, expected_version: u32) Error![]const u8 {
    if (data.len < expected_magic.len + @sizeOf(u32) * 3) return Error.Truncated;
    const payload = data[0 .. data.len - @sizeOf(u32)];
    const stored_checksum = std.mem.readInt(u32, data[data.len - 4 ..][0..4], .little);
    var checksum = std.hash.Crc32.init();
    checksum.update(payload);
    if (checksum.final() != stored_checksum) return Error.ChecksumMismatch;
    if (!std.mem.eql(u8, payload[0..expected_magic.len], expected_magic[0..])) return Error.MagicMismatch;
    const version_offset = expected_magic.len;
    if (std.mem.readInt(u32, payload[version_offset..][0..4], .little) != expected_version) return Error.VersionMismatch;
    if (std.mem.readInt(u32, payload[version_offset + 4 ..][0..4], .little) != ENDIAN_MARKER) return Error.EndiannessMismatch;
    return payload;
}

fn makeEnvelope(allocator: std.mem.Allocator, magic: [8]u8, version: u32, body: []const u8) ![]u8 {
    var bytes = std.ArrayList(u8).init(allocator);
    errdefer bytes.deinit();
    try bytes.appendSlice(magic[0..]);
    try bytes.writer().writeInt(u32, version, .little);
    try bytes.writer().writeInt(u32, 0x01020304, .little);
    try bytes.appendSlice(body);
    var checksum = std.hash.Crc32.init();
    checksum.update(bytes.items);
    try bytes.writer().writeInt(u32, checksum.final(), .little);
    return bytes.toOwnedSlice();
}

pub const Metadata = struct {
    global_step: u64,
    model_dim: u64,
    layer_count: u64,
    vocab_size: u64,
    local_batch_size: u64,
    learning_rate: f32,
    momentum: f32,
    fisher_gamma: f32,
    fisher_epsilon: f32,
    trust_ratio: f32,
    weight_floor: f32,
    optimizer_warmup_steps: u64,
    spectral_interval: u64,
    spectral_target_norm: f32,
    spectral_iterations: u64,
    reconstruction_alpha: f32,
    phase_a_steps: u64,
    phase_b_steps: u64,
    logdet_weight: f32,
    gradient_clip_norm: f32,
    grad_mean: bool,
    use_normalized_gradient_flow: bool,
    embedding_seed: u64,
    default_max_seq_len: u64,
    reasoning_cycles: u64,
    relational_pass_interval: u64,
    shuffle_target_control: bool,
    target_source_frozen: bool,
    spectral_depth_compensation: bool,
    shuffle_control_state: u64,
    relational_fast_mode: bool,
    clip_min: f32,
    clip_max: f32,
    rsf_optimizer_step: u64,
};

fn writeF32(writer: anytype, value: f32) !void {
    try writer.writeInt(u32, @as(u32, @bitCast(value)), .little);
}

fn readF32(reader: anytype) !f32 {
    const value: f32 = @bitCast(try reader.readInt(u32, .little));
    if (!std.math.isFinite(value)) return error.InvalidFloat;
    return value;
}

fn readBool(reader: anytype) !bool {
    return switch (try reader.readByte()) {
        0 => false,
        1 => true,
        else => error.InvalidBoolean,
    };
}

pub fn writeMetadata(writer: anytype, metadata: Metadata) !void {
    try writer.writeInt(u64, metadata.global_step, .little);
    try writer.writeInt(u64, metadata.model_dim, .little);
    try writer.writeInt(u64, metadata.layer_count, .little);
    try writer.writeInt(u64, metadata.vocab_size, .little);
    try writer.writeInt(u64, metadata.local_batch_size, .little);
    try writeF32(writer, metadata.learning_rate);
    try writeF32(writer, metadata.momentum);
    try writeF32(writer, metadata.fisher_gamma);
    try writeF32(writer, metadata.fisher_epsilon);
    try writeF32(writer, metadata.trust_ratio);
    try writeF32(writer, metadata.weight_floor);
    try writer.writeInt(u64, metadata.optimizer_warmup_steps, .little);
    try writer.writeInt(u64, metadata.spectral_interval, .little);
    try writeF32(writer, metadata.spectral_target_norm);
    try writer.writeInt(u64, metadata.spectral_iterations, .little);
    try writeF32(writer, metadata.reconstruction_alpha);
    try writer.writeInt(u64, metadata.phase_a_steps, .little);
    try writer.writeInt(u64, metadata.phase_b_steps, .little);
    try writeF32(writer, metadata.logdet_weight);
    try writeF32(writer, metadata.gradient_clip_norm);
    try writer.writeByte(@intFromBool(metadata.grad_mean));
    try writer.writeByte(@intFromBool(metadata.use_normalized_gradient_flow));
    try writer.writeInt(u64, metadata.embedding_seed, .little);
    try writer.writeInt(u64, metadata.default_max_seq_len, .little);
    try writer.writeInt(u64, metadata.reasoning_cycles, .little);
    try writer.writeInt(u64, metadata.relational_pass_interval, .little);
    try writer.writeByte(@intFromBool(metadata.shuffle_target_control));
    try writer.writeByte(@intFromBool(metadata.target_source_frozen));
    try writer.writeByte(@intFromBool(metadata.spectral_depth_compensation));
    try writer.writeInt(u64, metadata.shuffle_control_state, .little);
    try writer.writeByte(@intFromBool(metadata.relational_fast_mode));
    try writeF32(writer, metadata.clip_min);
    try writeF32(writer, metadata.clip_max);
    try writer.writeInt(u64, metadata.rsf_optimizer_step, .little);
}

pub fn readMetadata(reader: anytype) !Metadata {
    return .{
        .global_step = try reader.readInt(u64, .little),
        .model_dim = try reader.readInt(u64, .little),
        .layer_count = try reader.readInt(u64, .little),
        .vocab_size = try reader.readInt(u64, .little),
        .local_batch_size = try reader.readInt(u64, .little),
        .learning_rate = try readF32(reader),
        .momentum = try readF32(reader),
        .fisher_gamma = try readF32(reader),
        .fisher_epsilon = try readF32(reader),
        .trust_ratio = try readF32(reader),
        .weight_floor = try readF32(reader),
        .optimizer_warmup_steps = try reader.readInt(u64, .little),
        .spectral_interval = try reader.readInt(u64, .little),
        .spectral_target_norm = try readF32(reader),
        .spectral_iterations = try reader.readInt(u64, .little),
        .reconstruction_alpha = try readF32(reader),
        .phase_a_steps = try reader.readInt(u64, .little),
        .phase_b_steps = try reader.readInt(u64, .little),
        .logdet_weight = try readF32(reader),
        .gradient_clip_norm = try readF32(reader),
        .grad_mean = try readBool(reader),
        .use_normalized_gradient_flow = try readBool(reader),
        .embedding_seed = try reader.readInt(u64, .little),
        .default_max_seq_len = try reader.readInt(u64, .little),
        .reasoning_cycles = try reader.readInt(u64, .little),
        .relational_pass_interval = try reader.readInt(u64, .little),
        .shuffle_target_control = try readBool(reader),
        .target_source_frozen = try readBool(reader),
        .spectral_depth_compensation = try readBool(reader),
        .shuffle_control_state = try reader.readInt(u64, .little),
        .relational_fast_mode = try readBool(reader),
        .clip_min = try readF32(reader),
        .clip_max = try readF32(reader),
        .rsf_optimizer_step = try reader.readInt(u64, .little),
    };
}
