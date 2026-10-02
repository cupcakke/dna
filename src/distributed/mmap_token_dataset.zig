const std = @import("std");

pub const magic = "JAIDETOK";
pub const version: u32 = 1;
pub const token_type_u32: u32 = 0;
pub const header_size: u64 = 64;

pub const Error = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedTokenType,
    InvalidHeaderSize,
    InvalidFileSize,
    InvalidTokenCount,
    InvalidVocabularySize,
    InvalidMaximumSequenceLength,
    TokenOutOfRange,
    InvalidWorldSize,
    InvalidRank,
    InvalidBatchIndex,
    InsufficientTokens,
    IntegerOverflow,
    EmptyDataset,
};

pub const Header = struct {
    format_version: u32,
    token_type: u32,
    header_size: u64,
    total_token_count: u64,
    vocabulary_size: u64,
    maximum_sequence_length: u64,

    pub fn write(writer: anytype, value: Header) !void {
        try writer.writeAll(magic);
        try writer.writeInt(u32, value.format_version, .little);
        try writer.writeInt(u32, value.token_type, .little);
        try writer.writeInt(u64, value.header_size, .little);
        try writer.writeInt(u64, value.total_token_count, .little);
        try writer.writeInt(u64, value.vocabulary_size, .little);
        try writer.writeInt(u64, value.maximum_sequence_length, .little);
        try writer.writeInt(u64, 0, .little);
        try writer.writeInt(u64, 0, .little);
    }

    pub fn read(bytes: []const u8) Error!Header {
        if (bytes.len < header_size) return Error.InvalidHeaderSize;
        if (!std.mem.eql(u8, bytes[0..8], magic)) return Error.InvalidMagic;
        const result = Header{
            .format_version = std.mem.readInt(u32, bytes[8..12], .little),
            .token_type = std.mem.readInt(u32, bytes[12..16], .little),
            .header_size = std.mem.readInt(u64, bytes[16..24], .little),
            .total_token_count = std.mem.readInt(u64, bytes[24..32], .little),
            .vocabulary_size = std.mem.readInt(u64, bytes[32..40], .little),
            .maximum_sequence_length = std.mem.readInt(u64, bytes[40..48], .little),
        };
        if (result.format_version != version) return Error.UnsupportedVersion;
        if (result.token_type != token_type_u32) return Error.UnsupportedTokenType;
        if (result.header_size < header_size or result.header_size % 64 != 0) return Error.InvalidHeaderSize;
        if (result.vocabulary_size == 0) return Error.InvalidVocabularySize;
        if (result.maximum_sequence_length == 0) return Error.InvalidMaximumSequenceLength;
        return result;
    }
};

pub const Window = struct {
    tokens: []const u32,
    global_start: u64,
};

pub const MmapTokenDataset = struct {
    file: std.fs.File,
    mapping: []align(std.heap.page_size_min) u8,
    payload: []const u32,
    header: Header,

    pub fn open(path: []const u8, expected_vocabulary_size: ?u64) !MmapTokenDataset {
        const file = if (std.fs.path.isAbsolute(path))
            try std.fs.openFileAbsolute(path, .{ .mode = .read_only })
        else
            try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        errdefer file.close();
        const file_size = try file.getEndPos();
        const size = std.math.cast(usize, file_size) orelse return Error.InvalidFileSize;
        if (size < header_size) return Error.InvalidFileSize;
        const mapped = try std.posix.mmap(null, size, std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, file.handle, 0);
        errdefer std.posix.munmap(mapped);
        const parsed = try Header.read(mapped);
        if (parsed.header_size > file_size) return Error.InvalidFileSize;
        const payload_bytes = std.math.mul(u64, parsed.total_token_count, @sizeOf(u32)) catch return Error.IntegerOverflow;
        const expected_size = std.math.add(u64, parsed.header_size, payload_bytes) catch return Error.IntegerOverflow;
        if (expected_size != file_size) return Error.InvalidFileSize;
        if (expected_vocabulary_size) |vocab| {
            if (parsed.vocabulary_size != vocab) return Error.InvalidVocabularySize;
        }
        const payload_offset = std.math.cast(usize, parsed.header_size) orelse return Error.InvalidFileSize;
        const token_count = std.math.cast(usize, parsed.total_token_count) orelse return Error.InvalidTokenCount;
        const payload_bytes_usize = std.math.mul(usize, token_count, @sizeOf(u32)) catch return Error.IntegerOverflow;
        const raw = mapped[payload_offset .. payload_offset + payload_bytes_usize];
        const mapped_tokens = @as([*]const u32, @ptrCast(@alignCast(raw.ptr)))[0..token_count];
        return .{ .file = file, .mapping = mapped, .payload = mapped_tokens, .header = parsed };
    }

    pub fn close(self: *MmapTokenDataset) void {
        std.posix.munmap(self.mapping);
        self.file.close();
        self.* = undefined;
    }

    pub fn tokens(self: *const MmapTokenDataset) []const u32 {
        return self.payload;
    }

    pub fn rankSlice(self: *const MmapTokenDataset, rank: usize, world_size: usize) ![]const u32 {
        if (world_size == 0) return Error.InvalidWorldSize;
        if (rank >= world_size) return Error.InvalidRank;
        const total = self.payload.len;
        const base = total / world_size;
        const remainder = total % world_size;
        const count = base + @intFromBool(rank < remainder);
        const start = if (rank < remainder) rank * (base + 1) else remainder * (base + 1) + (rank - remainder) * base;
        return self.payload[start .. start + count];
    }

    pub fn window(self: *const MmapTokenDataset, start: usize, length: usize) ![]const u32 {
        const end = std.math.add(usize, start, length) catch return Error.IntegerOverflow;
        if (end > self.payload.len) return Error.InsufficientTokens;
        return self.payload[start..end];
    }

    pub fn batchWindow(self: *const MmapTokenDataset, epoch: u64, batch_index: u64, batch_size: usize, sequence_length: usize, rank: usize, world_size: usize) !Window {
        if (batch_size == 0 or sequence_length == 0) return Error.InvalidBatchIndex;
        if (world_size == 0 or rank >= world_size) return Error.InvalidWorldSize;
        const stride = std.math.mul(u64, @intCast(batch_size), @intCast(sequence_length)) catch return Error.IntegerOverflow;
        const rank_offset = std.math.mul(u64, @intCast(rank), stride) catch return Error.IntegerOverflow;
        const epoch_offset = std.math.mul(u64, epoch, @intCast(self.payload.len)) catch return Error.IntegerOverflow;
        const batch_offset = std.math.mul(u64, batch_index, stride) catch return Error.IntegerOverflow;
        const payload_len_u64 = @as(u64, @intCast(self.payload.len));
        const wrapped_rank_batch = std.math.add(u64, rank_offset, batch_offset) catch return Error.IntegerOverflow;
        const raw_start = std.math.add(
            u64,
            epoch_offset % payload_len_u64,
            wrapped_rank_batch % payload_len_u64,
        ) catch return Error.IntegerOverflow;
        const start = std.math.cast(usize, raw_start % payload_len_u64) orelse return Error.IntegerOverflow;
        const needed_u64 = std.math.add(u64, stride, 1) catch return Error.IntegerOverflow;
        const needed = std.math.cast(usize, needed_u64) orelse return Error.IntegerOverflow;
        if (needed > self.payload.len) return Error.InsufficientTokens;
        const adjusted_start = if (start + needed <= self.payload.len) start else 0;
        return .{ .tokens = self.payload[adjusted_start .. adjusted_start + needed], .global_start = adjusted_start };
    }
};
