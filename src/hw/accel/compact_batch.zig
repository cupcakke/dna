const std = @import("std");

pub const CompactError = error{
    InvalidDimensions,
};

pub fn compactSequenceLength(lengths: []const usize) usize {
    var maximum: usize = 0;
    for (lengths) |length| {
        if (length > maximum) maximum = length;
    }
    return if (maximum == 0) 1 else maximum;
}

pub fn packedTokenCount(batch: usize, compact_seq: usize) CompactError!usize {
    if (batch == 0 or compact_seq == 0) return CompactError.InvalidDimensions;
    return std.math.mul(usize, batch, compact_seq) catch CompactError.InvalidDimensions;
}

pub fn packTokens(
    dest: []u32,
    src: []const u32,
    batch: usize,
    padded_seq: usize,
    compact_seq: usize,
    lengths: []const usize,
) CompactError!void {
    if (lengths.len != batch) return CompactError.InvalidDimensions;
    const src_n = std.math.mul(usize, batch, padded_seq) catch return CompactError.InvalidDimensions;
    const dst_n = try packedTokenCount(batch, compact_seq);
    if (src.len != src_n or dest.len != dst_n) return CompactError.InvalidDimensions;
    @memset(dest, 0);
    var batch_index: usize = 0;
    while (batch_index < batch) : (batch_index += 1) {
        const length = lengths[batch_index];
        if (length > compact_seq or length > padded_seq) return CompactError.InvalidDimensions;
        const src_base = std.math.mul(usize, batch_index, padded_seq) catch return CompactError.InvalidDimensions;
        const dst_base = std.math.mul(usize, batch_index, compact_seq) catch return CompactError.InvalidDimensions;
        if (length > 0) {
            @memcpy(dest[dst_base .. dst_base + length], src[src_base .. src_base + length]);
        }
    }
}

pub fn scatterRows(
    dest: []f16,
    src: []const f16,
    batch: usize,
    padded_seq: usize,
    compact_seq: usize,
    dim: usize,
    lengths: []const usize,
) CompactError!void {
    if (dim == 0 or lengths.len != batch) return CompactError.InvalidDimensions;
    const src_rows = try packedTokenCount(batch, compact_seq);
    const dst_rows = std.math.mul(usize, batch, padded_seq) catch return CompactError.InvalidDimensions;
    const src_n = std.math.mul(usize, src_rows, dim) catch return CompactError.InvalidDimensions;
    const dst_n = std.math.mul(usize, dst_rows, dim) catch return CompactError.InvalidDimensions;
    if (src.len != src_n or dest.len != dst_n) return CompactError.InvalidDimensions;
    @memset(dest, 0);
    var batch_index: usize = 0;
    while (batch_index < batch) : (batch_index += 1) {
        const length = lengths[batch_index];
        if (length > compact_seq or length > padded_seq) return CompactError.InvalidDimensions;
        var row: usize = 0;
        while (row < length) : (row += 1) {
            const src_row = std.math.add(usize, std.math.mul(usize, batch_index, compact_seq) catch return CompactError.InvalidDimensions, row) catch return CompactError.InvalidDimensions;
            const dst_row = std.math.add(usize, std.math.mul(usize, batch_index, padded_seq) catch return CompactError.InvalidDimensions, row) catch return CompactError.InvalidDimensions;
            const src_off = std.math.mul(usize, src_row, dim) catch return CompactError.InvalidDimensions;
            const dst_off = std.math.mul(usize, dst_row, dim) catch return CompactError.InvalidDimensions;
            @memcpy(dest[dst_off .. dst_off + dim], src[src_off .. src_off + dim]);
        }
    }
}
