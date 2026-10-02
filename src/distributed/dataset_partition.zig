const std = @import("std");

pub const Bounds = struct {
    start: usize,
    count: usize,
};

pub const Error = error{
    EmptyDataset,
    InvalidWorldSize,
    InvalidRank,
    Overflow,
};

pub fn bounds(total: usize, world_size: usize, rank: usize) Error!Bounds {
    if (total == 0) return Error.EmptyDataset;
    if (world_size == 0) return Error.InvalidWorldSize;
    if (rank >= world_size) return Error.InvalidRank;
    const base = total / world_size;
    const remainder = total % world_size;
    const count = base + @intFromBool(rank < remainder);
    const start = if (rank < remainder)
        std.math.mul(usize, rank, base + 1) catch return Error.Overflow
    else
        std.math.add(
            usize,
            std.math.mul(usize, remainder, base + 1) catch return Error.Overflow,
            std.math.mul(usize, rank - remainder, base) catch return Error.Overflow,
        ) catch return Error.Overflow;
    return .{ .start = start, .count = count };
}
