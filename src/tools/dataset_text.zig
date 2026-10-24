//! Shared helpers for the offline corpus tools.
//!
//! Both the pre-tokenizer and the vocabulary trainer consume the benchmark
//! corpus, which is a JSONL stream whose records carry a `text` field. The
//! helpers below centralise path handling, atomic file replacement, buffered
//! line streaming and text extraction so both tools behave identically and
//! reject malformed input in exactly the same way.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Maximum accepted length of a single JSONL record.
pub const max_line_size: usize = 64 * 1024 * 1024;

/// Default read buffer size of `LineReader`. The buffer lives on the heap so
/// large record sizes can never overflow the thread stack.
pub const default_buffer_size: usize = 4 * 1024 * 1024;

pub const DatasetTextError = error{
    InvalidInput,
    InputTooLarge,
    InvalidPath,
};

pub fn openRead(path: []const u8) !std.fs.File {
    if (path.len == 0) return DatasetTextError.InvalidPath;
    if (std.fs.path.isAbsolute(path)) {
        return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
    }
    return std.fs.cwd().openFile(path, .{ .mode = .read_only });
}

pub fn openWrite(path: []const u8) !std.fs.File {
    if (path.len == 0) return DatasetTextError.InvalidPath;
    if (std.fs.path.isAbsolute(path)) {
        return std.fs.createFileAbsolute(path, .{ .truncate = true, .read = true });
    }
    return std.fs.cwd().createFile(path, .{ .truncate = true, .read = true });
}

pub fn deleteFile(path: []const u8) void {
    if (path.len == 0) return;
    if (std.fs.path.isAbsolute(path)) {
        std.fs.deleteFileAbsolute(path) catch {};
    } else {
        std.fs.cwd().deleteFile(path) catch {};
    }
}

pub fn renameFile(old_path: []const u8, new_path: []const u8) !void {
    if (old_path.len == 0 or new_path.len == 0) return DatasetTextError.InvalidPath;
    if (std.fs.path.isAbsolute(old_path) != std.fs.path.isAbsolute(new_path)) {
        return DatasetTextError.InvalidPath;
    }
    if (std.fs.path.isAbsolute(old_path)) {
        return std.fs.renameAbsolute(old_path, new_path);
    }
    return std.fs.cwd().rename(old_path, new_path);
}

pub fn sameFile(left: *const std.fs.File, right: *const std.fs.File) !bool {
    const left_stat = try left.stat();
    const right_stat = try right.stat();
    return left_stat.inode == right_stat.inode;
}

/// Heap-backed newline scanner.
///
/// `std.io.BufferedReader` stores its buffer inline, so a multi-megabyte read
/// buffer would live on the thread stack. This reader keeps the buffer on the
/// heap and appends only the bytes of the current record, which keeps the cost
/// of scanning a multi-gigabyte corpus to one `read` per buffer fill plus one
/// copy per record.
pub const LineReader = struct {
    file: std.fs.File,
    allocator: Allocator,
    buffer: []u8,
    start: usize,
    end: usize,
    eof: bool,

    pub fn init(allocator: Allocator, file: std.fs.File, buffer_size: usize) !LineReader {
        if (buffer_size == 0) return DatasetTextError.InvalidPath;
        return .{
            .file = file,
            .allocator = allocator,
            .buffer = try allocator.alloc(u8, buffer_size),
            .start = 0,
            .end = 0,
            .eof = false,
        };
    }

    pub fn deinit(self: *LineReader) void {
        self.allocator.free(self.buffer);
        self.* = undefined;
    }

    fn fill(self: *LineReader) !bool {
        if (self.eof) return false;
        self.start = 0;
        self.end = 0;
        while (true) {
            const read = self.file.read(self.buffer) catch |err| switch (err) {
                error.InputOutput => return err,
                else => return err,
            };
            if (read == 0) {
                self.eof = true;
                return false;
            }
            self.end = read;
            return true;
        }
    }

    /// Reads the next physical line into `out`. Returns false at end of file.
    pub fn nextLine(self: *LineReader, out: *std.ArrayList(u8)) !bool {
        out.clearRetainingCapacity();

        while (true) {
            if (self.start == self.end) {
                if (!try self.fill()) {
                    return out.items.len > 0;
                }
            }

            const slice = self.buffer[self.start..self.end];
            if (std.mem.indexOfScalar(u8, slice, '\n')) |offset| {
                try out.appendSlice(slice[0..offset]);
                self.start += offset + 1;
                return true;
            }

            try out.appendSlice(slice);
            self.start = self.end;
            if (out.items.len > max_line_size) return DatasetTextError.InputTooLarge;
        }
    }
};

/// Reads the next non-empty record, stripping a trailing carriage return.
pub fn readNextRecord(reader: *LineReader, out: *std.ArrayList(u8)) !bool {
    while (try reader.nextLine(out)) {
        var length = out.items.len;
        while (length > 0 and (out.items[length - 1] == '\r' or out.items[length - 1] == ' ' or out.items[length - 1] == '\t')) {
            length -= 1;
        }
        var begin: usize = 0;
        while (begin < length and (out.items[begin] == ' ' or out.items[begin] == '\t')) : (begin += 1) {}

        if (begin == length) continue;

        if (begin > 0) std.mem.copyForwards(u8, out.items[0 .. length - begin], out.items[begin..length]);
        out.shrinkRetainingCapacity(length - begin);
        return true;
    }
    return false;
}

/// Extracts the `text` field of one JSONL record.
///
/// The returned slice points into `arena`, which the caller resets between
/// records; no per-record free is required and the parser reuses the arena's
/// retained capacity, which keeps the tools free of allocation churn on
/// multi-gigabyte corpora.
pub fn extractText(arena: Allocator, line: []const u8) ![]const u8 {
    if (line.len == 0) return DatasetTextError.InvalidInput;

    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, line, .{
        .allocate = .alloc_always,
    }) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return DatasetTextError.InvalidInput,
    };

    const object = switch (parsed) {
        .object => |value| value,
        else => return DatasetTextError.InvalidInput,
    };
    const text_value = object.get("text") orelse return DatasetTextError.InvalidInput;
    const text = switch (text_value) {
        .string => |value| value,
        else => return DatasetTextError.InvalidInput,
    };
    if (text.len == 0) return DatasetTextError.InvalidInput;
    return text;
}
