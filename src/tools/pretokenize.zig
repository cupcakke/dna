const std = @import("std");
const MGT = @import("../tokenizer/mgt.zig").MGT;
const dataset = @import("../distributed/mmap_token_dataset.zig");

const CliError = error{
    InvalidArguments,
    InvalidInput,
    TokenOutOfRange,
    InputTooLarge,
};

const max_line_size: usize = 64 * 1024 * 1024;
const pad_token_id: u32 = 0;

fn openRead(path: []const u8) !std.fs.File {
    if (path.len == 0) return error.InvalidPath;
    if (std.fs.path.isAbsolute(path)) {
        return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
    }
    return std.fs.cwd().openFile(path, .{ .mode = .read_only });
}

fn openWrite(path: []const u8) !std.fs.File {
    if (path.len == 0) return error.InvalidPath;
    if (std.fs.path.isAbsolute(path)) {
        return std.fs.createFileAbsolute(path, .{ .truncate = true, .read = true });
    }
    return std.fs.cwd().createFile(path, .{ .truncate = true, .read = true });
}

fn deleteFile(path: []const u8) void {
    if (path.len == 0) return;
    if (std.fs.path.isAbsolute(path)) {
        std.fs.deleteFileAbsolute(path) catch {};
    } else {
        std.fs.cwd().deleteFile(path) catch {};
    }
}

fn renameFile(old_path: []const u8, new_path: []const u8) !void {
    if (std.fs.path.isAbsolute(old_path) != std.fs.path.isAbsolute(new_path)) {
        return error.InvalidPath;
    }
    if (std.fs.path.isAbsolute(old_path)) {
        return std.fs.renameAbsolute(old_path, new_path);
    }
    return std.fs.cwd().rename(old_path, new_path);
}

fn sameFile(left: *const std.fs.File, right: *const std.fs.File) !bool {
    const left_stat = try left.stat();
    const right_stat = try right.stat();
    return left_stat.inode == right_stat.inode;
}

fn extractText(allocator: std.mem.Allocator, line: []const u8) ![]u8 {
    if (line.len == 0) return CliError.InvalidInput;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{
        .allocate = .alloc_always,
    }) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return CliError.InvalidInput,
    };
    defer parsed.deinit();

    const object = switch (parsed.value) {
        .object => |value| value,
        else => return CliError.InvalidInput,
    };
    const text_value = object.get("text") orelse return CliError.InvalidInput;
    const text = switch (text_value) {
        .string => |value| value,
        else => return CliError.InvalidInput,
    };
    if (text.len == 0) return CliError.InvalidInput;
    return try allocator.dupe(u8, text);
}

fn writeToken(writer: anytype, value: u32, vocabulary_size: u32) !void {
    if (value >= vocabulary_size) return CliError.TokenOutOfRange;
    try writer.writeInt(u32, value, .little);
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll("usage: jaide-pretokenize <input.jsonl> <output.bin> <vocab.bin> <max_sequence_length> [sample_count]\n");
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        _ = gpa.deinit();
    }
    const allocator = gpa.allocator();

    var args = std.process.argsWithAllocator(allocator) catch |err| {
        if (err == error.OutOfMemory) return err;
        try printUsage(std.io.getStdErr().writer());
        return CliError.InvalidArguments;
    };
    defer args.deinit();

    _ = args.next();
    const input_path = args.next() orelse {
        try printUsage(std.io.getStdErr().writer());
        return CliError.InvalidArguments;
    };
    const output_path = args.next() orelse {
        try printUsage(std.io.getStdErr().writer());
        return CliError.InvalidArguments;
    };
    const vocabulary_path = args.next() orelse {
        try printUsage(std.io.getStdErr().writer());
        return CliError.InvalidArguments;
    };
    const max_sequence_raw = args.next() orelse {
        try printUsage(std.io.getStdErr().writer());
        return CliError.InvalidArguments;
    };
    const max_sequence = std.fmt.parseInt(usize, max_sequence_raw, 10) catch {
        try printUsage(std.io.getStdErr().writer());
        return CliError.InvalidArguments;
    };
    if (max_sequence == 0) {
        try printUsage(std.io.getStdErr().writer());
        return CliError.InvalidArguments;
    }
    const window_length = std.math.add(usize, max_sequence, 1) catch return CliError.InputTooLarge;

    const sample_count = if (args.next()) |raw| blk: {
        const value = std.fmt.parseInt(usize, raw, 10) catch {
            try printUsage(std.io.getStdErr().writer());
            return CliError.InvalidArguments;
        };
        if (value == 0) {
            try printUsage(std.io.getStdErr().writer());
            return CliError.InvalidArguments;
        }
        break :blk @as(?usize, value);
    } else null;
    if (args.next() != null) {
        try printUsage(std.io.getStdErr().writer());
        return CliError.InvalidArguments;
    }

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, null, .english);
    defer tokenizer.deinit();
    try tokenizer.loadVocab(vocabulary_path);
    if (tokenizer.next_token_id <= pad_token_id) return CliError.TokenOutOfRange;

    const input_file = try openRead(input_path);
    defer input_file.close();
    const vocabulary_file = try openRead(vocabulary_path);
    defer vocabulary_file.close();

    if (std.fs.path.isAbsolute(output_path)) {
        const existing_output = openRead(output_path) catch null;
        if (existing_output) |file| {
            var output_file = file;
            defer output_file.close();
            if (try sameFile(&input_file, &output_file) or try sameFile(&vocabulary_file, &output_file)) {
                return CliError.InvalidArguments;
            }
        }
    } else {
        const existing_output = std.fs.cwd().openFile(output_path, .{ .mode = .read_only }) catch null;
        if (existing_output) |file| {
            var output_file = file;
            defer output_file.close();
            if (try sameFile(&input_file, &output_file) or try sameFile(&vocabulary_file, &output_file)) {
                return CliError.InvalidArguments;
            }
        }
    }

    const temporary_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{output_path});
    defer allocator.free(temporary_path);
    const existing_temporary = openRead(temporary_path) catch null;
    if (existing_temporary) |file| {
        var temporary_file = file;
        defer temporary_file.close();
        if (try sameFile(&input_file, &temporary_file) or try sameFile(&vocabulary_file, &temporary_file)) {
            return CliError.InvalidArguments;
        }
    }
    deleteFile(temporary_path);
    var output_file = try openWrite(temporary_path);
    var temporary_is_present = true;
    var output_is_closed = false;
    defer {
        if (!output_is_closed) output_file.close();
        if (temporary_is_present) deleteFile(temporary_path);
    }

    var writer = output_file.writer();
    try writer.writeAll(dataset.magic);
    try writer.writeInt(u32, dataset.version, .little);
    try writer.writeInt(u32, dataset.token_type_u32, .little);
    try writer.writeInt(u64, dataset.header_size, .little);
    const count_position = try output_file.getPos();
    try writer.writeInt(u64, 0, .little);
    try writer.writeInt(u64, tokenizer.next_token_id, .little);
    try writer.writeInt(u64, @intCast(window_length), .little);
    try writer.writeInt(u64, 0, .little);
    try writer.writeInt(u64, 0, .little);
    if (try output_file.getPos() != dataset.header_size) return CliError.InvalidInput;

    var reader = input_file.reader();
    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();
    var samples_written: usize = 0;
    var total_tokens: u64 = 0;

    while (true) {
        if (sample_count) |limit| {
            if (samples_written >= limit) break;
        }
        const raw_line = reader.readUntilDelimiterOrEofAlloc(allocator, '\n', max_line_size) catch |err| switch (err) {
            error.StreamTooLong => return CliError.InputTooLarge,
            else => return err,
        } orelse break;
        defer allocator.free(raw_line);
        const line = std.mem.trimRight(u8, raw_line, "\r");
        if (line.len == 0) continue;

        const text = try extractText(allocator, line);
        defer allocator.free(text);
        tokens.clearRetainingCapacity();
        try tokenizer.encode(text, &tokens);

        var index: usize = 0;
        while (index < window_length) : (index += 1) {
            const token = if (index < tokens.items.len) tokens.items[index] else pad_token_id;
            try writeToken(writer, token, tokenizer.next_token_id);
        }
        total_tokens = std.math.add(u64, total_tokens, @intCast(window_length)) catch return CliError.InputTooLarge;
        samples_written += 1;
    }

    if (sample_count) |limit| {
        if (samples_written != limit) return CliError.InvalidInput;
    }
    try output_file.sync();
    try output_file.seekTo(count_position);
    var header_writer = output_file.writer();
    try header_writer.writeInt(u64, total_tokens, .little);
    try output_file.sync();
    output_file.close();
    output_is_closed = true;
    try renameFile(temporary_path, output_path);
    temporary_is_present = false;
}
