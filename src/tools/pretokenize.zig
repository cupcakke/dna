//! Offline pre-tokenizer for the JAIDE distributed trainer.
//!
//! The tool turns a JSONL corpus into the memory-mapped token dataset that the
//! Futhark trainer reads: a 64-byte little-endian header followed by fixed-size
//! windows of `max_sequence_length + 1` `u32` token ids.
//!
//! Operational contract:
//!   * encoding is bounded, so a single record can never encode more than one
//!     window and a pathological record cannot stall the process;
//!   * progress is reported on stderr every few seconds, which makes a slow
//!     shard visible instead of looking like a hang;
//!   * the output is written to `<output>.tmp` and atomically renamed, so a
//!     failed run never destroys a previously valid dataset.

const std = @import("std");
const MGT = @import("../tokenizer/mgt.zig").MGT;
const dataset = @import("../distributed/mmap_token_dataset.zig");
const text_tools = @import("dataset_text.zig");

const CliError = error{
    InvalidArguments,
    InvalidInput,
    TokenOutOfRange,
    InputTooLarge,
    VocabularyTooLarge,
};

const pad_token_id: u32 = 0;
const default_progress_interval_seconds: u64 = 5;

const Options = struct {
    input_path: []const u8,
    output_path: []const u8,
    vocabulary_path: []const u8,
    max_sequence_length: usize,
    sample_count: ?usize,
    progress_interval_seconds: u64,
};

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\usage: jaide-pretokenize <input.jsonl> <output.bin> <vocab.bin> <max_sequence_length> [sample_count]
        \\
        \\  input.jsonl          JSONL corpus; every record must carry a string "text" field
        \\  output.bin           destination token dataset (replaced atomically)
        \\  vocab.bin            vocabulary produced by jaide-vocab-train
        \\  max_sequence_length  context length; every stored window holds max_sequence_length + 1 tokens
        \\  sample_count         exact number of records to encode (default: until end of file)
        \\
        \\environment:
        \\  JAIDE_PRETOKENIZE_PROGRESS_SECONDS  progress heartbeat interval in seconds (default 5, 0 disables)
        \\
    );
}

fn parseOptions(allocator: std.mem.Allocator) !Options {
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next();
    const input_path = args.next() orelse return CliError.InvalidArguments;
    const output_path = args.next() orelse return CliError.InvalidArguments;
    const vocabulary_path = args.next() orelse return CliError.InvalidArguments;
    const max_sequence_raw = args.next() orelse return CliError.InvalidArguments;

    const max_sequence_length = std.fmt.parseInt(usize, max_sequence_raw, 10) catch return CliError.InvalidArguments;
    if (max_sequence_length == 0) return CliError.InvalidArguments;

    const sample_count = if (args.next()) |raw| blk: {
        const value = std.fmt.parseInt(usize, raw, 10) catch return CliError.InvalidArguments;
        if (value == 0) return CliError.InvalidArguments;
        break :blk @as(?usize, value);
    } else null;

    if (args.next() != null) return CliError.InvalidArguments;

    const progress_interval_seconds = blk: {
        const raw = std.process.getEnvVarOwned(allocator, "JAIDE_PRETOKENIZE_PROGRESS_SECONDS") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => break :blk default_progress_interval_seconds,
            else => return err,
        };
        defer allocator.free(raw);
        break :blk std.fmt.parseInt(u64, raw, 10) catch return CliError.InvalidArguments;
    };

    return .{
        .input_path = input_path,
        .output_path = output_path,
        .vocabulary_path = vocabulary_path,
        .max_sequence_length = max_sequence_length,
        .sample_count = sample_count,
        .progress_interval_seconds = progress_interval_seconds,
    };
}

/// Rejects output paths that alias the corpus or the vocabulary; truncating one
/// of the inputs would silently corrupt the run.
fn ensureOutputIsDistinct(options: Options) !void {
    const input_file = try text_tools.openRead(options.input_path);
    defer input_file.close();
    const vocabulary_file = try text_tools.openRead(options.vocabulary_path);
    defer vocabulary_file.close();

    const candidates = [_][]const u8{
        options.output_path,
        try std.fmt.allocPrint(std.heap.page_allocator, "{s}.tmp", .{options.output_path}),
    };
    defer std.heap.page_allocator.free(candidates[1]);

    for (candidates) |candidate| {
        const existing = text_tools.openRead(candidate) catch continue;
        var existing_file = existing;
        defer existing_file.close();
        if (try text_tools.sameFile(&input_file, &existing_file)) return CliError.InvalidArguments;
        if (try text_tools.sameFile(&vocabulary_file, &existing_file)) return CliError.InvalidArguments;
    }
}

fn writeToken(writer: anytype, value: u32, vocabulary_size: u32) !void {
    if (value >= vocabulary_size) return CliError.TokenOutOfRange;
    try writer.writeInt(u32, value, .little);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("jaide-pretokenize: internal allocation leak detected\n", .{});
        }
    }
    const allocator = gpa.allocator();

    const options = parseOptions(allocator) catch |err| switch (err) {
        CliError.InvalidArguments => {
            try printUsage(std.io.getStdErr().writer());
            return CliError.InvalidArguments;
        },
        else => return err,
    };

    const window_length = std.math.add(usize, options.max_sequence_length, 1) catch return CliError.InputTooLarge;
    const window_tokens_u64 = std.math.cast(u64, window_length) orelse return CliError.InputTooLarge;

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, null, .english);
    defer tokenizer.deinit();
    try tokenizer.loadVocab(options.vocabulary_path);

    if (tokenizer.next_token_id <= pad_token_id) return CliError.TokenOutOfRange;
    const vocabulary_size = tokenizer.next_token_id;

    const estimated_windows = options.sample_count orelse 0;
    const estimated_tokens = std.math.mul(u64, window_tokens_u64, @as(u64, estimated_windows)) catch 0;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const stderr = std.io.getStdErr().writer();
    if (options.sample_count) |count| {
        try stderr.print(
            "jaide-pretokenize: vocab={d} tokens, context={d}, window={d}, samples={d}, estimated_output={d} tokens\n",
            .{ vocabulary_size, options.max_sequence_length, window_length, count, estimated_tokens },
        );
    } else {
        try stderr.print(
            "jaide-pretokenize: vocab={d} tokens, context={d}, window={d}, samples=until-eof\n",
            .{ vocabulary_size, options.max_sequence_length, window_length },
        );
    }

    try ensureOutputIsDistinct(options);

    const input_file = try text_tools.openRead(options.input_path);
    defer input_file.close();

    const input_size = try input_file.getEndPos();

    const temporary_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{options.output_path});
    defer allocator.free(temporary_path);
    text_tools.deleteFile(temporary_path);

    var output_file = try text_tools.openWrite(temporary_path);
    var output_is_closed = false;
    var temporary_is_present = true;
    defer {
        if (!output_is_closed) output_file.close();
        if (temporary_is_present) text_tools.deleteFile(temporary_path);
    }

    try output_file.setEndPos(0);

    var writer = output_file.writer();
    try writer.writeAll(dataset.magic);
    try writer.writeInt(u32, dataset.version, .little);
    try writer.writeInt(u32, dataset.token_type_u32, .little);
    try writer.writeInt(u64, dataset.header_size, .little);
    const count_position = try output_file.getPos();
    try writer.writeInt(u64, 0, .little);
    try writer.writeInt(u64, vocabulary_size, .little);
    // The header records the context length. Every stored window carries one
    // additional token as the prediction target, so the payload length is
    // `(maximum_sequence_length + 1)` tokens per sample.
    try writer.writeInt(u64, @intCast(options.max_sequence_length), .little);
    try writer.writeInt(u64, 0, .little);
    try writer.writeInt(u64, 0, .little);
    if (try output_file.getPos() != dataset.header_size) return CliError.InvalidInput;

    var buffered_writer = std.io.bufferedWriter(writer);
    const buffered = buffered_writer.writer();

    var line_reader = try text_tools.LineReader.init(allocator, input_file, text_tools.default_buffer_size);
    defer line_reader.deinit();

    var record = std.ArrayList(u8).init(allocator);
    defer record.deinit();

    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();

    var samples_written: u64 = 0;
    var total_tokens: u64 = 0;
    var encoded_tokens: u64 = 0;
    var bytes_read: u64 = 0;
    var truncated_records: u64 = 0;

    var last_report_ns = std.time.nanoTimestamp();
    const started_ns = last_report_ns;
    const report_interval_ns: i128 = @as(i128, options.progress_interval_seconds) * std.time.ns_per_s;
    const progress_enabled = options.progress_interval_seconds > 0;

    while (true) {
        if (options.sample_count) |limit| {
            if (samples_written >= @as(u64, limit)) break;
        }

        const has_record = try text_tools.readNextRecord(&line_reader, &record);
        if (!has_record) {
            if (options.sample_count != null) {
                try stderr.print(
                    "jaide-pretokenize: corpus exhausted after {d} records, {d} required\n",
                    .{ samples_written, options.sample_count.? },
                );
                return CliError.InvalidInput;
            }
            break;
        }

        bytes_read = std.math.add(u64, bytes_read, @intCast(record.items.len + 1)) catch return CliError.InputTooLarge;

        _ = arena.reset(.retain_capacity);
        const text = try text_tools.extractText(arena.allocator(), record.items);

        tokens.clearRetainingCapacity();
        try tokenizer.encodeBounded(text, window_length, &tokens);

        if (tokens.items.len > window_length) {
            tokens.shrinkRetainingCapacity(window_length);
        }
        if (tokens.items.len == window_length) truncated_records += 1;
        encoded_tokens = std.math.add(u64, encoded_tokens, @intCast(tokens.items.len)) catch return CliError.InputTooLarge;

        var index: usize = 0;
        while (index < window_length) : (index += 1) {
            const token = if (index < tokens.items.len) tokens.items[index] else pad_token_id;
            try writeToken(buffered, token, vocabulary_size);
        }

        total_tokens = std.math.add(u64, total_tokens, window_tokens_u64) catch return CliError.InputTooLarge;
        samples_written += 1;

        if (progress_enabled) {
            const now = std.time.nanoTimestamp();
            if (now - last_report_ns >= report_interval_ns) {
                const elapsed_ns = now - started_ns;
                const elapsed_seconds = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);
                const sample_rate = if (elapsed_seconds > 0.0)
                    @as(f64, @floatFromInt(samples_written)) / elapsed_seconds
                else
                    0.0;
                try stderr.print(
                    "jaide-pretokenize: samples={d} tokens={d} encoded={d} truncated={d} read={d}/{d} bytes rate={d:.1} samples/s elapsed={d:.1}s\n",
                    .{
                        samples_written,
                        total_tokens,
                        encoded_tokens,
                        truncated_records,
                        bytes_read,
                        input_size,
                        sample_rate,
                        elapsed_seconds,
                    },
                );
                last_report_ns = now;
            }
        }
    }

    try buffered_writer.flush();
    try output_file.sync();
    try output_file.seekTo(count_position);
    var header_writer = output_file.writer();
    try header_writer.writeInt(u64, total_tokens, .little);
    try output_file.sync();
    output_file.close();
    output_is_closed = true;

    if (samples_written == 0) return CliError.InvalidInput;

    try text_tools.renameFile(temporary_path, options.output_path);
    temporary_is_present = false;

    const elapsed_ns = std.time.nanoTimestamp() - started_ns;
    const elapsed_seconds = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);
    const output_size = std.math.add(u64, dataset.header_size, std.math.mul(u64, total_tokens, 4) catch return CliError.InputTooLarge) catch return CliError.InputTooLarge;
    const tokens_per_second = if (elapsed_seconds > 0.0)
        @as(f64, @floatFromInt(total_tokens)) / elapsed_seconds
    else
        0.0;

    try stderr.print(
        "jaide-pretokenize: finished samples={d} tokens={d} encoded={d} truncated={d} bytes={d} rate={d:.1} tokens/s elapsed={d:.1}s\n",
        .{ samples_written, total_tokens, encoded_tokens, truncated_records, output_size, tokens_per_second, elapsed_seconds },
    );
}
