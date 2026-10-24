//! Offline vocabulary trainer for the JAIDE tokenizer.
//!
//! The tool streams a bounded slice of the JSONL corpus, builds the complete
//! Morpho-Graph Tokenizer vocabulary with the tokenizer's own trainer (special
//! tokens, affixes, byte tokens, morphological roots and then byte-pair
//! merges), verifies that the requested vocabulary size was reached and writes
//! the result atomically. The morphological roots are additionally exported as
//! a newline-separated anchor list next to the vocabulary, which is the format
//! the distributed trainer reads through `JAIDE_TOKENIZER_ANCHORS`. Running
//! this on a CPU worker keeps vocabulary construction off the GPU clock, which
//! is why the benchmark pipeline can prepare everything it needs for training
//! without holding a GPU allocation.

const std = @import("std");
const MGT = @import("../tokenizer/mgt.zig").MGT;
const text_tools = @import("dataset_text.zig");

const CliError = error{
    InvalidArguments,
    InvalidInput,
    VocabularyTooSmall,
    VocabularyTargetNotReached,
};

const default_corpus_bytes: u64 = 10 * 1024 * 1024;
const default_progress_interval_seconds: u64 = 5;

const Options = struct {
    input_path: []const u8,
    output_path: []const u8,
    vocabulary_size: u32,
    corpus_bytes: u64,
    progress_interval_seconds: u64,
    root_min_frequency: u64,
    root_capacity_divisor: usize,
    root_evidence: MGT.RootEvidence,
};

/// Reads an unsigned environment variable, falling back to `default_value`
/// when it is unset. A variable that is set to something other than a positive
/// decimal number is an error: silently keeping the default would make a
/// misconfigured run look identical to a correctly configured one.
fn readPositiveEnv(allocator: std.mem.Allocator, comptime name: []const u8, default_value: u64) !u64 {
    const raw = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return default_value,
        else => return err,
    };
    defer allocator.free(raw);

    const value = std.fmt.parseInt(u64, raw, 10) catch return CliError.InvalidArguments;
    if (value == 0) return CliError.InvalidArguments;
    return value;
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\usage: jaide-vocab-train <input.jsonl> <output.vocab> <vocabulary_size> [corpus_bytes]
        \\
        \\  input.jsonl      JSONL corpus; every record must carry a string "text" field
        \\  output.vocab     destination vocabulary (replaced atomically)
        \\  vocabulary_size  requested number of tokens (>= 260: four specials plus every byte)
        \\  corpus_bytes     maximum number of corpus bytes to read (default 10485760)
        \\
        \\environment:
        \\  JAIDE_VOCAB_PROGRESS_SECONDS         progress heartbeat interval in seconds (default 5, 0 disables)
        \\  JAIDE_VOCAB_MIN_ROOT_FREQUENCY       corpus frequency a stem needs to become a root (default 2)
        \\  JAIDE_VOCAB_ROOT_BUDGET_DIVISOR      roots may take 1/N of the free vocabulary capacity (default 8)
        \\  JAIDE_VOCAB_ROOT_EVIDENCE            auto (default) uses paradigm and falls back to frequency only
        \\                                       when the corpus proves no paradigm at all; paradigm requires a
        \\                                       stem to occur in more than one shape; frequency mines every stem
        \\                                       that reaches the minimum frequency
        \\
    );
}

fn parseOptions(allocator: std.mem.Allocator) !Options {
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next();
    const input_path = args.next() orelse return CliError.InvalidArguments;
    const output_path = args.next() orelse return CliError.InvalidArguments;
    const vocabulary_size = std.fmt.parseInt(u32, args.next() orelse return CliError.InvalidArguments, 10) catch return CliError.InvalidArguments;
    if (vocabulary_size <= 260) return CliError.VocabularyTooSmall;

    const corpus_bytes = if (args.next()) |raw| blk: {
        const value = std.fmt.parseInt(u64, raw, 10) catch return CliError.InvalidArguments;
        if (value == 0) return CliError.InvalidArguments;
        break :blk value;
    } else default_corpus_bytes;

    if (args.next() != null) return CliError.InvalidArguments;

    const progress_interval_seconds = blk: {
        const raw = std.process.getEnvVarOwned(allocator, "JAIDE_VOCAB_PROGRESS_SECONDS") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => break :blk default_progress_interval_seconds,
            else => return err,
        };
        defer allocator.free(raw);
        break :blk std.fmt.parseInt(u64, raw, 10) catch return CliError.InvalidArguments;
    };

    const root_min_frequency = readPositiveEnv(allocator, "JAIDE_VOCAB_MIN_ROOT_FREQUENCY", MGT.default_root_min_frequency) catch |err| switch (err) {
        CliError.InvalidArguments => return CliError.InvalidArguments,
        else => return err,
    };
    const root_capacity_divisor_u64 = readPositiveEnv(allocator, "JAIDE_VOCAB_ROOT_BUDGET_DIVISOR", MGT.default_root_capacity_divisor) catch |err| switch (err) {
        CliError.InvalidArguments => return CliError.InvalidArguments,
        else => return err,
    };
    const root_capacity_divisor = std.math.cast(usize, root_capacity_divisor_u64) orelse return CliError.InvalidArguments;

    const root_evidence = blk: {
        const raw = std.process.getEnvVarOwned(allocator, "JAIDE_VOCAB_ROOT_EVIDENCE") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => break :blk (MGT.RootMiningPolicy{}).evidence,
            else => return err,
        };
        defer allocator.free(raw);
        break :blk std.meta.stringToEnum(MGT.RootEvidence, raw) orelse return CliError.InvalidArguments;
    };

    return .{
        .input_path = input_path,
        .output_path = output_path,
        .vocabulary_size = vocabulary_size,
        .corpus_bytes = corpus_bytes,
        .progress_interval_seconds = progress_interval_seconds,
        .root_min_frequency = root_min_frequency,
        .root_capacity_divisor = root_capacity_divisor,
        .root_evidence = root_evidence,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("jaide-vocab-train: internal allocation leak detected\n", .{});
        }
    }
    const allocator = gpa.allocator();

    const options = parseOptions(allocator) catch |err| switch (err) {
        CliError.InvalidArguments, CliError.VocabularyTooSmall => {
            try printUsage(std.io.getStdErr().writer());
            return CliError.InvalidArguments;
        },
        else => return err,
    };

    const stderr = std.io.getStdErr().writer();

    const input_file = try text_tools.openRead(options.input_path);
    defer input_file.close();

    var line_reader = try text_tools.LineReader.init(allocator, input_file, text_tools.default_buffer_size);
    defer line_reader.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var record = std.ArrayList(u8).init(allocator);
    defer record.deinit();

    var corpus = std.ArrayList([]const u8).init(allocator);
    defer corpus.deinit();

    var corpus_bytes: u64 = 0;
    var records: u64 = 0;
    var last_report_ns = std.time.nanoTimestamp();
    const started_ns = last_report_ns;
    const report_interval_ns: i128 = @as(i128, options.progress_interval_seconds) * std.time.ns_per_s;
    const progress_enabled = options.progress_interval_seconds > 0;

    while (corpus_bytes < options.corpus_bytes) {
        const has_record = try text_tools.readNextRecord(&line_reader, &record);
        if (!has_record) break;

        const text = text_tools.extractText(arena.allocator(), record.items) catch |err| switch (err) {
            text_tools.DatasetTextError.InvalidInput => continue,
            else => return err,
        };
        try corpus.append(text);
        records += 1;
        corpus_bytes = std.math.add(u64, corpus_bytes, @intCast(text.len)) catch return CliError.InvalidInput;

        if (progress_enabled) {
            const now = std.time.nanoTimestamp();
            if (now - last_report_ns >= report_interval_ns) {
                const elapsed_seconds = @as(f64, @floatFromInt(now - started_ns)) / @as(f64, std.time.ns_per_s);
                try stderr.print(
                    "jaide-vocab-train: read records={d} bytes={d}/{d} elapsed={d:.1}s\n",
                    .{ records, corpus_bytes, options.corpus_bytes, elapsed_seconds },
                );
                last_report_ns = now;
            }
        }
    }

    if (corpus.items.len == 0) return CliError.InvalidInput;

    try stderr.print(
        "jaide-vocab-train: training bpe on records={d} bytes={d} target={d}\n",
        .{ records, corpus_bytes, options.vocabulary_size },
    );

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, options.vocabulary_size, .english);
    defer tokenizer.deinit();

    try tokenizer.configureRootMining(.{
        .minimum_frequency = options.root_min_frequency,
        .capacity_divisor = options.root_capacity_divisor,
        .evidence = options.root_evidence,
    });

    try stderr.print(
        "jaide-vocab-train: morphology policy evidence={s} min_root_frequency={d} root_budget=1/{d}\n",
        .{ @tagName(options.root_evidence), options.root_min_frequency, options.root_capacity_divisor },
    );

    try tokenizer.trainBPE(corpus.items, options.vocabulary_size);

    const reached = tokenizer.next_token_id;
    const root_stats = tokenizer.root_stats;
    try stderr.print(
        "jaide-vocab-train: trained vocab={d} merges={d} roots={d} anchors={d} words={d} productive_stems={d} added_roots={d}\n",
        .{
            reached,
            tokenizer.bpe_pairs.count(),
            tokenizer.rootCount(),
            tokenizer.anchorCount(),
            root_stats.words,
            root_stats.candidate_stems,
            root_stats.added_roots,
        },
    );

    if (reached != options.vocabulary_size) {
        try stderr.print(
            "jaide-vocab-train: requested vocabulary size {d} was not reachable, the corpus exhausted its mergeable pairs at {d}\n",
            .{ options.vocabulary_size, reached },
        );
        return CliError.VocabularyTargetNotReached;
    }

    const temporary_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{options.output_path});
    defer allocator.free(temporary_path);
    text_tools.deleteFile(temporary_path);

    var temporary_is_present = true;
    errdefer if (temporary_is_present) text_tools.deleteFile(temporary_path);

    try tokenizer.saveVocab(temporary_path);
    try text_tools.renameFile(temporary_path, options.output_path);
    temporary_is_present = false;

    // The anchor list carries the mined roots in the plain token-list format
    // the distributed trainer reads from JAIDE_TOKENIZER_ANCHORS, so a rank that
    // builds its tokenizer from a text vocabulary sees exactly the roots this
    // run mined. It is written after the vocabulary and through the same
    // temporary-file protocol, so a failed write can never leave a stale list
    // next to a fresh vocabulary.
    const anchors_path = try std.fmt.allocPrint(allocator, "{s}.anchors", .{options.output_path});
    defer allocator.free(anchors_path);

    const anchors_temporary_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{anchors_path});
    defer allocator.free(anchors_temporary_path);
    text_tools.deleteFile(anchors_temporary_path);

    var anchors_temporary_is_present = true;
    errdefer if (anchors_temporary_is_present) text_tools.deleteFile(anchors_temporary_path);

    const anchors_written = try tokenizer.writeAnchorList(anchors_temporary_path);
    if (anchors_written) {
        try text_tools.renameFile(anchors_temporary_path, anchors_path);
        anchors_temporary_is_present = false;
    } else {
        // The vocabulary carries no anchors, so no list is produced. A stale
        // list from an earlier run must not survive next to the fresh
        // vocabulary, because the trainer would then load anchors that this
        // vocabulary does not contain.
        text_tools.deleteFile(anchors_temporary_path);
        anchors_temporary_is_present = false;
        text_tools.deleteFile(anchors_path);
    }

    const anchors_size = if (anchors_written) blk: {
        const anchors_file = try text_tools.openRead(anchors_path);
        defer anchors_file.close();
        break :blk try anchors_file.getEndPos();
    } else 0;

    if (root_stats.fell_back) {
        try stderr.print(
            "jaide-vocab-train: the corpus proves no stem paradigm, so the frequency rule decided the root set; roots mined this way can be truncations\n",
            .{},
        );
    }

    if (root_stats.candidate_stems == 0) {
        try stderr.print(
            "jaide-vocab-train: warning: no stem passed the {s} evidence rule in this corpus slice; the morphological layer stays inactive and out-of-vocabulary words are encoded by the byte-pair layer\n",
            .{@tagName(root_stats.evidence)},
        );
    }

    const output_file = try text_tools.openRead(options.output_path);
    defer output_file.close();
    const output_size = try output_file.getEndPos();

    const elapsed_seconds = @as(f64, @floatFromInt(std.time.nanoTimestamp() - started_ns)) / @as(f64, std.time.ns_per_s);
    if (anchors_written) {
        try stderr.print(
            "jaide-vocab-train: finished vocabulary={d} bytes={d} anchors={s} anchor_bytes={d} elapsed={d:.1}s\n",
            .{ reached, output_size, anchors_path, anchors_size, elapsed_seconds },
        );
    } else {
        try stderr.print(
            "jaide-vocab-train: finished vocabulary={d} bytes={d} anchors=none elapsed={d:.1}s\n",
            .{ reached, output_size, elapsed_seconds },
        );
    }
}
