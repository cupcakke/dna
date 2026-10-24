//! Cross-module invariants for the Morpho-Graph Tokenizer.
//!
//! These tests protect the properties the training pipeline depends on:
//! byte-pair training terminates on a bounded corpus, the vocabulary file
//! round-trips exactly, encoding is lossless for arbitrary UTF-8 input and the
//! vocabulary produced by training reaches the requested size.

const std = @import("std");
const jaide = @import("jaide");
const MGT = jaide.tokenizer.MGT;

fn buildCorpus(allocator: std.mem.Allocator) ![]const []const u8 {
    const samples = [_][]const u8{
        "the reversible scatter flow stack reconstructs every activation",
        "the spectral fisher diagonalizer conditions the gradient update",
        "a magyar mondat toldalekokkal fejezi ki a nyelvtani viszonyokat",
        "surprise based memory replaces the key value cache entirely",
        "futhark kernels fuse prediction reconstruction and jacobian terms",
    };

    const corpus = try allocator.alloc([]const u8, 64);
    var index: usize = 0;
    while (index < corpus.len) : (index += 1) {
        corpus[index] = try allocator.dupe(u8, samples[index % samples.len]);
    }
    return corpus;
}

fn freeCorpus(allocator: std.mem.Allocator, corpus: []const []const u8) void {
    for (corpus) |sample| allocator.free(sample);
    allocator.free(corpus);
}

test "byte-pair training terminates and reaches the requested vocabulary size" {
    const allocator = std.testing.allocator;
    const corpus = try buildCorpus(allocator);
    defer freeCorpus(allocator, corpus);

    // The fixed corpus above exhausts its mergeable pairs shortly after 500
    // tokens, so this target is reachable and the assertion below also proves
    // that training stops exactly at the requested size.
    const target: u32 = 500;
    var tokenizer = try MGT.init(allocator, &.{}, &.{}, target, .english);
    defer tokenizer.deinit();

    try tokenizer.trainBPE(corpus, target);

    try std.testing.expectEqual(@as(u32, target), tokenizer.next_token_id);
    try std.testing.expect(tokenizer.vocabSize() == @as(usize, target));
    try std.testing.expect(tokenizer.bpe_pairs.count() > 0);

    // Every byte must be representable, otherwise encoded corpora would be
    // silently reduced to the unknown token.
    var byte_value: u32 = 0;
    while (byte_value < 256) : (byte_value += 1) {
        try std.testing.expect(tokenizer.byte_token_ids[byte_value] != null);
    }
}

test "encoding round-trips arbitrary text through the vocabulary" {
    const allocator = std.testing.allocator;
    const corpus = try buildCorpus(allocator);
    defer freeCorpus(allocator, corpus);

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, 800, .english);
    defer tokenizer.deinit();
    try tokenizer.trainBPE(corpus, 800);

    const probes = [_][]const u8{
        "the reversible scatter flow stack reconstructs every activation",
        "a magyar mondat toldalekokkal fejezi ki a nyelvtani viszonyokat",
        "unseen-tokenization: 0123456789 \t tab and\nnewline",
        "\u{1F600} emoji and \u{00E1}\u{00E9}\u{0151}\u{0171} accented characters",
    };

    for (probes) |probe| {
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();
        try tokenizer.encode(probe, &tokens);

        try std.testing.expect(tokens.items.len > 0);
        for (tokens.items) |token| {
            try std.testing.expect(tokenizer.id_to_token.contains(token));
        }

        const decoded = try tokenizer.detokenize(tokens.items);
        defer allocator.free(decoded);
        try std.testing.expectEqualStrings(probe, decoded);

        // A window must never contain more tokens than the corpus bytes plus
        // the bounded merge overhead allows.
        try std.testing.expect(tokens.items.len <= probe.len);
    }
}

test "bounded encoding never exceeds the requested window" {
    const allocator = std.testing.allocator;
    const corpus = try buildCorpus(allocator);
    defer freeCorpus(allocator, corpus);

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, 800, .english);
    defer tokenizer.deinit();
    try tokenizer.trainBPE(corpus, 800);

    const long_text = corpus[0];

    var index: usize = 0;
    while (index < 400) : (index += 1) {
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        try tokenizer.encodeBounded(long_text, 16, &tokens);
        try std.testing.expect(tokens.items.len <= 16);
        try std.testing.expect(tokens.items.len > 0);
    }
}

test "vocabulary persistence survives a save and load cycle" {
    const allocator = std.testing.allocator;
    const corpus = try buildCorpus(allocator);
    defer freeCorpus(allocator, corpus);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = "tokenizer.vocab";
    var trained = try MGT.init(allocator, &.{}, &.{}, 500, .english);
    defer trained.deinit();
    try trained.trainBPE(corpus, 500);

    {
        const file = try tmp.dir.createFile(path, .{ .truncate = true });
        file.close();
    }
    var absolute_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const absolute_path = try tmp.dir.realpath(path, &absolute_buffer);

    try trained.saveVocab(absolute_path);
    try std.testing.expectEqual(@as(u32, 500), trained.next_token_id);

    var loaded = try MGT.init(allocator, &.{}, &.{}, 500, .english);
    defer loaded.deinit();
    try loaded.loadVocab(absolute_path);

    try std.testing.expectEqual(trained.next_token_id, loaded.next_token_id);
    try std.testing.expectEqual(trained.bpe_pairs.count(), loaded.bpe_pairs.count());
    try std.testing.expectEqual(trained.max_token_len, loaded.max_token_len);

    const probe = "the spectral fisher diagonalizer conditions the gradient update";
    var trained_tokens = std.ArrayList(u32).init(allocator);
    defer trained_tokens.deinit();
    try trained.encode(probe, &trained_tokens);

    var loaded_tokens = std.ArrayList(u32).init(allocator);
    defer loaded_tokens.deinit();
    try loaded.encode(probe, &loaded_tokens);

    try std.testing.expectEqualSlices(u32, trained_tokens.items, loaded_tokens.items);
}

test "encoding a very long unbroken record stays bounded" {
    const allocator = std.testing.allocator;
    const corpus = try buildCorpus(allocator);
    defer freeCorpus(allocator, corpus);

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, 700, .english);
    defer tokenizer.deinit();
    try tokenizer.trainBPE(corpus, 700);

    var text = std.ArrayList(u8).init(allocator);
    defer text.deinit();
    var index: usize = 0;
    while (index < 64 * 1024) : (index += 1) {
        try text.append('a' + @as(u8, @intCast(index % 26)));
    }

    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();

    const started = std.time.nanoTimestamp();
    try tokenizer.encodeBounded(text.items, 128, &tokens);
    const elapsed_ms = @divTrunc(std.time.nanoTimestamp() - started, std.time.ns_per_ms);

    try std.testing.expect(tokens.items.len <= 128);
    // The match search is bounded by the longest vocabulary token, so a 64 KiB
    // record must be encoded in well under a second on any development machine.
    try std.testing.expect(elapsed_ms < 1000);
}

test "unknown words use byte-pair merges instead of collapsing to the unknown token" {
    const allocator = std.testing.allocator;
    const corpus = try buildCorpus(allocator);
    defer freeCorpus(allocator, corpus);

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, 900, .english);
    defer tokenizer.deinit();
    try tokenizer.trainBPE(corpus, 900);

    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();
    try tokenizer.encode("scatterflowspectralfisher", &tokens);

    const unknown: u32 = 1;
    var unknown_count: usize = 0;
    for (tokens.items) |token| {
        if (token == unknown) unknown_count += 1;
    }

    try std.testing.expect(unknown_count == 0);
    try std.testing.expect(tokens.items.len < "scatterflowspectralfisher".len);
}
