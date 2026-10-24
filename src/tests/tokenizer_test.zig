//! Cross-module invariants for the Morpho-Graph Tokenizer.
//!
//! These tests protect the properties the training pipeline depends on:
//! byte-pair training terminates on a bounded corpus, the vocabulary file
//! round-trips exactly, encoding is lossless for arbitrary UTF-8 input and the
//! vocabulary produced by training reaches the requested size. The
//! morphological layer is covered separately: mined roots must drive the
//! decomposition branch, the mined set must be a pure function of the corpus,
//! and the anchor positions reported for an encoded text must name the exact
//! byte offsets of the anchor tokens.

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

    // The morphological tables travel with the vocabulary: a rank that only
    // loads the file must decompose and anchor exactly like the trainer that
    // built it.
    try std.testing.expectEqual(trained.rootCount(), loaded.rootCount());
    try std.testing.expectEqual(trained.anchorCount(), loaded.anchorCount());

    var trained_anchor_list = std.ArrayList(u8).init(allocator);
    defer trained_anchor_list.deinit();
    var loaded_anchor_list = std.ArrayList(u8).init(allocator);
    defer loaded_anchor_list.deinit();

    const trained_path = try std.fmt.allocPrint(allocator, "{s}.trained.anchors", .{absolute_path});
    defer allocator.free(trained_path);
    const loaded_path = try std.fmt.allocPrint(allocator, "{s}.loaded.anchors", .{absolute_path});
    defer allocator.free(loaded_path);
    defer std.fs.deleteFileAbsolute(trained_path) catch {};
    defer std.fs.deleteFileAbsolute(loaded_path) catch {};

    const trained_wrote = try trained.writeAnchorList(trained_path);
    const loaded_wrote = try loaded.writeAnchorList(loaded_path);
    try std.testing.expectEqual(trained_wrote, loaded_wrote);

    if (trained_wrote) {
        const trained_file = try std.fs.openFileAbsolute(trained_path, .{ .mode = .read_only });
        defer trained_file.close();
        try trained_file.reader().readAllArrayList(&trained_anchor_list, 1 * 1024 * 1024);

        const loaded_file = try std.fs.openFileAbsolute(loaded_path, .{ .mode = .read_only });
        defer loaded_file.close();
        try loaded_file.reader().readAllArrayList(&loaded_anchor_list, 1 * 1024 * 1024);

        try std.testing.expect(trained_anchor_list.items.len > 0);
        try std.testing.expectEqualStrings(trained_anchor_list.items, loaded_anchor_list.items);
    }

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

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, morphology_target, .english);
    defer tokenizer.deinit();
    try tokenizer.trainBPE(corpus, morphology_target);

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

/// Vocabulary size the morphology corpus can fill. The corpus is deliberately
/// small, so its mergeable pairs run out shortly above this target; the tests
/// below need a target that training reaches exactly.
const morphology_target: u32 = 340;

fn buildMorphologyCorpus(allocator: std.mem.Allocator) ![]const []const u8 {
    const samples = [_][]const u8{
        "unflurbing the reflurbed stack",
        "reflurbing unflurbed the flurbed core",
        "the unflurbing reflurbing memory anchor",
    };

    const corpus = try allocator.alloc([]const u8, 24);
    var index: usize = 0;
    while (index < corpus.len) : (index += 1) {
        corpus[index] = try allocator.dupe(u8, samples[index % samples.len]);
    }
    return corpus;
}

test "morphological roots are mined from the corpus and drive the decomposition" {
    const allocator = std.testing.allocator;
    const corpus = try buildMorphologyCorpus(allocator);
    defer freeCorpus(allocator, corpus);

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, morphology_target, .english);
    defer tokenizer.deinit();

    try tokenizer.trainBPE(corpus, morphology_target);
    try std.testing.expectEqual(morphology_target, tokenizer.next_token_id);

    // The trainer must have scanned the corpus and registered stems; a
    // tokenizer that only performed byte-pair merges would leave both empty.
    const stats = tokenizer.root_stats;
    try std.testing.expect(stats.words >= 24);
    try std.testing.expect(stats.candidate_stems > 0);
    try std.testing.expect(stats.added_roots > 0);
    try std.testing.expectEqual(stats.minimum_frequency, 2);

    try std.testing.expect(tokenizer.rootCount() > 0);
    try std.testing.expect(tokenizer.anchors.contains("flurb"));

    // The mined stem, the affix inventory and the encoder must agree: the word
    // decomposes into exactly prefix, root and suffix and decodes losslessly.
    var decomposed = std.ArrayList(u32).init(allocator);
    defer decomposed.deinit();

    const did_decompose = try tokenizer.decompose("unflurbing", &decomposed);
    try std.testing.expect(did_decompose);
    try std.testing.expectEqual(@as(usize, 3), decomposed.items.len);

    const prefix_text = tokenizer.id_to_token.get(decomposed.items[0]) orelse return error.MissingToken;
    const root_text = tokenizer.id_to_token.get(decomposed.items[1]) orelse return error.MissingToken;
    const suffix_text = tokenizer.id_to_token.get(decomposed.items[2]) orelse return error.MissingToken;
    try std.testing.expectEqualStrings("un", prefix_text);
    try std.testing.expectEqualStrings("flurb", root_text);
    try std.testing.expectEqualStrings("ing", suffix_text);

    const decoded = try tokenizer.detokenize(decomposed.items);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("unflurbing", decoded);

    // Encoding takes the same path, so the word must reach the stream as three
    // tokens instead of a byte-pair fallback.
    var encoded = std.ArrayList(u32).init(allocator);
    defer encoded.deinit();
    try tokenizer.encode("unflurbing", &encoded);
    try std.testing.expectEqualSlices(u32, decomposed.items, encoded.items);
}

test "root mining and the anchor list are pure functions of the corpus" {
    const allocator = std.testing.allocator;
    const corpus = try buildMorphologyCorpus(allocator);
    defer freeCorpus(allocator, corpus);

    var first = try MGT.init(allocator, &.{}, &.{}, morphology_target, .english);
    defer first.deinit();
    try first.trainBPE(corpus, morphology_target);

    var second = try MGT.init(allocator, &.{}, &.{}, morphology_target, .english);
    defer second.deinit();
    try second.trainBPE(corpus, morphology_target);

    try std.testing.expectEqual(first.rootCount(), second.rootCount());
    try std.testing.expectEqual(first.anchorCount(), second.anchorCount());
    try std.testing.expectEqual(first.root_stats.candidate_stems, second.root_stats.candidate_stems);
    try std.testing.expectEqual(first.root_stats.added_roots, second.root_stats.added_roots);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const first_name = "first.anchors";
    const second_name = "second.anchors";
    {
        const file = try tmp.dir.createFile(first_name, .{ .truncate = true });
        file.close();
    }
    {
        const file = try tmp.dir.createFile(second_name, .{ .truncate = true });
        file.close();
    }

    var absolute_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const first_path = try tmp.dir.realpath(first_name, &absolute_buffer);
    var second_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const second_path = try tmp.dir.realpath(second_name, &second_buffer);

    const first_wrote = try first.writeAnchorList(first_path);
    const second_wrote = try second.writeAnchorList(second_path);
    try std.testing.expect(first_wrote);
    try std.testing.expect(second_wrote);

    const first_file = try std.fs.openFileAbsolute(first_path, .{ .mode = .read_only });
    defer first_file.close();
    const first_bytes = try first_file.readToEndAlloc(allocator, 1 * 1024 * 1024);
    defer allocator.free(first_bytes);

    const second_file = try std.fs.openFileAbsolute(second_path, .{ .mode = .read_only });
    defer second_file.close();
    const second_bytes = try second_file.readToEndAlloc(allocator, 1 * 1024 * 1024);
    defer allocator.free(second_bytes);

    try std.testing.expect(first_bytes.len > 0);
    try std.testing.expectEqualStrings(first_bytes, second_bytes);
    try std.testing.expect(std.mem.indexOf(u8, first_bytes, "flurb\n") != null);
}

test "anchor positions name the byte offsets of the anchor tokens" {
    const allocator = std.testing.allocator;
    const corpus = try buildMorphologyCorpus(allocator);
    defer freeCorpus(allocator, corpus);

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, 700, .english);
    defer tokenizer.deinit();
    try tokenizer.trainBPE(corpus, 700);

    const text = "the unflurbing stack reflurbed the core";

    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();

    var positions = std.ArrayList(usize).init(allocator);
    defer positions.deinit();

    try tokenizer.tokenizeWithAnchors(text, &tokens, &positions);

    const decoded = try tokenizer.detokenize(tokens.items);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(text, decoded);

    var expected = std.ArrayList(usize).init(allocator);
    defer expected.deinit();

    var cursor: usize = 0;
    for (tokens.items) |token| {
        const surface = tokenizer.byte_token_values.get(token);
        var is_anchor = false;
        if (surface == null) {
            if (tokenizer.id_to_token.get(token)) |token_text| {
                is_anchor = tokenizer.anchors.contains(token_text);
            }
        }
        if (is_anchor) try expected.append(cursor);
        cursor += if (surface) |_| 1 else tokenizer.id_to_token.get(token).?.len;
    }

    try std.testing.expect(cursor == text.len);
    try std.testing.expect(expected.items.len > 0);
    try std.testing.expectEqualSlices(usize, expected.items, positions.items);

    for (positions.items) |position| {
        try std.testing.expect(position < text.len);
    }
}

test "a stem with a single affixed form is not mined" {
    const allocator = std.testing.allocator;

    const samples = [_][]const u8{
        "brimtation and the core",
        "the core carries brimtation once",
    };

    const corpus = try allocator.alloc([]const u8, 16);
    for (corpus, 0..) |*slot, index| {
        slot.* = try allocator.dupe(u8, samples[index % samples.len]);
    }
    defer freeCorpus(allocator, corpus);

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, 330, .english);
    defer tokenizer.deinit();
    try tokenizer.configureRootMining(.{ .evidence = .paradigm });
    try tokenizer.trainBPE(corpus, 330);

    // "brimtation" splits into the suffix "tion" and the remainder "brimta",
    // but neither the remainder nor any second affixation of it exists in the
    // corpus. A miner that trusted a single split would register "brimta" and
    // spend a vocabulary slot on a truncation; the paradigm requirement rejects
    // it instead.
    try std.testing.expectEqual(@as(usize, 0), tokenizer.root_stats.candidate_stems);
    try std.testing.expectEqual(@as(usize, 0), tokenizer.rootCount());
    try std.testing.expectEqual(@as(usize, 0), tokenizer.anchorCount());
    try std.testing.expect(!tokenizer.anchors.contains("brimta"));
    try std.testing.expect(!tokenizer.token_to_id.contains("brimta"));

    // Encoding still succeeds: the word is handled by the byte-pair layer.
    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();
    try tokenizer.encode("brimtation", &tokens);
    try std.testing.expect(tokens.items.len > 0);

    const decoded = try tokenizer.detokenize(tokens.items);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("brimtation", decoded);
}

test "a productive english stem is mined and decomposes its inflections" {
    const allocator = std.testing.allocator;

    const samples = [_][]const u8{
        "connect the connected encoder and the connecting bridge",
        "reconnect the reconnected encoder connection",
        "connect disconnect connected reconnect",
    };

    const corpus = try allocator.alloc([]const u8, 18);
    for (corpus, 0..) |*slot, index| {
        slot.* = try allocator.dupe(u8, samples[index % samples.len]);
    }
    defer freeCorpus(allocator, corpus);

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, 360, .english);
    defer tokenizer.deinit();
    try tokenizer.trainBPE(corpus, 360);

    // "connect" is attested as a whole word and behind three different affix
    // combinations, so it is a root; "connec" only ever appears inside
    // "connection" and "disconnect" inside "disconnect", so neither is mined.
    // The default policy is `auto`, and a corpus that proves a paradigm must
    // never reach the lenient fallback.
    try std.testing.expect(tokenizer.root_stats.evidence == .paradigm);
    try std.testing.expect(!tokenizer.root_stats.fell_back);
    try std.testing.expect(tokenizer.rootCount() > 0);
    try std.testing.expect(tokenizer.anchors.contains("connect"));
    try std.testing.expect(!tokenizer.anchors.contains("connec"));

    var decomposed = std.ArrayList(u32).init(allocator);
    defer decomposed.deinit();

    try std.testing.expect(try tokenizer.decompose("reconnected", &decomposed));
    try std.testing.expectEqual(@as(usize, 3), decomposed.items.len);

    const prefix_text = tokenizer.id_to_token.get(decomposed.items[0]) orelse return error.MissingToken;
    const root_text = tokenizer.id_to_token.get(decomposed.items[1]) orelse return error.MissingToken;
    const suffix_text = tokenizer.id_to_token.get(decomposed.items[2]) orelse return error.MissingToken;
    try std.testing.expectEqualStrings("re", prefix_text);
    try std.testing.expectEqualStrings("connect", root_text);
    try std.testing.expectEqualStrings("ed", suffix_text);

    const decoded = try tokenizer.detokenize(decomposed.items);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("reconnected", decoded);
}

test "auto keeps the morphological layer alive when no paradigm can be proven" {
    const allocator = std.testing.allocator;

    // A repeated keyword list: every word appears in exactly one shape, so the
    // strict rule has no evidence to work with. With the default policy the
    // miner must fall back to the frequency rule instead of leaving the
    // morphological layer switched off for the whole run.
    const samples = [_][]const u8{
        "temporal memory surprise spectral relational gradient",
        "relational gradient temporal memory surprise spectral",
    };

    const corpus = try allocator.alloc([]const u8, 20);
    for (corpus, 0..) |*slot, index| {
        slot.* = try allocator.dupe(u8, samples[index % samples.len]);
    }
    defer freeCorpus(allocator, corpus);

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, 420, .english);
    defer tokenizer.deinit();
    try tokenizer.trainBPE(corpus, 420);

    try std.testing.expectEqual(MGT.default_root_capacity_divisor, tokenizer.rootMiningPolicy().capacity_divisor);
    try std.testing.expect(tokenizer.root_stats.evidence == .frequency);
    try std.testing.expect(tokenizer.root_stats.fell_back);
    try std.testing.expect(tokenizer.root_stats.candidate_stems > 0);
    try std.testing.expectEqual(tokenizer.root_stats.candidate_stems, tokenizer.rootCount());
    try std.testing.expect(tokenizer.anchorCount() > 0);

    // An explicit policy must not fall back: a caller that asked for the strict
    // rule gets the strict rule, even when that means no roots at all.
    var strict = try MGT.init(allocator, &.{}, &.{}, 420, .english);
    defer strict.deinit();
    try strict.configureRootMining(.{ .evidence = .paradigm });
    try strict.trainBPE(corpus, 420);

    try std.testing.expect(strict.root_stats.evidence == .paradigm);
    try std.testing.expect(!strict.root_stats.fell_back);
    try std.testing.expectEqual(@as(usize, 0), strict.rootCount());
    try std.testing.expectEqual(@as(usize, 0), strict.anchorCount());
}

test "the exported anchor list is accepted by the tokenizer anchor parameter" {
    const allocator = std.testing.allocator;

    const samples = [_][]const u8{
        "connect the connected encoder and the connecting bridge",
        "reconnect the reconnected encoder connection",
        "connect disconnect connected reconnect",
    };

    const corpus = try allocator.alloc([]const u8, 18);
    for (corpus, 0..) |*slot, index| {
        slot.* = try allocator.dupe(u8, samples[index % samples.len]);
    }
    defer freeCorpus(allocator, corpus);

    var trained = try MGT.init(allocator, &.{}, &.{}, 360, .english);
    defer trained.deinit();
    try trained.trainBPE(corpus, 360);
    try std.testing.expect(trained.anchorCount() > 0);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const name = "tokenizer.anchors";
    {
        const file = try tmp.dir.createFile(name, .{ .truncate = true });
        file.close();
    }

    var absolute_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const absolute_path = try tmp.dir.realpath(name, &absolute_buffer);

    try std.testing.expect(try trained.writeAnchorList(absolute_path));

    const anchor_text = try std.fs.openFileAbsolute(absolute_path, .{ .mode = .read_only });
    defer anchor_text.close();
    const content = try anchor_text.readToEndAlloc(allocator, 1 * 1024 * 1024);
    defer allocator.free(content);
    try std.testing.expect(std.mem.endsWith(u8, content, "\n"));

    // The distributed trainer reads exactly this shape: one token per line. The
    // lines are collected here and handed to the tokenizer's anchor parameter,
    // which is the call the trainer makes after loading its own vocabulary.
    var anchors = std.ArrayList([]const u8).init(allocator);
    defer anchors.deinit();

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        try anchors.append(line);
    }

    try std.testing.expectEqual(trained.anchorCount(), anchors.items.len);
    for (anchors.items) |anchor| {
        try std.testing.expect(trained.token_to_id.contains(anchor));
    }

    var rebuilt = try MGT.init(allocator, &.{}, anchors.items, 360, .english);
    defer rebuilt.deinit();
    try rebuilt.trainBPE(corpus, 360);

    try std.testing.expectEqual(trained.anchorCount(), rebuilt.anchorCount());
    for (anchors.items) |anchor| {
        try std.testing.expect(rebuilt.anchors.contains(anchor));
    }
}
