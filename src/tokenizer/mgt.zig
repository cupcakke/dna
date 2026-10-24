const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const core_tensor = @import("../core/tensor.zig");
const core_memory = @import("../core/memory.zig");
const core_io = @import("../core/io.zig");

pub const MGT = struct {
    token_to_id: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    prefixes: std.StringHashMap(u32),
    suffixes: std.StringHashMap(u32),
    roots: std.StringHashMap(u32),
    bpe_pairs: std.AutoHashMap(u64, BPEMerge),
    anchors: std.StringHashMap(u64),
    allocated_strings: std.ArrayList([]u8),
    allocator: Allocator,
    next_token_id: u32,
    language: Language,
    max_vocab_size: ?u32,
    allow_dynamic_tokens: bool,
    sorted_prefix_keys: std.ArrayList([]const u8),
    sorted_suffix_keys: std.ArrayList([]const u8),
    max_prefix_len: usize,
    max_suffix_len: usize,
    /// Length of the longest token string currently present in the vocabulary.
    /// `longestMatch` uses it to bound its scan window, which turns the match
    /// search from a quadratic walk over the remainder of the input into a
    /// bounded lookup per input position.
    max_token_len: usize,
    byte_token_ids: [256]?u32,
    byte_token_values: std.AutoHashMap(u32, u8),
    /// Scratch state reused by the byte-pair encoder. Keeping these buffers on
    /// the tokenizer removes one hash map and one array allocation per encoded
    /// fragment, which matters when millions of fragments are encoded.
    bpe_work_tokens: std.ArrayList(u32),
    bpe_output_tokens: std.ArrayList(u32),
    bpe_pair_cache: std.AutoHashMap(u64, BPEMerge),
    /// Statistics of the most recent root mining pass. `trainBPE` fills it and
    /// the offline vocabulary trainer reports it, so the share of the
    /// vocabulary that the morphological layer actually owns is observable.
    root_stats: RootMiningStats,
    /// Minimum corpus frequency a stem needs before it is mined.
    root_min_frequency: u64,
    /// Divisor that bounds the root share of the free vocabulary capacity.
    root_capacity_divisor: usize,
    /// Criterion the miner applies before it accepts a stem.
    root_evidence: RootEvidence,

    pub const Language = enum {
        english,
        hungarian,
    };

    const BPEMerge = struct {
        token_id: u32,
        priority: u32,
    };

    /// Affix split of one word: the encoder emits `[prefix][root][suffix]` from
    /// exactly these offsets, and root mining registers `root` as a vocabulary
    /// token so the split stays available at encode time.
    const AffixSplit = struct {
        prefix_len: usize,
        suffix_len: usize,
        root_start: usize,
        root_end: usize,
    };

    /// Result of one root mining pass.
    pub const RootMiningStats = struct {
        /// Number of word occurrences the pass scanned.
        words: u64 = 0,
        /// Number of distinct stems the affix split produced.
        candidate_stems: usize = 0,
        /// Number of stems this pass added to the vocabulary.
        added_roots: usize = 0,
        /// Frequency a stem has to reach before it is mined.
        minimum_frequency: u64 = 0,
        /// Criterion the pass applied. Under `auto` this is the rule that
        /// actually produced the root set, not the configured policy.
        evidence: RootEvidence = .auto,
        /// True when `auto` had to fall back from `paradigm` to `frequency`.
        fell_back: bool = false,
    };

    /// Default share of the remaining capacity the morphological layer may
    /// claim before byte-pair merges take the rest. Eight means at most one
    /// eighth of the free slots become roots, so training always has room for
    /// merges.
    pub const default_root_capacity_divisor: usize = 8;

    /// Default frequency a stem has to reach before it is mined.
    pub const default_root_min_frequency: u64 = 2;

    /// Shortest stem the miner accepts. A two-byte stem is almost always the
    /// residue of an over-eager affix strip, and the tokens it produces are
    /// shorter than the affixes that created them.
    pub const minimum_stem_bytes: usize = 3;

    /// Number of distinct affixations the miner remembers per stem. Two are
    /// enough to prove that a stem is productive; a small bound keeps the
    /// evidence table proportional to the number of distinct stems.
    pub const maximum_affixations: usize = 4;

    /// Marker for "this word carries no affix on this side".
    const absent_affix: u32 = std.math.maxInt(u32);

    /// Criterion a stem has to satisfy before it becomes a root.
    pub const RootEvidence = enum {
        /// A stem is mined when the corpus shows it in more than one shape:
        /// it also occurs as a whole word, or it appears behind at least two
        /// distinct prefix/suffix combinations. This is the default because it
        /// never invents a stem out of a single truncated word.
        paradigm,
        /// A stem is mined as soon as it reaches the minimum frequency. This is
        /// the right choice for corpora that repeat a fixed phrase list instead
        /// of natural text, where no stem can ever collect a paradigm; it can
        /// produce truncations such as `surpr` from `surprise`, which is why it
        /// is opt-in.
        frequency,
        /// `paradigm` first, `frequency` only when the corpus proves no
        /// paradigm at all. This is the default: on natural text it behaves
        /// exactly like `paradigm`, and on degenerate corpora (a repeated
        /// phrase list, a keyword dump) it keeps the morphological layer alive
        /// instead of silently disabling it. The fallback is recorded in
        /// `RootMiningStats.fell_back`, so a run never silently changes rule.
        auto,
    };

    /// Root mining configuration.
    pub const RootMiningPolicy = struct {
        minimum_frequency: u64 = default_root_min_frequency,
        capacity_divisor: usize = default_root_capacity_divisor,
        evidence: RootEvidence = .auto,
    };

    /// Morphological evidence collected for one candidate stem.
    ///
    /// A stem is only mined when the corpus shows it in more than one shape:
    /// either it also occurs as a whole word, or it appears behind at least two
    /// distinct prefix/suffix combinations. Requiring a paradigm is what keeps
    /// the miner from inventing stems such as `surpr` out of a single word like
    /// `surprise`: a stem that exists in exactly one affixed form is far more
    /// likely to be a truncation than a real root, and the token it would
    /// produce encodes nothing the word itself does not already encode.
    const StemEvidence = struct {
        /// Occurrences of this stem inside affixed words.
        frequency: u64 = 0,
        /// The stem also occurs as a whole word.
        standalone: bool = false,
        /// The stem was seen inside at least one affixed word.
        from_affixed: bool = false,
        signatures: [maximum_affixations]u64 = [_]u64{0} ** maximum_affixations,
        signature_count: usize = 0,

        fn addSignature(self: *StemEvidence, signature: u64) void {
            for (self.signatures[0..self.signature_count]) |existing| {
                if (existing == signature) return;
            }
            if (self.signature_count == maximum_affixations) return;
            self.signatures[self.signature_count] = signature;
            self.signature_count += 1;
        }

        fn isProductive(self: *const StemEvidence) bool {
            return self.standalone or self.signature_count >= 2;
        }
    };

    fn packPair(first: u32, second: u32) u64 {
        return (@as(u64, first) << 32) | @as(u64, second);
    }

    const SPECIAL_TOKENS = struct {
        const PAD: u32 = 0;
        const UNK: u32 = 1;
        const BOS: u32 = 2;
        const EOS: u32 = 3;
    };

    pub fn init(allocator: Allocator, vocab: []const []const u8, anchors: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        if (max_vocab_size) |max| {
            if (max < 4) return error.VocabularyTooSmall;
        }

        var mgt = initEmpty(allocator, max_vocab_size, language);
        errdefer mgt.deinit();

        _ = try mgt.addToken("[PAD]");
        _ = try mgt.addToken("[UNK]");
        _ = try mgt.addToken("[BOS]");
        _ = try mgt.addToken("[EOS]");

        for (vocab) |word| {
            if (!mgt.canAddToken()) break;
            _ = try mgt.addToken(word);
        }

        try mgt.initMorphemes();

        for (anchors) |anch| {
            if (!mgt.canAddToken() and !mgt.token_to_id.contains(anch)) break;
            const tid = mgt.token_to_id.get(anch) orelse try mgt.addToken(anch);
            const key = mgt.id_to_token.get(tid) orelse return error.InvalidData;
            try mgt.anchors.put(key, @as(u64, tid));
        }

        return mgt;
    }

    pub fn initWithArena(arena: *core_memory.ArenaAllocator, vocab: []const []const u8, anchors_list: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        return init(arena.allocator(), vocab, anchors_list, max_vocab_size, language);
    }

    pub fn initWithPool(pool: *core_memory.PoolAllocator, vocab: []const []const u8, anchors_list: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        return init(pool.allocator(), vocab, anchors_list, max_vocab_size, language);
    }

    pub fn initWithBuddy(buddy: *core_memory.BuddyAllocator, vocab: []const []const u8, anchors_list: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        return init(buddy.allocator(), vocab, anchors_list, max_vocab_size, language);
    }

    fn initEmpty(allocator: Allocator, max_vocab_size: ?u32, language: Language) MGT {
        return .{
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
            .prefixes = std.StringHashMap(u32).init(allocator),
            .suffixes = std.StringHashMap(u32).init(allocator),
            .roots = std.StringHashMap(u32).init(allocator),
            .bpe_pairs = std.AutoHashMap(u64, BPEMerge).init(allocator),
            .anchors = std.StringHashMap(u64).init(allocator),
            .allocated_strings = std.ArrayList([]u8).init(allocator),
            .allocator = allocator,
            .next_token_id = 0,
            .language = language,
            .max_vocab_size = max_vocab_size,
            .allow_dynamic_tokens = true,
            .sorted_prefix_keys = std.ArrayList([]const u8).init(allocator),
            .sorted_suffix_keys = std.ArrayList([]const u8).init(allocator),
            .max_prefix_len = 0,
            .max_suffix_len = 0,
            .max_token_len = 0,
            .byte_token_ids = [_]?u32{null} ** 256,
            .byte_token_values = std.AutoHashMap(u32, u8).init(allocator),
            .bpe_work_tokens = std.ArrayList(u32).init(allocator),
            .bpe_output_tokens = std.ArrayList(u32).init(allocator),
            .bpe_pair_cache = std.AutoHashMap(u64, BPEMerge).init(allocator),
            .root_stats = .{},
            .root_min_frequency = default_root_min_frequency,
            .root_capacity_divisor = default_root_capacity_divisor,
            .root_evidence = .auto,
        };
    }

    fn canAddToken(self: *const MGT) bool {
        if (self.next_token_id == std.math.maxInt(u32)) return false;
        if (self.max_vocab_size) |max| {
            return self.token_to_id.count() < @as(usize, max);
        }
        return true;
    }

    fn reset(self: *MGT) void {
        const allocator = self.allocator;
        const mvs = self.max_vocab_size;
        const lang = self.language;
        self.deinit();
        self.* = initEmpty(allocator, mvs, lang);
    }

    fn initMorphemes(self: *MGT) !void {
        const english_prefix_list = [_][]const u8{
            "un",  "re",    "pre",  "dis",  "mis",   "over",  "under", "out",
            "sub", "inter", "fore", "de",   "trans", "super", "semi",  "anti",
            "mid", "non",   "ex",   "post", "pro",   "co",    "en",    "em",
        };

        const hungarian_prefix_list = [_][]const u8{
            "meg",  "el",     "fel",  "le",     "be",     "ki",  "rá",      "át", "szét",  "vissza",
            "ide",  "oda",    "alá", "fölé", "közé", "egy", "össze",   "tul", "hozzá", "körül",
            "alig", "éppen", "majd", "csak",   "is",     "leg", "legesleg",
        };

        const english_suffix_list = [_][]const u8{
            "ing",  "ed",  "er",   "est", "ly",  "tion", "sion", "ness",
            "ment", "ful", "less", "ous", "ive", "able", "ible", "al",
            "ial",  "y",   "s",    "es",  "en",  "ize",  "ise",  "ate",
        };

        const hungarian_suffix_list = [_][]const u8{
            "ság",  "ség",  "ságú", "ségű", "é",    "je",     "ja",   "ban",   "ben",
            "ba",    "be",    "ból",   "ből",   "hoz",   "hez",    "höz", "tól",  "től",
            "nak",   "nek",   "val",    "vel",    "ért",  "ul",     "ül",  "ként", "án",
            "én",   "ig",    "at",     "et",     "tat",   "tet",    "ott",  "ett",   "atlan",
            "etlen", "talan", "telen",  "ál",    "él",   "oz",     "ez",   "öd",   "ed",
            "gyet",  "get",   "j",      "unk",    "jatok", "játok", "i",    "ni",    "nként",
            "kor",   "ra",    "re",
        };

        const prefix_list: []const []const u8 = switch (self.language) {
            .english => english_prefix_list[0..],
            .hungarian => hungarian_prefix_list[0..],
        };

        const suffix_list: []const []const u8 = switch (self.language) {
            .english => english_suffix_list[0..],
            .hungarian => hungarian_suffix_list[0..],
        };

        for (prefix_list) |prefix| {
            if (!self.canAddToken() and !self.token_to_id.contains(prefix)) break;
            const id = self.token_to_id.get(prefix) orelse try self.addToken(prefix);
            const key = self.id_to_token.get(id) orelse return error.InvalidData;
            try self.prefixes.put(key, id);
        }

        for (suffix_list) |suffix| {
            if (!self.canAddToken() and !self.token_to_id.contains(suffix)) break;
            const id = self.token_to_id.get(suffix) orelse try self.addToken(suffix);
            const key = self.id_to_token.get(id) orelse return error.InvalidData;
            try self.suffixes.put(key, id);
        }

        try self.rebuildSortedMorphemes();
    }

    fn rebuildSortedMorphemes(self: *MGT) !void {
        self.sorted_prefix_keys.clearRetainingCapacity();
        var pit = self.prefixes.iterator();
        while (pit.next()) |entry| {
            try self.sorted_prefix_keys.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, self.sorted_prefix_keys.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);

        self.max_prefix_len = 0;
        for (self.sorted_prefix_keys.items) |key| {
            self.max_prefix_len = @max(self.max_prefix_len, key.len);
        }

        self.sorted_suffix_keys.clearRetainingCapacity();
        var sit = self.suffixes.iterator();
        while (sit.next()) |entry| {
            try self.sorted_suffix_keys.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, self.sorted_suffix_keys.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);

        self.max_suffix_len = 0;
        for (self.sorted_suffix_keys.items) |key| {
            self.max_suffix_len = @max(self.max_suffix_len, key.len);
        }
    }

    pub fn deinit(self: *MGT) void {
        self.token_to_id.deinit();
        self.id_to_token.deinit();
        self.prefixes.deinit();
        self.suffixes.deinit();
        self.roots.deinit();
        self.bpe_pairs.deinit();
        self.anchors.deinit();
        self.sorted_prefix_keys.deinit();
        self.sorted_suffix_keys.deinit();
        self.byte_token_values.deinit();
        self.bpe_work_tokens.deinit();
        self.bpe_output_tokens.deinit();
        self.bpe_pair_cache.deinit();

        for (self.allocated_strings.items) |str| {
            self.allocator.free(str);
        }

        self.allocated_strings.deinit();
    }

    fn isWhitespace(c: u8) bool {
        return c == ' ' or c == '\n' or c == '\t' or c == '\r';
    }

    fn isPunctuation(c: u8) bool {
        return c == '.' or c == ',' or c == '!' or c == '?' or c == ';' or
            c == ':' or c == '"' or c == '\'' or c == '(' or c == ')' or
            c == '{' or c == '}';
    }

    fn isKnownSpecialTokenStart(self: *const MGT, text: []const u8, pos: usize) bool {
        if (pos >= text.len or text[pos] != '[') return false;

        const specials = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };
        for (specials) |special| {
            if (special.len <= text.len - pos and
                mem.eql(u8, text[pos .. pos + special.len], special) and
                self.token_to_id.contains(special))
            {
                return true;
            }
        }

        return false;
    }

    fn getKnownSpecialTokenLen(self: *const MGT, text: []const u8, pos: usize) ?usize {
        if (pos >= text.len or text[pos] != '[') return null;

        const specials = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };
        for (specials) |special| {
            if (special.len <= text.len - pos and
                mem.eql(u8, text[pos .. pos + special.len], special) and
                self.token_to_id.contains(special))
            {
                return special.len;
            }
        }

        return null;
    }

    fn utf8CharLen(first_byte: u8) u8 {
        if (first_byte & 0x80 == 0) return 1;
        if (first_byte & 0xE0 == 0xC0) return 2;
        if (first_byte & 0xF0 == 0xE0) return 3;
        if (first_byte & 0xF8 == 0xF0) return 4;
        return 1;
    }

    fn safeUtf8SequenceLenAt(text: []const u8, pos: usize) usize {
        if (pos >= text.len) return 0;

        const sequence_len: usize = utf8CharLen(text[pos]);
        if (sequence_len > text.len - pos) return 1;
        if (sequence_len == 1) return 1;

        var i: usize = 1;
        while (i < sequence_len) : (i += 1) {
            if ((text[pos + i] & 0xC0) != 0x80) return 1;
        }

        return sequence_len;
    }

    fn truncateUtf8Input(text: []const u8, maximum_bytes: usize) []const u8 {
        const length = @min(text.len, maximum_bytes);
        var boundary = length;
        while (boundary > 0 and boundary < text.len and (text[boundary] & 0xC0) == 0x80) {
            boundary -= 1;
        }
        return text[0..boundary];
    }

    /// Span of one word inside an input text.
    const WordSpan = struct {
        start: usize,
        end: usize,
    };

    /// Returns the next `[start, end)` span the encoder treats as a word and
    /// advances `position` past it.
    ///
    /// Whitespace, punctuation and known special tokens separate words, and the
    /// same UTF-8 validation the encoder applies is used here. Root mining walks
    /// the corpus with this function, which is what makes the mined stems agree
    /// with the fragments the encoder looks up at training time.
    fn nextWordSpan(self: *const MGT, text: []const u8, position: *usize) ?WordSpan {
        var cursor = position.*;

        while (cursor < text.len) {
            if (self.getKnownSpecialTokenLen(text, cursor)) |special_len| {
                cursor += special_len;
                continue;
            }

            const c = text[cursor];
            if (isWhitespace(c) or isPunctuation(c)) {
                cursor += 1;
                continue;
            }

            break;
        }

        if (cursor >= text.len) {
            position.* = text.len;
            return null;
        }

        const start = cursor;
        var end = cursor;

        while (end < text.len) {
            if (self.isKnownSpecialTokenStart(text, end)) break;

            const c = text[end];
            if (isWhitespace(c) or isPunctuation(c)) break;

            const char_len = safeUtf8SequenceLenAt(text, end);
            if (char_len == 0 or char_len > text.len - end) break;
            end += char_len;
        }

        if (end == start) {
            position.* = text.len;
            return null;
        }

        position.* = end;
        return WordSpan{ .start = start, .end = end };
    }

    fn emitToken(self: *const MGT, tid: u32, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        try out_tokens.append(tid);

        if (out_anchors) |anchors_out| {
            if (self.id_to_token.get(tid)) |token_str| {
                if (self.anchors.contains(token_str)) {
                    try anchors_out.append(byte_pos);
                }
            }
        }
    }

    fn appendUnknownForSlice(self: *const MGT, slice: []const u8, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        const tid = self.unknownReplacement(slice);
        try self.emitToken(tid, byte_pos, out_tokens, out_anchors);
    }

    fn appendBPEOrUnknown(self: *MGT, slice: []const u8, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        if (slice.len == 0) return;

        try self.encodeBPEInto(slice, &self.bpe_output_tokens);

        if (self.bpe_output_tokens.items.len == 0) {
            try self.appendUnknownForSlice(slice, byte_pos, out_tokens, out_anchors);
            return;
        }

        for (self.bpe_output_tokens.items) |tid| {
            try self.emitToken(tid, byte_pos, out_tokens, out_anchors);
        }
    }

    fn encodeInternal(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        var i: usize = 0;

        while (i < text.len) {
            if (self.getKnownSpecialTokenLen(text, i)) |special_len| {
                const special_token = text[i .. i + special_len];
                const tid = self.token_to_id.get(special_token) orelse return error.InvalidData;
                try self.emitToken(tid, i, out_tokens, out_anchors);
                i += special_len;
                continue;
            }

            if (isWhitespace(text[i])) {
                const whitespace = text[i .. i + 1];

                // Whitespace falls back to the byte-pair encoder so that a
                // space, tab or newline is represented by its byte token when
                // the vocabulary has no dedicated token for it. Emitting the
                // unknown token here would collapse every separator in the
                // corpus into a single symbol and destroy the information the
                // model needs to learn word boundaries.
                if (self.token_to_id.get(whitespace)) |tid| {
                    try self.emitToken(tid, i, out_tokens, out_anchors);
                } else {
                    try self.appendBPEOrUnknown(whitespace, i, out_tokens, out_anchors);
                }

                i += 1;
                continue;
            }

            if (isPunctuation(text[i])) {
                const punctuation = text[i .. i + 1];

                if (self.token_to_id.get(punctuation)) |tid| {
                    try self.emitToken(tid, i, out_tokens, out_anchors);
                } else {
                    try self.appendBPEOrUnknown(punctuation, i, out_tokens, out_anchors);
                }

                i += 1;
                continue;
            }

            var word_end = i;
            while (word_end < text.len) {
                if (self.isKnownSpecialTokenStart(text, word_end)) break;

                const c = text[word_end];
                if (isWhitespace(c) or isPunctuation(c)) break;

                const char_len = safeUtf8SequenceLenAt(text, word_end);
                if (char_len == 0) break;
                word_end += char_len;
            }

            if (word_end == i) {
                const char_len = safeUtf8SequenceLenAt(text, i);
                if (char_len == 0 or char_len > text.len - i) return error.InvalidData;
                try self.appendBPEOrUnknown(text[i .. i + char_len], i, out_tokens, out_anchors);
                i += char_len;
                continue;
            }

            const word = text[i..word_end];

            if (self.token_to_id.get(word)) |tid| {
                try self.emitToken(tid, i, out_tokens, out_anchors);
            } else if (!try self.morphDecompose(word, i, out_tokens, out_anchors)) {
                // Only words that cannot be explained by a known prefix, root
                // and suffix combination fall through to the subword layer.
                try self.subwordSplitInto(word, i, out_tokens, out_anchors);
            }

            i = word_end;
        }
    }

    pub fn encode(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32)) !void {
        try self.encodeInternal(text, out_tokens, null);
    }

    pub fn encodeBounded(
        self: *MGT,
        text: []const u8,
        max_tokens: usize,
        out_tokens: *std.ArrayList(u32),
    ) !void {
        if (max_tokens == 0) return;
        const max_bytes = std.math.mul(usize, max_tokens, 4) catch return error.InputTooLarge;
        const bounded = truncateUtf8Input(text, max_bytes);
        const original_len = out_tokens.items.len;
        try self.encodeInternal(bounded, out_tokens, null);
        if (out_tokens.items.len - original_len > max_tokens) {
            out_tokens.shrinkRetainingCapacity(original_len + max_tokens);
        }
    }

    fn binarySearchString(sorted: []const []const u8, target: []const u8) bool {
        var lo: usize = 0;
        var hi: usize = sorted.len;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;

            if (std.mem.lessThan(u8, sorted[mid], target)) {
                lo = mid + 1;
            } else if (std.mem.lessThan(u8, target, sorted[mid])) {
                hi = mid;
            } else {
                return true;
            }
        }

        return false;
    }

    fn findLongestPrefix(self: *MGT, word: []const u8) ?struct { prefix: []const u8, len: usize } {
        if (word.len < 2) return null;

        var test_len = @min(self.max_prefix_len, word.len - 1);
        while (test_len > 0) : (test_len -= 1) {
            const candidate = word[0..test_len];

            if (self.prefixes.get(candidate)) |id| {
                const key = self.id_to_token.get(id) orelse return null;
                return .{ .prefix = key, .len = test_len };
            }
        }

        return null;
    }

    fn findLongestSuffix(self: *MGT, word: []const u8) ?struct { suffix: []const u8, len: usize } {
        if (word.len < 2) return null;

        var test_len = @min(self.max_suffix_len, word.len - 1);
        while (test_len > 0) : (test_len -= 1) {
            const candidate = word[word.len - test_len ..];

            if (self.suffixes.get(candidate)) |id| {
                const key = self.id_to_token.get(id) orelse return null;
                return .{ .suffix = key, .len = test_len };
            }
        }

        return null;
    }

    /// Splits the longest known affixes off `word`.
    ///
    /// The split is the single definition of a morphological decomposition in
    /// this tokenizer: the encoder and the root miner both use it, so a stem
    /// that is mined from the corpus is exactly the stem the encoder will look
    /// up later. A word without a known prefix or suffix, a word shorter than
    /// four bytes, or a word whose stem would be shorter than two bytes has no
    /// split.
    fn splitAffixes(self: *MGT, word: []const u8) ?AffixSplit {
        if (word.len < 4) return null;

        const prefix_result = self.findLongestPrefix(word);
        const suffix_result = self.findLongestSuffix(word);

        const prefix_len = if (prefix_result) |prefix| prefix.len else 0;
        const suffix_len = if (suffix_result) |suffix| suffix.len else 0;

        if (prefix_len == 0 and suffix_len == 0) return null;
        if (prefix_len > word.len or suffix_len > word.len - prefix_len) return null;

        const root_start = prefix_len;
        const root_end = word.len - suffix_len;

        if (root_end <= root_start or root_end - root_start < 2) return null;

        return AffixSplit{
            .prefix_len = prefix_len,
            .suffix_len = suffix_len,
            .root_start = root_start,
            .root_end = root_end,
        };
    }

    fn morphDecompose(self: *MGT, word: []const u8, word_start: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !bool {
        const split = self.splitAffixes(word) orelse return false;

        // Every part is resolved before the first token is emitted, so a word
        // is either decomposed completely or not at all. A partial emission
        // would desynchronise the token stream from the input text.
        const root = word[split.root_start..split.root_end];
        const root_tid = self.token_to_id.get(root) orelse return false;

        var prefix_tid: ?u32 = null;
        if (split.prefix_len > 0) {
            prefix_tid = self.token_to_id.get(word[0..split.prefix_len]) orelse return false;
        }

        var suffix_tid: ?u32 = null;
        if (split.suffix_len > 0) {
            suffix_tid = self.token_to_id.get(word[word.len - split.suffix_len ..]) orelse return false;
        }

        if (prefix_tid) |tid| {
            try self.emitToken(tid, word_start, out_tokens, out_anchors);
        }

        try self.emitToken(root_tid, word_start + split.root_start, out_tokens, out_anchors);

        if (suffix_tid) |tid| {
            try self.emitToken(tid, word_start + split.root_start + root.len, out_tokens, out_anchors);
        }

        return true;
    }

    fn addByteToken(self: *MGT, byte: u8) !u32 {
        if (self.byte_token_ids[byte]) |existing| return existing;
        if (!self.allow_dynamic_tokens) return SPECIAL_TOKENS.UNK;

        var buffer: [16]u8 = undefined;
        const byte_string = try std.fmt.bufPrint(&buffer, "<{x:0>2}>", .{byte});
        const id = try self.addToken(byte_string);

        try self.byte_token_values.put(id, byte);
        self.byte_token_ids[byte] = id;

        return id;
    }

    fn addToken(self: *MGT, token: []const u8) !u32 {
        if (self.token_to_id.get(token)) |existing| {
            return existing;
        }

        if (!self.canAddToken()) {
            return error.VocabularyFull;
        }

        const id = self.next_token_id;
        const next_id = std.math.add(u32, id, 1) catch return error.TokenIdOverflow;
        const token_copy = try self.allocator.dupe(u8, token);
        errdefer self.allocator.free(token_copy);

        try self.token_to_id.put(token_copy, id);
        errdefer _ = self.token_to_id.remove(token_copy);

        try self.id_to_token.put(id, token_copy);
        errdefer _ = self.id_to_token.remove(id);

        try self.allocated_strings.append(token_copy);
        self.next_token_id = next_id;
        self.max_token_len = @max(self.max_token_len, token_copy.len);

        return id;
    }

    fn adoptTokenWithId(self: *MGT, token: []u8, id: u32) !void {
        if (id == std.math.maxInt(u32)) return error.TokenIdOverflow;
        if (self.token_to_id.contains(token) or self.id_to_token.contains(id)) {
            return error.InvalidData;
        }

        errdefer self.allocator.free(token);

        try self.token_to_id.put(token, id);
        errdefer _ = self.token_to_id.remove(token);

        try self.id_to_token.put(id, token);
        errdefer _ = self.id_to_token.remove(id);

        try self.allocated_strings.append(token);

        if (id >= self.next_token_id) {
            self.next_token_id = std.math.add(u32, id, 1) catch return error.TokenIdOverflow;
        }
        self.max_token_len = @max(self.max_token_len, token.len);
    }

    fn getCanonicalTokenForLoad(self: *MGT, raw: []u8, id: u32) ![]const u8 {
        if (self.id_to_token.get(id)) |canonical| {
            if (!mem.eql(u8, canonical, raw)) return error.InvalidData;
            self.allocator.free(raw);
            return canonical;
        }

        if (self.token_to_id.get(raw)) |existing_id| {
            if (existing_id != id) return error.InvalidData;
            const canonical = self.id_to_token.get(existing_id) orelse return error.InvalidData;
            self.allocator.free(raw);
            return canonical;
        }

        try self.adoptTokenWithId(raw, id);
        return self.id_to_token.get(id) orelse return error.InvalidData;
    }

    /// Encodes one fragment with the trained byte-pair merges and appends the
    /// resulting token ids to `out`. The working structures live on the
    /// tokenizer, so steady-state encoding performs no heap allocation and the
    /// merge cache is reused for the whole fragment instead of one hash map per
    /// merge pass.
    fn encodeBPEInto(self: *MGT, text: []const u8, out: *std.ArrayList(u32)) !void {
        out.clearRetainingCapacity();
        if (text.len == 0) return;

        const current = &self.bpe_work_tokens;
        current.clearRetainingCapacity();
        try current.ensureTotalCapacity(text.len);

        for (text) |byte| {
            const tid = self.byte_token_ids[byte] orelse try self.addByteToken(byte);
            current.appendAssumeCapacity(tid);
        }

        const cache = &self.bpe_pair_cache;
        cache.clearRetainingCapacity();

        const no_merge = BPEMerge{
            .token_id = std.math.maxInt(u32),
            .priority = std.math.maxInt(u32),
        };

        while (current.items.len > 1) {
            var best_priority: u32 = std.math.maxInt(u32);
            var best_left: u32 = 0;
            var best_right: u32 = 0;
            var best_merge_id: u32 = 0;
            var found = false;

            var i: usize = 0;
            while (i < current.items.len - 1) : (i += 1) {
                const left_id = current.items[i];
                const right_id = current.items[i + 1];
                const key = packPair(left_id, right_id);

                const merge = cache.getPtr(key) orelse blk: {
                    const entry = try cache.getOrPut(key);
                    if (!entry.found_existing) {
                        entry.value_ptr.* = self.bpe_pairs.get(key) orelse no_merge;
                    }
                    break :blk entry.value_ptr;
                };

                if (merge.priority >= best_priority) continue;
                if (merge.token_id == std.math.maxInt(u32)) continue;

                best_priority = merge.priority;
                best_left = left_id;
                best_right = right_id;
                best_merge_id = merge.token_id;
                found = true;
            }

            if (!found) break;

            var write: usize = 0;
            var read: usize = 0;

            while (read < current.items.len) {
                if (read < current.items.len - 1 and
                    current.items[read] == best_left and
                    current.items[read + 1] == best_right)
                {
                    current.items[write] = best_merge_id;
                    write += 1;
                    read += 2;
                } else {
                    if (write != read) {
                        current.items[write] = current.items[read];
                    }
                    write += 1;
                    read += 1;
                }
            }

            current.shrinkRetainingCapacity(write);
        }

        try out.appendSlice(current.items);
    }

    /// Returns an owned token slice for one fragment.
    fn encodeBPE(self: *MGT, text: []const u8) ![]u32 {
        var owned = std.ArrayList(u32).init(self.allocator);
        errdefer owned.deinit();
        try owned.ensureTotalCapacity(text.len);
        try self.encodeBPEInto(text, &owned);
        return owned.toOwnedSlice();
    }

    fn validateAllocationCount(comptime T: type, count: usize) !void {
        const bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.InputTooLarge;
        _ = std.math.add(usize, bytes, 65535) catch return error.InputTooLarge;
    }

    fn validateCorpus(corpus: []const []const u8) !void {
        var total_tokens: usize = 0;
        for (corpus) |text| {
            try validateAllocationCount(u32, text.len);
            total_tokens = std.math.add(usize, total_tokens, text.len) catch return error.CorpusTooLarge;
        }
        try validateAllocationCount(u32, total_tokens);
    }

    /// Returns true when `text` has the exact shape of a byte-token label.
    fn isByteLabelText(text: []const u8) bool {
        if (text.len != 4) return false;
        if (text[0] != '<' or text[3] != '>') return false;
        _ = std.fmt.parseInt(u8, text[1..3], 16) catch return false;
        return true;
    }

    /// Appends the raw text a token stands for.
    ///
    /// Byte tokens are stored as `<xx>` labels, so a byte-pair merge must be
    /// materialised from the bytes those labels represent. Concatenating the
    /// labels instead would register tokens whose text is `<74><74>` rather
    /// than `tt`, which would then decode to the label instead of the bytes.
    fn appendSurfaceText(self: *const MGT, token_id: u32, out: *std.ArrayList(u8)) !void {
        if (self.byte_token_values.get(token_id)) |byte| {
            try out.append(byte);
            return;
        }
        const token_string = self.id_to_token.get(token_id) orelse return error.InvalidData;
        try out.appendSlice(token_string);
    }

    fn initialBpePriority(self: *const MGT) !u32 {
        var next_priority: u32 = 0;
        var iterator = self.bpe_pairs.iterator();

        while (iterator.next()) |entry| {
            const priority = entry.value_ptr.priority;
            if (priority >= next_priority) {
                next_priority = std.math.add(u32, priority, 1) catch return error.PriorityOverflow;
            }
        }

        return next_priority;
    }

    /// Mines the stems of the corpus and registers them as morphological roots.
    ///
    /// The pass counts every word, splits it with the same affix inventory the
    /// encoder uses and accumulates the frequency of the stem that remains.
    /// Stems that occur at least `minimum_frequency` times are then registered
    /// in frequency order (ties broken by the stem text, so the outcome is
    /// independent of hash-map iteration order). A stem that is already a
    /// vocabulary token becomes a root immediately; other stems are added as
    /// tokens while `token_budget` and the vocabulary capacity allow it.
    ///
    /// Every registered stem is also an anchor, which is what makes the
    /// morphological layer observable outside this file: the anchor set is
    /// saved with the vocabulary, can be exported as a token list and marks the
    /// token positions the encoder reports through `tokenizeWithAnchors`.
    pub fn mineRoots(self: *MGT, corpus: []const []const u8, token_budget: usize, minimum_frequency: u64) !RootMiningStats {
        if (self.root_evidence != .auto) {
            return self.mineRootsWithEvidence(corpus, token_budget, minimum_frequency, self.root_evidence);
        }

        // `auto` first asks the strict rule. A corpus that proves a paradigm is
        // the common case on natural text and the strict result is kept as is.
        const strict = try self.mineRootsWithEvidence(corpus, token_budget, minimum_frequency, .paradigm);
        if (strict.candidate_stems > 0) return strict;

        // No stem could be proven productive, so the strict rule has no signal
        // to work with and the lenient rule decides. The fallback is recorded
        // rather than hidden: callers report it, and a run that used it is
        // distinguishable from a run that simply found no stems at all.
        const lenient = try self.mineRootsWithEvidence(corpus, token_budget, minimum_frequency, .frequency);
        var stats = lenient;
        stats.fell_back = true;
        self.root_stats = stats;
        return stats;
    }

    /// Runs one mining pass with an explicit evidence rule.
    fn mineRootsWithEvidence(
        self: *MGT,
        corpus: []const []const u8,
        token_budget: usize,
        minimum_frequency: u64,
        evidence: RootEvidence,
    ) !RootMiningStats {
        var stats = RootMiningStats{
            .minimum_frequency = minimum_frequency,
            .evidence = evidence,
        };
        if (minimum_frequency == 0 or corpus.len == 0) {
            self.root_stats = stats;
            return stats;
        }

        // The frequency tables are transient: they are released before the
        // merge trainer starts, so the transient allocator returns their pages
        // instead of holding them for the rest of the run.
        const allocator = transientAllocator();

        var word_frequencies = std.StringHashMap(u64).init(allocator);
        defer word_frequencies.deinit();

        for (corpus) |text| {
            var cursor: usize = 0;
            while (self.nextWordSpan(text, &cursor)) |span| {
                stats.words = std.math.add(u64, stats.words, 1) catch return error.FrequencyOverflow;

                const entry = try word_frequencies.getOrPut(text[span.start..span.end]);
                if (!entry.found_existing) entry.value_ptr.* = 0;
                entry.value_ptr.* = std.math.add(u64, entry.value_ptr.*, 1) catch return error.FrequencyOverflow;
            }
        }

        // One pass over the distinct words collects the paradigm evidence of
        // every candidate stem. Whole words that carry no affix are only
        // recorded as supporting evidence; a stem is mined from the affixed
        // words that contain it.
        var stem_evidence_table = std.StringHashMap(StemEvidence).init(allocator);
        defer stem_evidence_table.deinit();

        var word_iterator = word_frequencies.iterator();
        while (word_iterator.next()) |entry| {
            const word = entry.key_ptr.*;
            const frequency = entry.value_ptr.*;
            if (frequency < minimum_frequency) continue;

            if (self.splitAffixes(word)) |split| {
                const stem = word[split.root_start..split.root_end];
                if (stem.len < minimum_stem_bytes) continue;

                const prefix_id: u32 = if (split.prefix_len == 0)
                    absent_affix
                else
                    self.prefixes.get(word[0..split.prefix_len]) orelse absent_affix;
                const suffix_id: u32 = if (split.suffix_len == 0)
                    absent_affix
                else
                    self.suffixes.get(word[word.len - split.suffix_len ..]) orelse absent_affix;

                const stem_entry = try stem_evidence_table.getOrPut(stem);
                if (!stem_entry.found_existing) stem_entry.value_ptr.* = .{};
                stem_entry.value_ptr.frequency = std.math.add(u64, stem_entry.value_ptr.frequency, frequency) catch return error.FrequencyOverflow;
                stem_entry.value_ptr.from_affixed = true;
                stem_entry.value_ptr.addSignature((@as(u64, prefix_id) << 32) | @as(u64, suffix_id));
                continue;
            }

            // A word with no known affix at all proves that its text is a word
            // in its own right, which is exactly the evidence a stem needs to
            // be considered productive.
            if (self.findLongestPrefix(word) == null and self.findLongestSuffix(word) == null) {
                const stem_entry = try stem_evidence_table.getOrPut(word);
                if (!stem_entry.found_existing) stem_entry.value_ptr.* = .{};
                stem_entry.value_ptr.standalone = true;
            }
        }

        const Candidate = struct {
            stem: []const u8,
            frequency: u64,
        };

        var candidates = std.ArrayList(Candidate).init(allocator);
        defer candidates.deinit();

        var evidence_iterator = stem_evidence_table.iterator();
        while (evidence_iterator.next()) |entry| {
            const stem_evidence = entry.value_ptr.*;
            if (!stem_evidence.from_affixed) continue;
            if (stem_evidence.frequency < minimum_frequency) continue;
            if (entry.key_ptr.*.len < minimum_stem_bytes) continue;
            if (evidence == .paradigm and !stem_evidence.isProductive()) continue;

            try candidates.append(.{
                .stem = entry.key_ptr.*,
                .frequency = stem_evidence.frequency,
            });
        }

        stats.candidate_stems = candidates.items.len;

        // Descending frequency, and the smaller stem text first when two stems
        // are equally frequent. The comparator never depends on the hash-map
        // layout, so the mined root set is a pure function of the corpus.
        std.mem.sort(Candidate, candidates.items, {}, struct {
            fn lessThan(_: void, a: Candidate, b: Candidate) bool {
                if (a.frequency != b.frequency) return a.frequency > b.frequency;
                return std.mem.lessThan(u8, a.stem, b.stem);
            }
        }.lessThan);

        for (candidates.items) |candidate| {
            var token_id: u32 = undefined;

            if (self.token_to_id.get(candidate.stem)) |existing_id| {
                if (self.roots.contains(candidate.stem)) continue;
                token_id = existing_id;
            } else {
                if (stats.added_roots >= token_budget) continue;
                if (!self.canAddToken()) continue;

                token_id = self.addToken(candidate.stem) catch |err| switch (err) {
                    error.VocabularyFull => continue,
                    else => return err,
                };
                stats.added_roots += 1;
            }

            const key = self.id_to_token.get(token_id) orelse return error.InvalidData;
            try self.roots.put(key, token_id);
            try self.anchors.put(key, @as(u64, token_id));
        }

        self.root_stats = stats;
        return stats;
    }

    /// Returns the allocator used for the large, short-lived structures of
    /// training and mining.
    ///
    /// They are allocated and released millions of times during a run. The C
    /// allocator returns freed pages to the operating system, which keeps
    /// resident memory flat at roughly the size of the corpus representation;
    /// the page allocator retains the freed mappings and grows resident memory
    /// by gigabytes on the same work. Binaries that use this trainer link libc,
    /// and the page allocator remains the fallback for any libc-free consumer.
    fn transientAllocator() Allocator {
        return if (@import("builtin").link_libc)
            std.heap.c_allocator
        else
            std.heap.page_allocator;
    }

    const no_node: u32 = std.math.maxInt(u32);

    /// One token position inside the linked representation of the training
    /// corpus. Merges splice nodes out of their sequence instead of shifting
    /// the surrounding bytes, which keeps every merge proportional to the
    /// number of occurrences it actually rewrites.
    const BpeNode = struct {
        token: u32,
        prev: u32,
        next: u32,
    };

    /// A candidate merge in the frequency queue.
    const BpeCandidate = struct {
        frequency: u64,
        key: u64,
    };

    /// `std.PriorityQueue` pops the element that compares as `.lt` against the
    /// rest, so a candidate compares as "less" when it must be merged earlier:
    /// higher frequency first, and the smallest packed pair key to break ties.
    const BpeCandidateOrder = struct {
        fn lessThan(_: BpeCandidateOrder, a: BpeCandidate, b: BpeCandidate) std.math.Order {
            if (a.frequency != b.frequency) return std.math.order(b.frequency, a.frequency);
            if (a.key != b.key) return std.math.order(a.key, b.key);
            return .eq;
        }
    };

    const BpeQueue = std.PriorityQueue(BpeCandidate, BpeCandidateOrder, BpeCandidateOrder.lessThan);

    /// Builds the complete vocabulary on `corpus`.
    ///
    /// The order is fixed and deterministic: the four special tokens and the
    /// morphemes are already present when this is called, every byte receives a
    /// dedicated token, the morphological layer mines the stems of the corpus
    /// (`mineRoots`, bounded to one eighth of the free capacity), and byte-pair
    /// merges fill the remaining slots up to `target_vocab_size`.
    ///
    /// Pair frequencies are maintained incrementally: every merge only updates
    /// the pair counts that touch the rewritten positions, so a full training
    /// pass costs O(corpus) instead of rescanning the whole corpus for each one
    /// of the tens of thousands of merges.
    pub fn trainBPE(self: *MGT, corpus: []const []const u8, target_vocab_size: u32) !void {
        const configured_target = if (self.max_vocab_size) |limit|
            @min(target_vocab_size, limit)
        else
            target_vocab_size;

        if (configured_target <= SPECIAL_TOKENS.EOS) return;

        try validateCorpus(corpus);

        // Every byte gets a dedicated token up front so the resulting
        // vocabulary can encode arbitrary UTF-8 input without falling back to
        // the unknown token. The set of byte tokens is therefore identical for
        // every corpus, which keeps vocabulary construction deterministic.
        var byte_index: u32 = 0;
        while (byte_index < 256) : (byte_index += 1) {
            if (self.canAddToken() == false and self.byte_token_ids[byte_index] == null) break;
            _ = self.addByteToken(@intCast(byte_index)) catch |err| switch (err) {
                error.VocabularyFull => break,
                else => return err,
            };
        }

        if (self.vocabSize() >= @as(usize, configured_target)) return;

        // The morphological layer reserves a bounded share of the remaining
        // capacity for the stems of the corpus. Without this pass the affix
        // inventory would only ever match words whose stem happened to be a
        // byte-pair merge, and the decomposition branch would stay unused on
        // real text.
        const remaining_capacity = @as(usize, configured_target) - self.vocabSize();
        const divisor = @max(self.root_capacity_divisor, 1);
        const root_budget = remaining_capacity / divisor;
        if (root_budget > 0 and self.root_min_frequency > 0) {
            _ = try self.mineRoots(corpus, root_budget, self.root_min_frequency);
        }

        if (self.vocabSize() >= @as(usize, configured_target)) return;

        const allocator = transientAllocator();

        var nodes = std.ArrayList(BpeNode).init(allocator);
        defer nodes.deinit();

        var total_bytes: usize = 0;
        for (corpus) |text| {
            total_bytes = std.math.add(usize, total_bytes, text.len) catch return error.CorpusTooLarge;
        }
        try validateAllocationCount(BpeNode, total_bytes);
        try nodes.ensureTotalCapacity(total_bytes);

        for (corpus) |text| {
            if (text.len == 0) continue;

            const sequence_start = nodes.items.len;
            for (text, 0..) |byte, index| {
                const token = self.byte_token_ids[byte] orelse break;
                nodes.appendAssumeCapacity(.{
                    .token = token,
                    .prev = if (index == 0) no_node else @intCast(sequence_start + index - 1),
                    .next = if (index + 1 == text.len) no_node else @intCast(sequence_start + index + 1),
                });
            }
        }

        if (nodes.items.len < 2) return;

        var pair_counts = std.AutoHashMap(u64, u64).init(allocator);
        defer pair_counts.deinit();

        var pair_positions = std.AutoHashMap(u64, std.ArrayList(u32)).init(allocator);
        defer {
            var position_iterator = pair_positions.valueIterator();
            while (position_iterator.next()) |positions| positions.deinit();
            pair_positions.deinit();
        }

        var queue = BpeQueue.init(allocator, .{});
        defer queue.deinit();

        var node_index: usize = 0;
        while (node_index < nodes.items.len) : (node_index += 1) {
            const left = nodes.items[node_index];
            if (left.next == no_node) continue;

            const right = nodes.items[left.next];
            const key = packPair(left.token, right.token);

            const count_entry = try pair_counts.getOrPut(key);
            if (!count_entry.found_existing) count_entry.value_ptr.* = 0;
            count_entry.value_ptr.* = std.math.add(u64, count_entry.value_ptr.*, 1) catch return error.FrequencyOverflow;

            const positions_entry = try pair_positions.getOrPut(key);
            if (!positions_entry.found_existing) {
                positions_entry.value_ptr.* = std.ArrayList(u32).init(allocator);
            }
            try positions_entry.value_ptr.append(@intCast(node_index));
        }

        var queued_frequency = std.AutoHashMap(u64, u64).init(allocator);
        defer queued_frequency.deinit();

        var touched = std.ArrayList(u64).init(allocator);
        defer touched.deinit();

        var occurrence_scratch = std.ArrayList(u32).init(allocator);
        defer occurrence_scratch.deinit();

        var initial_iterator = pair_counts.iterator();
        while (initial_iterator.next()) |entry| {
            if (entry.value_ptr.* < 2) continue;
            try queued_frequency.put(entry.key_ptr.*, entry.value_ptr.*);
            try queue.add(.{
                .frequency = entry.value_ptr.*,
                .key = entry.key_ptr.*,
            });
        }

        var merge_priority = try self.initialBpePriority();
        var scratch = std.ArrayList(u8).init(allocator);
        defer scratch.deinit();

        const state = BpeMergeState{
            .allocator = allocator,
            .nodes = &nodes,
            .pair_counts = &pair_counts,
            .pair_positions = &pair_positions,
            .queue = &queue,
            .queued_frequency = &queued_frequency,
            .touched = &touched,
            .occurrence_scratch = &occurrence_scratch,
        };

        while (self.vocabSize() < @as(usize, configured_target)) {
            var selected: ?BpeCandidate = null;

            while (queue.removeOrNull()) |candidate| {
                const current = pair_counts.get(candidate.key) orelse 0;
                if (current < 2) continue;
                if (current != candidate.frequency) {
                    // The entry is stale: only re-queue it when the pair grew
                    // beyond the frequency that is currently registered.
                    const queued = try queued_frequency.getOrPut(candidate.key);
                    if (!queued.found_existing) queued.value_ptr.* = 0;
                    if (current > queued.value_ptr.*) {
                        queued.value_ptr.* = current;
                        try queue.add(.{ .frequency = current, .key = candidate.key });
                    }
                    continue;
                }
                selected = candidate;
                break;
            }

            const candidate = selected orelse break;
            const first_id: u32 = @intCast(candidate.key >> 32);
            const second_id: u32 = @truncate(candidate.key);

            scratch.clearRetainingCapacity();
            try self.appendSurfaceText(first_id, &scratch);
            try self.appendSurfaceText(second_id, &scratch);

            // A merge that spells a byte label would be indistinguishable from a
            // byte token after the vocabulary is reloaded, so it is not
            // materialised. The pair stays consumed and the next candidate is
            // considered.
            if (isByteLabelText(scratch.items)) {
                if (pair_counts.getPtr(candidate.key)) |count| count.* = 0;
                continue;
            }

            const merged_id = try self.addToken(scratch.items);
            try self.bpe_pairs.put(candidate.key, .{
                .token_id = merged_id,
                .priority = merge_priority,
            });

            try applyBpeMerge(state, candidate.key, merged_id);

            if (merge_priority == std.math.maxInt(u32)) {
                if (self.vocabSize() < @as(usize, configured_target)) return error.PriorityOverflow;
            } else {
                merge_priority += 1;
            }
        }
    }

    const BpeMergeState = struct {
        allocator: Allocator,
        nodes: *std.ArrayList(BpeNode),
        pair_counts: *std.AutoHashMap(u64, u64),
        pair_positions: *std.AutoHashMap(u64, std.ArrayList(u32)),
        queue: *BpeQueue,
        /// Frequency that is currently queued for a pair. A pair is only
        /// re-queued when its frequency rises above this value, which keeps the
        /// queue proportional to the number of distinct pair updates instead of
        /// the number of merged occurrences.
        queued_frequency: *std.AutoHashMap(u64, u64),
        /// Pairs whose frequency changed during the merge that is in flight.
        touched: *std.ArrayList(u64),
        /// Occurrence list of the pair being merged. Reused between merges so
        /// a long run performs two allocations in total instead of one per
        /// merge.
        occurrence_scratch: *std.ArrayList(u32),
    };

    fn validOccurrence(nodes: []const BpeNode, position: u32, first: u32, second: u32) bool {
        if (position >= nodes.len) return false;
        const node = nodes[position];
        if (node.token != first) return false;
        if (node.next == no_node) return false;
        return nodes[node.next].token == second;
    }

    fn recordPairFrequency(state: BpeMergeState, left: u32, right: u32) !void {
        const key = packPair(left, right);
        const entry = try state.pair_counts.getOrPut(key);
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* = std.math.add(u64, entry.value_ptr.*, 1) catch return error.FrequencyOverflow;
        try state.touched.append(key);
    }

    /// Queues every pair whose frequency rose during the merge that just
    /// finished. Calling this once per merge instead of once per occurrence is
    /// what keeps the queue small on corpora with millions of tokens.
    fn flushTouchedCandidates(state: BpeMergeState) !void {
        for (state.touched.items) |key| {
            const count = state.pair_counts.get(key) orelse 0;
            if (count < 2) continue;

            const queued = try state.queued_frequency.getOrPut(key);
            if (!queued.found_existing) queued.value_ptr.* = 0;
            if (count <= queued.value_ptr.*) continue;

            queued.value_ptr.* = count;
            try state.queue.add(.{ .frequency = count, .key = key });
        }
        state.touched.clearRetainingCapacity();
    }

    fn forgetPairFrequency(state: BpeMergeState, left: u32, right: u32) void {
        const key = packPair(left, right);
        if (state.pair_counts.getPtr(key)) |count| {
            if (count.* > 0) count.* -= 1;
        }
    }

    fn recordPairPosition(state: BpeMergeState, left: u32, right: u32, position: u32) !void {
        const key = packPair(left, right);
        const entry = try state.pair_positions.getOrPut(key);
        if (!entry.found_existing) {
            entry.value_ptr.* = std.ArrayList(u32).init(state.allocator);
        }
        try entry.value_ptr.append(position);
    }

    /// Rewrites every live occurrence of `key` into `merged_id`. The occurrence
    /// list is detached from the map before it is walked, so registering the
    /// pairs that appear around the rewritten positions cannot invalidate the
    /// iterator.
    fn applyBpeMerge(state: BpeMergeState, key: u64, merged_id: u32) !void {
        const removed = state.pair_positions.fetchRemove(key) orelse return;

        // The detached list becomes the scratch buffer and the previous scratch
        // storage is released, so a training run reuses one buffer for every
        // merge instead of allocating a fresh one each time.
        var previous_scratch = removed.value;
        defer previous_scratch.deinit();
        std.mem.swap(std.ArrayList(u32), &previous_scratch, state.occurrence_scratch);

        const first: u32 = @intCast(key >> 32);
        const second: u32 = @truncate(key);
        const nodes = state.nodes;
        const occurrences = state.occurrence_scratch.items;

        var index: usize = 0;
        while (index < occurrences.len) : (index += 1) {
            const position = occurrences[index];
            if (!validOccurrence(nodes.items, position, first, second)) continue;

            const left = nodes.items[position].prev;
            const right_node = nodes.items[position].next;
            const right = nodes.items[right_node].next;

            if (left != no_node) {
                forgetPairFrequency(state, nodes.items[left].token, first);
            }
            if (right != no_node) {
                forgetPairFrequency(state, second, nodes.items[right].token);
            }

            nodes.items[position].token = merged_id;
            nodes.items[position].next = right;
            if (right != no_node) nodes.items[right].prev = position;

            nodes.items[right_node].prev = no_node;
            nodes.items[right_node].next = no_node;

            if (left != no_node) {
                try recordPairPosition(state, nodes.items[left].token, merged_id, left);
                try recordPairFrequency(state, nodes.items[left].token, merged_id);
            }
            if (right != no_node) {
                try recordPairPosition(state, merged_id, nodes.items[right].token, position);
                try recordPairFrequency(state, merged_id, nodes.items[right].token);
            }
        }

        // No occurrence of the merged pair can survive this rewrite, and the
        // merge can never recreate it because both merged token strings are
        // non-empty, so the merged id always differs from both source ids.
        if (state.pair_counts.getPtr(key)) |count| count.* = 0;

        state.occurrence_scratch.clearRetainingCapacity();
        try flushTouchedCandidates(state);
    }

    pub fn decode(self: *MGT, tokens: []const u32, out_text: *std.ArrayList(u8)) !void {
        for (tokens) |token| {
            if (self.byte_token_values.get(token)) |byte| {
                try out_text.append(byte);
            } else if (self.id_to_token.get(token)) |token_string| {
                try out_text.appendSlice(token_string);
            } else {
                const unknown = self.id_to_token.get(SPECIAL_TOKENS.UNK) orelse "[UNK]";
                try out_text.appendSlice(unknown);
            }
        }
    }

    pub fn longestMatch(self: *MGT, text: []const u8, start: usize) usize {
        if (start >= text.len) return 0;
        if (self.max_token_len == 0) return 0;

        const window_end = @min(text.len, start + self.max_token_len);
        var max_len: usize = 0;
        var end = start;

        while (end < window_end) {
            const step = safeUtf8SequenceLenAt(text, end);
            if (step == 0 or step > window_end - end) break;

            end += step;

            if (self.token_to_id.contains(text[start..end])) {
                max_len = end - start;
            }
        }

        return max_len;
    }

    pub fn vocabSize(self: *const MGT) usize {
        return self.token_to_id.count();
    }

    /// Sets the root mining policy.
    ///
    /// `minimum_frequency` is the corpus frequency a stem needs, and
    /// `capacity_divisor` bounds the root share of the free vocabulary capacity
    /// to `1 / capacity_divisor`. Both must be non-zero; a zero divisor would
    /// let the morphological layer consume the whole vocabulary and leave the
    /// byte-pair trainer without room for the merges that make the vocabulary
    /// useful on unseen words.
    pub fn configureRootMining(self: *MGT, policy: RootMiningPolicy) !void {
        if (policy.minimum_frequency == 0) return error.InvalidRootMiningPolicy;
        if (policy.capacity_divisor == 0) return error.InvalidRootMiningPolicy;
        self.root_min_frequency = policy.minimum_frequency;
        self.root_capacity_divisor = policy.capacity_divisor;
        self.root_evidence = policy.evidence;
    }

    /// Returns the active root mining policy.
    pub fn rootMiningPolicy(self: *const MGT) RootMiningPolicy {
        return .{
            .minimum_frequency = self.root_min_frequency,
            .capacity_divisor = self.root_capacity_divisor,
            .evidence = self.root_evidence,
        };
    }

    /// Number of stems the morphological layer owns.
    pub fn rootCount(self: *const MGT) usize {
        return self.roots.count();
    }

    /// Number of anchor tokens in the vocabulary.
    pub fn anchorCount(self: *const MGT) usize {
        return self.anchors.count();
    }

    /// Public entry point of the morphological layer.
    ///
    /// Decomposes `word` into `[prefix][root][suffix]` and appends the token ids
    /// to `out_tokens`. Returns false when the word carries no known affix or
    /// when the remaining stem is not a vocabulary token, in which case nothing
    /// is appended and the caller falls back to the subword layer.
    pub fn decompose(self: *MGT, word: []const u8, out_tokens: *std.ArrayList(u32)) !bool {
        return self.morphDecompose(word, 0, out_tokens, null);
    }

    /// Writes the anchor tokens as a newline-separated token list.
    ///
    /// This is the format the distributed trainer reads through
    /// `JAIDE_TOKENIZER_ANCHORS`, so a vocabulary trained offline can hand its
    /// morphological roots to every rank without the binary vocabulary format.
    /// Tokens are ordered by id, which makes the file byte-identical for
    /// identical vocabularies.
    ///
    /// Returns false, and writes nothing at all, when the vocabulary has no
    /// anchors: the token-list reader rejects an empty file, so creating one
    /// would turn a legitimate "this corpus had no productive stems" outcome
    /// into a startup failure for whoever points the trainer at it.
    pub fn writeAnchorList(self: *MGT, path: []const u8) !bool {
        const AnchorItem = struct {
            key: []const u8,
            value: u64,
        };

        const AnchorContext = struct {
            fn lessThan(_: @This(), a: AnchorItem, b: AnchorItem) bool {
                if (a.value != b.value) return a.value < b.value;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var items = std.ArrayList(AnchorItem).init(self.allocator);
        defer items.deinit();

        var iterator = self.anchors.iterator();
        while (iterator.next()) |entry| {
            try items.append(.{
                .key = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }

        std.mem.sort(AnchorItem, items.items, AnchorContext{}, AnchorContext.lessThan);

        if (items.items.len == 0) return false;

        var file = try core_io.createFilePath(path, .{ .truncate = true });
        defer file.close();

        var buffered = std.io.bufferedWriter(file.writer());
        const writer = buffered.writer();

        for (items.items) |item| {
            // A token that contains a newline could not be read back from a
            // line-oriented token list, so such a vocabulary is rejected
            // instead of being written in a form that loses tokens.
            if (std.mem.indexOfScalar(u8, item.key, '\n') != null) return error.InvalidData;
            try writer.print("{s}\n", .{item.key});
        }

        try buffered.flush();
        return true;
    }

    pub fn addVocabWord(self: *MGT, word: []const u8, is_anchor: bool) !void {
        if (!self.canAddToken() and !self.token_to_id.contains(word)) {
            return error.VocabularyFull;
        }

        const id = try self.addToken(word);

        if (is_anchor) {
            const key = self.id_to_token.get(id) orelse return error.InvalidData;
            try self.anchors.put(key, @as(u64, id));
        }
    }

    pub fn removeVocabWord(self: *MGT, word: []const u8) void {
        if (mem.eql(u8, word, "[PAD]") or
            mem.eql(u8, word, "[UNK]") or
            mem.eql(u8, word, "[BOS]") or
            mem.eql(u8, word, "[EOS]"))
        {
            return;
        }

        if (self.token_to_id.get(word)) |id| {
            if (self.id_to_token.get(id)) |allocated_pointer| {
                _ = self.token_to_id.remove(word);
                _ = self.id_to_token.remove(id);
                _ = self.anchors.remove(word);
                _ = self.prefixes.remove(word);
                _ = self.suffixes.remove(word);
                _ = self.roots.remove(word);

                if (self.byte_token_values.fetchRemove(id)) |entry| {
                    self.byte_token_ids[entry.value] = null;
                }

                var bpe_remove = std.ArrayList(u64).init(self.allocator);
                defer bpe_remove.deinit();

                var bpe_iterator = self.bpe_pairs.iterator();
                while (bpe_iterator.next()) |entry| {
                    const key = entry.key_ptr.*;
                    const merge = entry.value_ptr.*;

                    if (merge.token_id == id) {
                        bpe_remove.append(key) catch return;
                    }
                }

                for (bpe_remove.items) |key| {
                    _ = self.bpe_pairs.remove(key);
                }

                var index: usize = 0;
                while (index < self.allocated_strings.items.len) : (index += 1) {
                    const str = self.allocated_strings.items[index];

                    if (str.ptr == allocated_pointer.ptr and str.len == allocated_pointer.len) {
                        const removed_length = str.len;
                        self.allocator.free(str);
                        _ = self.allocated_strings.swapRemove(index);
                        if (removed_length == self.max_token_len) self.recomputeMaxTokenLen();
                        break;
                    }
                }
            }
        }

        self.rebuildSortedMorphemes() catch {};
    }

    fn recomputeMaxTokenLen(self: *MGT) void {
        var longest: usize = 0;
        for (self.allocated_strings.items) |str| {
            longest = @max(longest, str.len);
        }
        self.max_token_len = longest;
    }

    pub fn tokenizeWithAnchors(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: *std.ArrayList(usize)) !void {
        try self.encodeInternal(text, out_tokens, out_anchors);
    }

    pub fn detokenize(self: *MGT, tokens: []const u32) ![]u8 {
        return self.detokenizeAlloc(tokens, self.allocator);
    }

    fn detokenizeAlloc(self: *MGT, tokens: []const u32, allocator: Allocator) ![]u8 {
        var text = std.ArrayList(u8).init(allocator);
        defer text.deinit();

        try self.decode(tokens, &text);
        return text.toOwnedSlice();
    }

    pub fn encodeBatch(self: *MGT, texts: []const []const u8, allocator: Allocator) ![][]u32 {
        const results = try allocator.alloc([]u32, texts.len);
        errdefer allocator.free(results);

        var initialized: usize = 0;
        errdefer {
            for (results[0..initialized]) |result| {
                allocator.free(result);
            }
        }

        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();

            try self.encode(text, &tokens);
            results[initialized] = try tokens.toOwnedSlice();
            initialized += 1;
        }

        return results;
    }

    pub fn batchDetokenize(self: *MGT, token_lists: []const []const u32, allocator: Allocator) ![][]u8 {
        const results = try allocator.alloc([]u8, token_lists.len);
        errdefer allocator.free(results);

        var initialized: usize = 0;
        errdefer {
            for (results[0..initialized]) |result| {
                allocator.free(result);
            }
        }

        for (token_lists) |tokens| {
            results[initialized] = try self.detokenizeAlloc(tokens, allocator);
            initialized += 1;
        }

        return results;
    }

    fn usizeToU32(value: usize) !u32 {
        if (value > std.math.maxInt(u32)) return error.DataTooLarge;
        return @intCast(value);
    }

    fn writeStringMapSorted(map: std.StringHashMap(u32), writer: anytype, allocator: Allocator) !void {
        const Item = struct {
            key: []const u8,
            value: u32,
        };

        const Context = struct {
            fn lessThan(_: @This(), a: Item, b: Item) bool {
                if (a.value != b.value) return a.value < b.value;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var items = std.ArrayList(Item).init(allocator);
        defer items.deinit();

        var iterator = map.iterator();
        while (iterator.next()) |entry| {
            try items.append(.{
                .key = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }

        std.mem.sort(Item, items.items, Context{}, Context.lessThan);
        try writer.writeInt(u32, try usizeToU32(items.items.len), .little);

        for (items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.key.len), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u32, item.value, .little);
        }
    }

    pub fn saveVocab(self: *MGT, path: []const u8) !void {
        var file = try core_io.createFilePath(path, .{ .truncate = true });
        defer file.close();

        var writer = file.writer();

        const TokenItem = struct {
            id: u32,
            token: []const u8,
        };

        const TokenContext = struct {
            fn lessThan(_: @This(), a: TokenItem, b: TokenItem) bool {
                if (a.id != b.id) return a.id < b.id;
                return std.mem.lessThan(u8, a.token, b.token);
            }
        };

        var token_items = std.ArrayList(TokenItem).init(self.allocator);
        defer token_items.deinit();

        var token_iterator = self.id_to_token.iterator();
        while (token_iterator.next()) |entry| {
            try token_items.append(.{
                .id = entry.key_ptr.*,
                .token = entry.value_ptr.*,
            });
        }

        std.mem.sort(TokenItem, token_items.items, TokenContext{}, TokenContext.lessThan);
        try writer.writeInt(u32, try usizeToU32(token_items.items.len), .little);

        for (token_items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.token.len), .little);
            try writer.writeAll(item.token);
            try writer.writeInt(u32, item.id, .little);
        }

        const BpeItem = struct {
            key: u64,
            merge: BPEMerge,
        };

        const BpeContext = struct {
            fn lessThan(_: @This(), a: BpeItem, b: BpeItem) bool {
                if (a.merge.priority != b.merge.priority) {
                    return a.merge.priority < b.merge.priority;
                }
                return a.key < b.key;
            }
        };

        var bpe_items = std.ArrayList(BpeItem).init(self.allocator);
        defer bpe_items.deinit();

        var bpe_iterator = self.bpe_pairs.iterator();
        while (bpe_iterator.next()) |entry| {
            try bpe_items.append(.{
                .key = entry.key_ptr.*,
                .merge = entry.value_ptr.*,
            });
        }

        std.mem.sort(BpeItem, bpe_items.items, BpeContext{}, BpeContext.lessThan);
        try writer.writeInt(u32, try usizeToU32(bpe_items.items.len), .little);

        for (bpe_items.items) |item| {
            try writer.writeInt(u64, item.key, .little);
            try writer.writeInt(u32, item.merge.token_id, .little);
            try writer.writeInt(u32, item.merge.priority, .little);
        }

        try writeStringMapSorted(self.prefixes, writer, self.allocator);
        try writeStringMapSorted(self.suffixes, writer, self.allocator);
        try writeStringMapSorted(self.roots, writer, self.allocator);

        const AnchorItem = struct {
            key: []const u8,
            value: u64,
        };

        const AnchorContext = struct {
            fn lessThan(_: @This(), a: AnchorItem, b: AnchorItem) bool {
                if (a.value != b.value) return a.value < b.value;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var anchor_items = std.ArrayList(AnchorItem).init(self.allocator);
        defer anchor_items.deinit();

        var anchor_iterator = self.anchors.iterator();
        while (anchor_iterator.next()) |entry| {
            try anchor_items.append(.{
                .key = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }

        std.mem.sort(AnchorItem, anchor_items.items, AnchorContext{}, AnchorContext.lessThan);
        try writer.writeInt(u32, try usizeToU32(anchor_items.items.len), .little);

        for (anchor_items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.key.len), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u64, item.value, .little);
        }
    }

    pub fn loadVocab(self: *MGT, path: []const u8) !void {
        self.reset();

        var file = try core_io.openFilePath(path, .{});
        defer file.close();

        var reader = file.reader();
        const token_count = try reader.readInt(u32, .little);

        if (self.max_vocab_size) |limit| {
            if (token_count > limit) return error.VocabularyFull;
        }

        var token_index: u32 = 0;
        while (token_index < token_count) : (token_index += 1) {
            const word_len = try reader.readInt(u32, .little);
            const word_buffer = try self.allocator.alloc(u8, @as(usize, word_len));
            var word_owned = true;
            errdefer if (word_owned) self.allocator.free(word_buffer);

            try reader.readNoEof(word_buffer);
            const id = try reader.readInt(u32, .little);

            if (id == std.math.maxInt(u32) or
                self.token_to_id.contains(word_buffer) or
                self.id_to_token.contains(id))
            {
                return error.InvalidData;
            }

            try self.adoptTokenWithId(word_buffer, id);
            word_owned = false;
        }

        const bpe_count = try reader.readInt(u32, .little);
        var bpe_index: u32 = 0;

        while (bpe_index < bpe_count) : (bpe_index += 1) {
            const pair_key = try reader.readInt(u64, .little);
            const token_id = try reader.readInt(u32, .little);
            const priority = try reader.readInt(u32, .little);

            try self.bpe_pairs.put(pair_key, .{
                .token_id = token_id,
                .priority = priority,
            });
        }

        const ReadStringMap = struct {
            fn read(self_mgt: *MGT, map: *std.StringHashMap(u32), input_reader: anytype) !void {
                const count = try input_reader.readInt(u32, .little);
                var index: u32 = 0;

                while (index < count) : (index += 1) {
                    const length = try input_reader.readInt(u32, .little);
                    const buffer = try self_mgt.allocator.alloc(u8, @as(usize, length));
                    var buffer_owned = true;
                    errdefer if (buffer_owned) self_mgt.allocator.free(buffer);

                    try input_reader.readNoEof(buffer);

                    const id = try input_reader.readInt(u32, .little);
                    const canonical = try self_mgt.getCanonicalTokenForLoad(buffer, id);
                    buffer_owned = false;

                    try map.put(canonical, id);
                }
            }
        };

        try ReadStringMap.read(self, &self.prefixes, reader);
        try ReadStringMap.read(self, &self.suffixes, reader);
        try ReadStringMap.read(self, &self.roots, reader);

        const anchor_count = try reader.readInt(u32, .little);
        var anchor_index: u32 = 0;

        while (anchor_index < anchor_count) : (anchor_index += 1) {
            const key_len = try reader.readInt(u32, .little);
            const key_buffer = try self.allocator.alloc(u8, @as(usize, key_len));
            var key_owned = true;
            errdefer if (key_owned) self.allocator.free(key_buffer);

            try reader.readNoEof(key_buffer);

            const value = try reader.readInt(u64, .little);
            if (value > std.math.maxInt(u32)) return error.InvalidData;

            const canonical = try self.getCanonicalTokenForLoad(key_buffer, @intCast(value));
            key_owned = false;

            try self.anchors.put(canonical, value);
        }

        try self.rebuildSortedMorphemes();
        try self.rebuildByteTokenLookup();
        self.allow_dynamic_tokens = false;
    }

    fn rebuildByteTokenLookup(self: *MGT) !void {
        self.byte_token_ids = [_]?u32{null} ** 256;
        self.byte_token_values.clearRetainingCapacity();

        var iterator = self.id_to_token.iterator();
        while (iterator.next()) |entry| {
            const token_string = entry.value_ptr.*;
            const id = entry.key_ptr.*;

            if (token_string.len == 4 and
                token_string[0] == '<' and
                token_string[3] == '>')
            {
                const hex = token_string[1..3];

                if (std.fmt.parseInt(u8, hex, 16)) |byte_value| {
                    if (self.byte_token_ids[byte_value] != null) {
                        return error.InvalidData;
                    }

                    self.byte_token_ids[byte_value] = id;
                    try self.byte_token_values.put(id, byte_value);
                } else |_| {}
            }
        }
    }

    pub fn unknownReplacement(self: *const MGT, context: []const u8) u32 {
        _ = self;
        _ = context;
        return SPECIAL_TOKENS.UNK;
    }

    /// Splits an out-of-vocabulary word into known vocabulary tokens and
    /// byte-pair encoded runs.
    ///
    /// The scan alternates between two regimes: while the longest vocabulary
    /// match at the cursor is non-empty the match is emitted directly, and as
    /// soon as no vocabulary token starts at the cursor the maximal run without
    /// any vocabulary match is handed to the byte-pair encoder in one piece, so
    /// trained merges actually apply across the whole unknown fragment instead
    /// of being limited to isolated characters.
    fn subwordSplitInto(self: *MGT, word: []const u8, word_start: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        if (word.len == 0) return;

        var i: usize = 0;
        while (i < word.len) {
            const match_len = self.longestMatch(word, i);

            if (match_len > 0 and match_len <= word.len - i) {
                const found_word = word[i .. i + match_len];

                if (self.token_to_id.get(found_word)) |tid| {
                    try self.emitToken(tid, word_start + i, out_tokens, out_anchors);
                    i += match_len;
                    continue;
                }
            }

            var run_end = i;
            while (run_end < word.len and self.longestMatch(word, run_end) == 0) {
                const step = safeUtf8SequenceLenAt(word, run_end);
                if (step == 0 or step > word.len - run_end) return error.InvalidData;
                run_end += step;
            }

            if (run_end == i) {
                const step = safeUtf8SequenceLenAt(word, i);
                if (step == 0 or step > word.len - i) return error.InvalidData;
                run_end = i + step;
            }

            try self.appendBPEOrUnknown(word[i..run_end], word_start + i, out_tokens, out_anchors);
            i = run_end;
        }
    }

    pub fn subwordSplit(self: *MGT, word: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        defer tokens.deinit();

        try self.subwordSplitInto(word, 0, &tokens, null);
        return tokens.toOwnedSlice();
    }

    pub fn mergeSubwords(self: *MGT, subwords: []const []const u32) ![]u32 {
        var merged = std.ArrayList(u32).init(self.allocator);
        defer merged.deinit();

        var total_length: usize = 0;
        for (subwords) |subword| {
            total_length = std.math.add(usize, total_length, subword.len) catch return error.InputTooLarge;
        }

        try merged.ensureTotalCapacity(total_length);

        for (subwords) |subword| {
            try merged.appendSlice(subword);
        }

        return merged.toOwnedSlice();
    }

    pub fn validateTokens(self: *MGT, tokens: []const u32) bool {
        for (tokens) |token| {
            if (!self.id_to_token.contains(token)) return false;
        }

        return true;
    }

    pub fn coverage(self: *MGT, corpus: []const u8) f32 {
        if (corpus.len == 0) return 0.0;

        var covered: usize = 0;
        var i: usize = 0;

        while (i < corpus.len) {
            if (self.getKnownSpecialTokenLen(corpus, i)) |special_len| {
                covered = std.math.add(usize, covered, special_len) catch return 0.0;
                i += special_len;
                continue;
            }

            if (isWhitespace(corpus[i]) or isPunctuation(corpus[i])) {
                const slice = corpus[i .. i + 1];

                if (self.token_to_id.contains(slice) or
                    (corpus[i] == ' ' and self.token_to_id.contains(" ")))
                {
                    covered = std.math.add(usize, covered, 1) catch return 0.0;
                }

                i += 1;
                continue;
            }

            var word_end = i;
            while (word_end < corpus.len) {
                if (self.isKnownSpecialTokenStart(corpus, word_end)) break;

                const c = corpus[word_end];
                if (isWhitespace(c) or isPunctuation(c)) break;

                const char_len = safeUtf8SequenceLenAt(corpus, word_end);
                if (char_len == 0 or char_len > corpus.len - word_end) break;
                word_end += char_len;
            }

            if (word_end == i) {
                const char_len = safeUtf8SequenceLenAt(corpus, i);
                if (char_len == 0 or char_len > corpus.len - i) break;

                const maybe_bpe = self.encodeBPE(corpus[i .. i + char_len]) catch null;
                if (maybe_bpe) |bpe| {
                    defer self.allocator.free(bpe);

                    var all_unknown = true;
                    for (bpe) |tid| {
                        if (tid != SPECIAL_TOKENS.UNK) {
                            all_unknown = false;
                            break;
                        }
                    }

                    if (!all_unknown) {
                        covered = std.math.add(usize, covered, char_len) catch return 0.0;
                    }
                }

                i += char_len;
                continue;
            }

            const word = corpus[i..word_end];

            if (self.token_to_id.contains(word)) {
                covered = std.math.add(usize, covered, word.len) catch return 0.0;
            } else {
                var temporary = std.ArrayList(u32).init(self.allocator);
                defer temporary.deinit();

                if (self.morphDecompose(word, i, &temporary, null) catch false) {
                    covered = std.math.add(usize, covered, word.len) catch return 0.0;
                } else {
                    const maybe_subwords = self.subwordSplit(word) catch null;

                    if (maybe_subwords) |subwords| {
                        defer self.allocator.free(subwords);

                        var all_unknown = true;
                        for (subwords) |tid| {
                            if (tid != SPECIAL_TOKENS.UNK) {
                                all_unknown = false;
                                break;
                            }
                        }

                        if (!all_unknown) {
                            covered = std.math.add(usize, covered, word.len) catch return 0.0;
                        }
                    }
                }
            }

            i = word_end;
        }

        return @as(f32, @floatFromInt(covered)) /
            @as(f32, @floatFromInt(corpus.len));
    }

    pub fn encodeToTensor(self: *MGT, text: []const u8, allocator: Allocator) !core_tensor.Tensor {
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        try self.encode(text, &tokens);

        const shape = [_]usize{tokens.items.len};
        var tensor = try core_tensor.Tensor.init(allocator, &shape);

        for (tokens.items, 0..) |token, index| {
            tensor.data[index] = @floatFromInt(token);
        }

        return tensor;
    }

    pub fn encodeBatchToTensor(self: *MGT, texts: []const []const u8, allocator: Allocator) !core_tensor.Tensor {
        var max_len: usize = 0;
        var per_row = std.ArrayList([]u32).init(allocator);

        defer {
            for (per_row.items) |row| {
                allocator.free(row);
            }
            per_row.deinit();
        }

        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();

            try self.encode(text, &tokens);

            const owned = try tokens.toOwnedSlice();
            errdefer allocator.free(owned);

            try per_row.append(owned);
            max_len = @max(max_len, owned.len);
        }

        if (max_len == 0) max_len = 1;
        _ = std.math.mul(usize, texts.len, max_len) catch return error.InputTooLarge;

        const shape = [_]usize{ texts.len, max_len };
        var tensor = try core_tensor.Tensor.init(allocator, &shape);
        @memset(tensor.data, @as(@TypeOf(tensor.data[0]), 0));

        for (per_row.items, 0..) |row, row_index| {
            for (row, 0..) |token, column_index| {
                const row_offset = std.math.mul(usize, row_index, max_len) catch return error.InputTooLarge;
                const tensor_index = std.math.add(usize, row_offset, column_index) catch return error.InputTooLarge;
                tensor.data[tensor_index] = @floatFromInt(token);
            }
        }

        return tensor;
    }

    pub fn decodeFromTensor(self: *MGT, tensor: *const core_tensor.Tensor, allocator: Allocator) ![]u8 {
        const tokens = try allocator.alloc(u32, tensor.data.len);
        defer allocator.free(tokens);

        for (tensor.data, 0..) |value, index| {
            if (std.math.isNan(value) or
                std.math.isInf(value) or
                value < 0.0 or
                value > @as(@TypeOf(value), @floatFromInt(std.math.maxInt(u32))))
            {
                tokens[index] = SPECIAL_TOKENS.UNK;
            } else {
                tokens[index] = @intFromFloat(value);

                if (!self.id_to_token.contains(tokens[index])) {
                    tokens[index] = SPECIAL_TOKENS.UNK;
                }
            }
        }

        return self.detokenizeAlloc(tokens, allocator);
    }
};
