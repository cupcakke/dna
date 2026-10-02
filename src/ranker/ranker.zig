const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const BitSet = types.BitSet;
const Tensor = @import("../core/tensor.zig").Tensor;
const SSI = @import("../index/ssi.zig").SSI;
const SSIScoringConfig = @import("../index/ssi.zig").SSIScoringConfig;
const core_io = @import("../core/io.zig");
const stableHash = core_io.stableHash;
const Error = types.Error;
const rsf_mod = @import("../processor/rsf.zig");
const RSF = rsf_mod.RSF;
const RSFLatentState = rsf_mod.RSFLatentState;
const SFD = @import("../optimizer/sfd.zig").SFD;

pub const RankerConfig = struct {
    pub const STREAMING_BUFFER_SIZE: usize = 1024;
    pub const STREAMING_WINDOW_SIZE: usize = 512;
    pub const DEFAULT_TOP_N_RETRIEVAL: usize = 1000;
    pub const HASH_SEED_MULTIPLIER_A: u64 = 0x9e3779b97f4a7c15;
    pub const HASH_SEED_MULTIPLIER_B: u64 = 0x517cc1b727220a95;
    pub const LEARNING_RATE: f32 = 0.01;
    pub const DIVERSITY_WEIGHT: f32 = 0.3;
    pub const PROXIMITY_WEIGHT: f32 = 0.3;
    pub const MAX_RAW_SCORE: f32 = 100.0;
    pub const BASE_SCORE_WEIGHT: f32 = 0.4;
    pub const OVERLAP_WEIGHT: f32 = 0.3;
    pub const JACCARD_WEIGHT: f32 = 0.3;
    pub const SCORE_RETRIEVAL_LIMIT: usize = 32;
    pub const MAX_NGRAM_ORDER: usize = 64;
    pub const PROXIMITY_SCALE: u64 = 1024;
    pub const SCORE_SIGMOID_CENTER: f32 = 0.8;
    pub const SCORE_SIGMOID_WIDTH: f32 = 0.4;
    pub const HEAD_DIM: usize = 2;
    pub const HEAD_LAYERS: usize = 1;
    pub const HEAD_STATE: usize = 4;
    pub const HEAD_COUPLING_WIDTH: usize = 2;
    pub const EXPORT_VERSION: u8 = 3;
    pub const EXPORT_VERSION_LEGACY: u8 = 2;
    pub const RAW_FUSE_DIVISOR: f32 = 10.0;
};

const head_pair_len: usize = RankerConfig.HEAD_DIM * RankerConfig.HEAD_COUPLING_WIDTH;

fn tokenToLEBytes(token: u32) [4]u8 {
    return mem.toBytes(mem.nativeToLittle(u32, token));
}

fn encodeNgramLE(ngram: []const u32, buf: []u8) void {
    for (ngram, 0..) |token, i| {
        const le = tokenToLEBytes(token);
        buf[i * 4 + 0] = le[0];
        buf[i * 4 + 1] = le[1];
        buf[i * 4 + 2] = le[2];
        buf[i * 4 + 3] = le[3];
    }
}

fn sigmoidScale(raw: f32) f32 {
    const z = (raw - RankerConfig.SCORE_SIGMOID_CENTER) / RankerConfig.SCORE_SIGMOID_WIDTH;
    if (z >= 0.0) {
        return 1.0 / (1.0 + @exp(-z));
    } else {
        const e = @exp(z);
        return e / (1.0 + e);
    }
}

fn sigmoidDerivative(out: f32) f32 {
    return out * (1.0 - out) / RankerConfig.SCORE_SIGMOID_WIDTH;
}

fn freeRankedSegments(segments: []types.RankedSegment, allocator: Allocator) void {
    for (segments) |*seg| seg.deinit(allocator);
    allocator.free(segments);
}

const RankedSegmentComparator = struct {
    pub fn lessThan(_: void, a: types.RankedSegment, b: types.RankedSegment) std.math.Order {
        if (math.isNan(a.score) and math.isNan(b.score)) return .eq;
        if (math.isNan(a.score)) return .lt;
        if (math.isNan(b.score)) return .gt;
        return std.math.order(a.score, b.score);
    }
};

pub const CandidateEvaluation = struct {
    score: f32,
    latent_similarity: f32,
    reconstruction_confidence: f32,
    volume_surprise: f32,
};

pub const Ranker = struct {
    ngram_weights: []f32,
    lsh_hash_params: []u64,
    num_hash_functions: usize,
    num_ngrams: usize,
    seed: u64,
    allocator: Allocator,
    head: RSF,
    head_sfd: SFD,
    model_id: u64,
    model_dim: usize,
    model_global_diffusion: bool,

    fn seedHeadWeights(head: *RSF, seed: u64, allocator: Allocator) !void {
        const plen = head_pair_len;
        const s_buf = try allocator.alloc(f32, plen);
        defer allocator.free(s_buf);
        const t_buf = try allocator.alloc(f32, plen);
        defer allocator.free(t_buf);
        var prng = types.PRNG.init(seed ^ 0xC0FFEE00FF00FF00);
        const bound: f32 = @sqrt(6.0 / 4.0);
        var i: usize = 0;
        while (i < plen) : (i += 1) {
            s_buf[i] = prng.float() * (2.0 * bound) - bound;
            t_buf[i] = prng.float() * (2.0 * bound) - bound;
        }
        var d: usize = 0;
        while (d < RankerConfig.HEAD_DIM) : (d += 1) {
            s_buf[d * RankerConfig.HEAD_COUPLING_WIDTH + 1] = 0.0;
            t_buf[d * RankerConfig.HEAD_COUPLING_WIDTH + 1] = 0.0;
        }
        try head.writeLayerWeights(0, s_buf, t_buf);
    }

    fn createHead(allocator: Allocator, seed: u64) !RSF {
        var head = try RSF.initWithConfig(allocator, RankerConfig.HEAD_DIM, RankerConfig.HEAD_LAYERS, .{
            .global_diffusion = false,
            .clip_min = -5.0,
            .clip_max = 5.0,
        });
        errdefer head.deinit();
        try seedHeadWeights(&head, seed, allocator);
        return head;
    }

    fn resetHead(self: *Ranker) !void {
        self.head_sfd.deinit();
        self.head.deinit();
        self.head = try createHead(self.allocator, self.seed);
        errdefer self.head.deinit();
        self.head_sfd = try SFD.init(self.allocator, &self.head);
    }

    pub fn init(allocator: Allocator, num_ngrams: usize, num_hash_funcs: usize, seed: u64, model: *const RSF) !Ranker {
        if (num_ngrams == 0) return error.InvalidParameter;
        if (num_ngrams > RankerConfig.MAX_NGRAM_ORDER) return error.InvalidParameter;
        if (num_hash_funcs == 0) return error.InvalidParameter;
        if (model.id == 0) return Error.RSFModelMismatch;
        const model_dim = try model.dim();
        if (model_dim == 0) return error.InvalidParameter;
        const model_gd = try model.globalDiffusionEnabled();

        const weights = try allocator.alloc(f32, num_ngrams);
        errdefer allocator.free(weights);

        var i: usize = 0;
        while (i < weights.len) : (i += 1) {
            const decay = 1.0 / @as(f32, @floatFromInt(i + 1));
            weights[i] = decay;
        }

        const hash_params = try allocator.alloc(u64, num_hash_funcs * 2);
        errdefer allocator.free(hash_params);

        i = 0;
        while (i < num_hash_funcs) : (i += 1) {
            const i_u64: u64 = @intCast(i);
            const i_plus_one: u64 = @intCast(i + 1);
            hash_params[i * 2] = seed +% (i_u64 *% RankerConfig.HASH_SEED_MULTIPLIER_A);
            hash_params[i * 2 + 1] = seed +% (i_plus_one *% RankerConfig.HASH_SEED_MULTIPLIER_B);
        }

        var head = try createHead(allocator, seed);
        errdefer head.deinit();
        var head_sfd = try SFD.init(allocator, &head);
        errdefer head_sfd.deinit();

        return .{
            .ngram_weights = weights,
            .lsh_hash_params = hash_params,
            .num_hash_functions = num_hash_funcs,
            .num_ngrams = num_ngrams,
            .seed = seed,
            .allocator = allocator,
            .head = head,
            .head_sfd = head_sfd,
            .model_id = model.id,
            .model_dim = model_dim,
            .model_global_diffusion = model_gd,
        };
    }

    pub fn deinit(self: *Ranker) void {
        self.head_sfd.deinit();
        self.head.deinit();
        self.allocator.free(self.ngram_weights);
        self.allocator.free(self.lsh_hash_params);
    }

    fn requireModel(self: *const Ranker, model: *const RSF, ssi: *const SSI) !void {
        if (model.id == 0 or model.id != self.model_id) return Error.RSFModelMismatch;
        const dim = try model.dim();
        if (dim != self.model_dim) return Error.RSFModelMismatch;
        const gd = try model.globalDiffusionEnabled();
        if (gd != self.model_global_diffusion) return Error.RSFModelMismatch;
        if (ssi.identity_set) {
            if (ssi.dim != self.model_dim) return Error.RSFModelMismatch;
            if (ssi.global_diffusion != self.model_global_diffusion) return Error.RSFModelMismatch;
            if (ssi.model_id != 0 and ssi.model_id != self.model_id) return Error.RSFModelMismatch;
        }
    }

    fn requireQuery(self: *const Ranker, model: *const RSF, query_state: *const RSFLatentState, ssi: *const SSI) !void {
        try self.requireModel(model, ssi);
        try query_state.requireModel(model);
        if (query_state.data.shape.dims.len != 3) return error.InvalidShape;
        if (query_state.data.shape.dims[0] == 0) return error.EmptyInput;
        if (query_state.data.shape.dims[1] != self.model_dim or query_state.data.shape.dims[2] != 2) return Error.RSFDimMismatch;
    }

    fn packStateBlocked(state: *const RSFLatentState, dim: usize, out: []f32) !void {
        if (state.data.shape.dims.len != 3) return error.InvalidShape;
        if (state.data.shape.dims[1] != dim or state.data.shape.dims[2] != 2) return error.ShapeMismatch;
        if (state.data.shape.dims[0] == 0) return error.EmptyInput;
        if (out.len < dim * 2) return error.DataLengthMismatch;
        const data = state.data.data;
        var d: usize = 0;
        while (d < dim) : (d += 1) {
            out[d] = data[d * 2];
            out[dim + d] = data[d * 2 + 1];
        }
    }

    fn latentStateFromBlocked(allocator: Allocator, model: *const RSF, blocked: []const f32) !RSFLatentState {
        const dim = try model.dim();
        if (blocked.len != dim * 2) return error.ShapeMismatch;
        var x1 = try Tensor.init(allocator, &[_]usize{ 1, dim });
        defer x1.deinit();
        var x2 = try Tensor.init(allocator, &[_]usize{ 1, dim });
        defer x2.deinit();
        @memcpy(x1.data[0..dim], blocked[0..dim]);
        @memcpy(x2.data[0..dim], blocked[dim .. dim * 2]);
        return RSFLatentState.fromHalves(allocator, model, &x1, &x2);
    }

    fn hashEmbedTokens(tokens: []const u32, dim: usize, seed: u64, out: []f32) void {
        const total = dim * 2;
        if (out.len < total or dim == 0) return;
        var d: usize = 0;
        while (d < total) : (d += 1) {
            var acc: u64 = seed ^ (@as(u64, @intCast(d)) *% RankerConfig.HASH_SEED_MULTIPLIER_A);
            var ti: usize = 0;
            while (ti < tokens.len) : (ti += 1) {
                const le = tokenToLEBytes(tokens[ti]);
                acc = stableHash(&le, acc);
            }
            const top = acc >> 40;
            const u = @as(f32, @floatFromInt(top)) / 16777216.0;
            var v = u * 2.0 - 1.0;
            if (!std.math.isFinite(v)) v = 0.0;
            out[d] = v;
        }
    }

    fn cosineClamped(a: []const f32, b: []const f32) f32 {
        if (a.len == 0 or b.len == 0 or a.len != b.len) return 0.0;
        var dot: f64 = 0.0;
        var na: f64 = 0.0;
        var nb: f64 = 0.0;
        var i: usize = 0;
        while (i < a.len) : (i += 1) {
            if (!std.math.isFinite(a[i]) or !std.math.isFinite(b[i])) continue;
            const x: f64 = a[i];
            const y: f64 = b[i];
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        const denom = @sqrt(na) * @sqrt(nb) + @as(f64, SSIScoringConfig.cosine_eps);
        const cos: f32 = @floatCast(dot / denom);
        if (!std.math.isFinite(cos)) return 0.0;
        return math.clamp(cos, 0.0, 1.0);
    }

    fn volumeSurprise(segment_log_det: f32, query_log_det: f32) f32 {
        const sl = if (std.math.isFinite(segment_log_det)) segment_log_det else 0.0;
        const ql = if (std.math.isFinite(query_log_det)) query_log_det else 0.0;
        const delta = @abs(sl - ql) / SSIScoringConfig.volume_kappa;
        const v: f32 = @floatCast(@exp(@as(f64, -delta)));
        if (!std.math.isFinite(v)) return 0.0;
        return math.clamp(v, 0.0, 1.0);
    }

    fn queryStateFromTokens(self: *const Ranker, model: *RSF, tokens: []const u32, allocator: Allocator) !RSFLatentState {
        const dim = self.model_dim;
        const blocked = try allocator.alloc(f32, dim * 2);
        defer allocator.free(blocked);
        hashEmbedTokens(tokens, dim, self.seed, blocked);
        var state = try latentStateFromBlocked(allocator, model, blocked);
        errdefer state.deinit();
        try model.forwardLatent(&state);
        return state;
    }

    fn candidateLatent(self: *const Ranker, model: *RSF, tokens: []const u32, position: u64, ssi: *const SSI, allocator: Allocator, out: []f32, log_det_out: *f32) !void {
        const dim = self.model_dim;
        if (out.len < dim * 2) return error.DataLengthMismatch;
        if (ssi.getSegment(position)) |seg| {
            if (seg.latent_version == 1 and seg.latent.len == dim * 2) {
                @memcpy(out[0 .. dim * 2], seg.latent[0 .. dim * 2]);
                log_det_out.* = seg.log_det;
                return;
            }
        }
        hashEmbedTokens(tokens, dim, self.seed, out);
        var state = try latentStateFromBlocked(allocator, model, out[0 .. dim * 2]);
        defer state.deinit();
        try model.forwardLatent(&state);
        try packStateBlocked(&state, dim, out);
        log_det_out.* = state.log_det;
    }

    fn headForwardAlloc(self: *Ranker, z: [4]f32, allocator: Allocator) !f32 {
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            if (!std.math.isFinite(z[i])) return error.NonFinite;
        }
        var x = try Tensor.init(allocator, &[_]usize{ 1, RankerConfig.HEAD_STATE });
        defer x.deinit();
        @memcpy(x.data[0..RankerConfig.HEAD_STATE], &z);
        try self.head.forward(&x);
        const readout = x.data[0] + x.data[1];
        if (!std.math.isFinite(readout)) return error.NonFinite;
        return readout;
    }

    pub fn headForward(self: *Ranker, z: [4]f32) !f32 {
        return self.headForwardAlloc(z, self.allocator);
    }

    fn fusedFromZ(self: *Ranker, z: [4]f32, allocator: Allocator) !f32 {
        const raw = try self.headForwardAlloc(z, allocator);
        return sigmoidScale(raw * (RankerConfig.MAX_RAW_SCORE / RankerConfig.RAW_FUSE_DIVISOR));
    }

    fn criteriaVector(
        self: *Ranker,
        model: *RSF,
        query_state: *const RSFLatentState,
        query_tokens: []const u32,
        tokens: []const u32,
        position: u64,
        ssi: *const SSI,
        allocator: Allocator,
    ) ![4]f32 {
        const dim = self.model_dim;
        const z_token = if (query_tokens.len > 0)
            try self.scoreSequenceWithQueryAlloc(tokens, query_tokens, ssi, allocator)
        else
            try self.scoreSequenceAlloc(tokens, ssi, allocator);
        const q_blocked = try allocator.alloc(f32, dim * 2);
        defer allocator.free(q_blocked);
        try packStateBlocked(query_state, dim, q_blocked);
        const c_blocked = try allocator.alloc(f32, dim * 2);
        defer allocator.free(c_blocked);
        var cand_log_det: f32 = 0.0;
        try self.candidateLatent(model, tokens, position, ssi, allocator, c_blocked, &cand_log_det);
        const z_latent = cosineClamped(q_blocked, c_blocked);
        var cand_state = try latentStateFromBlocked(allocator, model, c_blocked);
        defer cand_state.deinit();
        const rt = cand_state.roundtripError(model, allocator) catch math.inf(f32);
        const z_recon = if (std.math.isFinite(rt)) math.clamp(1.0 - rt, 0.0, 1.0) else 0.0;
        const z_volume = volumeSurprise(cand_log_det, query_state.log_det);
        const zt = if (std.math.isFinite(z_token)) math.clamp(z_token, 0.0, 1.0) else 0.0;
        return .{ zt, z_latent, z_recon, z_volume };
    }

    fn evaluateCandidate(
        self: *Ranker,
        model: *RSF,
        query_state: *const RSFLatentState,
        query_tokens: []const u32,
        tokens: []const u32,
        position: u64,
        ssi: *const SSI,
        allocator: Allocator,
    ) !CandidateEvaluation {
        const z = try self.criteriaVector(model, query_state, query_tokens, tokens, position, ssi, allocator);
        const score = try self.fusedFromZ(z, allocator);
        return .{
            .score = score,
            .latent_similarity = z[1],
            .reconstruction_confidence = z[2],
            .volume_surprise = z[3],
        };
    }

    fn applyEvaluation(seg: *types.RankedSegment, evaluated: CandidateEvaluation) void {
        seg.score = evaluated.score;
        seg.latent_similarity = evaluated.latent_similarity;
        seg.reconstruction_confidence = evaluated.reconstruction_confidence;
        seg.volume_surprise = evaluated.volume_surprise;
    }

    fn windowHash(self: *const Ranker, window: []const u32) u64 {
        const cap_tokens = @min(window.len, RankerConfig.MAX_NGRAM_ORDER);
        var buf: [RankerConfig.MAX_NGRAM_ORDER * 4]u8 = undefined;
        encodeNgramLE(window[0..cap_tokens], buf[0 .. cap_tokens * 4]);
        return stableHash(buf[0 .. cap_tokens * 4], self.seed);
    }

    fn baseRaw(self: *const Ranker, tokens: []const u32, candidate_pos: u64, exclude_pos: ?u64, retrieved: []const types.RankedSegment, grad_buf: ?[]f32, allocator: Allocator) !f32 {
        if (tokens.len == 0) return 0.0;

        const gram_limit = @min(self.num_ngrams, tokens.len);
        const rates = try allocator.alloc(f32, gram_limit);
        defer allocator.free(rates);

        var ngram_score: f32 = 0.0;
        var weight_sum: f32 = 0.0;
        var gram: usize = 1;
        while (gram <= gram_limit) : (gram += 1) {
            const weight_idx = @min(gram - 1, self.ngram_weights.len - 1);
            const w = self.ngram_weights[weight_idx];
            const cand_windows: usize = tokens.len - gram + 1;
            var ref_count: usize = retrieved.len;
            if (exclude_pos) |ep| {
                ref_count = 0;
                for (retrieved) |seg| {
                    if (seg.position == ep) continue;
                    ref_count += 1;
                }
            }
            const total_compare: usize = cand_windows * @max(ref_count, 1);

            var token_windows = std.AutoHashMap(u64, void).init(allocator);
            defer token_windows.deinit();

            var ti: usize = 0;
            while (ti + gram <= tokens.len) : (ti += 1) {
                try token_windows.put(self.windowHash(tokens[ti .. ti + gram]), {});
            }

            var matches_total: usize = 0;
            for (retrieved) |seg| {
                if (exclude_pos) |ep| {
                    if (seg.position == ep) continue;
                }
                const sim = seg.score;
                if (math.isNan(sim) or math.isInf(sim)) continue;
                var matches: usize = 0;
                var si: usize = 0;
                while (si + gram <= seg.tokens.len) : (si += 1) {
                    if (token_windows.contains(self.windowHash(seg.tokens[si .. si + gram]))) {
                        matches += 1;
                    }
                }
                matches_total += matches;
            }

            var rate: f32 = 0.0;
            if (total_compare > 0) {
                rate = @as(f32, @floatFromInt(matches_total)) / @as(f32, @floatFromInt(total_compare));
            }
            rates[gram - 1] = rate;
            weight_sum += w;
            ngram_score += w * rate;
        }

        const ngram_norm = if (weight_sum > 0.0) ngram_score / weight_sum else 0.0;
        const diversity_score = try self.computeTokenDiversity(tokens, allocator);
        const proximity = self.anchorProximity(candidate_pos, exclude_pos, retrieved);
        const raw = ngram_norm + RankerConfig.DIVERSITY_WEIGHT * diversity_score + RankerConfig.PROXIMITY_WEIGHT * proximity;

        if (grad_buf) |gb| {
            if (weight_sum > 0.0) {
                var gi: usize = 0;
                while (gi < gram_limit) : (gi += 1) {
                    gb[gi] = (rates[gi] - ngram_norm) / weight_sum;
                }
            }
            if (gram_limit < gb.len) {
                @memset(gb[gram_limit..], 0.0);
            }
        }

        return raw;
    }

    fn scoreSequenceRaw(self: *const Ranker, tokens: []const u32, ssi: *const SSI, grad_buf: ?[]f32, allocator: Allocator) !f32 {
        const retrieved = try ssi.retrieveTopK(tokens, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
        defer freeRankedSegments(retrieved, allocator);
        return self.baseRaw(tokens, 0, null, retrieved, grad_buf, allocator);
    }

    fn scoreSequenceAlloc(self: *const Ranker, tokens: []const u32, ssi: *const SSI, allocator: Allocator) !f32 {
        const retrieved = try ssi.retrieveTopK(tokens, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
        defer freeRankedSegments(retrieved, allocator);
        const raw = try self.baseRaw(tokens, 0, null, retrieved, null, allocator);
        return sigmoidScale(raw);
    }

    pub fn scoreSequence(self: *const Ranker, tokens: []const u32, ssi: *const SSI) !f32 {
        return self.scoreSequenceAlloc(tokens, ssi, self.allocator);
    }

    fn combinedScore(self: *const Ranker, tokens: []const u32, query: []const u32, candidate_pos: u64, exclude_pos: ?u64, retrieved: []const types.RankedSegment, grad_buf: ?[]f32, allocator: Allocator) !f32 {
        const raw = try self.baseRaw(tokens, candidate_pos, exclude_pos, retrieved, grad_buf, allocator);
        const base_scaled = sigmoidScale(raw);
        if (query.len == 0) return base_scaled;

        var overlap: f32 = 0.0;
        var jaccard: f32 = 0.0;
        if (tokens.len > 0 and query.len > 0) {
            const m = try self.setMetrics(tokens, query, allocator);
            overlap = @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(@max(@min(m.card_a, m.card_b), 1)));
            const denom = m.card_a + m.card_b - m.intersection;
            jaccard = @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(@max(denom, 1)));
        } else if (tokens.len == 0 and query.len == 0) {
            jaccard = 1.0;
        }

        return math.clamp(base_scaled * RankerConfig.BASE_SCORE_WEIGHT + overlap * RankerConfig.OVERLAP_WEIGHT + jaccard * RankerConfig.JACCARD_WEIGHT, 0.0, 1.0);
    }

    fn scoreSequenceWithQueryAlloc(self: *const Ranker, tokens: []const u32, query: []const u32, ssi: *const SSI, allocator: Allocator) !f32 {
        const retrieved = try ssi.retrieveTopK(query, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
        defer freeRankedSegments(retrieved, allocator);
        return self.combinedScore(tokens, query, 0, null, retrieved, null, allocator);
    }

    pub fn scoreSequenceWithQuery(self: *const Ranker, tokens: []const u32, query: []const u32, ssi: *const SSI) !f32 {
        return self.scoreSequenceWithQueryAlloc(tokens, query, ssi, self.allocator);
    }

    fn computeTokenDiversity(self: *const Ranker, tokens: []const u32, allocator: Allocator) !f32 {
        _ = self;
        if (tokens.len == 0) return 0.0;

        var unique_tokens = std.AutoHashMap(u32, void).init(allocator);
        defer unique_tokens.deinit();

        for (tokens) |token| {
            try unique_tokens.put(token, {});
        }

        const unique_count = unique_tokens.count();
        const diversity = @as(f32, @floatFromInt(unique_count)) / @as(f32, @floatFromInt(tokens.len));

        return diversity;
    }

    fn setMetrics(self: *const Ranker, tokens: []const u32, query: []const u32, allocator: Allocator) !struct { intersection: usize, card_a: usize, card_b: usize } {
        _ = self;
        var set_a = std.AutoHashMap(u32, void).init(allocator);
        defer set_a.deinit();

        for (tokens) |token| {
            try set_a.put(token, {});
        }

        var set_b = std.AutoHashMap(u32, void).init(allocator);
        defer set_b.deinit();

        for (query) |qtoken| {
            try set_b.put(qtoken, {});
        }

        var intersection: usize = 0;
        var it = set_a.keyIterator();
        while (it.next()) |key| {
            if (set_b.contains(key.*)) {
                intersection += 1;
            }
        }

        return .{
            .intersection = intersection,
            .card_a = set_a.count(),
            .card_b = set_b.count(),
        };
    }

    fn computeTokenOverlap(self: *const Ranker, tokens: []const u32, query: []const u32, allocator: Allocator) !f32 {
        if (tokens.len == 0 or query.len == 0) return 0.0;

        const m = try self.setMetrics(tokens, query, allocator);
        const denom = @min(m.card_a, m.card_b);
        if (denom == 0) return 0.0;
        return @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(denom));
    }

    fn computeJaccardSimilarity(self: *const Ranker, tokens: []const u32, query: []const u32, allocator: Allocator) !f32 {
        if (tokens.len == 0 and query.len == 0) return 1.0;
        if (tokens.len == 0 or query.len == 0) return 0.0;

        const m = try self.setMetrics(tokens, query, allocator);
        const denom = m.card_a + m.card_b - m.intersection;
        if (denom == 0) return 0.0;
        return @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(denom));
    }

    fn anchorProximity(self: *const Ranker, candidate_pos: u64, exclude_pos: ?u64, retrieved: []const types.RankedSegment) f32 {
        _ = self;
        if (retrieved.len == 0) return 0.0;

        var anchors: usize = 0;
        var total_dist: u64 = 0;
        for (retrieved) |seg| {
            if (exclude_pos) |ep| {
                if (seg.position == ep) continue;
            }
            if (!seg.anchor) continue;
            anchors += 1;
            const d = if (seg.position >= candidate_pos) seg.position - candidate_pos else candidate_pos - seg.position;
            total_dist += @min(d, RankerConfig.PROXIMITY_SCALE);
        }
        if (anchors == 0) return 0.0;
        const mean = total_dist / @as(u64, @intCast(anchors));
        return 1.0 - math.clamp(@as(f32, @floatFromInt(mean)) / @as(f32, @floatFromInt(RankerConfig.PROXIMITY_SCALE)), 0.0, 1.0);
    }

    fn rearrangeCandidatesByIndices(candidates: []types.RankedSegment, indices: []const usize, scores: []const f32, allocator: Allocator) !void {
        if (candidates.len == 0) return;
        if (indices.len != candidates.len) return error.LengthMismatch;
        if (scores.len != candidates.len) return error.LengthMismatch;

        const copy = try allocator.alloc(types.RankedSegment, candidates.len);
        defer allocator.free(copy);

        for (candidates, 0..) |c, idx| {
            copy[idx] = c;
        }

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const src = indices[i];
            candidates[i] = copy[src];
            candidates[i].score = scores[src];
        }
    }

    fn sortCandidatesByScore(candidates: []types.RankedSegment, scores: []const f32, allocator: Allocator) !void {
        if (candidates.len == 0) return;

        const indices = try allocator.alloc(usize, candidates.len);
        defer allocator.free(indices);

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            indices[i] = i;
        }

        const SortContext = struct {
            scores: []const f32,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const score_a = ctx.scores[a];
                const score_b = ctx.scores[b];
                if (math.isNan(score_a)) return false;
                if (math.isNan(score_b)) return true;
                return score_a > score_b;
            }
        };
        std.mem.sort(usize, indices, SortContext{ .scores = scores }, SortContext.lessThan);

        try rearrangeCandidatesByIndices(candidates, indices, scores, allocator);
    }

    pub fn rankCandidatesWithModel(self: *Ranker, model: *RSF, query_state: *const RSFLatentState, candidates: []types.RankedSegment, ssi: *const SSI, allocator: Allocator) !void {
        try self.requireQuery(model, query_state, ssi);
        if (candidates.len == 0) return;

        const scores = try allocator.alloc(f32, candidates.len);
        defer allocator.free(scores);

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const evaluated = try self.evaluateCandidate(model, query_state, &.{}, candidates[i].tokens, candidates[i].position, ssi, allocator);
            applyEvaluation(&candidates[i], evaluated);
            scores[i] = evaluated.score;
        }

        try sortCandidatesByScore(candidates, scores, allocator);
    }

    pub fn batchScore(self: *Ranker, sequences: []const []const u32, ssi: *const SSI, model: *RSF, query_state: *const RSFLatentState, allocator: Allocator) ![]f32 {
        try self.requireQuery(model, query_state, ssi);
        if (sequences.len == 0) return allocator.alloc(f32, 0);

        const batch_size = sequences.len;
        const scores = try allocator.alloc(f32, batch_size);
        errdefer allocator.free(scores);

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const evaluated = try self.evaluateCandidate(model, query_state, &.{}, sequences[b], 0, ssi, allocator);
            scores[b] = evaluated.score;
        }
        return scores;
    }

    pub fn topKHeap(self: *Ranker, ssi: *const SSI, query: []const u32, query_state: *const RSFLatentState, model: *RSF, k: usize, allocator: Allocator) ![]types.RankedSegment {
        try self.requireQuery(model, query_state, ssi);
        if (k == 0) return allocator.alloc(types.RankedSegment, 0);

        const retrieval_count = @max(k, RankerConfig.DEFAULT_TOP_N_RETRIEVAL);

        var heap = std.PriorityQueue(types.RankedSegment, void, RankedSegmentComparator.lessThan).init(allocator, {});
        defer {
            while (heap.removeOrNull()) |item| {
                var m = item;
                m.deinit(allocator);
            }
            heap.deinit();
        }

        const dim = self.model_dim;
        const q_blocked = try allocator.alloc(f32, dim * 2);
        defer allocator.free(q_blocked);
        try packStateBlocked(query_state, dim, q_blocked);

        const candidates = ssi.retrieveTopKLatent(q_blocked, query_state.log_det, retrieval_count, allocator) catch |err| switch (err) {
            error.RSFModelMismatch => return err,
            else => try ssi.retrieveTopK(query, retrieval_count, allocator),
        };
        defer freeRankedSegments(candidates, allocator);

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const cand = candidates[i];
            const evaluated = try self.evaluateCandidate(model, query_state, query, cand.tokens, cand.position, ssi, allocator);
            const score = evaluated.score;

            if (math.isNan(score) or math.isInf(score)) continue;

            if (heap.count() < k) {
                var ranked = try types.RankedSegment.init(allocator, cand.tokens, score, cand.position, cand.anchor);
                applyEvaluation(&ranked, evaluated);
                heap.add(ranked) catch |err| {
                    ranked.deinit(allocator);
                    return err;
                };
            } else if (heap.peek()) |top| {
                if (math.isNan(top.score) or score > top.score) {
                    var removed = heap.remove();
                    removed.deinit(allocator);
                    var ranked = try types.RankedSegment.init(allocator, cand.tokens, score, cand.position, cand.anchor);
                    applyEvaluation(&ranked, evaluated);
                    heap.add(ranked) catch |err| {
                        ranked.deinit(allocator);
                        return err;
                    };
                }
            }
        }

        const result_count = heap.count();
        const top_n = try allocator.alloc(types.RankedSegment, result_count);
        var n_placed: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < n_placed) : (j += 1) {
                top_n[result_count - 1 - j].deinit(allocator);
            }
            allocator.free(top_n);
        }

        var idx: usize = result_count;
        while (heap.removeOrNull()) |item| {
            if (idx > 0) {
                idx -= 1;
                top_n[idx] = item;
                n_placed += 1;
            } else {
                var mutable_item = item;
                mutable_item.deinit(allocator);
            }
        }

        return top_n;
    }

    pub fn minHashSignature(self: *const Ranker, tokens: []const u32) ![]u64 {
        if (tokens.len == 0) {
            const sig = try self.allocator.alloc(u64, self.num_hash_functions);
            @memset(sig, std.math.maxInt(u64));
            return sig;
        }

        const sig = try self.allocator.alloc(u64, self.num_hash_functions);
        errdefer self.allocator.free(sig);

        const token_hashes = try self.allocator.alloc(u64, tokens.len);
        defer self.allocator.free(token_hashes);

        var ti: usize = 0;
        while (ti < tokens.len) : (ti += 1) {
            const le_bytes = tokenToLEBytes(tokens[ti]);
            token_hashes[ti] = stableHash(&le_bytes, self.seed);
        }

        var h: usize = 0;
        while (h < self.num_hash_functions) : (h += 1) {
            var min_hash: u64 = std.math.maxInt(u64);
            const seed_a = self.lsh_hash_params[h * 2];
            const seed_b = self.lsh_hash_params[h * 2 + 1];
            for (token_hashes) |th| {
                const hash_val = (th ^ seed_a) *% RankerConfig.HASH_SEED_MULTIPLIER_B +% seed_b;
                if (hash_val < min_hash) {
                    min_hash = hash_val;
                }
            }
            sig[h] = min_hash;
        }
        return sig;
    }

    pub fn jaccardSimilarityFromSignatures(sig1: []const u64, sig2: []const u64) f32 {
        if (sig1.len != sig2.len) return 0.0;
        if (sig1.len == 0) return 0.0;

        var matches: usize = 0;
        var i: usize = 0;
        while (i < sig1.len) : (i += 1) {
            if (sig1[i] == sig2[i]) {
                matches += 1;
            }
        }
        return @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(sig1.len));
    }

    pub fn minHashBitmask(self: *const Ranker, tokens: []const u32) ![]u64 {
        const words = (self.num_hash_functions + 63) / 64;
        const mask = try self.allocator.alloc(u64, words);
        errdefer self.allocator.free(mask);
        @memset(mask, 0);

        const sig = try self.minHashSignature(tokens);
        defer self.allocator.free(sig);

        var h: usize = 0;
        while (h < self.num_hash_functions) : (h += 1) {
            if ((sig[h] & 1) != 0) {
                mask[h / 64] |= @as(u64, 1) << @intCast(h % 64);
            }
        }
        return mask;
    }

    pub fn jaccardFromBitmasks(mask1: []const u64, mask2: []const u64, valid_bits: usize) f32 {
        if (mask1.len != mask2.len) return 0.0;
        if (mask1.len == 0 or valid_bits == 0) return 0.0;

        const vector_len: usize = 4;
        const full_words = valid_bits / 64;
        var matches: usize = 0;
        var w: usize = 0;
        while (w + vector_len <= full_words) : (w += vector_len) {
            const a: @Vector(vector_len, u64) = mask1[w..][0..vector_len].*;
            const b: @Vector(vector_len, u64) = mask2[w..][0..vector_len].*;
            const agree = ~(a ^ b);
            var v: usize = 0;
            while (v < vector_len) : (v += 1) {
                matches += @popCount(agree[v]);
            }
        }
        while (w < full_words) : (w += 1) {
            matches += @popCount(~(mask1[w] ^ mask2[w]));
        }
        if (full_words < mask1.len) {
            const rem = valid_bits % 64;
            const valid_mask: u64 = if (rem == 0) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(rem)) - 1;
            matches += @popCount((~(mask1[full_words] ^ mask2[full_words])) & valid_mask);
        }
        const denom = valid_bits;
        if (denom == 0) return 0.0;
        const ratio = @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(denom));
        const estimate = 2.0 * ratio - 1.0;
        return math.clamp(estimate, 0.0, 1.0);
    }

    pub fn jaccardSignatureBitmask(self: *const Ranker, tokens1: []const u32, tokens2: []const u32) !f32 {
        const mask1 = try self.minHashBitmask(tokens1);
        defer self.allocator.free(mask1);
        const mask2 = try self.minHashBitmask(tokens2);
        defer self.allocator.free(mask2);
        return jaccardFromBitmasks(mask1, mask2, self.num_hash_functions);
    }

    pub fn estimateJaccard(set1: BitSet, set2: BitSet) f32 {
        const len1 = set1.bits.len;
        const len2 = set2.bits.len;
        const max_words = @max(len1, len2);

        if (max_words == 0) return 1.0;

        var intersect: usize = 0;
        var union_count: usize = 0;
        var i: usize = 0;
        while (i < max_words) : (i += 1) {
            const w1: u64 = if (i < len1) set1.bits[i] else 0;
            const w2: u64 = if (i < len2) set2.bits[i] else 0;
            intersect += @popCount(w1 & w2);
            union_count += @popCount(w1 | w2);
        }
        return if (union_count == 0) 1.0 else @as(f32, @floatFromInt(intersect)) / @as(f32, @floatFromInt(union_count));
    }

    pub fn weightedAverage(scores: []const f32, weights: []const f32) !f32 {
        if (scores.len != weights.len) return error.LengthMismatch;
        if (scores.len == 0) return 0.0;

        var num: f32 = 0.0;
        var den: f32 = 0.0;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            const s = scores[i];
            const w = weights[i];

            if (math.isNan(s) or math.isNan(w)) continue;
            if (math.isInf(w) or w < 0.0) return error.InvalidParameter;
            if (math.isInf(s)) continue;

            num += s * w;
            den += w;
        }

        if (den == 0.0) return 0.0;
        return num / den;
    }

    pub fn exponentialDecay(scores: []f32, decay_factor: f32) !void {
        if (scores.len == 0) return;
        if (decay_factor <= 0.0 or decay_factor >= 1.0) return error.InvalidParameter;

        var current_decay: f32 = 1.0;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            if (!math.isNan(scores[i]) and !math.isInf(scores[i])) {
                scores[i] *= current_decay;
            }
            current_decay *= decay_factor;
        }
    }

    pub fn normalizeScores(self: *const Ranker, scores: []f32) void {
        _ = self;
        normalizeScoresStatic(scores);
    }

    fn normalizeScoresStatic(scores: []f32) void {
        if (scores.len == 0) return;

        var min_score: f32 = math.inf(f32);
        var max_score: f32 = -math.inf(f32);
        var valid_count: usize = 0;

        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            const s = scores[i];
            if (math.isNan(s) or math.isInf(s)) continue;
            valid_count += 1;
            if (s < min_score) min_score = s;
            if (s > max_score) max_score = s;
        }

        if (valid_count == 0) {
            i = 0;
            while (i < scores.len) : (i += 1) {
                scores[i] = 0.0;
            }
            return;
        }
        if (max_score == min_score) {
            i = 0;
            while (i < scores.len) : (i += 1) {
                if (!math.isNan(scores[i]) and !math.isInf(scores[i])) {
                    scores[i] = 0.5;
                } else {
                    scores[i] = 0.0;
                }
            }
            return;
        }

        const range = max_score - min_score;
        i = 0;
        while (i < scores.len) : (i += 1) {
            if (math.isNan(scores[i]) or math.isInf(scores[i])) {
                scores[i] = 0.0;
            } else {
                scores[i] = (scores[i] - min_score) / range;
            }
        }
    }

    pub fn rankByMultipleCriteria(self: *const Ranker, candidates: []types.RankedSegment, criteria: []const []const f32, weights: []const f32, allocator: Allocator) !void {
        _ = self;
        if (candidates.len == 0) return;
        if (criteria.len == 0) return;
        if (weights.len == 0) return;

        const num_cand = candidates.len;
        const num_crit = @min(criteria.len, weights.len);

        var cr: usize = 0;
        while (cr < num_crit) : (cr += 1) {
            if (criteria[cr].len < num_cand) return error.LengthMismatch;
        }

        const combined = try allocator.alloc(f32, num_cand);
        defer allocator.free(combined);

        var c: usize = 0;
        while (c < num_cand) : (c += 1) {
            var crit_score: f32 = 0.0;
            cr = 0;
            while (cr < num_crit) : (cr += 1) {
                const score_val = criteria[cr][c];
                const weight_val = weights[cr];
                if (!math.isNan(score_val) and !math.isNan(weight_val) and !math.isInf(score_val) and !math.isInf(weight_val)) {
                    crit_score += score_val * weight_val;
                }
            }
            combined[c] = crit_score;
        }

        try sortCandidatesByScore(candidates, combined, allocator);
    }

    pub fn streamingRank(self: *Ranker, reader: anytype, ssi: *const SSI, model: *RSF, query_state: *const RSFLatentState, k: usize, allocator: Allocator) ![]types.RankedSegment {
        try self.requireQuery(model, query_state, ssi);
        if (k == 0) return allocator.alloc(types.RankedSegment, 0);

        var rolling_buffer = std.ArrayList(u32).init(allocator);
        defer rolling_buffer.deinit();

        var heap = std.PriorityQueue(types.RankedSegment, void, RankedSegmentComparator.lessThan).init(allocator, {});
        defer {
            while (heap.removeOrNull()) |item| {
                var m = item;
                m.deinit(allocator);
            }
            heap.deinit();
        }

        var leftover_bytes: [3]u8 = undefined;
        var leftover_len: usize = 0;
        var position: u64 = 0;
        var read_buf: [RankerConfig.STREAMING_BUFFER_SIZE * @sizeOf(u32)]u8 = undefined;

        while (true) {
            const bytes_read = reader.read(&read_buf) catch |err| return err;
            if (bytes_read == 0) break;

            var combined_buf: []u8 = undefined;
            var combined_len: usize = 0;
            var combined_alloc: ?[]u8 = null;
            defer {
                if (combined_alloc) |ca| allocator.free(ca);
            }

            if (leftover_len > 0) {
                combined_len = leftover_len + bytes_read;
                combined_alloc = try allocator.alloc(u8, combined_len);
                combined_buf = combined_alloc.?;
                var ci: usize = 0;
                while (ci < leftover_len) : (ci += 1) {
                    combined_buf[ci] = leftover_bytes[ci];
                }
                var ri: usize = 0;
                while (ri < bytes_read) : (ri += 1) {
                    combined_buf[leftover_len + ri] = read_buf[ri];
                }
                leftover_len = 0;
            } else {
                combined_buf = read_buf[0..bytes_read];
                combined_len = bytes_read;
            }

            const full_tokens = combined_len / @sizeOf(u32);
            const remainder = combined_len % @sizeOf(u32);

            if (remainder > 0) {
                var ri: usize = 0;
                while (ri < remainder) : (ri += 1) {
                    leftover_bytes[ri] = combined_buf[full_tokens * @sizeOf(u32) + ri];
                }
                leftover_len = remainder;
            }

            var ti: usize = 0;
            while (ti < full_tokens) : (ti += 1) {
                const offset = ti * @sizeOf(u32);
                var token_bytes: [4]u8 = undefined;
                token_bytes[0] = combined_buf[offset + 0];
                token_bytes[1] = combined_buf[offset + 1];
                token_bytes[2] = combined_buf[offset + 2];
                token_bytes[3] = combined_buf[offset + 3];
                const token = mem.readInt(u32, &token_bytes, .little);
                try rolling_buffer.append(token);
            }

            while (rolling_buffer.items.len >= RankerConfig.STREAMING_WINDOW_SIZE) {
                const window = rolling_buffer.items[0..RankerConfig.STREAMING_WINDOW_SIZE];
                const evaluated = try self.evaluateCandidate(model, query_state, &.{}, window, position, ssi, allocator);
                const score = evaluated.score;

                if (!math.isNan(score) and !math.isInf(score)) {
                    if (heap.count() < k) {
                        var seg = try types.RankedSegment.init(allocator, window, score, position, false);
                        applyEvaluation(&seg, evaluated);
                        heap.add(seg) catch |err| {
                            seg.deinit(allocator);
                            return err;
                        };
                    } else if (heap.peek()) |top| {
                        if (math.isNan(top.score) or score > top.score) {
                            var removed = heap.remove();
                            removed.deinit(allocator);
                            var seg = try types.RankedSegment.init(allocator, window, score, position, false);
                        applyEvaluation(&seg, evaluated);
                            heap.add(seg) catch |err| {
                                seg.deinit(allocator);
                                return err;
                            };
                        }
                    }
                }

                const shift = @min(rolling_buffer.items.len, RankerConfig.STREAMING_WINDOW_SIZE / 2);
                const remaining = rolling_buffer.items.len - shift;
                if (remaining > 0) {
                    std.mem.copyForwards(u32, rolling_buffer.items[0..remaining], rolling_buffer.items[shift..rolling_buffer.items.len]);
                }
                rolling_buffer.shrinkRetainingCapacity(remaining);
                position += shift;
            }
        }

        if (leftover_len > 0) return error.InvalidData;

        if (rolling_buffer.items.len > 0) {
            const tail = rolling_buffer.items;
            const evaluated = try self.evaluateCandidate(model, query_state, &.{}, tail, position, ssi, allocator);
            const score = evaluated.score;
            if (!math.isNan(score) and !math.isInf(score)) {
                if (heap.count() < k) {
                    var seg = try types.RankedSegment.init(allocator, tail, score, position, false);
                    applyEvaluation(&seg, evaluated);
                    heap.add(seg) catch |err| {
                        seg.deinit(allocator);
                        return err;
                    };
                } else if (heap.peek()) |top| {
                    if (math.isNan(top.score) or score > top.score) {
                        var removed = heap.remove();
                        removed.deinit(allocator);
                        var seg = try types.RankedSegment.init(allocator, tail, score, position, false);
                    applyEvaluation(&seg, evaluated);
                        heap.add(seg) catch |err| {
                            seg.deinit(allocator);
                            return err;
                        };
                    }
                }
            }
        }

        const result_count = heap.count();
        const result = try allocator.alloc(types.RankedSegment, result_count);
        var n_placed: usize = 0;
        errdefer {
            var ei: usize = 0;
            while (ei < n_placed) : (ei += 1) {
                result[result_count - 1 - ei].deinit(allocator);
            }
            allocator.free(result);
        }

        var idx: usize = result_count;
        while (heap.removeOrNull()) |item| {
            if (idx > 0) {
                idx -= 1;
                result[idx] = item;
                n_placed += 1;
            } else {
                var m = item;
                m.deinit(allocator);
            }
        }

        return result;
    }

    pub fn parallelScore(self: *Ranker, sequences: []const []const u32, ssi: *const SSI, model: *RSF, query_state: *const RSFLatentState, num_threads: usize) ![]f32 {
        try self.requireQuery(model, query_state, ssi);
        if (sequences.len == 0) return self.allocator.alloc(f32, 0);

        const scores = try self.allocator.alloc(f32, sequences.len);
        errdefer self.allocator.free(scores);

        if (num_threads <= 1 or sequences.len <= 1) {
            var i: usize = 0;
            while (i < sequences.len) : (i += 1) {
                const evaluated = try self.evaluateCandidate(model, query_state, &.{}, sequences[i], 0, ssi, self.allocator);
                scores[i] = evaluated.score;
            }
            return scores;
        }

        const cpu_count = std.Thread.getCpuCount() catch @as(usize, 1);
        const effective_threads = @min(@min(num_threads, sequences.len), cpu_count);
        const chunk_size = sequences.len / effective_threads;
        const remainder_count = sequences.len % effective_threads;

        const ThreadContext = struct {
            ranker: *Ranker,
            seqs: []const []const u32,
            ssi_ptr: *const SSI,
            model: *RSF,
            query_state: *const RSFLatentState,
            out: []f32,
            start: usize,
            end: usize,
            err_flag: bool,
        };

        const contexts = try self.allocator.alloc(ThreadContext, effective_threads);
        defer self.allocator.free(contexts);

        const threads = try self.allocator.alloc(std.Thread, effective_threads);
        defer self.allocator.free(threads);

        const thread_spawned = try self.allocator.alloc(bool, effective_threads);
        defer self.allocator.free(thread_spawned);
        @memset(thread_spawned, false);

        var offset: usize = 0;
        var t: usize = 0;
        while (t < effective_threads) : (t += 1) {
            const this_chunk = chunk_size + @as(usize, if (t < remainder_count) 1 else 0);
            contexts[t] = .{
                .ranker = self,
                .seqs = sequences,
                .ssi_ptr = ssi,
                .model = model,
                .query_state = query_state,
                .out = scores,
                .start = offset,
                .end = offset + this_chunk,
                .err_flag = false,
            };
            offset += this_chunk;
        }

        t = 0;
        while (t < effective_threads) : (t += 1) {
            threads[t] = std.Thread.spawn(.{}, struct {
                fn work(ctx: *ThreadContext) void {
                    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                    defer arena.deinit();
                    var si: usize = ctx.start;
                    while (si < ctx.end) : (si += 1) {
                        _ = arena.reset(.retain_capacity);
                        const evaluated = ctx.ranker.evaluateCandidate(ctx.model, ctx.query_state, &.{}, ctx.seqs[si], 0, ctx.ssi_ptr, arena.allocator()) catch {
                            ctx.err_flag = true;
                            return;
                        };
                        ctx.out[si] = evaluated.score;
                    }
                }
            }.work, .{&contexts[t]}) catch {
                var si: usize = contexts[t].start;
                while (si < contexts[t].end) : (si += 1) {
                    const evaluated = self.evaluateCandidate(model, query_state, &.{}, sequences[si], 0, ssi, self.allocator) catch {
                        contexts[t].err_flag = true;
                        break;
                    };
                    scores[si] = evaluated.score;
                }
                thread_spawned[t] = false;
                continue;
            };
            thread_spawned[t] = true;
        }

        t = 0;
        while (t < effective_threads) : (t += 1) {
            if (thread_spawned[t]) {
                threads[t].join();
            }
        }

        var had_error = false;
        t = 0;
        while (t < effective_threads) : (t += 1) {
            if (contexts[t].err_flag) had_error = true;
        }

        if (had_error) {
            return error.ScoringFailed;
        }

        return scores;
    }

    fn logisticPairGrads(score_pos: f32, score_neg: f32) struct { d_pos: f32, d_neg: f32 } {
        const diff = score_pos - score_neg;
        var p: f32 = undefined;
        if (diff >= 0.0) {
            const e = @exp(-diff);
            p = e / (1.0 + e);
        } else {
            const e = @exp(diff);
            p = 1.0 / (1.0 + e);
        }
        if (!std.math.isFinite(p)) p = 0.0;
        return .{ .d_pos = -p, .d_neg = p };
    }

    fn backwardHeadReadout(self: *Ranker, z: [4]f32, d_readout: f32, allocator: Allocator) !void {
        if (!std.math.isFinite(d_readout) or d_readout == 0.0) return;
        var input = try Tensor.init(allocator, &[_]usize{ 1, RankerConfig.HEAD_STATE });
        defer input.deinit();
        var output = try Tensor.init(allocator, &[_]usize{ 1, RankerConfig.HEAD_STATE });
        defer output.deinit();
        @memcpy(input.data[0..RankerConfig.HEAD_STATE], &z);
        @memcpy(output.data[0..RankerConfig.HEAD_STATE], &z);
        try self.head.forward(&output);
        var grad_out = try Tensor.init(allocator, &[_]usize{ 1, RankerConfig.HEAD_STATE });
        defer grad_out.deinit();
        @memset(grad_out.data, 0.0);
        grad_out.data[0] = d_readout;
        grad_out.data[1] = d_readout;
        var grad_in = try Tensor.init(allocator, &[_]usize{ 1, RankerConfig.HEAD_STATE });
        defer grad_in.deinit();
        try self.head.backward(&grad_out, &input, &output, &grad_in);
    }

    pub fn trainHead(self: *Ranker, training_data: []const []const u32, labels: []const f32, ssi: *const SSI, model: *RSF, epochs: usize) !void {
        if (training_data.len == 0 or labels.len == 0) return error.InvalidParameter;
        if (training_data.len != labels.len) return error.LengthMismatch;
        try self.requireModel(model, ssi);

        const n = training_data.len;
        const zs = try self.allocator.alloc([4]f32, n);
        defer self.allocator.free(zs);

        var i: usize = 0;
        while (i < n) : (i += 1) {
            var qstate = try self.queryStateFromTokens(model, training_data[i], self.allocator);
            defer qstate.deinit();
            const retrieved = try ssi.retrieveTopK(training_data[i], RankerConfig.SCORE_RETRIEVAL_LIMIT, self.allocator);
            defer freeRankedSegments(retrieved, self.allocator);
            if (retrieved.len > 0) {
                zs[i] = try self.criteriaVector(model, &qstate, training_data[i], retrieved[0].tokens, retrieved[0].position, ssi, self.allocator);
            } else {
                zs[i] = try self.criteriaVector(model, &qstate, training_data[i], training_data[i], 0, ssi, self.allocator);
            }
        }

        const order = try self.allocator.alloc(usize, n);
        defer self.allocator.free(order);
        i = 0;
        while (i < n) : (i += 1) {
            order[i] = i;
        }
        const SortCtx = struct {
            labels: []const f32,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const la = ctx.labels[a];
                const lb = ctx.labels[b];
                if (math.isNan(la)) return false;
                if (math.isNan(lb)) return true;
                return la < lb;
            }
        };
        std.mem.sort(usize, order, SortCtx{ .labels = labels }, SortCtx.lessThan);

        const scale = RankerConfig.MAX_RAW_SCORE / RankerConfig.RAW_FUSE_DIVISOR;
        var epoch: usize = 0;
        while (epoch < epochs) : (epoch += 1) {
            try self.head.zeroGradients();
            try self.head.ensureGradients();
            var pair_count: usize = 0;
            var p: usize = 0;
            while (p + 1 < n) : (p += 1) {
                const ineg = order[p];
                const ipos = order[p + 1];
                const lneg = labels[ineg];
                const lpos = labels[ipos];
                if (math.isNan(lneg) or math.isNan(lpos) or math.isInf(lneg) or math.isInf(lpos)) continue;
                if (!(lpos > lneg)) continue;
                const score_neg = self.fusedFromZ(zs[ineg], self.allocator) catch continue;
                const score_pos = self.fusedFromZ(zs[ipos], self.allocator) catch continue;
                if (math.isNan(score_neg) or math.isNan(score_pos) or math.isInf(score_neg) or math.isInf(score_pos)) continue;
                const grads = logisticPairGrads(score_pos, score_neg);
                const d_raw_pos = grads.d_pos * sigmoidDerivative(score_pos) * scale;
                const d_raw_neg = grads.d_neg * sigmoidDerivative(score_neg) * scale;
                try self.backwardHeadReadout(zs[ipos], d_raw_pos, self.allocator);
                try self.backwardHeadReadout(zs[ineg], d_raw_neg, self.allocator);
                pair_count += 1;
            }
            if (pair_count == 0) continue;
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(pair_count));
            try self.head.scaleLayerGradients(inv);
            _ = try self.head_sfd.step(&self.head, RankerConfig.LEARNING_RATE);
        }
    }

    pub fn exportModel(self: *const Ranker, path: []const u8) !void {
        const file = try core_io.createFilePath(path, .{});
        defer file.close();
        const writer = file.writer();
        try writer.writeInt(u8, RankerConfig.EXPORT_VERSION, .little);
        try writer.writeInt(u64, @intCast(self.ngram_weights.len), .little);
        try writer.writeInt(u64, @intCast(self.num_ngrams), .little);
        var i: usize = 0;
        while (i < self.ngram_weights.len) : (i += 1) {
            const bits: u32 = @bitCast(self.ngram_weights[i]);
            try writer.writeInt(u32, bits, .little);
        }
        try writer.writeInt(u64, @intCast(self.num_hash_functions), .little);
        i = 0;
        while (i < self.lsh_hash_params.len) : (i += 1) {
            try writer.writeInt(u64, self.lsh_hash_params[i], .little);
        }
        try writer.writeInt(u64, self.seed, .little);
        try writer.writeInt(u64, self.model_id, .little);
        try writer.writeInt(u64, @intCast(self.model_dim), .little);
        try writer.writeByte(if (self.model_global_diffusion) 1 else 0);
        const head_gd = self.head.globalDiffusionEnabled() catch false;
        try writer.writeByte(if (head_gd) 1 else 0);
        const head_dim = self.head.dim() catch RankerConfig.HEAD_DIM;
        const head_layers = self.head.layerCount() catch RankerConfig.HEAD_LAYERS;
        try writer.writeInt(u64, @intCast(head_dim), .little);
        try writer.writeInt(u64, @intCast(head_layers), .little);
        const plen = head_pair_len;
        const s_buf = try self.allocator.alloc(f32, plen);
        defer self.allocator.free(s_buf);
        const t_buf = try self.allocator.alloc(f32, plen);
        defer self.allocator.free(t_buf);
        try self.head.readLayerWeights(0, s_buf, t_buf);
        try writer.writeInt(u64, @intCast(plen), .little);
        i = 0;
        while (i < plen) : (i += 1) {
            const bits: u32 = @bitCast(s_buf[i]);
            try writer.writeInt(u32, bits, .little);
        }
        try writer.writeInt(u64, @intCast(plen), .little);
        i = 0;
        while (i < plen) : (i += 1) {
            const bits: u32 = @bitCast(t_buf[i]);
            try writer.writeInt(u32, bits, .little);
        }
        const flat = try self.head_sfd.exportFlatState(self.allocator);
        defer self.allocator.free(flat);
        try writer.writeInt(u64, @intCast(flat.len), .little);
        i = 0;
        while (i < flat.len) : (i += 1) {
            const bits: u32 = @bitCast(flat[i]);
            try writer.writeInt(u32, bits, .little);
        }
    }

    pub fn importModel(self: *Ranker, path: []const u8) !void {
        const file = try core_io.openFilePath(path, .{});
        defer file.close();
        const reader = file.reader();
        const version = try reader.readInt(u8, .little);
        if (version != RankerConfig.EXPORT_VERSION and version != RankerConfig.EXPORT_VERSION_LEGACY) return error.InvalidVersion;

        const num_w = try reader.readInt(u64, .little);
        const num_ng = try reader.readInt(u64, .little);

        if (num_w == 0 or num_ng == 0) return error.InvalidParameter;
        if (num_w != num_ng) return error.InvalidParameter;
        if (num_w > std.math.maxInt(usize)) return error.InvalidParameter;
        if (num_w > RankerConfig.MAX_NGRAM_ORDER) return error.InvalidParameter;

        const num_w_usize: usize = @intCast(num_w);
        const num_ng_usize: usize = @intCast(num_ng);

        const new_weights = try self.allocator.alloc(f32, num_w_usize);
        errdefer self.allocator.free(new_weights);

        var i: usize = 0;
        while (i < num_w_usize) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            var weight: f32 = @bitCast(bits);
            if (math.isNan(weight) or math.isInf(weight)) weight = 0.0;
            new_weights[i] = weight;
        }

        const num_h = try reader.readInt(u64, .little);
        if (num_h == 0) return error.InvalidParameter;
        if (num_h > std.math.maxInt(usize) / 2) return error.InvalidParameter;

        const num_h_usize: usize = @intCast(num_h);

        const new_params = try self.allocator.alloc(u64, num_h_usize * 2);
        errdefer self.allocator.free(new_params);

        i = 0;
        while (i < new_params.len) : (i += 1) {
            new_params[i] = try reader.readInt(u64, .little);
        }
        const new_seed = try reader.readInt(u64, .little);

        self.allocator.free(self.ngram_weights);
        self.allocator.free(self.lsh_hash_params);
        self.ngram_weights = new_weights;
        self.lsh_hash_params = new_params;
        self.num_ngrams = num_ng_usize;
        self.num_hash_functions = num_h_usize;
        self.seed = new_seed;

        if (version == RankerConfig.EXPORT_VERSION_LEGACY) {
            try self.resetHead();
            return;
        }

        self.model_id = try reader.readInt(u64, .little);
        const model_dim_u64 = try reader.readInt(u64, .little);
        if (model_dim_u64 == 0 or model_dim_u64 > std.math.maxInt(usize)) return error.InvalidParameter;
        self.model_dim = @intCast(model_dim_u64);
        const model_gd = try reader.readByte();
        self.model_global_diffusion = model_gd != 0;
        const head_gd = try reader.readByte();
        _ = head_gd;
        const head_dim_u64 = try reader.readInt(u64, .little);
        const head_layers_u64 = try reader.readInt(u64, .little);
        if (head_dim_u64 != RankerConfig.HEAD_DIM) return error.InvalidParameter;
        if (head_layers_u64 != RankerConfig.HEAD_LAYERS) return error.InvalidParameter;

        const s_len = try reader.readInt(u64, .little);
        const plen = head_pair_len;
        if (s_len != plen) return error.InvalidParameter;
        const s_buf = try self.allocator.alloc(f32, plen);
        defer self.allocator.free(s_buf);
        i = 0;
        while (i < plen) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            var v: f32 = @bitCast(bits);
            if (math.isNan(v) or math.isInf(v)) v = 0.0;
            s_buf[i] = v;
        }
        const t_len = try reader.readInt(u64, .little);
        if (t_len != plen) return error.InvalidParameter;
        const t_buf = try self.allocator.alloc(f32, plen);
        defer self.allocator.free(t_buf);
        i = 0;
        while (i < plen) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            var v: f32 = @bitCast(bits);
            if (math.isNan(v) or math.isInf(v)) v = 0.0;
            t_buf[i] = v;
        }
        try self.head.writeLayerWeights(0, s_buf, t_buf);

        const flat_len_u64 = try reader.readInt(u64, .little);
        if (flat_len_u64 > std.math.maxInt(usize)) return error.InvalidParameter;
        const flat_len: usize = @intCast(flat_len_u64);
        const flat = try self.allocator.alloc(f32, flat_len);
        defer self.allocator.free(flat);
        i = 0;
        while (i < flat_len) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            var v: f32 = @bitCast(bits);
            if (math.isNan(v) or math.isInf(v)) v = 0.0;
            flat[i] = v;
        }
        self.head_sfd.importFlatState(flat) catch {
            try self.head.writeLayerWeights(0, s_buf, t_buf);
            self.head_sfd.deinit();
            self.head_sfd = try SFD.init(self.allocator, &self.head);
            return;
        };
    }

};
