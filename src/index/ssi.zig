const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const Error = types.Error;

pub const SSIScoringConfig = struct {
    pub const cosine_weight: f32 = 0.7;
    pub const volume_weight: f32 = 0.3;
    pub const volume_kappa: f32 = 1.0;
    pub const cosine_eps: f32 = 1.0e-12;
};

pub const SSI = struct {
    root: ?*Node,
    allocator: Allocator,
    height: usize = 0,
    size: usize = 0,
    max_height: usize = 6,
    dim: usize = 0,
    model_id: u64 = 0,
    global_diffusion: bool = false,
    identity_set: bool = false,

    pub const bucket_width: usize = 6;
    pub const bucket_count: usize = 1 << bucket_width;
    const tensor_width: usize = 134;
    const format_magic: u32 = 0x32535349;
    const format_version: u32 = 2;

    const Segment = struct {
        tokens: []u32,
        position: u64,
        score: f32,
        anchor_hash: u64,
        signature: u64,
        latent: []f32,
        log_det: f32,
        latent_version: u32,

        pub fn init(
            allocator: Allocator,
            tokens: []const u32,
            position: u64,
            score: f32,
            anchor_hash: u64,
            latent: []const f32,
            log_det: f32,
            latent_version: u32,
        ) !Segment {
            const token_copy = try allocator.dupe(u32, tokens);
            errdefer allocator.free(token_copy);
            const latent_copy = try allocator.dupe(f32, latent);
            errdefer allocator.free(latent_copy);
            const token_sig = computeMinHashSignature(tokens);
            const latent_sig = if (latent_version == 1) types.rsfLatentSimHash(latent_copy) else 0;
            const signature = if (latent_version == 1) mixHash(token_sig, latent_sig) else token_sig;
            return .{
                .tokens = token_copy,
                .position = position,
                .score = score,
                .anchor_hash = anchor_hash,
                .signature = signature,
                .latent = latent_copy,
                .log_det = log_det,
                .latent_version = latent_version,
            };
        }

        pub fn deinit(self: *Segment, allocator: Allocator) void {
            allocator.free(self.tokens);
            self.tokens = &.{};
            if (self.latent.len > 0) {
                allocator.free(self.latent);
            }
            self.latent = &.{};
        }

        pub fn tokenHash(self: *const Segment) u64 {
            return hashTokens(self.tokens);
        }

        pub fn fullHash(self: *const Segment) u64 {
            var state: u64 = 0;
            state = mixHash(state, self.position);
            state = mixHash(state, @as(u64, scoreBits(self.score)));
            state = mixHash(state, self.anchor_hash);
            state = mixHash(state, self.signature);
            state = mixHash(state, @as(u64, @intCast(self.tokens.len)));
            for (self.tokens) |tok| {
                state = mixHash(state, tok);
            }
            state = mixHash(state, self.latent_version);
            state = mixHash(state, @as(u64, scoreBits(self.log_det)));
            state = mixHash(state, @as(u64, @intCast(self.latent.len)));
            for (self.latent) |value| {
                state = mixHash(state, @as(u64, scoreBits(value)));
            }
            return state;
        }
    };

    const CollisionNode = struct {
        seg: Segment,
        next: ?*CollisionNode,
    };

    const Node = struct {
        hash: u64,
        children: ?[]?*Node,
        segment: ?Segment,
        collision_chain: ?*CollisionNode,
        height: usize,
        is_leaf: bool,

        pub fn init(allocator: Allocator, height: usize) !Node {
            var children: ?[]?*Node = null;
            if (height > 0) {
                const allocated = try allocator.alloc(?*Node, bucket_count);
                @memset(allocated, null);
                children = allocated;
            }
            return .{
                .hash = 0,
                .children = children,
                .segment = null,
                .collision_chain = null,
                .height = height,
                .is_leaf = height == 0,
            };
        }

        pub fn deinit(self: *Node, allocator: Allocator) void {
            if (self.segment) |*seg| {
                seg.deinit(allocator);
                self.segment = null;
            }
            var chain = self.collision_chain;
            while (chain) |c| {
                const next = c.next;
                c.seg.deinit(allocator);
                allocator.destroy(c);
                chain = next;
            }
            self.collision_chain = null;
            if (self.children) |children| {
                allocator.free(children);
                self.children = null;
            }
        }
    };

    pub fn init(allocator: Allocator) SSI {
        return .{
            .root = null,
            .allocator = allocator,
            .height = 0,
            .size = 0,
            .max_height = bucket_width,
            .dim = 0,
            .model_id = 0,
            .global_diffusion = false,
            .identity_set = false,
        };
    }

    pub fn mixHash(state: u64, value: u64) u64 {
        return state *% 0x9E3779B185EBCA87 +% value +% 0x517CC1B727220A95;
    }

    fn scoreBits(value: f32) u32 {
        return @as(u32, @bitCast(value));
    }

    pub fn hashTokens(tokens: []const u32) u64 {
        var state: u64 = 0;
        state = mixHash(state, @as(u64, @intCast(tokens.len)));
        for (tokens) |tok| {
            state = mixHash(state, tok);
        }
        return state;
    }

    fn minHashSeedA(lane: u64) u64 {
        return 0x9E3779B185EBCA87 +% lane *% 0xC2B2AE3D27D4EB4F +% 0x165667B19E3779F9;
    }

    fn minHashSeedB(lane: u64) u64 {
        return 0x517CC1B727220A95 +% lane *% 0xD6E8FEB86659FD93 +% 0x2545F4914F6CDD1D;
    }

    fn minHashLaneHash(token: u32, lane: u64) u64 {
        var h = @as(u64, token) *% minHashSeedA(lane) +% minHashSeedB(lane);
        h = (h ^ (h >> 30)) *% 0xbf58476d1ce4e5b9;
        h = (h ^ (h >> 27)) *% 0x94d049bb133111eb;
        h = h ^ (h >> 31);
        return h;
    }

    pub fn computeMinHashSignature(tokens: []const u32) u64 {
        const vector_len: usize = 8;
        const lane_count: usize = 64 / vector_len;
        var minima: [lane_count]@Vector(vector_len, u64) = undefined;
        var lane: usize = 0;
        while (lane < lane_count) : (lane += 1) {
            minima[lane] = @splat(std.math.maxInt(u64));
        }
        for (tokens) |token| {
            lane = 0;
            while (lane < lane_count) : (lane += 1) {
                var hashes: @Vector(vector_len, u64) = undefined;
                var lane_bit: usize = 0;
                while (lane_bit < vector_len) : (lane_bit += 1) {
                    hashes[lane_bit] = minHashLaneHash(token, lane * vector_len + lane_bit);
                }
                minima[lane] = @min(minima[lane], hashes);
            }
        }
        var signature: u64 = 0;
        lane = 0;
        while (lane < lane_count) : (lane += 1) {
            const parities = minima[lane] & @as(@Vector(vector_len, u64), @splat(1));
            var lane_bit: usize = 0;
            while (lane_bit < vector_len) : (lane_bit += 1) {
                if (parities[lane_bit] != 0) {
                    signature |= @as(u64, 1) << @intCast(lane * vector_len + lane_bit);
                }
            }
        }
        return signature;
    }

    pub fn latentSimHash(latent: []const f32) u64 {
        return types.rsfLatentSimHash(latent);
    }

    pub fn signatureSimilarity(query_signature: u64, segment_signature: u64) f32 {
        const mismatch = @popCount(query_signature ^ segment_signature);
        const matches = 64 - @as(i64, @intCast(mismatch));
        if (matches <= 0) return 0.0;
        const ratio = @as(f32, @floatFromInt(matches)) / 64.0;
        const estimate = 2.0 * ratio - 1.0;
        return std.math.clamp(estimate, @as(f32, 0.0), @as(f32, 1.0));
    }

    pub fn computeAnchorHash(tokens: []const u32, position: u64) u64 {
        var state: u64 = position;
        state = mixHash(state, @as(u64, @intCast(tokens.len)));
        for (tokens) |tok| {
            state = mixHash(state, tok);
        }
        return state;
    }

    pub fn bucketIndex(position: u64) usize {
        var h = position *% 0x9E3779B185EBCA87;
        h = (h ^ (h >> 30)) *% 0xbf58476d1ce4e5b9;
        h = (h ^ (h >> 27)) *% 0x94d049bb133111eb;
        h = h ^ (h >> 31);
        return @as(usize, @intCast(h & (bucket_count - 1)));
    }

    fn low32(value: u64) u32 {
        return @as(u32, @intCast(value & 0xFFFF_FFFF));
    }

    fn high32(value: u64) u32 {
        return @as(u32, @intCast(value >> 32));
    }

    fn joinU64(lo: u32, hi: u32) u64 {
        return (@as(u64, hi) << 32) | @as(u64, lo);
    }

    fn bitsToFloat(bits: u32) f32 {
        return @as(f32, @bitCast(bits));
    }

    fn floatToBits(value: f32) u32 {
        return @as(u32, @bitCast(value));
    }

    fn safeFloat(v: f32) f32 {
        if (std.math.isNan(v) or std.math.isInf(v)) return 0.0;
        return std.math.clamp(v, -3.4e38, 3.4e38);
    }

    fn recursiveDeinit(node: *Node, allocator: Allocator) void {
        if (node.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    recursiveDeinit(child, allocator);
                }
            }
        }
        node.deinit(allocator);
        allocator.destroy(node);
    }

    pub fn deinit(self: *SSI) void {
        if (self.root) |root| {
            recursiveDeinit(root, self.allocator);
        }
        self.root = null;
        self.height = 0;
        self.size = 0;
        self.dim = 0;
        self.model_id = 0;
        self.global_diffusion = false;
        self.identity_set = false;
    }

    fn computeLeafHash(node: *const Node) u64 {
        var acc: u64 = 0;
        if (node.segment) |seg| {
            acc +%= seg.fullHash();
        }
        var chain = node.collision_chain;
        while (chain) |c| {
            acc +%= c.seg.fullHash();
            chain = c.next;
        }
        return acc;
    }

    fn computeBranchHash(node: *const Node) u64 {
        var acc: u64 = 0;
        if (node.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    acc +%= child.hash;
                }
            }
        }
        return acc;
    }

    fn refreshHash(node: *Node) void {
        node.hash = if (node.is_leaf) computeLeafHash(node) else computeBranchHash(node);
    }

    fn ensureRoot(self: *SSI) !*Node {
        if (self.root == null) {
            const root = try self.allocator.create(Node);
            root.* = try Node.init(self.allocator, bucket_width);
            root.is_leaf = false;
            root.height = bucket_width;
            refreshHash(root);
            self.root = root;
            self.height = bucket_width;
        }
        return self.root.?;
    }

    fn adoptIdentity(self: *SSI, dim: usize, model_id: u64, global_diffusion: bool) !void {
        if (self.identity_set) {
            if (self.dim != dim or self.model_id != model_id or self.global_diffusion != global_diffusion) {
                return Error.RSFModelMismatch;
            }
            return;
        }
        self.dim = dim;
        self.model_id = model_id;
        self.global_diffusion = global_diffusion;
        self.identity_set = true;
    }

    fn copyIdentityInto(self: *const SSI, target: *SSI) !void {
        if (!self.identity_set) return;
        try target.adoptIdentity(self.dim, self.model_id, self.global_diffusion);
    }

    fn insertIntoLeaf(
        self: *SSI,
        leaf: *Node,
        tokens: []const u32,
        position: u64,
        score: f32,
        anchor_hash: u64,
        latent: []const f32,
        log_det: f32,
        latent_version: u32,
    ) !bool {
        if (!leaf.is_leaf or leaf.height != 0) {
            return error.InvalidNodeState;
        }
        if (leaf.segment == null) {
            leaf.segment = try Segment.init(self.allocator, tokens, position, score, anchor_hash, latent, log_det, latent_version);
            refreshHash(leaf);
            return true;
        }
        if (leaf.segment.?.position == position) {
            var old = leaf.segment.?;
            old.deinit(self.allocator);
            leaf.segment = try Segment.init(self.allocator, tokens, position, score, anchor_hash, latent, log_det, latent_version);
            refreshHash(leaf);
            return false;
        }
        var chain = leaf.collision_chain;
        while (chain) |c| {
            if (c.seg.position == position) {
                c.seg.deinit(self.allocator);
                c.seg = try Segment.init(self.allocator, tokens, position, score, anchor_hash, latent, log_det, latent_version);
                refreshHash(leaf);
                return false;
            }
            chain = c.next;
        }
        const collision = try self.allocator.create(CollisionNode);
        collision.* = .{
            .seg = try Segment.init(self.allocator, tokens, position, score, anchor_hash, latent, log_det, latent_version),
            .next = leaf.collision_chain,
        };
        leaf.collision_chain = collision;
        refreshHash(leaf);
        return true;
    }

    fn addSequenceWithMetadata(
        self: *SSI,
        tokens: []const u32,
        position: u64,
        score: f32,
        anchor_hash: u64,
        latent: []const f32,
        log_det: f32,
        latent_version: u32,
    ) !void {
        const root = try self.ensureRoot();
        const idx = bucketIndex(position);
        if (root.children.?[idx] == null) {
            const leaf = try self.allocator.create(Node);
            leaf.* = try Node.init(self.allocator, 0);
            root.children.?[idx] = leaf;
        }
        const leaf = root.children.?[idx].?;
        const inserted_new = try self.insertIntoLeaf(leaf, tokens, position, score, anchor_hash, latent, log_det, latent_version);
        refreshHash(root);
        if (inserted_new) {
            self.size += 1;
        }
    }

    fn copyInto(self: *const SSI, target: *SSI) !void {
        try self.copyIdentityInto(target);
        if (self.root == null) {
            return;
        }
        const root = self.root.?;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment) |seg| {
                        try target.addSequenceWithMetadata(seg.tokens, seg.position, seg.score, seg.anchor_hash, seg.latent, seg.log_det, seg.latent_version);
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        try target.addSequenceWithMetadata(c.seg.tokens, c.seg.position, c.seg.score, c.seg.anchor_hash, c.seg.latent, c.seg.log_det, c.seg.latent_version);
                        chain = c.next;
                    }
                }
            }
        }
    }

    pub fn addSequence(self: *SSI, tokens: []const u32, position: u64, is_anchor: bool) !void {
        const anchor_hash = if (is_anchor) computeAnchorHash(tokens, position) else 0;
        try self.addSequenceWithMetadata(tokens, position, 0.0, anchor_hash, &.{}, 0.0, 0);
        if (self.size > 0) {
            const load_factor = @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(bucket_count));
            if (load_factor > 8.0) {
                try self.compact();
            }
        }
    }

    pub fn addSequenceWithLatent(self: *SSI, tokens: []const u32, position: u64, is_anchor: bool, latent: []const f32, log_det: f32) !void {
        if (latent.len < 2 or latent.len % 2 != 0) return Error.InvalidShape;
        const dim = latent.len / 2;
        try self.adoptIdentity(dim, self.model_id, self.global_diffusion);
        if (self.dim != dim) return Error.RSFModelMismatch;
        const anchor_hash = if (is_anchor) computeAnchorHash(tokens, position) else 0;
        try self.addSequenceWithMetadata(tokens, position, 0.0, anchor_hash, latent, log_det, 1);
        if (self.size > 0) {
            const load_factor = @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(bucket_count));
            if (load_factor > 8.0) {
                try self.compact();
            }
        }
    }

    pub fn addLatent(self: *SSI, model: anytype, state: anytype, tokens: []const u32, position: u64, is_anchor: bool) !void {
        const dim = try model.dim();
        const gd = try model.globalDiffusionEnabled();
        const binding = try model.latentBinding();
        try state.binding.requireModel(binding.model_id);
        try state.binding.requireSpace(.latent_state);
        if (state.binding.dim != dim) return Error.RSFDimMismatch;
        if (state.data.shape.dims.len != 3) return Error.InvalidShape;
        if (state.data.shape.dims[1] != dim or state.data.shape.dims[2] != 2) return Error.ShapeMismatch;
        if (state.data.shape.dims[0] == 0) return Error.EmptyInput;
        const packed_len = dim * 2;
        const latent = try self.allocator.alloc(f32, packed_len);
        defer self.allocator.free(latent);
        var d: usize = 0;
        while (d < dim) : (d += 1) {
            latent[d] = state.data.data[d * 2];
            latent[dim + d] = state.data.data[d * 2 + 1];
        }
        try self.adoptIdentity(dim, binding.model_id, gd);
        try self.addSequenceWithLatent(tokens, position, is_anchor, latent, state.log_det);
    }

    fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        if (a.len == 0 or b.len == 0 or a.len != b.len) return 0.0;
        var dot: f64 = 0.0;
        var na: f64 = 0.0;
        var nb: f64 = 0.0;
        var i: usize = 0;
        while (i < a.len) : (i += 1) {
            const x: f64 = a[i];
            const y: f64 = b[i];
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        const denom = @sqrt(na) * @sqrt(nb) + @as(f64, SSIScoringConfig.cosine_eps);
        return @floatCast(dot / denom);
    }

    fn volumeScore(segment_log_det: f32, query_log_det: f32) f32 {
        const delta = @abs(segment_log_det - query_log_det) / SSIScoringConfig.volume_kappa;
        return @floatCast(@exp(@as(f64, -delta)));
    }

    fn latentScore(seg: Segment, query_latent: []const f32, query_log_det: f32, query_hash: u64, query_signature: u64) f32 {
        if (seg.latent_version == 0 or seg.latent.len == 0) {
            return signatureSimilarity(query_signature, seg.signature);
        }
        const cos = cosineSimilarity(query_latent, seg.latent);
        const vol = volumeScore(seg.log_det, query_log_det);
        _ = query_hash;
        return SSIScoringConfig.cosine_weight * cos + SSIScoringConfig.volume_weight * vol;
    }

    pub fn retrieveTopK(self: *const SSI, query_tokens: []const u32, k: usize, allocator: Allocator) ![]types.RankedSegment {
        if (k == 0) {
            return allocator.alloc(types.RankedSegment, 0);
        }
        var heap = std.PriorityQueue(types.RankedSegment, void, struct {
            pub fn lessThan(_: void, a: types.RankedSegment, b: types.RankedSegment) std.math.Order {
                return std.math.order(a.score, b.score);
            }
        }.lessThan).init(allocator, {});
        errdefer {
            while (heap.removeOrNull()) |item| {
                var mut = item;
                mut.deinit(allocator);
            }
            heap.deinit();
        }
        defer heap.deinit();
        const query_hash = hashTokens(query_tokens);
        const query_signature = computeMinHashSignature(query_tokens);
        try self.traverse(self.root, query_hash, query_signature, &heap, k, allocator);
        const result_len = @min(k, heap.count());
        var top_n = try allocator.alloc(types.RankedSegment, result_len);
        var index = result_len;
        while (heap.removeOrNull()) |item| {
            index -= 1;
            top_n[index] = item;
        }
        return top_n;
    }

    pub fn retrieveTopKLatent(self: *const SSI, query_latent: []const f32, query_log_det: f32, k: usize, allocator: Allocator) ![]types.RankedSegment {
        if (k == 0) {
            return allocator.alloc(types.RankedSegment, 0);
        }
        if (self.identity_set and query_latent.len != self.dim * 2) {
            return Error.RSFModelMismatch;
        }
        var heap = std.PriorityQueue(types.RankedSegment, void, struct {
            pub fn lessThan(_: void, a: types.RankedSegment, b: types.RankedSegment) std.math.Order {
                return std.math.order(a.score, b.score);
            }
        }.lessThan).init(allocator, {});
        errdefer {
            while (heap.removeOrNull()) |item| {
                var mut = item;
                mut.deinit(allocator);
            }
            heap.deinit();
        }
        defer heap.deinit();
        const query_signature = mixHash(0, types.rsfLatentSimHash(query_latent));
        try self.traverseLatent(self.root, query_latent, query_log_det, query_signature, &heap, k, allocator);
        const result_len = @min(k, heap.count());
        var top_n = try allocator.alloc(types.RankedSegment, result_len);
        var index = result_len;
        while (heap.removeOrNull()) |item| {
            index -= 1;
            top_n[index] = item;
        }
        return top_n;
    }

    fn traverse(self: *const SSI, node: ?*Node, query_hash: u64, query_signature: u64, heap: anytype, k: usize, allocator: Allocator) !void {
        if (node == null) {
            return;
        }
        const current = node.?;
        if (current.is_leaf) {
            if (current.segment) |seg| {
                try addSegmentToHeap(seg, query_hash, query_signature, heap, k, allocator);
            }
            var chain = current.collision_chain;
            while (chain) |c| {
                try addSegmentToHeap(c.seg, query_hash, query_signature, heap, k, allocator);
                chain = c.next;
            }
            return;
        }
        if (current.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    try traverse(self, child, query_hash, query_signature, heap, k, allocator);
                }
            }
        }
    }

    fn traverseLatent(self: *const SSI, node: ?*Node, query_latent: []const f32, query_log_det: f32, query_signature: u64, heap: anytype, k: usize, allocator: Allocator) !void {
        if (node == null) {
            return;
        }
        const current = node.?;
        if (current.is_leaf) {
            if (current.segment) |seg| {
                try addLatentSegmentToHeap(seg, query_latent, query_log_det, query_signature, heap, k, allocator);
            }
            var chain = current.collision_chain;
            while (chain) |c| {
                try addLatentSegmentToHeap(c.seg, query_latent, query_log_det, query_signature, heap, k, allocator);
                chain = c.next;
            }
            return;
        }
        if (current.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    try traverseLatent(self, child, query_latent, query_log_det, query_signature, heap, k, allocator);
                }
            }
        }
    }

    fn addSegmentToHeap(seg: Segment, query_hash: u64, query_signature: u64, heap: anytype, k: usize, allocator: Allocator) !void {
        const similarity = computeFusedSimilarity(query_hash, query_signature, seg.tokenHash(), seg.signature);
        if (heap.count() >= k) {
            if (heap.peek()) |top| {
                if (similarity <= top.score) {
                    return;
                }
            }
        }
        const ranked = types.RankedSegment{
            .tokens = try allocator.dupe(u32, seg.tokens),
            .score = similarity,
            .position = seg.position,
            .anchor = seg.anchor_hash != 0,
            .latent_similarity = 0,
            .reconstruction_confidence = 0,
            .volume_surprise = 0,
        };
        errdefer allocator.free(ranked.tokens);
        if (heap.count() < k) {
            try heap.add(ranked);
            return;
        }
        try heap.add(ranked);
        var removed = heap.remove();
        removed.deinit(allocator);
    }

    fn addLatentSegmentToHeap(seg: Segment, query_latent: []const f32, query_log_det: f32, query_signature: u64, heap: anytype, k: usize, allocator: Allocator) !void {
        const query_hash = hashTokens(seg.tokens);
        const similarity = latentScore(seg, query_latent, query_log_det, query_hash, query_signature);
        if (heap.count() >= k) {
            if (heap.peek()) |top| {
                if (similarity <= top.score) {
                    return;
                }
            }
        }
        const ranked = types.RankedSegment{
            .tokens = try allocator.dupe(u32, seg.tokens),
            .score = similarity,
            .position = seg.position,
            .anchor = seg.anchor_hash != 0,
            .latent_similarity = cosineSimilarity(query_latent, seg.latent),
            .reconstruction_confidence = 0,
            .volume_surprise = volumeScore(seg.log_det, query_log_det),
        };
        errdefer allocator.free(ranked.tokens);
        if (heap.count() < k) {
            try heap.add(ranked);
            return;
        }
        try heap.add(ranked);
        var removed = heap.remove();
        removed.deinit(allocator);
    }

    fn computeSimilarity(h1: u64, h2: u64) f32 {
        const pc1 = @popCount(h1);
        const pc2 = @popCount(h2);
        if (pc1 == 0 and pc2 == 0) return 1.0;
        if (pc1 == 0 or pc2 == 0) return 0.0;
        const intersection = @popCount(h1 & h2);
        const denom = @sqrt(@as(f32, @floatFromInt(pc1)) * @as(f32, @floatFromInt(pc2)));
        return @as(f32, @floatFromInt(intersection)) / denom;
    }

    fn computeFusedSimilarity(query_hash: u64, query_signature: u64, segment_hash: u64, segment_signature: u64) f32 {
        const hash_cosine = computeSimilarity(query_hash, segment_hash);
        const jaccard_estimate = signatureSimilarity(query_signature, segment_signature);
        return 0.5 * hash_cosine + 0.5 * jaccard_estimate;
    }

    pub fn compact(self: *SSI) !void {
        if (self.size < 1000) {
            return;
        }
        var rebuilt = SSI.init(self.allocator);
        rebuilt.max_height = self.max_height;
        errdefer rebuilt.deinit();
        try self.copyInto(&rebuilt);
        self.deinit();
        self.* = rebuilt;
    }

    pub fn updateScore(self: *SSI, position: u64, new_score: f32) !void {
        const root = self.root orelse return Error.OutOfBounds;
        const child = root.children.?[bucketIndex(position)] orelse return Error.OutOfBounds;
        if (child.segment) |*seg| {
            if (seg.position == position) {
                seg.score = new_score;
                refreshHash(child);
                refreshHash(root);
                return;
            }
        }
        var chain = child.collision_chain;
        while (chain) |c| {
            if (c.seg.position == position) {
                c.seg.score = new_score;
                refreshHash(child);
                refreshHash(root);
                return;
            }
            chain = c.next;
        }
        return Error.OutOfBounds;
    }

    pub fn getSegment(self: *const SSI, position: u64) ?Segment {
        const root = self.root orelse return null;
        const child = root.children.?[bucketIndex(position)] orelse return null;
        if (child.segment) |seg| {
            if (seg.position == position) {
                return seg;
            }
        }
        var chain = child.collision_chain;
        while (chain) |c| {
            if (c.seg.position == position) {
                return c.seg;
            }
            chain = c.next;
        }
        return null;
    }

    fn countSegments(self: *const SSI) usize {
        const root = self.root orelse return 0;
        var count: usize = 0;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment != null) {
                        count += 1;
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        count += 1;
                        chain = c.next;
                    }
                }
            }
        }
        return count;
    }

    fn writeBoolFlag(writer: anytype, value: bool) !void {
        try writer.writeInt(u8, if (value) 1 else 0, .little);
    }

    fn readBoolFlag(reader: anytype) !bool {
        return (try reader.readInt(u8, .little)) != 0;
    }

    fn writeSegment(writer: anytype, seg: Segment) !void {
        try writer.writeInt(u64, seg.position, .little);
        try writer.writeInt(u32, floatToBits(seg.score), .little);
        try writer.writeInt(u64, seg.anchor_hash, .little);
        try writer.writeInt(u64, seg.signature, .little);
        try writer.writeInt(u64, @as(u64, seg.tokens.len), .little);
        for (seg.tokens) |tok| {
            try writer.writeInt(u32, tok, .little);
        }
        try writer.writeInt(u32, seg.latent_version, .little);
        try writer.writeInt(u32, floatToBits(seg.log_det), .little);
        try writer.writeInt(u64, @as(u64, seg.latent.len), .little);
        for (seg.latent) |value| {
            try writer.writeInt(u32, floatToBits(value), .little);
        }
    }

    fn readSegment(allocator: Allocator, reader: anytype, version: u32) !Segment {
        const position = try reader.readInt(u64, .little);
        const score = bitsToFloat(try reader.readInt(u32, .little));
        const anchor_hash = try reader.readInt(u64, .little);
        const stored_signature = try reader.readInt(u64, .little);
        const token_len_raw = try reader.readInt(u64, .little);
        if (token_len_raw > std.math.maxInt(usize)) return error.InvalidData;
        const token_len: usize = @intCast(token_len_raw);
        const tokens = try allocator.alloc(u32, token_len);
        errdefer allocator.free(tokens);
        for (tokens) |*tok| {
            tok.* = try reader.readInt(u32, .little);
        }
        var latent_version: u32 = 0;
        var log_det: f32 = 0.0;
        var latent: []f32 = &.{};
        if (version >= format_version) {
            latent_version = try reader.readInt(u32, .little);
            log_det = bitsToFloat(try reader.readInt(u32, .little));
            const latent_len_raw = try reader.readInt(u64, .little);
            if (latent_len_raw > std.math.maxInt(usize)) return error.InvalidData;
            const latent_len: usize = @intCast(latent_len_raw);
            latent = try allocator.alloc(f32, latent_len);
            errdefer allocator.free(latent);
            for (latent) |*value| {
                value.* = bitsToFloat(try reader.readInt(u32, .little));
            }
        }
        const token_sig = computeMinHashSignature(tokens);
        const expected = if (latent_version == 1) mixHash(token_sig, types.rsfLatentSimHash(latent)) else token_sig;
        if (expected != stored_signature) return error.InvalidData;
        return .{
            .tokens = tokens,
            .position = position,
            .score = score,
            .anchor_hash = anchor_hash,
            .signature = stored_signature,
            .latent = latent,
            .log_det = log_det,
            .latent_version = latent_version,
        };
    }

    fn serializeNode(node: *const Node, writer: anytype) !void {
        try writeBoolFlag(writer, node.is_leaf);
        try writer.writeInt(u64, @as(u64, node.height), .little);
        try writer.writeInt(u64, node.hash, .little);
        if (node.is_leaf) {
            try writeBoolFlag(writer, node.segment != null);
            if (node.segment) |seg| {
                try writeSegment(writer, seg);
            }
            var chain_len: usize = 0;
            var chain = node.collision_chain;
            while (chain) |c| {
                chain_len += 1;
                chain = c.next;
            }
            try writer.writeInt(u64, @as(u64, chain_len), .little);
            chain = node.collision_chain;
            while (chain) |c| {
                try writeSegment(writer, c.seg);
                chain = c.next;
            }
            return;
        }
        const children = node.children orelse return error.InvalidNodeState;
        try writer.writeInt(u64, @as(u64, children.len), .little);
        for (children) |maybe_child| {
            try writeBoolFlag(writer, maybe_child != null);
            if (maybe_child) |child| {
                try serializeNode(child, writer);
            }
        }
    }

    fn deserializeNode(allocator: Allocator, reader: anytype, version: u32) !*Node {
        const is_leaf = try readBoolFlag(reader);
        const height_raw = try reader.readInt(u64, .little);
        if (height_raw > std.math.maxInt(usize)) return error.InvalidData;
        const height: usize = @intCast(height_raw);
        const stored_hash = try reader.readInt(u64, .little);
        const node = try allocator.create(Node);
        var cleanup = true;
        errdefer {
            if (cleanup) {
                recursiveDeinit(node, allocator);
            }
        }
        node.* = try Node.init(allocator, if (is_leaf) 0 else height);
        if (node.is_leaf != is_leaf) {
            return error.InvalidData;
        }
        if (is_leaf) {
            const has_segment = try readBoolFlag(reader);
            if (has_segment) {
                node.segment = try readSegment(allocator, reader, version);
            }
            const chain_len_raw = try reader.readInt(u64, .little);
            if (chain_len_raw > std.math.maxInt(usize)) return error.InvalidData;
            const chain_len: usize = @intCast(chain_len_raw);
            var head: ?*CollisionNode = null;
            var tail: ?*CollisionNode = null;
            var index: usize = 0;
            while (index < chain_len) : (index += 1) {
                const collision = try allocator.create(CollisionNode);
                collision.* = .{
                    .seg = try readSegment(allocator, reader, version),
                    .next = null,
                };
                if (head == null) {
                    head = collision;
                    tail = collision;
                } else {
                    tail.?.next = collision;
                    tail = collision;
                }
            }
            node.collision_chain = head;
        } else {
            const children_len_raw = try reader.readInt(u64, .little);
            if (children_len_raw > std.math.maxInt(usize)) return error.InvalidData;
            const children_len: usize = @intCast(children_len_raw);
            if (children_len != bucket_count) {
                return error.InvalidData;
            }
            for (0..children_len) |i| {
                const has_child = try readBoolFlag(reader);
                if (has_child) {
                    node.children.?[i] = try deserializeNode(allocator, reader, version);
                }
            }
        }
        refreshHash(node);
        if (node.hash != stored_hash) {
            return error.InvalidData;
        }
        cleanup = false;
        return node;
    }

    pub fn serialize(self: *SSI, writer: anytype) !void {
        const header = (@as(u64, format_version) << 32) | @as(u64, format_magic);
        try writer.writeInt(u64, header, .little);
        try writer.writeInt(u64, @as(u64, self.dim), .little);
        try writer.writeInt(u64, self.model_id, .little);
        try writeBoolFlag(writer, self.global_diffusion);
        try writeBoolFlag(writer, self.identity_set);
        try writer.writeInt(u64, @as(u64, self.max_height), .little);
        try writer.writeInt(u64, @as(u64, self.height), .little);
        try writer.writeInt(u64, @as(u64, self.size), .little);
        try writeBoolFlag(writer, self.root != null);
        if (self.root) |root| {
            try serializeNode(root, writer);
        }
    }

    pub fn deserialize(allocator: Allocator, reader: anytype) !SSI {
        var ssi = SSI.init(allocator);
        errdefer ssi.deinit();
        const first = try reader.readInt(u64, .little);
        const magic: u32 = @truncate(first);
        var version: u32 = 1;
        if (magic == format_magic) {
            version = @intCast(first >> 32);
            const dim_raw = try reader.readInt(u64, .little);
            if (dim_raw > std.math.maxInt(usize)) return error.InvalidData;
            ssi.dim = @intCast(dim_raw);
            ssi.model_id = try reader.readInt(u64, .little);
            ssi.global_diffusion = try readBoolFlag(reader);
            ssi.identity_set = try readBoolFlag(reader);
        } else {
            ssi.max_height = @intCast(first);
        }
        if (magic == format_magic) {
            const max_height_raw = try reader.readInt(u64, .little);
            const height_raw = try reader.readInt(u64, .little);
            const size_raw = try reader.readInt(u64, .little);
            if (max_height_raw > std.math.maxInt(usize) or height_raw > std.math.maxInt(usize) or size_raw > std.math.maxInt(usize)) {
                return error.InvalidData;
            }
            ssi.max_height = @intCast(max_height_raw);
            ssi.height = @intCast(height_raw);
            ssi.size = @intCast(size_raw);
        } else {
            const height_raw = try reader.readInt(u64, .little);
            const size_raw = try reader.readInt(u64, .little);
            if (height_raw > std.math.maxInt(usize) or size_raw > std.math.maxInt(usize)) {
                return error.InvalidData;
            }
            ssi.height = @intCast(height_raw);
            ssi.size = @intCast(size_raw);
        }
        const has_root = try readBoolFlag(reader);
        if (has_root) {
            ssi.root = try deserializeNode(allocator, reader, version);
        }
        if (ssi.countSegments() != ssi.size) {
            return error.InvalidData;
        }
        return ssi;
    }

    pub fn exportToTensor(self: *SSI, allocator: Allocator) !Tensor {
        const segment_count = self.countSegments();
        const dim = if (self.dim == 0) @as(usize, 0) else self.dim;
        const cols = if (dim == 0) tensor_width else dim * 2 + 4;
        const rows = if (segment_count == 0) 1 else segment_count;
        var tensor = try Tensor.init(allocator, &.{ rows, cols });
        @memset(tensor.data, 0);
        const root = self.root orelse return tensor;
        var row: usize = 0;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment) |seg| {
                        encodeSegmentRow(&tensor, row, seg, dim, cols);
                        row += 1;
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        encodeSegmentRow(&tensor, row, c.seg, dim, cols);
                        row += 1;
                        chain = c.next;
                    }
                }
            }
        }
        return tensor;
    }

    fn encodeSegmentRow(tensor: *Tensor, row: usize, seg: Segment, dim: usize, cols: usize) void {
        const offset = row * cols;
        if (dim == 0) {
            tensor.data[offset + 0] = safeFloat(@as(f32, @floatFromInt(seg.tokens.len)));
            tensor.data[offset + 1] = safeFloat(bitsToFloat(low32(seg.position)));
            tensor.data[offset + 2] = safeFloat(bitsToFloat(high32(seg.position)));
            tensor.data[offset + 3] = safeFloat(seg.score);
            tensor.data[offset + 4] = safeFloat(bitsToFloat(low32(seg.anchor_hash)));
            tensor.data[offset + 5] = safeFloat(bitsToFloat(high32(seg.anchor_hash)));
            var i: usize = 0;
            while (i < seg.tokens.len and i < 128) : (i += 1) {
                tensor.data[offset + 6 + i] = safeFloat(std.math.clamp(bitsToFloat(seg.tokens[i]), -3.4e38, 3.4e38));
            }
            return;
        }
        const latent_len = dim * 2;
        var i: usize = 0;
        while (i < latent_len) : (i += 1) {
            const value: f32 = if (i < seg.latent.len) seg.latent[i] else 0.0;
            tensor.data[offset + i] = safeFloat(value);
        }
        tensor.data[offset + latent_len + 0] = safeFloat(seg.score);
        tensor.data[offset + latent_len + 1] = safeFloat(seg.log_det);
        tensor.data[offset + latent_len + 2] = safeFloat(bitsToFloat(low32(seg.position)));
        tensor.data[offset + latent_len + 3] = safeFloat(bitsToFloat(high32(seg.position)));
    }

    pub fn importFromTensor(self: *SSI, tensor: *const Tensor) !void {
        const saved_dim = self.dim;
        const saved_model = self.model_id;
        const saved_gd = self.global_diffusion;
        const saved_id = self.identity_set;
        self.deinit();
        self.dim = saved_dim;
        self.model_id = saved_model;
        self.global_diffusion = saved_gd;
        self.identity_set = saved_id;
        if (tensor.shape.dims.len < 2) {
            return;
        }
        const cols = tensor.shape.dims[1];
        const rows = tensor.shape.dims[0];
        if (cols == tensor_width) {
            var tokens_buffer: [128]u32 = undefined;
            var row: usize = 0;
            while (row < rows) : (row += 1) {
                const offset = row * tensor_width;
                if (offset + tensor_width > tensor.data.len) {
                    break;
                }
                const token_len_float = tensor.data[offset + 0];
                if (!(token_len_float >= 0) or std.math.isInf(token_len_float)) {
                    continue;
                }
                const token_len_raw: usize = @intFromFloat(token_len_float);
                const token_len = @min(token_len_raw, 128);
                const position = joinU64(floatToBits(tensor.data[offset + 1]), floatToBits(tensor.data[offset + 2]));
                const score = tensor.data[offset + 3];
                const anchor_hash = joinU64(floatToBits(tensor.data[offset + 4]), floatToBits(tensor.data[offset + 5]));
                var i: usize = 0;
                while (i < token_len) : (i += 1) {
                    tokens_buffer[i] = floatToBits(tensor.data[offset + 6 + i]);
                }
                try self.addSequenceWithMetadata(tokens_buffer[0..token_len], position, score, anchor_hash, &.{}, 0.0, 0);
            }
            return;
        }
        if (cols < 4) return error.InvalidData;
        const latent_len = cols - 4;
        if (latent_len % 2 != 0) return error.InvalidData;
        const dim = latent_len / 2;
        if (self.identity_set and self.dim != dim) return Error.RSFModelMismatch;
        if (!self.identity_set and dim > 0) {
            try self.adoptIdentity(dim, self.model_id, self.global_diffusion);
        }
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            const offset = row * cols;
            if (offset + cols > tensor.data.len) break;
            const latent = tensor.data[offset .. offset + latent_len];
            const score = tensor.data[offset + latent_len + 0];
            const log_det = tensor.data[offset + latent_len + 1];
            const position = joinU64(floatToBits(tensor.data[offset + latent_len + 2]), floatToBits(tensor.data[offset + latent_len + 3]));
            var all_zero = true;
            for (latent) |v| {
                if (v != 0.0) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero and score == 0.0 and log_det == 0.0 and position == 0) continue;
            try self.addSequenceWithMetadata(&.{}, position, score, 0, latent, log_det, 1);
        }
    }

    pub fn merge(self: *SSI, other: *const SSI) !void {
        if (self.identity_set and other.identity_set) {
            if (self.dim != other.dim or self.model_id != other.model_id or self.global_diffusion != other.global_diffusion) {
                return Error.RSFModelMismatch;
            }
        }
        try other.copyInto(self);
    }

    pub fn split(self: *SSI, threshold: f32) !SSI {
        var result = SSI.init(self.allocator);
        result.max_height = self.max_height;
        try self.copyIdentityInto(&result);
        if (self.root == null) {
            return result;
        }
        const root = self.root.?;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment) |seg| {
                        if (seg.score > threshold) {
                            try result.addSequenceWithMetadata(seg.tokens, seg.position, seg.score, seg.anchor_hash, seg.latent, seg.log_det, seg.latent_version);
                        }
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        if (c.seg.score > threshold) {
                            try result.addSequenceWithMetadata(c.seg.tokens, c.seg.position, c.seg.score, c.seg.anchor_hash, c.seg.latent, c.seg.log_det, c.seg.latent_version);
                        }
                        chain = c.next;
                    }
                }
            }
        }
        return result;
    }

    pub fn balance(self: *SSI) void {
        if (self.root == null) {
            return;
        }
        var rebuilt = SSI.init(self.allocator);
        rebuilt.max_height = self.max_height;
        self.copyInto(&rebuilt) catch {
            rebuilt.deinit();
            return;
        };
        if (rebuilt.countSegments() != self.size) {
            rebuilt.deinit();
            return;
        }
        const old_root = self.root;
        const old_height = self.height;
        const old_size = self.size;
        const old_dim = self.dim;
        const old_model = self.model_id;
        const old_gd = self.global_diffusion;
        const old_id = self.identity_set;
        self.root = rebuilt.root;
        self.height = rebuilt.height;
        self.size = rebuilt.size;
        self.dim = rebuilt.dim;
        self.model_id = rebuilt.model_id;
        self.global_diffusion = rebuilt.global_diffusion;
        self.identity_set = rebuilt.identity_set;
        rebuilt.root = old_root;
        rebuilt.height = old_height;
        rebuilt.size = old_size;
        rebuilt.dim = old_dim;
        rebuilt.model_id = old_model;
        rebuilt.global_diffusion = old_gd;
        rebuilt.identity_set = old_id;
        rebuilt.deinit();
    }

    pub fn stats(self: *const SSI) struct { nodes: usize, leaves: usize, depth: usize } {
        var nodes: usize = 0;
        var leaves: usize = 0;
        var depth: usize = 0;
        const root = self.root orelse return .{ .nodes = 0, .leaves = 0, .depth = 0 };
        var stack = std.ArrayList(struct { node: *const Node, d: usize }).init(self.allocator);
        defer stack.deinit();
        stack.append(.{ .node = root, .d = 0 }) catch return .{ .nodes = nodes, .leaves = leaves, .depth = depth };
        while (stack.pop()) |entry| {
            nodes += 1;
            if (entry.node.is_leaf) {
                leaves += 1;
            }
            if (entry.d > depth) {
                depth = entry.d;
            }
            if (entry.node.children) |children| {
                for (children) |maybe_child| {
                    if (maybe_child) |child| {
                        stack.append(.{ .node = child, .d = entry.d + 1 }) catch {};
                    }
                }
            }
        }
        return .{ .nodes = nodes, .leaves = leaves, .depth = depth };
    }

    fn validateLeaf(node: *const Node) bool {
        if (!node.is_leaf) {
            return false;
        }
        if (node.height != 0) {
            return false;
        }
        if (node.children != null) {
            return false;
        }
        if (node.segment == null and node.collision_chain == null) {
            return true;
        }
        return computeLeafHash(node) == node.hash;
    }

    fn validateNode(node: *const Node, position_set: anytype) !bool {
        if (node.is_leaf) {
            if (!validateLeaf(node)) {
                return false;
            }
            if (node.segment) |seg| {
                if (position_set.contains(seg.position)) return false;
                try position_set.put(seg.position, {});
            }
            var chain = node.collision_chain;
            while (chain) |c| {
                if (position_set.contains(c.seg.position)) return false;
                try position_set.put(c.seg.position, {});
                chain = c.next;
            }
            return true;
        }
        if (node.height != bucket_width) {
            return false;
        }
        const children = node.children orelse return false;
        if (children.len != bucket_count) {
            return false;
        }
        var acc: u64 = 0;
        for (children) |maybe_child| {
            if (maybe_child) |child| {
                if (!(try validateNode(child, position_set))) {
                    return false;
                }
                acc +%= child.hash;
            }
        }
        return acc == node.hash;
    }

    pub fn validate(self: *SSI) bool {
        const root = self.root orelse return self.size == 0;
        if (self.height != bucket_width) {
            return false;
        }
        var position_set = std.AutoHashMap(u64, void).init(self.allocator);
        defer position_set.deinit();
        const valid = validateNode(root, &position_set) catch return false;
        if (!valid) return false;
        const counted = position_set.count();
        if (counted != self.size) {
            return false;
        }
        return true;
    }
};
