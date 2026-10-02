const std = @import("std");
const SSI = @import("../../index/ssi.zig").SSI;

extern fn hs_init(argc: *c_int, argv: *[*c][*c]u8) void;
extern fn hs_exit() void;

extern fn jaide_rtl_abi_version() c_uint;
extern fn jaide_rtl_mix_hash(state: c_ulonglong, value: c_ulonglong) c_ulonglong;
extern fn jaide_rtl_count_bits64(value: c_ulonglong) c_uint;
extern fn jaide_rtl_isqrt32(value: c_uint) c_uint;
extern fn jaide_rtl_signature_similarity(a: c_ulonglong, b: c_ulonglong) c_uint;
extern fn jaide_rtl_compute_similarity(a: c_ulonglong, b: c_ulonglong) c_uint;
extern fn jaide_rtl_bucket_index(position: c_ulonglong) c_uint;
extern fn jaide_rtl_hash_tokens(tokens: [*]const c_uint, count: c_uint) c_ulonglong;
extern fn jaide_rtl_fuse_scores(base: f64, overlap: f64, jaccard: f64, proximity: f64, diversity: f64) f64;
extern fn jaide_rtl_arbiter_first_grant(mask: c_uint) c_int;
extern fn jaide_rtl_arbiter_service_cycles() c_uint;
extern fn jaide_rtl_max_search_depth() c_uint;

const expected_abi_version: c_uint = 1;
const expected_service_cycles: c_uint = 4;
const expected_max_search_depth: c_uint = 64;

var checks_passed: usize = 0;
var checks_failed: usize = 0;

fn report(name: []const u8, ok: bool) void {
    if (ok) {
        checks_passed += 1;
        std.debug.print("[PASS] {s}\n", .{name});
    } else {
        checks_failed += 1;
        std.debug.print("[FAIL] {s}\n", .{name});
    }
}

fn reportEqualU64(name: []const u8, hardware: u64, reference: u64) void {
    const ok = hardware == reference;
    if (ok) {
        checks_passed += 1;
        std.debug.print("[PASS] {s}\n", .{name});
    } else {
        checks_failed += 1;
        std.debug.print("[FAIL] {s}: rtl={d} reference={d}\n", .{ name, hardware, reference });
    }
}

fn referenceIsqrt32(value: u32) u32 {
    var result: u32 = 0;
    var index: u5 = 15;
    while (true) {
        const candidate = result | (@as(u32, 1) << index);
        if (candidate * candidate <= value) {
            result = candidate;
        }
        if (index == 0) break;
        index -= 1;
    }
    return result;
}

fn referenceSignatureSimilarityQ16(query: u64, segment: u64) u32 {
    const mismatch: u32 = @popCount(query ^ segment);
    if (mismatch >= 32) return 0;
    return (32 - mismatch) * 2048;
}

fn referenceComputeSimilarityQ16(first: u64, second: u64) u32 {
    const popcount_first: u32 = @popCount(first);
    const popcount_second: u32 = @popCount(second);
    const intersection: u32 = @popCount(first & second);
    if (popcount_first == 0 and popcount_second == 0) return 65536;
    if (popcount_first == 0 or popcount_second == 0) return 0;
    const root = referenceIsqrt32(popcount_first * popcount_second);
    if (root == 0) return 0;
    return @intCast((@as(u64, intersection) * 65536) / root);
}

fn referenceFuseScores(base: f64, overlap: f64, jaccard: f64, proximity: f64, diversity: f64) f64 {
    const raw_base = std.math.clamp(base, 0.0, 100.0);
    const scaled_base = raw_base * 0.010009765625;
    const combined = scaled_base * 0.25 +
        std.math.clamp(overlap, 0.0, 1.0) * 0.1875 +
        std.math.clamp(jaccard, 0.0, 1.0) * 0.1875 +
        std.math.clamp(proximity, 0.0, 1.0) * 0.1875 +
        std.math.clamp(diversity, 0.0, 1.0) * 0.1875;
    return std.math.clamp(combined, 0.0, 1.0);
}

fn referenceFirstGrant(mask: u32) i32 {
    var client: u5 = 0;
    while (client < 4) : (client += 1) {
        if ((mask & (@as(u32, 1) << client)) != 0) return @intCast(client);
    }
    return -1;
}

fn runHashEquivalence(allocator: std.mem.Allocator, token_count: usize, seed: u64) !void {
    const tokens = try allocator.alloc(u32, token_count);
    defer allocator.free(tokens);
    const c_tokens = try allocator.alloc(c_uint, token_count);
    defer allocator.free(c_tokens);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (tokens, 0..) |*token, index| {
        token.* = random.int(u32);
        c_tokens[index] = @intCast(token.*);
    }

    const reference = SSI.hashTokens(tokens);
    const hardware: u64 = @intCast(jaide_rtl_hash_tokens(c_tokens.ptr, @intCast(token_count)));

    var name_buffer: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buffer, "hashTokens equivalence over {d} tokens", .{token_count});
    reportEqualU64(name, hardware, reference);
}

pub fn main() !u8 {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var argc: c_int = 0;
    var argv_storage: [*c][*c]u8 = null;
    hs_init(&argc, &argv_storage);
    defer hs_exit();

    std.debug.print("JAIDE RTL equivalence checker (MemoryArbiter + RankerCore + SSISearch)\n", .{});

    report("RTL ABI version", jaide_rtl_abi_version() == expected_abi_version);
    report("arbiter service cycles", jaide_rtl_arbiter_service_cycles() == expected_service_cycles);
    report("SSI maximum search depth", jaide_rtl_max_search_depth() == expected_max_search_depth);

    var prng = std.Random.DefaultPrng.init(0x5EED_1234_ABCD_9876);
    const random = prng.random();

    var iteration: usize = 0;
    var mix_matches = true;
    var popcount_matches = true;
    var isqrt_matches = true;
    var signature_matches = true;
    var similarity_matches = true;
    var bucket_matches = true;
    var fuse_matches = true;

    while (iteration < 4096) : (iteration += 1) {
        const state = random.int(u64);
        const value = random.int(u64);

        if (@as(u64, @intCast(jaide_rtl_mix_hash(state, value))) != SSI.mixHash(state, value)) {
            mix_matches = false;
        }
        if (@as(u32, @intCast(jaide_rtl_count_bits64(state))) != @as(u32, @popCount(state))) {
            popcount_matches = false;
        }

        const narrow: u32 = @truncate(value);
        if (@as(u32, @intCast(jaide_rtl_isqrt32(narrow))) != referenceIsqrt32(narrow)) {
            isqrt_matches = false;
        }
        if (@as(u32, @intCast(jaide_rtl_signature_similarity(state, value))) != referenceSignatureSimilarityQ16(state, value)) {
            signature_matches = false;
        }
        if (@as(u32, @intCast(jaide_rtl_compute_similarity(state, value))) != referenceComputeSimilarityQ16(state, value)) {
            similarity_matches = false;
        }
        if (@as(usize, @intCast(jaide_rtl_bucket_index(state))) != SSI.bucketIndex(state)) {
            bucket_matches = false;
        }

        const base = random.float(f64) * 100.0;
        const overlap = random.float(f64);
        const jaccard = random.float(f64);
        const proximity = random.float(f64);
        const diversity = random.float(f64);
        const hardware_score = jaide_rtl_fuse_scores(base, overlap, jaccard, proximity, diversity);
        const reference_score = referenceFuseScores(base, overlap, jaccard, proximity, diversity);
        if (@abs(hardware_score - reference_score) > 1.0e-3) {
            fuse_matches = false;
        }
    }

    report("mixHash equivalence over 4096 vectors", mix_matches);
    report("countBits64 equivalence over 4096 vectors", popcount_matches);
    report("isqrt32 equivalence over 4096 vectors", isqrt_matches);
    report("signatureSimilarity equivalence over 4096 vectors", signature_matches);
    report("computeSimilarity equivalence over 4096 vectors", similarity_matches);
    report("bucketIndex equivalence over 4096 vectors", bucket_matches);
    report("fuseScores equivalence over 4096 vectors", fuse_matches);

    var mask: u32 = 0;
    var grant_matches = true;
    while (mask < 16) : (mask += 1) {
        if (jaide_rtl_arbiter_first_grant(mask) != referenceFirstGrant(mask)) {
            grant_matches = false;
        }
    }
    report("arbiter grant priority over every request mask", grant_matches);

    try runHashEquivalence(allocator, 1, 0x1111_2222_3333_4444);
    try runHashEquivalence(allocator, 8, 0x2222_3333_4444_5555);
    try runHashEquivalence(allocator, 64, 0x3333_4444_5555_6666);
    try runHashEquivalence(allocator, 512, 0x4444_5555_6666_7777);

    const weight_sum = 0.25 + 0.1875 * 4.0;
    report("ranker fusion weights sum to one", @abs(weight_sum - 1.0) < 1.0e-12);
    report(
        "ranker fusion saturates at one",
        @abs(referenceFuseScores(100.0, 1.0, 1.0, 1.0, 1.0) - 1.0) < 1.0e-3,
    );

    std.debug.print("jaide-rtl-sim: {d} passed, {d} failed\n", .{ checks_passed, checks_failed });
    return if (checks_failed == 0) 0 else 1;
}
