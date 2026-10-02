const std = @import("std");
const GPUCoordinator = @import("distributed/gpu_coordinator.zig").GPUCoordinator;
const dataset_partition = @import("distributed/dataset_partition.zig");
const mmap_dataset = @import("distributed/mmap_token_dataset.zig");
const dtf = @import("distributed/distributed_trainer_futhark.zig");
const DistributedTrainerFuthark = dtf.DistributedTrainerFuthark;
const TrainerConfig = dtf.TrainerConfig;
const TrainerComponents = dtf.TrainerComponents;
const MGT = @import("tokenizer/mgt.zig").MGT;
const nccl = @import("distributed/nccl_bindings.zig");
const modal_gpu = @import("distributed/modal_gpu.zig");
const core_relational = @import("core_relational/mod.zig");
const accel_interface = @import("hw/accel/accel_interface.zig");
const core_memory = @import("core/memory.zig");

fn extractDatasetText(
    arena: *core_memory.ArenaAllocator,
    line: []const u8,
) !?[]const u8 {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        arena.allocator(),
        line,
        .{ .allocate = .alloc_if_needed },
    ) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return null,
    };

    return switch (parsed.value) {
        .object => |obj| blk: {
            const text_value = obj.get("text") orelse break :blk null;
            break :blk switch (text_value) {
                .string => |text| if (text.len > 0) text else null,
                else => null,
            };
        },
        else => null,
    };
}

fn datasetTextLimitBytes(allocator: std.mem.Allocator) !usize {
    const raw = std.process.getEnvVarOwned(allocator, "JAIDE_MAX_SEQ_LEN") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return 256 * 4,
        else => return err,
    };
    defer allocator.free(raw);
    const sequence_length = std.fmt.parseInt(usize, raw, 10) catch return error.InvalidEnvironmentValue;
    if (sequence_length == 0) return error.InvalidEnvironmentValue;
    return std.math.mul(usize, sequence_length, 4) catch return error.InvalidEnvironmentValue;
}

fn truncateDatasetText(text: []const u8, maximum_bytes: usize) []const u8 {
    const length = @min(text.len, maximum_bytes);
    var boundary = length;
    while (boundary > 0 and boundary < text.len and (text[boundary] & 0xC0) == 0x80) {
        boundary -= 1;
    }
    return text[0..boundary];
}

fn openDatasetFile(dataset_path: []const u8) !std.fs.File {
    if (dataset_path.len == 0) return error.InvalidDatasetPath;
    if (std.fs.path.isAbsolute(dataset_path)) {
        return std.fs.openFileAbsolute(dataset_path, .{ .mode = .read_only });
    }
    return std.fs.cwd().openFile(dataset_path, .{ .mode = .read_only });
}

fn appendDatasetRange(
    allocator: std.mem.Allocator,
    dataset_path: []const u8,
    max_line_size: usize,
    start_valid_index: usize,
    end_valid_index: usize,
    samples: *std.ArrayList([]const u8),
) !usize {
    if (end_valid_index <= start_valid_index) return 0;
    const text_limit_bytes = try datasetTextLimitBytes(allocator);

    const file = try openDatasetFile(dataset_path);
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    var stream = buffered_reader.reader();
    var valid_index: usize = 0;
    var appended: usize = 0;

    while (true) {
        var arena = core_memory.ArenaAllocator.init(allocator, 64 * 1024);
        defer arena.deinit();

        const maybe_line = try stream.readUntilDelimiterOrEofAlloc(
            arena.allocator(),
            '\n',
            max_line_size,
        );
        const line = maybe_line orelse break;

        if (valid_index >= end_valid_index) break;

        const maybe_text = try extractDatasetText(&arena, line);
        if (maybe_text) |text| {
            if (valid_index >= start_valid_index and valid_index < end_valid_index) {
                const persistent_text = try allocator.dupe(u8, truncateDatasetText(text, text_limit_bytes));
                samples.append(persistent_text) catch |err| {
                    allocator.free(persistent_text);
                    return err;
                };
                appended = try std.math.add(usize, appended, 1);
            }

            valid_index = try std.math.add(usize, valid_index, 1);
        }
    }

    const expected = end_valid_index - start_valid_index;
    if (appended != expected) {
        return error.DatasetSampleCountMismatch;
    }

    return appended;
}

fn fnv1aHashBytes(data: []const u8) u64 {
    var hash: u64 = 14695981039346656037;
    for (data) |byte| {
        hash ^= @as(u64, byte);
        hash *%= 1099511628211;
    }
    return hash;
}

fn parseOptionalEnvironmentUsize(allocator: std.mem.Allocator, name: []const u8) !?usize {
    const owned = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        else => return err,
    };
    defer allocator.free(owned);
    if (owned.len == 0) return error.InvalidEnvironmentValue;
    return std.fmt.parseInt(usize, owned, 10) catch return error.InvalidEnvironmentValue;
}

fn parseOptionalEnvironmentF32(allocator: std.mem.Allocator, name: []const u8) !?f32 {
    const owned = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        else => return err,
    };
    defer allocator.free(owned);
    if (owned.len == 0) return error.InvalidEnvironmentValue;
    const value = std.fmt.parseFloat(f32, owned) catch return error.InvalidEnvironmentValue;
    if (!std.math.isFinite(value)) return error.InvalidEnvironmentValue;
    return value;
}

fn parseOptionalEnvironmentBool(allocator: std.mem.Allocator, name: []const u8) !?bool {
    const owned = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        else => return err,
    };
    defer allocator.free(owned);
    if (std.mem.eql(u8, owned, "1") or std.mem.eql(u8, owned, "true")) return true;
    if (std.mem.eql(u8, owned, "0") or std.mem.eql(u8, owned, "false")) return false;
    return error.InvalidEnvironmentValue;
}

fn loadDataset(
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    dataset_path: []const u8,
    max_line_size: usize,
) ![][]const u8 {
    if (coordinator.world_size == 0) return error.InvalidWorldSize;
    if (coordinator.rank >= coordinator.world_size) return error.InvalidRank;

    const environment_total = try parseOptionalEnvironmentUsize(allocator, "JAIDE_TOTAL_SAMPLES");
    const environment_maximum = try parseOptionalEnvironmentUsize(allocator, "JAIDE_MAX_SAMPLES");
    var valid_sample_count = environment_total orelse 0;
    if (environment_total != null and valid_sample_count == 0) return error.InvalidEnvironmentValue;

    if (valid_sample_count == 0) {
        std.debug.print(
            "[Rank {d}] WARN: JAIDE_TOTAL_SAMPLES not provided, scanning valid records\n",
            .{coordinator.rank},
        );

        const count_file = try openDatasetFile(dataset_path);
        defer count_file.close();

        var buffered_reader = std.io.bufferedReader(count_file.reader());
        var stream = buffered_reader.reader();

        while (true) {
            var arena = core_memory.ArenaAllocator.init(allocator, 64 * 1024);
            defer arena.deinit();

            const maybe_line = try stream.readUntilDelimiterOrEofAlloc(
                arena.allocator(),
                '\n',
                max_line_size,
            );
            const line = maybe_line orelse break;

            const maybe_text = extractDatasetText(&arena, line) catch |err| switch (err) {
                error.OutOfMemory => return err,
                else => null,
            };

            if (maybe_text != null) {
                valid_sample_count = try std.math.add(
                    usize,
                    valid_sample_count,
                    1,
                );
            }
        }
    }

    if (environment_maximum) |maximum| {
        if (maximum == 0) return error.InvalidEnvironmentValue;
        if (maximum < valid_sample_count) valid_sample_count = maximum;
    }

    if (valid_sample_count == 0) {
        std.debug.print(
            "[Rank {d}] ERROR: dataset is empty or contains no valid records\n",
            .{coordinator.rank},
        );
        return error.EmptyDataset;
    }

    const partition = try dataset_partition.bounds(valid_sample_count, coordinator.world_size, coordinator.rank);
    const samples_per_rank = partition.count;
    const start_valid_index = partition.start;
    const end_valid_index = try std.math.add(usize, start_valid_index, samples_per_rank);

    var samples = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (samples.items) |sample| {
            allocator.free(sample);
        }
        samples.deinit();
    }

    try samples.ensureTotalCapacity(samples_per_rank);
    _ = try appendDatasetRange(
        allocator,
        dataset_path,
        max_line_size,
        start_valid_index,
        end_valid_index,
        &samples,
    );

    if (samples.items.len != samples_per_rank) {
        std.debug.print(
            "[Rank {d}] ERROR: loaded {d} samples, expected {d}\n",
            .{
                coordinator.rank,
                samples.items.len,
                samples_per_rank,
            },
        );
        return error.DatasetSampleCountMismatch;
    }

    std.debug.print(
        "[Rank {d}] Loaded {d} samples from {d} valid records\n",
        .{
            coordinator.rank,
            samples.items.len,
            valid_sample_count,
        },
    );

    return samples.toOwnedSlice();
}

fn loadDatasetHashes(
    allocator: std.mem.Allocator,
    dataset_path: []const u8,
    max_line_size: usize,
) ![]u64 {
    const maximum = (try parseOptionalEnvironmentUsize(allocator, "JAIDE_MAX_SAMPLES")) orelse 0;
    const file = try openDatasetFile(dataset_path);
    defer file.close();
    var buffered_reader = std.io.bufferedReader(file.reader());
    var stream = buffered_reader.reader();
    var hashes = std.ArrayList(u64).init(allocator);
    errdefer hashes.deinit();
    var seen = std.AutoHashMap(u64, void).init(allocator);
    defer seen.deinit();
    var valid_count: usize = 0;
    while (maximum == 0 or valid_count < maximum) {
        var arena = core_memory.ArenaAllocator.init(allocator, 64 * 1024);
        defer arena.deinit();
        const line = try stream.readUntilDelimiterOrEofAlloc(arena.allocator(), '\n', max_line_size) orelse break;
        const text = try extractDatasetText(&arena, line) orelse continue;
        valid_count = try std.math.add(usize, valid_count, 1);
        const hash = fnv1aHashBytes(text);
        if (seen.contains(hash)) continue;
        try seen.put(hash, {});
        try hashes.append(hash);
    }
    if (hashes.items.len == 0) return error.EmptyDataset;
    return hashes.toOwnedSlice();
}

fn loadTokenizerDataset(
    allocator: std.mem.Allocator,
    dataset_path: []const u8,
    max_line_size: usize,
) ![][]const u8 {
    const maximum_samples = (try parseOptionalEnvironmentUsize(allocator, "JAIDE_MAX_SAMPLES")) orelse 0;
    const text_limit_bytes = try datasetTextLimitBytes(allocator);

    const file = try openDatasetFile(dataset_path);
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    var stream = buffered_reader.reader();
    var samples = std.ArrayList([]const u8).init(allocator);

    errdefer {
        for (samples.items) |sample| {
            allocator.free(sample);
        }
        samples.deinit();
    }

    while (true) {
        var arena = core_memory.ArenaAllocator.init(allocator, 64 * 1024);
        defer arena.deinit();

        const maybe_line = try stream.readUntilDelimiterOrEofAlloc(
            arena.allocator(),
            '\n',
            max_line_size,
        );
        const line = maybe_line orelse break;

        const maybe_text = try extractDatasetText(&arena, line);
        if (maybe_text) |text| {
            const persistent_text = try allocator.dupe(u8, truncateDatasetText(text, text_limit_bytes));
            samples.append(persistent_text) catch |err| {
                allocator.free(persistent_text);
                return err;
            };

            if (maximum_samples > 0 and samples.items.len >= maximum_samples) {
                break;
            }
        }
    }

    if (samples.items.len == 0) {
        return error.EmptyDataset;
    }

    return samples.toOwnedSlice();
}

const StageMarkerState = enum {
    pending,
    ok,
    fail,
};

const EpochArtifactWorker = struct {
    allocator: std.mem.Allocator,
    trainer: *DistributedTrainerFuthark,
    checkpoint_path: []u8,
    mutex: std.Thread.Mutex,
    condition: std.Thread.Condition,
    capture_done: bool,
    capture_failure: ?anyerror,
    write_failure: ?anyerror,
    capture_duration_ns: u64,
    write_duration_ns: u64,
    thread: ?std.Thread,

    fn start(
        allocator: std.mem.Allocator,
        trainer: *DistributedTrainerFuthark,
        checkpoint_path: []u8,
    ) !*EpochArtifactWorker {
        const worker = try allocator.create(EpochArtifactWorker);
        errdefer allocator.destroy(worker);
        worker.* = EpochArtifactWorker{
            .allocator = allocator,
            .trainer = trainer,
            .checkpoint_path = checkpoint_path,
            .mutex = .{},
            .condition = .{},
            .capture_done = false,
            .capture_failure = null,
            .write_failure = null,
            .capture_duration_ns = 0,
            .write_duration_ns = 0,
            .thread = null,
        };
        worker.thread = try std.Thread.spawn(.{}, EpochArtifactWorker.run, .{worker});
        return worker;
    }

    fn run(self: *EpochArtifactWorker) void {
        const capture_started = std.time.nanoTimestamp();
        const snapshot = self.trainer.captureCheckpointSnapshot() catch |err| {
            self.mutex.lock();
            self.capture_failure = err;
            self.capture_done = true;
            self.condition.broadcast();
            self.mutex.unlock();
            return;
        };
        const capture_elapsed = std.time.nanoTimestamp() - capture_started;
        self.mutex.lock();
        self.capture_duration_ns = if (capture_elapsed > 0) @intCast(capture_elapsed) else 0;
        self.capture_done = true;
        self.condition.broadcast();
        self.mutex.unlock();

        const write_started = std.time.nanoTimestamp();
        DistributedTrainerFuthark.writeCheckpointSnapshotFile(snapshot, self.checkpoint_path) catch |err| {
            self.mutex.lock();
            self.write_failure = err;
            self.mutex.unlock();
        };
        const write_elapsed = std.time.nanoTimestamp() - write_started;
        snapshot.deinit();
        self.allocator.destroy(snapshot);

        self.mutex.lock();
        self.write_duration_ns = if (write_elapsed > 0) @intCast(write_elapsed) else 0;
        const failed = self.capture_failure != null or self.write_failure != null;
        self.mutex.unlock();
        if (!failed) {
            std.debug.print(
                "Checkpoint saved: {s} capture_ms={d} write_ms={d}\n",
                .{ self.checkpoint_path, self.capture_duration_ns / std.time.ns_per_ms, self.write_duration_ns / std.time.ns_per_ms },
            );
        }
    }

    fn awaitCapture(self: *EpochArtifactWorker) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (!self.capture_done) {
            self.condition.wait(&self.mutex);
        }
        if (self.capture_failure) |err| return err;
    }

    fn joinAndDestroy(self: *EpochArtifactWorker) ?anyerror {
        if (self.thread) |thread| {
            thread.join();
            self.thread = null;
        }
        self.mutex.lock();
        const failure: ?anyerror = if (self.capture_failure) |err|
            err
        else
            self.write_failure;
        self.mutex.unlock();
        self.allocator.free(self.checkpoint_path);
        self.allocator.destroy(self);
        return failure;
    }
};

const STAGE_SYNC_TIMEOUT_NS: i128 = 30 * 60 * std.time.ns_per_s;
const STAGE_SYNC_POLL_NS: u64 = 50 * std.time.ns_per_ms;

fn readStageMarker(path: []const u8) StageMarkerState {
    const file = std.fs.openFileAbsolute(path, .{ .mode = .read_only }) catch return .pending;
    defer file.close();

    var status_buffer: [4]u8 = undefined;
    const bytes_read = file.readAll(&status_buffer) catch return .pending;
    if (std.mem.eql(u8, status_buffer[0..bytes_read], "ok")) return .ok;
    if (std.mem.eql(u8, status_buffer[0..bytes_read], "fail")) return .fail;
    return .pending;
}

fn writeStageMarker(path: []const u8, state: StageMarkerState) !void {
    const marker = switch (state) {
        .ok => "ok",
        .fail => "fail",
        .pending => return error.InvalidStageMarker,
    };

    const file = try std.fs.createFileAbsolute(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(marker);
    try file.sync();
}

fn waitForStageMarker(path: []const u8, deadline_ns: i128) !StageMarkerState {
    while (true) {
        const state = readStageMarker(path);
        if (state != .pending) return state;
        if (std.time.nanoTimestamp() >= deadline_ns) return error.StageSynchronizationTimeout;
        std.time.sleep(STAGE_SYNC_POLL_NS);
    }
}

fn waitForAllRankMarkers(
    allocator: std.mem.Allocator,
    status_base_path: []const u8,
    stage_name: []const u8,
    world_size: usize,
    suffix: []const u8,
    deadline_ns: i128,
) !bool {
    while (true) {
        var every_rank_reported = true;
        var every_rank_succeeded = true;
        var inspected_rank: usize = 0;

        while (inspected_rank < world_size) : (inspected_rank += 1) {
            const marker_path = try std.fmt.allocPrint(
                allocator,
                "{s}.{s}.rank_{d}.{s}",
                .{ status_base_path, stage_name, inspected_rank, suffix },
            );
            const marker_state = readStageMarker(marker_path);
            allocator.free(marker_path);

            switch (marker_state) {
                .pending => {
                    every_rank_reported = false;
                    break;
                },
                .ok => {},
                .fail => every_rank_succeeded = false,
            }
        }

        if (every_rank_reported) return every_rank_succeeded;
        if (std.time.nanoTimestamp() >= deadline_ns) return error.StageSynchronizationTimeout;
        std.time.sleep(STAGE_SYNC_POLL_NS);
    }
}

fn removeStageMarkers(
    allocator: std.mem.Allocator,
    status_base_path: []const u8,
    stage_name: []const u8,
    world_size: usize,
) void {
    const ready_path = std.fmt.allocPrint(
        allocator,
        "{s}.{s}.ready",
        .{ status_base_path, stage_name },
    ) catch return;
    defer allocator.free(ready_path);
    std.fs.deleteFileAbsolute(ready_path) catch {};

    const global_path = std.fmt.allocPrint(
        allocator,
        "{s}.{s}.global.status",
        .{ status_base_path, stage_name },
    ) catch return;
    defer allocator.free(global_path);
    std.fs.deleteFileAbsolute(global_path) catch {};

    var cleanup_rank: usize = 0;
    while (cleanup_rank < world_size) : (cleanup_rank += 1) {
        const status_path = std.fmt.allocPrint(
            allocator,
            "{s}.{s}.rank_{d}.status",
            .{ status_base_path, stage_name, cleanup_rank },
        ) catch continue;
        defer allocator.free(status_path);
        std.fs.deleteFileAbsolute(status_path) catch {};

        const ack_path = std.fmt.allocPrint(
            allocator,
            "{s}.{s}.rank_{d}.ack",
            .{ status_base_path, stage_name, cleanup_rank },
        ) catch continue;
        defer allocator.free(ack_path);
        std.fs.deleteFileAbsolute(ack_path) catch {};
    }
}

fn synchronizeStageStatus(
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    status_base_path: []const u8,
    stage_name: []const u8,
    local_error: ?anyerror,
) !void {
    const ready_path = try std.fmt.allocPrint(
        allocator,
        "{s}.{s}.ready",
        .{ status_base_path, stage_name },
    );
    defer allocator.free(ready_path);

    const rank_status_path = try std.fmt.allocPrint(
        allocator,
        "{s}.{s}.rank_{d}.status",
        .{ status_base_path, stage_name, coordinator.rank },
    );
    defer allocator.free(rank_status_path);

    const global_status_path = try std.fmt.allocPrint(
        allocator,
        "{s}.{s}.global.status",
        .{ status_base_path, stage_name },
    );
    defer allocator.free(global_status_path);

    const rank_ack_path = try std.fmt.allocPrint(
        allocator,
        "{s}.{s}.rank_{d}.ack",
        .{ status_base_path, stage_name, coordinator.rank },
    );
    defer allocator.free(rank_ack_path);

    const deadline_ns = std.time.nanoTimestamp() + STAGE_SYNC_TIMEOUT_NS;

    if (coordinator.isRoot()) {
        removeStageMarkers(allocator, status_base_path, stage_name, coordinator.world_size);
        try writeStageMarker(ready_path, .ok);
    } else {
        _ = try waitForStageMarker(ready_path, deadline_ns);
    }

    const local_state: StageMarkerState = if (local_error == null) .ok else .fail;
    try writeStageMarker(rank_status_path, local_state);

    var global_state: StageMarkerState = undefined;
    if (coordinator.isRoot()) {
        const all_succeeded = try waitForAllRankMarkers(
            allocator,
            status_base_path,
            stage_name,
            coordinator.world_size,
            "status",
            deadline_ns,
        );
        global_state = if (all_succeeded) .ok else .fail;
        try writeStageMarker(global_status_path, global_state);
    } else {
        global_state = try waitForStageMarker(global_status_path, deadline_ns);
    }

    try writeStageMarker(rank_ack_path, .ok);

    if (coordinator.isRoot()) {
        _ = try waitForAllRankMarkers(
            allocator,
            status_base_path,
            stage_name,
            coordinator.world_size,
            "ack",
            deadline_ns,
        );
        removeStageMarkers(allocator, status_base_path, stage_name, coordinator.world_size);
    }

    if (global_state != .ok) {
        return local_error orelse error.DistributedStageFailed;
    }
}

fn deployToModal(allocator: std.mem.Allocator, args: [][:0]u8) !void {
    const api_token = try std.process.getEnvVarOwned(
        allocator,
        "MODAL_API_TOKEN",
    );
    defer allocator.free(api_token);

    const model_path: []const u8 = if (args.len > 0)
        args[0]
    else
        "/checkpoints/latest";

    const dataset_path: []const u8 = if (args.len > 1)
        args[1]
    else
        "/data/dataset/train.jsonl";

    var job_config = try modal_gpu.TrainingJobConfig.fromEnvironment(allocator);
    defer job_config.deinit();

    var client = try modal_gpu.ModalGPUClient.init(allocator, api_token, &job_config);
    defer client.deinit();

    const job_id = try client.deployTrainingJob(model_path, dataset_path);
    defer allocator.free(job_id);

    std.debug.print("Deployed training job: {s}\n", .{job_id});

    const max_poll_attempts: usize = 360;
    var poll_attempt: usize = 0;

    while (poll_attempt < max_poll_attempts) : (poll_attempt += 1) {
        const status_raw = try client.getJobStatus(job_id);
        defer allocator.free(status_raw);

        std.debug.print("Job status raw: {s}\n", .{status_raw});

        const parsed_status = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            status_raw,
            .{ .allocate = .alloc_always },
        ) catch null;

        if (parsed_status) |parsed| {
            defer parsed.deinit();

            const status_field: ?[]const u8 = switch (parsed.value) {
                .object => |object| blk: {
                    const status_value = object.get("status") orelse break :blk null;
                    break :blk switch (status_value) {
                        .string => |status| status,
                        else => null,
                    };
                },
                else => null,
            };

            if (status_field) |status| {
                std.debug.print("Job status: {s}\n", .{status});

                if (std.mem.eql(u8, status, "completed")) {
                    return;
                }

                if (std.mem.eql(u8, status, "failed")) {
                    return error.ModalJobFailed;
                }
            }
        }

        std.time.sleep(30 * std.time.ns_per_s);
    }

    std.debug.print(
        "Timeout waiting for Modal job completion after {d} polls\n",
        .{max_poll_attempts},
    );
    return error.ModalJobTimeout;
}

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = general_purpose_allocator.deinit();
    const allocator = general_purpose_allocator.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len > 1 and std.mem.eql(u8, args[1], "--deploy")) {
        return deployToModal(allocator, args[2..]);
    }

    const startup_started = std.time.nanoTimestamp();

    const world_size_string = try std.process.getEnvVarOwned(
        allocator,
        "WORLD_SIZE",
    );
    defer allocator.free(world_size_string);

    const world_size = try std.fmt.parseInt(
        usize,
        world_size_string,
        10,
    );

    if (world_size == 0) {
        return error.InvalidWorldSize;
    }

    const rank_string = try std.process.getEnvVarOwned(
        allocator,
        "RANK",
    );
    defer allocator.free(rank_string);

    const rank = try std.fmt.parseInt(usize, rank_string, 10);

    if (rank >= world_size) {
        return error.InvalidRank;
    }

    var local_rank_string_owned: ?[]u8 = null;
    const local_rank: usize = blk: {
        local_rank_string_owned = std.process.getEnvVarOwned(
            allocator,
            "LOCAL_RANK",
        ) catch null;

        if (local_rank_string_owned) |owned| {
            break :blk try std.fmt.parseInt(usize, owned, 10);
        }

        std.debug.print(
            "[Rank {d}] WARN: LOCAL_RANK not set, using RANK ({d}) for device selection\n",
            .{ rank, rank },
        );
        break :blk rank;
    };
    defer if (local_rank_string_owned) |owned| allocator.free(owned);

    const master_addr = try std.process.getEnvVarOwned(
        allocator,
        "MASTER_ADDR",
    );
    defer allocator.free(master_addr);

    const master_port = try std.process.getEnvVarOwned(
        allocator,
        "MASTER_PORT",
    );
    defer allocator.free(master_port);

    std.debug.print(
        "============================================================\n",
        .{},
    );
    std.debug.print(
        "JAIDE v40 Distributed Training (Futhark GPU Acceleration)\n",
        .{},
    );
    std.debug.print(
        "============================================================\n",
        .{},
    );
    std.debug.print("Rank: {d}/{d}\n", .{ rank, world_size });
    std.debug.print(
        "Master addr/port: {s}:{s}\n",
        .{ master_addr, master_port },
    );
    std.debug.print(
        "============================================================\n\n",
        .{},
    );

    var nccl_id_path_owned: ?[]u8 = null;
    const nccl_id_path: []const u8 = blk: {
        nccl_id_path_owned = std.process.getEnvVarOwned(
            allocator,
            "JAIDE_NCCL_ID_PATH",
        ) catch null;
        break :blk nccl_id_path_owned orelse "/tmp/jaide_nccl_id";
    };
    defer if (nccl_id_path_owned) |owned| allocator.free(owned);

    std.debug.print(
        "[Rank {d}] NCCL ID exchange path: {s}\n",
        .{ rank, nccl_id_path },
    );

    const nccl_ready_path = try std.fmt.allocPrint(
        allocator,
        "{s}.ready",
        .{nccl_id_path},
    );
    defer allocator.free(nccl_ready_path);

    var nccl_id: nccl.ncclUniqueId = std.mem.zeroes(nccl.ncclUniqueId);

    if (world_size > 1) {
        if (rank == 0) {
            std.fs.deleteFileAbsolute(nccl_ready_path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };

            std.fs.deleteFileAbsolute(nccl_id_path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };

            const result = nccl.ncclGetUniqueId(&nccl_id);
            if (result != .ncclSuccess) {
                return error.NCCLGetUniqueIdFailed;
            }

            {
                const id_file = try std.fs.createFileAbsolute(
                    nccl_id_path,
                    .{ .truncate = true },
                );
                defer id_file.close();

                try id_file.writeAll(std.mem.asBytes(&nccl_id));
                try id_file.sync();
            }

            {
                const ready_file = try std.fs.createFileAbsolute(
                    nccl_ready_path,
                    .{ .truncate = true },
                );
                defer ready_file.close();

                try ready_file.writeAll("ready");
                try ready_file.sync();
            }

            std.debug.print(
                "[Rank 0] Generated NCCL ID at {s}\n",
                .{nccl_id_path},
            );
        } else {
            const maximum_attempts: usize = 3000;
            var attempts: usize = 0;

            while (attempts < maximum_attempts) : (attempts += 1) {
                const ready_file = std.fs.openFileAbsolute(
                    nccl_ready_path,
                    .{ .mode = .read_only },
                ) catch {
                    std.time.sleep(100 * std.time.ns_per_ms);
                    continue;
                };
                ready_file.close();
                break;
            }

            if (attempts >= maximum_attempts) {
                return error.NCCLIdTimeout;
            }

            const id_file = try std.fs.openFileAbsolute(
                nccl_id_path,
                .{ .mode = .read_only },
            );
            defer id_file.close();

            const bytes_read = try id_file.readAll(std.mem.asBytes(&nccl_id));
            if (bytes_read != @sizeOf(nccl.ncclUniqueId)) {
                return error.NCCLIdReadFailed;
            }

            std.debug.print(
                "[Rank {d}] Loaded NCCL ID from rank 0\n",
                .{rank},
            );
        }
    }

    var coordinator = try GPUCoordinator.init(
        world_size,
        rank,
        local_rank,
        nccl_id,
    );
    defer coordinator.deinit();

    std.debug.print(
        "[Rank {d}] GPU coordinator initialized\n",
        .{rank},
    );

    const model_dim_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_MODEL_DIM",
    ) catch null;
    defer if (model_dim_string_owned) |owned| allocator.free(owned);

    const model_dim: usize = if (model_dim_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_MODEL_DIM='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        98304;

    if (model_dim == 0) {
        return error.InvalidConfig;
    }

    const num_layers_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_LAYERS",
    ) catch null;
    defer if (num_layers_string_owned) |owned| allocator.free(owned);

    const num_layers: usize = if (num_layers_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_LAYERS='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        24;

    if (num_layers == 0) {
        return error.InvalidConfig;
    }

    const batch_size_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_BATCH_SIZE",
    ) catch null;
    defer if (batch_size_string_owned) |owned| allocator.free(owned);

    const local_batch_size: usize = if (batch_size_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_BATCH_SIZE='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        16;

    if (local_batch_size == 0) {
        return error.InvalidConfig;
    }

    const epochs_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_EPOCHS",
    ) catch null;
    defer if (epochs_string_owned) |owned| allocator.free(owned);

    const num_epochs: usize = if (epochs_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_EPOCHS='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        1;

    const checkpoint_interval_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_CHECKPOINT_INTERVAL_EPOCHS",
    ) catch null;
    defer if (checkpoint_interval_owned) |owned| allocator.free(owned);
    const checkpoint_interval_epochs: usize = if (checkpoint_interval_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch return error.InvalidConfig
    else
        5;

    const learning_rate_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_LEARNING_RATE",
    ) catch null;
    defer if (learning_rate_string_owned) |owned| allocator.free(owned);

    const learning_rate: f32 = if (learning_rate_string_owned) |value|
        std.fmt.parseFloat(f32, value) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_LEARNING_RATE='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        0.0003;

    if (!std.math.isFinite(learning_rate) or learning_rate <= 0.0) {
        return error.InvalidConfig;
    }

    const clip_min_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_CLIP_MIN",
    ) catch null;
    defer if (clip_min_string_owned) |owned| allocator.free(owned);

    const clip_min: f32 = if (clip_min_string_owned) |value|
        std.fmt.parseFloat(f32, value) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_CLIP_MIN='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        -5.0;

    const clip_max_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_CLIP_MAX",
    ) catch null;
    defer if (clip_max_string_owned) |owned| allocator.free(owned);

    const clip_max: f32 = if (clip_max_string_owned) |value|
        std.fmt.parseFloat(f32, value) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_CLIP_MAX='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        5.0;

    if (!std.math.isFinite(clip_min) or !std.math.isFinite(clip_max) or clip_min >= clip_max) {
        return error.InvalidConfig;
    }

    const reasoning_cycles_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_REASONING_CYCLES",
    ) catch null;
    defer if (reasoning_cycles_string_owned) |owned| allocator.free(owned);

    const reasoning_cycles: usize = if (reasoning_cycles_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_REASONING_CYCLES='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        1;

    if (reasoning_cycles == 0) {
        return error.InvalidConfig;
    }

    const relational_pass_interval_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_RELATIONAL_PASS_INTERVAL",
    ) catch null;
    defer if (relational_pass_interval_string_owned) |owned| allocator.free(owned);

    const relational_pass_interval: usize = if (relational_pass_interval_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_RELATIONAL_PASS_INTERVAL='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        1000;

    if (relational_pass_interval == 0) {
        return error.InvalidConfig;
    }
    const training_sample_count = (try parseOptionalEnvironmentUsize(allocator, "JAIDE_MAX_SAMPLES")) orelse 0;

    const reconstruction_alpha_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_RECONSTRUCTION_ALPHA",
    ) catch null;
    defer if (reconstruction_alpha_string_owned) |owned| allocator.free(owned);

    const reconstruction_alpha: f32 = if (reconstruction_alpha_string_owned) |value|
        std.fmt.parseFloat(f32, value) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_RECONSTRUCTION_ALPHA='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        0.3;

    if (!std.math.isFinite(reconstruction_alpha) or reconstruction_alpha < 0.0 or reconstruction_alpha > 1.0) {
        std.debug.print(
            "[Rank {d}] ERROR: JAIDE_RECONSTRUCTION_ALPHA must be within [0.0, 1.0], got {d}\n",
            .{ rank, reconstruction_alpha },
        );
        return error.InvalidConfig;
    }

    const phase_a_steps_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_PHASE_A_STEPS",
    ) catch null;
    defer if (phase_a_steps_string_owned) |owned| allocator.free(owned);

    const phase_a_steps: u64 = if (phase_a_steps_string_owned) |value|
        std.fmt.parseInt(u64, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_PHASE_A_STEPS='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        500;

    const phase_b_steps_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_PHASE_B_STEPS",
    ) catch null;
    defer if (phase_b_steps_string_owned) |owned| allocator.free(owned);

    const phase_b_steps: u64 = if (phase_b_steps_string_owned) |value|
        std.fmt.parseInt(u64, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_PHASE_B_STEPS='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        2000;

    _ = std.math.add(u64, phase_a_steps, phase_b_steps) catch {
        std.debug.print(
            "[Rank {d}] ERROR: JAIDE_PHASE_A_STEPS + JAIDE_PHASE_B_STEPS overflows\n",
            .{rank},
        );
        return error.InvalidConfig;
    };

    const shuffle_target_control = (try parseOptionalEnvironmentBool(allocator, "JAIDE_SHUFFLE_TARGET_CONTROL")) orelse false;
    const target_source_frozen = (try parseOptionalEnvironmentBool(allocator, "JAIDE_TARGET_SOURCE_FROZEN")) orelse true;
    const spectral_depth_compensation = (try parseOptionalEnvironmentBool(allocator, "JAIDE_SPECTRAL_DEPTH_COMPENSATION")) orelse true;
    const grad_mean = (try parseOptionalEnvironmentBool(allocator, "JAIDE_GRAD_MEAN")) orelse true;
    const use_normalized_gradient_flow = (try parseOptionalEnvironmentBool(allocator, "JAIDE_NORMALIZED_GRADIENT_FLOW")) orelse true;
    const spectral_target_norm = (try parseOptionalEnvironmentF32(allocator, "JAIDE_SPECTRAL_NORM_TARGET")) orelse 0.9;
    const spectral_iterations = (try parseOptionalEnvironmentUsize(allocator, "JAIDE_SPECTRAL_POWER_ITERATIONS")) orelse 30;
    const init_spectral_iterations = (try parseOptionalEnvironmentUsize(allocator, "JAIDE_INIT_SPECTRAL_ITERS")) orelse 0;
    const spectral_interval = (try parseOptionalEnvironmentUsize(allocator, "JAIDE_SPECTRAL_INTERVAL")) orelse 10;
    const trust_ratio = (try parseOptionalEnvironmentF32(allocator, "JAIDE_SFD_TRUST_RATIO")) orelse 0.1;
    const weight_floor = (try parseOptionalEnvironmentF32(allocator, "JAIDE_SFD_WEIGHT_FLOOR")) orelse 1e-3;
    const gradient_clip_norm = (try parseOptionalEnvironmentF32(allocator, "JAIDE_GRADIENT_CLIP_NORM")) orelse 1.0;
    const logdet_weight = (try parseOptionalEnvironmentF32(allocator, "JAIDE_LOGDET_WEIGHT")) orelse dtf.fused_logdet_weight_default;
    const checkpoint_version = (try parseOptionalEnvironmentUsize(allocator, "JAIDE_CHECKPOINT_VERSION")) orelse dtf.CHECKPOINT_VERSION;
    const seed_offset = (try parseOptionalEnvironmentUsize(allocator, "JAIDE_SEED_OFFSET")) orelse 0;
    const embedding_seed = std.math.add(u64, 42, std.math.cast(u64, seed_offset) orelse return error.InvalidConfig) catch return error.InvalidConfig;
    if (checkpoint_version != dtf.CHECKPOINT_VERSION) return error.InvalidConfig;
    if (spectral_target_norm <= 0.0 or spectral_interval == 0 or trust_ratio <= 0.0 or trust_ratio > 1.0 or weight_floor <= 0.0 or gradient_clip_norm <= 0.0) return error.InvalidConfig;

    const dataset_path_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_DATASET",
    ) catch null;
    defer if (dataset_path_owned) |owned| allocator.free(owned);

    const dataset_path: []const u8 = dataset_path_owned orelse
        "/data/dataset/finephrase_bench.bin";

    std.debug.print(
        "[Rank {d}] Loading dataset from {s}\n",
        .{ rank, dataset_path },
    );

    const vocab_path_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_TOKENIZER_VOCAB",
    ) catch null;
    defer if (vocab_path_owned) |owned| allocator.free(owned);
    const vocab_path = vocab_path_owned orelse "/checkpoints/tokenizer.vocab";

    const vocab_size_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_VOCAB_SIZE",
    ) catch null;
    defer if (vocab_size_string_owned) |owned| allocator.free(owned);
    const vocab_size: u32 = if (vocab_size_string_owned) |value|
        std.fmt.parseInt(u32, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_VOCAB_SIZE='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        32000;
    if (vocab_size == 0) return error.InvalidConfig;

    const vocab_ready_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_VOCAB_READY",
    ) catch null;
    defer if (vocab_ready_string_owned) |owned| allocator.free(owned);

    const vocab_ready = if (vocab_ready_string_owned) |value|
        std.mem.eql(u8, value, "1")
    else
        false;
    if (!vocab_ready) return error.TokenizerNotPrecomputed;

    const tokenizer_started = std.time.nanoTimestamp();
    if (!vocab_ready) {
        var vocabulary_error: ?anyerror = null;

        if (coordinator.isRoot()) {
            vocabulary_training: {
                const vocabulary_samples = loadTokenizerDataset(
                    allocator,
                    dataset_path,
                    10 * 1024 * 1024,
                ) catch |err| {
                    vocabulary_error = err;
                    break :vocabulary_training;
                };
                defer {
                    for (vocabulary_samples) |sample| {
                        allocator.free(sample);
                    }
                    allocator.free(vocabulary_samples);
                }

                var temporary_tokenizer = MGT.init(
                    allocator,
                    &.{},
                    &.{},
                    vocab_size,
                    .english,
                ) catch |err| {
                    vocabulary_error = err;
                    break :vocabulary_training;
                };
                defer temporary_tokenizer.deinit();

                temporary_tokenizer.trainBPE(
                    vocabulary_samples,
                    vocab_size,
                ) catch |err| {
                    vocabulary_error = err;
                    break :vocabulary_training;
                };

                std.fs.makeDirAbsolute("/checkpoints") catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => {
                        vocabulary_error = err;
                        break :vocabulary_training;
                    },
                };

                temporary_tokenizer.saveVocab(vocab_path) catch |err| {
                    vocabulary_error = err;
                    break :vocabulary_training;
                };

                std.debug.print(
                    "[Rank 0] Tokenizer trained on {d} records and saved to {s}\n",
                    .{
                        vocabulary_samples.len,
                        vocab_path,
                    },
                );
            }
        }

        synchronizeStageStatus(
            allocator,
            &coordinator,
            nccl_id_path,
            "vocabulary",
            vocabulary_error,
        ) catch |err| {
            std.debug.print(
                "[Rank {d}] vocabulary stage failed: {}\n",
                .{ rank, err },
            );
            if (@errorReturnTrace()) |trace| {
                std.debug.dumpStackTrace(trace.*);
            }
            return err;
        };
    } else {
        std.debug.print(
            "[Rank {d}] Reusing tokenizer vocabulary at {s}\n",
            .{ rank, vocab_path },
        );
    }

    var model_initialization_elapsed: i128 = 0;
    var tokenizer = try MGT.init(
        allocator,
        &.{},
        &.{},
        vocab_size,
        .english,
    );

    var trainer = trainer_initialization: {
        errdefer tokenizer.deinit();

        var tokenizer_load_error: ?anyerror = null;

        tokenizer.loadVocab(vocab_path) catch |err| {
            tokenizer_load_error = err;
        };

        synchronizeStageStatus(
            allocator,
            &coordinator,
            nccl_id_path,
            "tokenizer_load",
            tokenizer_load_error,
        ) catch |err| {
            std.debug.print(
                "[Rank {d}] tokenizer load failed: {}\n",
                .{ rank, err },
            );
            if (@errorReturnTrace()) |trace| {
                std.debug.dumpStackTrace(trace.*);
            }
            return err;
        };

        const tokenizer_elapsed = std.time.nanoTimestamp() - tokenizer_started;
        std.debug.print(
            "[Rank {d}] Tokenizer loaded, next_token_id={d} tokenizer_ms={d}\n",
            .{ rank, tokenizer.next_token_id, @divTrunc(tokenizer_elapsed, std.time.ns_per_ms) },
        );

        var trainer_config: TrainerConfig = .{};
        trainer_config.checkpoint_version = @intCast(checkpoint_version);
        trainer_config.learning_rate = learning_rate;
        trainer_config.embedding_seed = embedding_seed;
        trainer_config.spectral_iterations = spectral_iterations;
        trainer_config.spectral_target_norm = spectral_target_norm;
        trainer_config.spectral_interval = @intCast(spectral_interval);
        trainer_config.clip_min = clip_min;
        trainer_config.clip_max = clip_max;
        trainer_config.grad_mean = grad_mean;
        trainer_config.use_normalized_gradient_flow = use_normalized_gradient_flow;
        trainer_config.gradient_clip_norm = gradient_clip_norm;
        trainer_config.reasoning_cycles = reasoning_cycles;
        trainer_config.relational_pass_interval = relational_pass_interval;
        trainer_config.training_sample_count = training_sample_count;
        trainer_config.reconstruction_alpha = reconstruction_alpha;
        trainer_config.phase_a_steps = phase_a_steps;
        trainer_config.phase_b_steps = phase_b_steps;
        trainer_config.shuffle_target_control = shuffle_target_control;
        trainer_config.target_source_frozen = target_source_frozen;
        trainer_config.spectral_depth_compensation = spectral_depth_compensation;
        trainer_config.trust_ratio = trust_ratio;
        trainer_config.weight_floor = weight_floor;
        trainer_config.logdet_weight = logdet_weight;

        const components = TrainerComponents{
            .tokenizer = tokenizer,
        };

        const model_initialization_started = std.time.nanoTimestamp();
        std.debug.print(
            "[Rank {d}] Initializing model weights (this is not training yet)\n",
            .{rank},
        );
        const initialized_trainer = try DistributedTrainerFuthark.initWithComponents(
            allocator,
            &coordinator,
            model_dim,
            num_layers,
            local_batch_size,
            trainer_config,
            components,
        );
        model_initialization_elapsed = std.time.nanoTimestamp() - model_initialization_started;
        break :trainer_initialization initialized_trainer;
    };
    defer trainer.deinit();
    var token_dataset = try mmap_dataset.MmapTokenDataset.open(dataset_path, tokenizer.next_token_id);
    defer token_dataset.close();
    std.debug.print("[Rank {d}] model_compile_initialization_ms={d}\n", .{ rank, @divTrunc(model_initialization_elapsed, std.time.ns_per_ms) });

    const resume_checkpoint_owned = std.process.getEnvVarOwned(allocator, "JAIDE_RESUME_CHECKPOINT") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => null,
        else => return err,
    };
    defer if (resume_checkpoint_owned) |path| allocator.free(path);
    var resume_error: ?anyerror = null;
    if (resume_checkpoint_owned) |path| {
        if (path.len == 0) return error.InvalidEnvironmentValue;
        trainer.loadCheckpoint(path) catch |err| {
            resume_error = err;
        };
    }
    synchronizeStageStatus(
        allocator,
        &coordinator,
        nccl_id_path,
        "checkpoint_restore",
        resume_error,
    ) catch |err| {
        std.debug.print("[Rank {d}] checkpoint restore stage failed: {}\n", .{ rank, err });
        return err;
    };
    const resumed_from_checkpoint = resume_checkpoint_owned != null;

    std.debug.print(
        "[Rank {d}] learning_rate={d}\n",
        .{ rank, learning_rate },
    );
    std.debug.print(
        "[Rank {d}] reconstruction_alpha={d} phase_a_steps={d} phase_b_steps={d}\n",
        .{ rank, reconstruction_alpha, phase_a_steps, phase_b_steps },
    );
    std.debug.print(
        "[Rank {d}] target_source_frozen={} shuffle_target_control={} spectral_depth_compensation={}\n",
        .{ rank, target_source_frozen, shuffle_target_control, spectral_depth_compensation },
    );
    std.debug.print(
        "[Rank {d}] grad_mean={} normalized_gradient_flow={} gradient_clip={d} trust_ratio={d} weight_floor={d} logdet_weight={d} spectral_target={d} spectral_iterations={d} spectral_interval={d}\n",
        .{ rank, grad_mean, use_normalized_gradient_flow, gradient_clip_norm, trust_ratio, weight_floor, logdet_weight, spectral_target_norm, spectral_iterations, spectral_interval },
    );
    std.debug.print(
        "[Rank {d}] init_spectral_iterations={d} skip_knowledge_graph default=1\n",
        .{ rank, init_spectral_iterations },
    );
    std.debug.print(
        "[Rank {d}] Futhark trainer initialized with model_dim={d}, layers={d}\n",
        .{
            rank,
            model_dim,
            num_layers,
        },
    );

    if (coordinator.isRoot()) {
        std.debug.print(
            "\n============================================================\n",
            .{},
        );
        std.debug.print(
            "Starting Futhark-accelerated training\n",
            .{},
        );
        std.debug.print(
            "Dataset: {d} samples per rank\n",
            .{token_dataset.payload.len},
        );
        std.debug.print(
            "Batch size: {d} per rank\n",
            .{local_batch_size},
        );
        std.debug.print("Epochs: {d}\n", .{num_epochs});
        std.debug.print(
            "============================================================\n\n",
            .{},
        );
    }

    const graph_started = std.time.nanoTimestamp();
    var graph_stage_error: ?anyerror = null;

    graph_construction: {
        const skip_knowledge_graph = (parseOptionalEnvironmentBool(allocator, "JAIDE_SKIP_KNOWLEDGE_GRAPH") catch |err| {
            graph_stage_error = err;
            break :graph_construction;
        }) orelse true;
        const graph_chunk_size = (parseOptionalEnvironmentUsize(allocator, "JAIDE_GRAPH_CHUNK_SIZE") catch |err| {
            graph_stage_error = err;
            break :graph_construction;
        }) orelse 4096;
        if (skip_knowledge_graph or resumed_from_checkpoint) {
            if (coordinator.isRoot()) {
                std.debug.print(
                    "[Rank {d}] Skipping offline knowledge-graph import samples={d} skip={} resumed={} chunk={d}\n",
                    .{ rank, token_dataset.payload.len, skip_knowledge_graph, resumed_from_checkpoint, graph_chunk_size },
                );
            }
            break :graph_construction;
        }
        if (graph_chunk_size == 0) {
            graph_stage_error = error.InvalidEnvironmentValue;
            break :graph_construction;
        }
        const hashes = loadDatasetHashes(allocator, dataset_path, 10 * 1024 * 1024) catch |err| {
            graph_stage_error = err;
            break :graph_construction;
        };
        defer allocator.free(hashes);
        var offset: usize = 0;
        while (offset < hashes.len) {
            const remaining = hashes.len - offset;
            const take = if (remaining < graph_chunk_size) remaining else graph_chunk_size;
            const chunk = hashes[offset .. offset + take];
            var encoded = accel_interface.batchEncodeGraph(&trainer.accelerator.ctx, chunk, embedding_seed, allocator) catch |err| {
                graph_stage_error = err;
                break :graph_construction;
            };
            defer encoded.deinit();
            trainer.knowledge_nsir_graph.bulkImportFromGPU(
                encoded.hashes,
                encoded.re_a,
                encoded.im_a,
                encoded.re_b,
                encoded.im_b,
                encoded.edge_srcs,
                encoded.edge_tgts,
            ) catch |err| {
                graph_stage_error = err;
                break :graph_construction;
            };
            offset += take;
            if (coordinator.isRoot()) {
                std.debug.print(
                    "[Rank {d}] knowledge graph chunk imported {d}/{d}\n",
                    .{ rank, offset, hashes.len },
                );
            }
        }
        trainer.r_gpu.distributeGraphFast(trainer.knowledge_nsir_graph) catch |err| {
            graph_stage_error = err;
            break :graph_construction;
        };
    }

    synchronizeStageStatus(
        allocator,
        &coordinator,
        nccl_id_path,
        "knowledge_graph",
        graph_stage_error,
    ) catch |err| {
        std.debug.print(
            "[Rank {d}] knowledge graph stage failed: {}\n",
            .{ rank, err },
        );
        if (@errorReturnTrace()) |trace| {
            std.debug.dumpStackTrace(trace.*);
        }
        return err;
    };

    const graph_elapsed = std.time.nanoTimestamp() - graph_started;
    std.debug.print(
        "[Rank {d}] Knowledge graph stage finished graph_ms={d}\n",
        .{ rank, @divTrunc(graph_elapsed, std.time.ns_per_ms) },
    );
    const startup_elapsed = std.time.nanoTimestamp() - startup_started;
    std.debug.print("[Rank {d}] startup_total_ms={d}\n", .{ rank, @divTrunc(startup_elapsed, std.time.ns_per_ms) });

    var loss_history = std.ArrayList(EpochMetric).init(allocator);
    defer loss_history.deinit();

    var checkpoint_failures: usize = 0;
    var metrics_failures: usize = 0;
    var epoch: usize = 0;
    var artifact_worker: ?*EpochArtifactWorker = null;
    var pending_artifact_error: ?anyerror = null;
    defer if (artifact_worker) |worker| {
        _ = worker.joinAndDestroy();
    };

    while (epoch < num_epochs) : (epoch += 1) {
        var epoch_timer = try std.time.Timer.start();

        if (artifact_worker) |pending_worker| {
            pending_worker.awaitCapture() catch |err| {
                std.debug.print(
                    "[Rank {d}] Async checkpoint capture failed before epoch {d}: {}\n",
                    .{ rank, epoch + 1, err },
                );
                checkpoint_failures += 1;
                pending_artifact_error = err;
            };
        }

        if (coordinator.isRoot()) {
            std.debug.print(
                "[Epoch {d}/{d}] Starting\n",
                .{ epoch + 1, num_epochs },
            );
        }

        const average_loss = trainer.trainEpochTokens(&token_dataset, @intCast(epoch)) catch |err| {
            std.debug.print(
                "[Rank {d}] trainEpoch ERROR at epoch {d}: {}\n",
                .{
                    rank,
                    epoch + 1,
                    err,
                },
            );
            if (@errorReturnTrace()) |trace| {
                std.debug.dumpStackTrace(trace.*);
            }
            return err;
        };

        const elapsed_nanoseconds = epoch_timer.read();
        const elapsed_seconds =
            @as(f64, @floatFromInt(elapsed_nanoseconds)) / 1.0e9;

        var epoch_artifact_error: ?anyerror = null;

        if (coordinator.isRoot()) {
            epoch_artifact_error = epoch_artifact_error orelse pending_artifact_error;
            pending_artifact_error = null;
            if (artifact_worker) |previous_worker| {
                if (previous_worker.joinAndDestroy()) |err| {
                    std.debug.print(
                        "[Rank 0] Async checkpoint write failed: {}\n",
                        .{err},
                    );
                    checkpoint_failures += 1;
                    epoch_artifact_error = epoch_artifact_error orelse err;
                }
                artifact_worker = null;
            }

            loss_history.append(.{
                .epoch = epoch + 1,
                .loss = average_loss,
                .time_s = elapsed_seconds,
            }) catch |err| {
                std.debug.print(
                    "[Rank 0] Failed to append loss metric: {}\n",
                    .{err},
                );
                metrics_failures += 1;
                epoch_artifact_error = err;
            };

            std.debug.print(
                "[Epoch {d}/{d}] Loss: {d:.6} | Time: {d:.2}s\n",
                .{
                    epoch + 1,
                    num_epochs,
                    average_loss,
                    elapsed_seconds,
                },
            );

            checkpoint_creation: {
                const epoch_number = epoch + 1;
                if (checkpoint_interval_epochs == 0 or (epoch_number % checkpoint_interval_epochs != 0 and epoch_number != num_epochs)) break :checkpoint_creation;
                std.fs.makeDirAbsolute("/checkpoints") catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => {
                        std.debug.print(
                            "[Rank 0] Failed to create /checkpoints: {}\n",
                            .{err},
                        );
                        checkpoint_failures += 1;
                        epoch_artifact_error = epoch_artifact_error orelse err;
                        break :checkpoint_creation;
                    },
                };

                var directory_buffer: [256]u8 = undefined;
                const directory_path = std.fmt.bufPrint(
                    &directory_buffer,
                    "/checkpoints/epoch_{d:0>3}",
                    .{epoch + 1},
                ) catch |err| {
                    std.debug.print(
                        "[Rank 0] Failed to format checkpoint directory: {}\n",
                        .{err},
                    );
                    checkpoint_failures += 1;
                    epoch_artifact_error = epoch_artifact_error orelse err;
                    break :checkpoint_creation;
                };

                std.fs.makeDirAbsolute(directory_path) catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => {
                        std.debug.print(
                            "[Rank 0] Failed to create {s}: {}\n",
                            .{ directory_path, err },
                        );
                        checkpoint_failures += 1;
                        epoch_artifact_error = epoch_artifact_error orelse err;
                        break :checkpoint_creation;
                    },
                };

                var checkpoint_path_buffer: [256]u8 = undefined;
                const checkpoint_path = std.fmt.bufPrint(
                    &checkpoint_path_buffer,
                    "/checkpoints/epoch_{d:0>3}/model.ckpt",
                    .{epoch + 1},
                ) catch |err| {
                    std.debug.print(
                        "[Rank 0] Failed to format checkpoint path: {}\n",
                        .{err},
                    );
                    checkpoint_failures += 1;
                    epoch_artifact_error = epoch_artifact_error orelse err;
                    break :checkpoint_creation;
                };

                const checkpoint_path_owned = allocator.dupe(u8, checkpoint_path) catch |err| {
                    std.debug.print(
                        "[Rank 0] Failed to allocate checkpoint path: {}\n",
                        .{err},
                    );
                    checkpoint_failures += 1;
                    epoch_artifact_error = epoch_artifact_error orelse err;
                    break :checkpoint_creation;
                };

                artifact_worker = EpochArtifactWorker.start(
                    allocator,
                    &trainer,
                    checkpoint_path_owned,
                ) catch |err| {
                    allocator.free(checkpoint_path_owned);
                    std.debug.print(
                        "[Rank 0] Failed to start async checkpoint worker: {}\n",
                        .{err},
                    );
                    checkpoint_failures += 1;
                    epoch_artifact_error = epoch_artifact_error orelse err;
                    break :checkpoint_creation;
                };
                artifact_worker.?.awaitCapture() catch |err| {
                    checkpoint_failures += 1;
                    epoch_artifact_error = epoch_artifact_error orelse err;
                    break :checkpoint_creation;
                };
            }

            if (epoch_artifact_error == null) {
                writeTrainingMetrics(
                    allocator,
                    loss_history.items,
                    model_dim,
                    num_layers,
                    local_batch_size,
                    learning_rate,
                    token_dataset.payload.len,
                    num_epochs,
                ) catch |err| {
                    std.debug.print(
                        "[Rank 0] Failed to write training metrics: {}\n",
                        .{err},
                    );
                    metrics_failures += 1;
                    epoch_artifact_error = err;
                };
            }
        }

        synchronizeStageStatus(
            allocator,
            &coordinator,
            nccl_id_path,
            "epoch_artifacts",
            epoch_artifact_error,
        ) catch |err| {
            std.debug.print(
                "[Rank {d}] checkpoint/metrics stage failed after epoch {d}: {}\n",
                .{ rank, epoch + 1, err },
            );
            return err;
        };
    }

    var final_artifact_error: ?anyerror = null;
    if (artifact_worker) |final_worker| {
        if (final_worker.joinAndDestroy()) |err| {
            std.debug.print(
                "[Rank 0] Final async checkpoint flush failed: {}\n",
                .{err},
            );
            checkpoint_failures += 1;
            final_artifact_error = err;
        }
        artifact_worker = null;
    }
    synchronizeStageStatus(
        allocator,
        &coordinator,
        nccl_id_path,
        "final_checkpoint_write",
        final_artifact_error,
    ) catch |err| {
        std.debug.print("[Rank {d}] final checkpoint write stage failed: {}\n", .{ rank, err });
        return err;
    };

    if (coordinator.isRoot()) {
        std.debug.print(
            "\n============================================================\n",
            .{},
        );
        std.debug.print(
            "Futhark-accelerated training completed\n",
            .{},
        );
        std.debug.print(
            "Checkpoint failures: {d}\n",
            .{checkpoint_failures},
        );
        std.debug.print(
            "Metrics failures: {d}\n",
            .{metrics_failures},
        );
        std.debug.print(
            "============================================================\n",
            .{},
        );
    }
}

const EpochMetric = struct {
    epoch: usize,
    loss: f64,
    time_s: f64,
};

fn writeTrainingMetrics(
    allocator: std.mem.Allocator,
    epoch_metrics: []const EpochMetric,
    model_dim: usize,
    num_layers: usize,
    batch_size: usize,
    learning_rate: f64,
    sample_count: usize,
    planned_epochs: usize,
) !void {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const writer = buffer.writer();
    const safe_learning_rate: f64 = if (std.math.isFinite(learning_rate))
        learning_rate
    else
        0.0;

    try writer.print(
        "{{\n  \"model_dim\": {d},\n  \"num_layers\": {d},\n  \"batch_size\": {d},\n  \"learning_rate\": {d},\n  \"sample_count\": {d},\n  \"planned_epochs\": {d},\n  \"loss_curve\": [\n",
        .{
            model_dim,
            num_layers,
            batch_size,
            safe_learning_rate,
            sample_count,
            planned_epochs,
        },
    );

    for (epoch_metrics, 0..) |metric, index| {
        const safe_loss: f64 = if (std.math.isFinite(metric.loss))
            metric.loss
        else
            0.0;

        const safe_time: f64 = if (std.math.isFinite(metric.time_s))
            metric.time_s
        else
            0.0;

        try writer.print(
            "    {{ \"epoch\": {d}, \"loss\": {d:.6}, \"time_s\": {d:.2} }}{s}\n",
            .{
                metric.epoch,
                safe_loss,
                safe_time,
                if (index + 1 < epoch_metrics.len) "," else "",
            },
        );
    }

    try writer.print("  ]\n}}\n", .{});

    const temporary_path = "/checkpoints/training_metrics.json.tmp";
    const final_path = "/checkpoints/training_metrics.json";

    {
        const temporary_file = try std.fs.createFileAbsolute(
            temporary_path,
            .{ .truncate = true },
        );
        errdefer temporary_file.close();

        try temporary_file.writeAll(buffer.items);
        try temporary_file.sync();
        temporary_file.close();
    }

    std.fs.deleteFileAbsolute(final_path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };

    try std.fs.renameAbsolute(temporary_path, final_path);
}
