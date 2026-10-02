const std = @import("std");
const http = std.http;

pub const TrainingJobConfig = struct {
    allocator: std.mem.Allocator,
    gpu_count: usize,
    gpu_preferences: [][]u8,
    image: []u8,
    batch_size: usize,
    epochs: usize,

    pub fn fromEnvironment(allocator: std.mem.Allocator) !TrainingJobConfig {
        const gpu_count_text = try requiredEnvironment(allocator, "JAIDE_MODAL_GPU_COUNT");
        defer allocator.free(gpu_count_text);
        const gpu_count = std.fmt.parseInt(usize, gpu_count_text, 10) catch return error.InvalidModalConfiguration;
        if (gpu_count == 0) return error.InvalidModalConfiguration;

        const preferences_text = try requiredEnvironment(allocator, "JAIDE_MODAL_GPU_PREFERENCES");
        defer allocator.free(preferences_text);
        var preferences = std.ArrayList([]u8).init(allocator);
        errdefer {
            for (preferences.items) |item| allocator.free(item);
            preferences.deinit();
        }

        var preferences_iter = std.mem.splitScalar(u8, preferences_text, ',');
        while (preferences_iter.next()) |raw_preference| {
            const preference = std.mem.trim(u8, raw_preference, " \t\r\n");
            if (preference.len == 0) continue;
            try preferences.append(try allocator.dupe(u8, preference));
        }
        if (preferences.items.len == 0) return error.InvalidModalConfiguration;

        const image = try requiredEnvironment(allocator, "JAIDE_MODAL_IMAGE");
        errdefer allocator.free(image);
        if (image.len == 0) return error.InvalidModalConfiguration;

        const batch_size_text = try requiredEnvironment(allocator, "JAIDE_MODAL_BATCH_SIZE");
        defer allocator.free(batch_size_text);
        const batch_size = std.fmt.parseInt(usize, batch_size_text, 10) catch return error.InvalidModalConfiguration;
        if (batch_size == 0) return error.InvalidModalConfiguration;

        const epochs_text = try requiredEnvironment(allocator, "JAIDE_MODAL_EPOCHS");
        defer allocator.free(epochs_text);
        const epochs = std.fmt.parseInt(usize, epochs_text, 10) catch return error.InvalidModalConfiguration;
        if (epochs == 0) return error.InvalidModalConfiguration;

        return .{
            .allocator = allocator,
            .gpu_count = gpu_count,
            .gpu_preferences = try preferences.toOwnedSlice(),
            .image = image,
            .batch_size = batch_size,
            .epochs = epochs,
        };
    }

    pub fn deinit(self: *TrainingJobConfig) void {
        for (self.gpu_preferences) |preference| self.allocator.free(preference);
        self.allocator.free(self.gpu_preferences);
        self.allocator.free(self.image);
    }
};

fn requiredEnvironment(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    return std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => error.MissingModalConfiguration,
        else => err,
    };
}

pub const ModalGPUClient = struct {
    allocator: std.mem.Allocator,
    api_token: []const u8,
    http_client: http.Client,
    job_config: *const TrainingJobConfig,

    /// `job_config` remains owned by the caller and must outlive the client.
    /// This makes every resource and training parameter explicit at deployment
    /// time instead of silently selecting a hardware or training profile.
    pub fn init(
        allocator: std.mem.Allocator,
        api_token: []const u8,
        job_config: *const TrainingJobConfig,
    ) !ModalGPUClient {
        return .{
            .allocator = allocator,
            .api_token = try allocator.dupe(u8, api_token),
            .http_client = http.Client{ .allocator = allocator },
            .job_config = job_config,
        };
    }

    pub fn deinit(self: *ModalGPUClient) void {
        self.allocator.free(self.api_token);
        self.http_client.deinit();
    }

    pub fn deployTrainingJob(self: *ModalGPUClient, model_path: []const u8, dataset_path: []const u8) ![]const u8 {
        const config = self.job_config;
        const payload = try std.json.stringifyAlloc(self.allocator, .{
            .gpu = config.gpu_preferences,
            .gpu_count = config.gpu_count,
            .image = config.image,
            .model_path = model_path,
            .dataset_path = dataset_path,
            .batch_size = config.batch_size,
            .epochs = config.epochs,
        }, .{});
        defer self.allocator.free(payload);

        return try self.sendRequest(.POST, "https://api.modal.com/v1/functions/deploy", payload);
    }

    pub fn getJobStatus(self: *ModalGPUClient, job_id: []const u8) ![]const u8 {
        const uri_str = try std.fmt.allocPrint(self.allocator, "https://api.modal.com/v1/functions/{s}/status", .{job_id});
        defer self.allocator.free(uri_str);

        return try self.sendRequest(.GET, uri_str, null);
    }

    fn sendRequest(self: *ModalGPUClient, method: http.Method, url: []const u8, body: ?[]const u8) ![]const u8 {
        const authorization_value = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_token});
        defer self.allocator.free(authorization_value);

        var response_storage = std.ArrayList(u8).init(self.allocator);
        errdefer response_storage.deinit();

        const fetch_options = if (body) |b|
            http.Client.FetchOptions{
                .method = method,
                .location = .{ .url = url },
                .headers = .{
                    .authorization = .{ .override = authorization_value },
                    .content_type = .{ .override = "application/json" },
                },
                .payload = b,
                .response_storage = .{ .dynamic = &response_storage },
                .max_append_size = 1024 * 1024,
            }
        else
            http.Client.FetchOptions{
                .method = method,
                .location = .{ .url = url },
                .headers = .{
                    .authorization = .{ .override = authorization_value },
                },
                .response_storage = .{ .dynamic = &response_storage },
                .max_append_size = 1024 * 1024,
            };

        const result = try self.http_client.fetch(fetch_options);
        _ = result;

        return response_storage.toOwnedSlice();
    }
};
