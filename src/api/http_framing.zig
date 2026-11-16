const std = @import("std");

pub const Frame = struct {
    method_end: usize,
    path_start: usize,
    path_end: usize,
    headers_end: usize,
    body_start: usize,
    request_end: usize,
    keep_alive: bool,
};

pub fn parse(data: []const u8, maximum_size: usize) !?Frame {
    const headers_end = std.mem.indexOf(u8, data, "\r\n\r\n") orelse {
        if (data.len >= maximum_size) return error.RequestTooLarge;
        return null;
    };
    const body_start = try std.math.add(usize, headers_end, 4);
    if (body_start > maximum_size) return error.RequestTooLarge;
    const headers = data[0..headers_end];
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    const request_line = lines.next() orelse return error.InvalidRequest;
    const method_end = std.mem.indexOfScalar(u8, request_line, ' ') orelse return error.InvalidRequest;
    if (method_end == 0) return error.InvalidRequest;
    const path_start = method_end + 1;
    const path_end = std.mem.indexOfScalarPos(u8, request_line, path_start, ' ') orelse return error.InvalidRequest;
    if (path_end == path_start) return error.InvalidRequest;
    const version = request_line[path_end + 1 ..];
    if (!std.mem.eql(u8, version, "HTTP/1.1") and !std.mem.eql(u8, version, "HTTP/1.0")) return error.InvalidRequest;
    var length: ?usize = null;
    var connection_close = false;
    var connection_keep_alive = false;
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.InvalidRequest;
        if (colon == 0) return error.InvalidRequest;
        const name = line[0..colon];
        for (name) |c| {
            if (!std.ascii.isAlphanumeric(c) and std.mem.indexOfScalar(u8, "!#$%&'*+-.^_`|~", c) == null) return error.InvalidRequest;
        }
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            if (length != null or value.len == 0) return error.InvalidRequest;
            for (value) |c| if (!std.ascii.isDigit(c)) return error.InvalidRequest;
            length = std.fmt.parseInt(usize, value, 10) catch return error.InvalidRequest;
        } else if (std.ascii.eqlIgnoreCase(name, "transfer-encoding")) {
            return error.UnsupportedTransferEncoding;
        } else if (std.ascii.eqlIgnoreCase(name, "connection")) {
            var options = std.mem.splitScalar(u8, value, ',');
            while (options.next()) |option| {
                const token = std.mem.trim(u8, option, " \t");
                if (std.ascii.eqlIgnoreCase(token, "close")) connection_close = true;
                if (std.ascii.eqlIgnoreCase(token, "keep-alive")) connection_keep_alive = true;
            }
        }
    }
    const content_length = length orelse 0;
    if (content_length > maximum_size - body_start) return error.RequestTooLarge;
    return .{
        .method_end = method_end,
        .path_start = path_start,
        .path_end = path_end,
        .headers_end = headers_end,
        .body_start = body_start,
        .request_end = body_start + content_length,
        .keep_alive = !connection_close and connection_keep_alive,
    };
}

pub fn consume(buffer: *std.ArrayList(u8), request_end: usize) !void {
    if (request_end > buffer.items.len) return error.IncompleteRequest;
    const remaining = buffer.items.len - request_end;
    std.mem.copyForwards(u8, buffer.items[0..remaining], buffer.items[request_end..]);
    buffer.shrinkRetainingCapacity(remaining);
}

test "fragmented headers remain incomplete until the delimiter arrives" {
    try std.testing.expectEqual(@as(?Frame, null), try parse("POST /v1/inference HTTP/1.1\r\nContent-Length: 4\r\n", 1024));
    const frame = (try parse("POST /v1/inference HTTP/1.1\r\nContent-Length: 4\r\n\r\n", 1024)).?;
    try std.testing.expectEqual(@as(usize, 4), frame.request_end - frame.body_start);
}

test "body slices are formed after buffer relocation and exclude pipelined bytes" {
    var buffer = std.ArrayList(u8).init(std.testing.allocator);
    defer buffer.deinit();
    try buffer.appendSlice("POST /v1/inference HTTP/1.1\r\nContent-Length: 8192\r\n\r\na");
    const frame = (try parse(buffer.items, 16384)).?;
    try buffer.appendNTimes('b', 8191);
    try buffer.appendSlice("GET /v1/health HTTP/1.1\r\n\r\n");
    const body = buffer.items[frame.body_start..frame.request_end];
    try std.testing.expectEqual(@as(usize, 8192), body.len);
    try std.testing.expectEqual(@as(u8, 'a'), body[0]);
    try std.testing.expectEqual(@as(u8, 'b'), body[8191]);
    try consume(&buffer, frame.request_end);
    const next = (try parse(buffer.items, 1024)).?;
    try std.testing.expectEqualStrings("GET", buffer.items[0..next.method_end]);
    try std.testing.expectEqualStrings("/v1/health", buffer.items[next.path_start..next.path_end]);
}

test "malformed or ambiguous body lengths are rejected" {
    for ([_][]const u8{ "-1", "+1", "abc", "", "184467440737095516160" }) |value| {
        const request = try std.fmt.allocPrint(std.testing.allocator, "POST / HTTP/1.1\r\nContent-Length: {s}\r\n\r\n", .{value});
        defer std.testing.allocator.free(request);
        try std.testing.expectError(error.InvalidRequest, parse(request, 1024));
    }
    try std.testing.expectError(error.InvalidRequest, parse("POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\n", 1024));
    try std.testing.expectError(error.UnsupportedTransferEncoding, parse("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", 1024));
}

test "frame limits apply per request rather than the entire pipeline" {
    const request = "GET / HTTP/1.1\r\n\r\n";
    const frame = (try parse(request ++ request, request.len)).?;
    try std.testing.expectEqual(request.len, frame.request_end);
    try std.testing.expectError(error.RequestTooLarge, parse(request, request.len - 1));
    try std.testing.expectError(error.RequestTooLarge, parse("POST / HTTP/1.1\r\nContent-Length: 9999\r\n\r\n", 100));
}

test "connection close takes precedence over keep alive" {
    const closed = (try parse("GET / HTTP/1.1\r\nConnection: Keep-Alive, close\r\n\r\n", 1024)).?;
    try std.testing.expect(!closed.keep_alive);
    const persistent = (try parse("GET / HTTP/1.1\r\nConnection: keep-alive\r\n\r\n", 1024)).?;
    try std.testing.expect(persistent.keep_alive);
}
