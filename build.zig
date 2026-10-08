const std = @import("std");

const AccelConfig = struct {
    include: std.Build.LazyPath,
    kernels_c: std.Build.LazyPath,
    abi_check_c: std.Build.LazyPath,
    cflags: []const []const u8,
    codegen: ?*std.Build.Step,
    gpu: bool,
    options: *std.Build.Step.Options,
    jaide: *std.Build.Module,
};

fn linkCudaRuntime(artifact: *std.Build.Step.Compile, with_nccl: bool) void {
    artifact.addIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
    artifact.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
    artifact.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64/stubs" });
    artifact.linkSystemLibrary("cuda");
    artifact.linkSystemLibrary("cudart");
    artifact.linkSystemLibrary("nvrtc");
    artifact.linkSystemLibrary("cublas");
    artifact.linkSystemLibrary("cublasLt");
    if (with_nccl) artifact.linkSystemLibrary("nccl");
    artifact.linkSystemLibrary("m");
    artifact.linkSystemLibrary("pthread");
    artifact.linkSystemLibrary("dl");
}

fn applyAccel(artifact: *std.Build.Step.Compile, cfg: AccelConfig, with_nccl: bool) void {
    artifact.linkLibC();
    artifact.addIncludePath(cfg.include);
    artifact.addCSourceFile(.{ .file = cfg.kernels_c, .flags = cfg.cflags });
    artifact.addCSourceFile(.{ .file = cfg.abi_check_c, .flags = cfg.cflags });
    if (cfg.codegen) |step| artifact.step.dependOn(step);
    if (cfg.gpu) linkCudaRuntime(artifact, with_nccl);
    artifact.root_module.addOptions("build_options", cfg.options);
    artifact.root_module.addImport("jaide", cfg.jaide);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gpu_enabled = b.option(bool, "gpu", "Enable GPU/CUDA via the Futhark CUDA backend") orelse false;
    const rtl_enabled = b.option(bool, "rtl", "Compile the Clash RTL modules and link the RTL simulator") orelse false;
    const skip_futhark = b.option(bool, "skip-futhark", "Assume the generated Futhark C sources are already present") orelse false;
    const clash_bin = b.option([]const u8, "clash", "Clash compiler executable") orelse "clash";

    const build_options = b.addOptions();
    build_options.addOption(bool, "gpu_acceleration", gpu_enabled);
    build_options.addOption(bool, "rtl_enabled", rtl_enabled);

    const accel_dir = "src/hw/accel";
    const futhark_include = b.path(accel_dir);

    const futhark_sync_step = b.addSystemCommand(&.{ "futhark", "pkg", "sync" });
    futhark_sync_step.setCwd(b.path(accel_dir));

    const main_cpu_step = b.addSystemCommand(&.{ "futhark", "c", "--library", "main.fut", "-o", "main_cpu" });
    main_cpu_step.setCwd(b.path(accel_dir));
    main_cpu_step.step.dependOn(&futhark_sync_step.step);

    const main_gpu_step = b.addSystemCommand(&.{ "futhark", "cuda", "--library", "main.fut", "-o", "main_gpu" });
    main_gpu_step.setCwd(b.path(accel_dir));
    main_gpu_step.step.dependOn(&futhark_sync_step.step);

    const kernels_cpu_step = b.addSystemCommand(&.{ "futhark", "c", "--library", "futhark_kernels.fut", "-o", "futhark_kernels" });
    kernels_cpu_step.setCwd(b.path(accel_dir));
    kernels_cpu_step.step.dependOn(&futhark_sync_step.step);

    const kernels_gpu_step = b.addSystemCommand(&.{ "futhark", "cuda", "--library", "futhark_kernels.fut", "-o", "futhark_kernels_cuda" });
    kernels_gpu_step.setCwd(b.path(accel_dir));
    kernels_gpu_step.step.dependOn(&futhark_sync_step.step);

    const futhark_check_step = b.addSystemCommand(&.{ "futhark", "check", "main.fut", "futhark_kernels.fut" });
    futhark_check_step.setCwd(b.path(accel_dir));
    futhark_check_step.step.dependOn(&futhark_sync_step.step);

    const regenerate_futhark_step = b.step("regen-futhark", "Regenerate every Futhark C source from the .fut definitions");
    regenerate_futhark_step.dependOn(&main_cpu_step.step);
    regenerate_futhark_step.dependOn(&main_gpu_step.step);
    regenerate_futhark_step.dependOn(&kernels_cpu_step.step);
    regenerate_futhark_step.dependOn(&kernels_gpu_step.step);

    const futhark_lint_step = b.step("futhark-check", "Type check every Futhark source");
    futhark_lint_step.dependOn(&futhark_check_step.step);

    const cpu_cflags = [_][]const u8{ "-O2", "-std=c11" };
    const gpu_cflags = [_][]const u8{ "-O2", "-std=c11", "-DJAIDE_FUTHARK_CUDA" };

    const jaide_mod = b.createModule(.{
        .root_source_file = b.path("src/lib_root.zig"),
        .target = target,
        .optimize = optimize,
    });
    jaide_mod.addOptions("build_options", build_options);

    const accel: AccelConfig = .{
        .include = futhark_include,
        .kernels_c = if (gpu_enabled) b.path(accel_dir ++ "/main_gpu.c") else b.path(accel_dir ++ "/main_cpu.c"),
        .abi_check_c = b.path(accel_dir ++ "/futhark_abi_check.c"),
        .cflags = if (gpu_enabled) &gpu_cflags else &cpu_cflags,
        .codegen = if (skip_futhark) null else if (gpu_enabled) &main_gpu_step.step else &main_cpu_step.step,
        .gpu = gpu_enabled,
        .options = build_options,
        .jaide = jaide_mod,
    };

    const inference_server_exe = b.addExecutable(.{
        .name = "jaide-inference-server",
        .root_source_file = b.path("src/inference_server_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    applyAccel(inference_server_exe, accel, false);
    b.installArtifact(inference_server_exe);
    const inference_server_step = b.step("inference-server", "Build the inference server");
    inference_server_step.dependOn(&inference_server_exe.step);

    const distributed_futhark_step = b.step("distributed-futhark", "Build the Futhark-accelerated distributed trainer");
    if (gpu_enabled) {
        const distributed_futhark_exe = b.addExecutable(.{
            .name = "jaide-distributed-futhark",
            .root_source_file = b.path("src/main_distributed_futhark.zig"),
            .target = target,
            .optimize = optimize,
        });
        applyAccel(distributed_futhark_exe, accel, true);
        b.installArtifact(distributed_futhark_exe);
        distributed_futhark_step.dependOn(&distributed_futhark_exe.step);
    } else {
        const distributed_futhark_unavailable = b.addFail("jaide-distributed-futhark requires CUDA and NCCL; configure with -Dgpu=true");
        distributed_futhark_step.dependOn(&distributed_futhark_unavailable.step);
    }

    const pretokenize_exe = b.addExecutable(.{
        .name = "jaide-pretokenize",
        .root_source_file = b.path("src/pretokenize_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    pretokenize_exe.linkLibC();
    pretokenize_exe.root_module.addOptions("build_options", build_options);

    b.installArtifact(pretokenize_exe);
    const pretokenize_step = b.step("pretokenize", "Build the binary dataset pre-tokenizer");
    pretokenize_step.dependOn(&pretokenize_exe.step);

    const c_api_lib = b.addStaticLibrary(.{
        .name = "jaide",
        .root_source_file = b.path("src/core_relational/c_api.zig"),
        .target = target,
        .optimize = optimize,
    });
    c_api_lib.linkLibC();
    c_api_lib.root_module.addOptions("build_options", build_options);
    c_api_lib.installHeader(b.path("src/core_relational/jaide.h"), "jaide.h");
    b.installArtifact(c_api_lib);
    const c_api_step = b.step("c-api", "Build the JAIDE C API static library");
    c_api_step.dependOn(&c_api_lib.step);

    const semantic_check_obj = b.addObject(.{
        .name = "jaide-semantic-check",
        .root_source_file = b.path("src/semantic_check_root.zig"),
        .target = target,
        .optimize = optimize,
    });
    semantic_check_obj.linkLibC();
    semantic_check_obj.addIncludePath(futhark_include);
    semantic_check_obj.root_module.addOptions("build_options", build_options);

    const distributed_check_obj = b.addObject(.{
        .name = "jaide-distributed-check",
        .root_source_file = b.path("src/main_distributed_futhark.zig"),
        .target = target,
        .optimize = optimize,
    });
    distributed_check_obj.linkLibC();
    distributed_check_obj.addIncludePath(futhark_include);
    distributed_check_obj.root_module.addOptions("build_options", build_options);

    const check_step = b.step("check", "Semantically analyse every Zig module without linking");
    check_step.dependOn(&semantic_check_obj.step);
    check_step.dependOn(&distributed_check_obj.step);

    const rsf_substrate_tests = b.addTest(.{
        .name = "rsf-substrate-tests",
        .root_source_file = b.path("src/tests/rsf_substrate_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    rsf_substrate_tests.root_module.addOptions("build_options", build_options);
    rsf_substrate_tests.root_module.addImport("jaide", jaide_mod);
    const run_rsf_substrate_tests = b.addRunArtifact(rsf_substrate_tests);
    const test_rsf_step = b.step("test-rsf", "Run the RSF tensor, causal, and spectral substrate tests");
    test_rsf_step.dependOn(&run_rsf_substrate_tests.step);

    const rsf_native_tests = b.addTest(.{
        .name = "rsf-native-tests",
        .root_source_file = b.path("src/tests/rsf_native_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    rsf_native_tests.linkLibC();
    rsf_native_tests.root_module.addOptions("build_options", build_options);
    rsf_native_tests.root_module.addImport("jaide", jaide_mod);
    const run_rsf_native_tests = b.addRunArtifact(rsf_native_tests);
    const test_rsf_native_step = b.step("test-rsf-native", "Run the accel-free RSF-native cross-module invariant suite");
    test_rsf_native_step.dependOn(&run_rsf_native_tests.step);

    // The accel-backed half of the invariant suite. Its import closure reaches
    // `src/hw/accel/accel_interface.zig`, whose `pub extern "c" fn futhark_*`
    // declarations are called from its destructors, so this artifact needs the
    // Futhark-generated C to link. It is registered WITHOUT the Futhark
    // codegen step so the failure surfaces as a link error rather than a
    // missing-tool error, and it is reported as blocked when that happens.
    const rsf_native_accel_tests = b.addTest(.{
        .name = "rsf-native-accel-tests",
        .root_source_file = b.path("src/tests/rsf_native_accel_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    rsf_native_accel_tests.linkLibC();
    rsf_native_accel_tests.root_module.addOptions("build_options", build_options);
    rsf_native_accel_tests.root_module.addImport("jaide", jaide_mod);
    const run_rsf_native_accel_tests = b.addRunArtifact(rsf_native_accel_tests);
    const test_rsf_native_accel_step = b.step("test-rsf-native-accel", "Run the accelerator-backed RSF-native invariants (needs the Futhark-generated C)");
    test_rsf_native_accel_step.dependOn(&run_rsf_native_accel_tests.step);

    if (rtl_enabled) {
        const clash_step = b.addSystemCommand(&.{clash_bin});
        clash_step.addArg("--verilog");
        clash_step.addArg("-isrc/hw/rtl");
        clash_step.addArg("-outputdir");
        clash_step.addArg("src/hw/rtl/verilog");
        clash_step.addArg("src/hw/rtl/MemoryArbiter.hs");
        clash_step.addArg("src/hw/rtl/RankerCore.hs");
        clash_step.addArg("src/hw/rtl/SSISearch.hs");

        const ghc_step = b.addSystemCommand(&.{
            "ghc",
            "-O2",
            "-dynamic",
            "-shared",
            "-fPIC",
            "-no-hs-main",
            "-package",
            "clash-prelude",
            "-package",
            "base",
            "-isrc/hw/rtl",
            "-outputdir",
            "src/hw/rtl/build",
            "-hidir",
            "src/hw/rtl/build",
            "src/hw/rtl/MemoryArbiter.hs",
            "src/hw/rtl/RankerCore.hs",
            "src/hw/rtl/SSISearch.hs",
            "src/hw/rtl/RtlExports.hs",
            "-o",
            "src/hw/rtl/librtl_sim.so",
        });

        const rtl_exe = b.addExecutable(.{
            .name = "jaide-rtl-sim",
            .root_source_file = b.path("src/hw/rtl/rtl_sim_main.zig"),
            .target = target,
            .optimize = optimize,
        });
        rtl_exe.linkLibC();
        rtl_exe.root_module.addOptions("build_options", build_options);
        rtl_exe.root_module.addImport("jaide", jaide_mod);
        rtl_exe.addLibraryPath(b.path("src/hw/rtl"));
        rtl_exe.linkSystemLibrary("rtl_sim");
        rtl_exe.addRPath(b.path("src/hw/rtl"));
        rtl_exe.step.dependOn(&ghc_step.step);

        const rtl_install = b.addInstallArtifact(rtl_exe, .{});

        const rtl_step = b.step("rtl", "Build the Clash RTL library and the RTL simulator");
        rtl_step.dependOn(&rtl_install.step);

        const rtl_verilog_step = b.step("rtl-verilog", "Generate Verilog from the Clash RTL modules");
        rtl_verilog_step.dependOn(&clash_step.step);
    }

    const bench_deps = b.createModule(.{
        .root_source_file = b.path("src/_bench_deps.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_deps.addOptions("build_options", build_options);
    bench_deps.addImport("jaide", jaide_mod);

    const bench_step = b.step("bench", "Run every benchmark");

    const bench_sources = [_]struct {
        name: []const u8,
        path: []const u8,
    }{
        .{ .name = "bench-rsf", .path = "src/bench/bench_rsf.zig" },
        .{ .name = "bench-matmul", .path = "src/bench/bench_matmul.zig" },
        .{ .name = "bench-tensor-ops", .path = "src/bench/bench_tensor_ops.zig" },
        .{ .name = "bench-sfd", .path = "src/bench/bench_sfd.zig" },
    };

    inline for (bench_sources) |source| {
        const executable = b.addExecutable(.{
            .name = source.name,
            .root_source_file = b.path(source.path),
            .target = target,
            .optimize = optimize,
        });
        applyAccel(executable, accel, gpu_enabled);
        executable.root_module.addImport("deps", bench_deps);
        b.installArtifact(executable);

        const run = b.addRunArtifact(executable);
        bench_step.dependOn(&run.step);
    }
}
