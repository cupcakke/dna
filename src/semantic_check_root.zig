const std = @import("std");

pub const api_inference_server = @import("api/inference_server.zig");
pub const core_io = @import("core/io.zig");
pub const core_learned_embedding = @import("core/learned_embedding.zig");
pub const core_memory = @import("core/memory.zig");
pub const core_model_io = @import("core/model_io.zig");
pub const core_tensor = @import("core/tensor.zig");
pub const core_types = @import("core/types.zig");
pub const core_relational_c_api = @import("core_relational/c_api.zig");
pub const core_relational_chaos_core = @import("core_relational/chaos_core.zig");
pub const core_relational_crev_pipeline = @import("core_relational/crev_pipeline.zig");
pub const core_relational_dataset_obfuscation = @import("core_relational/dataset_obfuscation.zig");
pub const core_relational_esso_optimizer = @import("core_relational/esso_optimizer.zig");
pub const core_relational_fnds = @import("core_relational/fnds.zig");
pub const core_relational_formal_verification = @import("core_relational/formal_verification.zig");
pub const core_relational_ibm_quantum = @import("core_relational/ibm_quantum.zig");
pub const core_relational_mod = @import("core_relational/mod.zig");
pub const core_relational_nsir_core = @import("core_relational/nsir_core.zig");
pub const core_relational_quantum_hardware = @import("core_relational/quantum_hardware.zig");
pub const core_relational_quantum_logic = @import("core_relational/quantum_logic.zig");
pub const core_relational_quantum_task_adapter = @import("core_relational/quantum_task_adapter.zig");
pub const core_relational_r_gpu = @import("core_relational/r_gpu.zig");
pub const core_relational_reasoning_orchestrator = @import("core_relational/reasoning_orchestrator.zig");
pub const core_relational_safety = @import("core_relational/safety.zig");
pub const core_relational_security_proofs = @import("core_relational/security_proofs.zig");
pub const core_relational_signal_propagation = @import("core_relational/signal_propagation.zig");
pub const core_relational_surprise_memory = @import("core_relational/surprise_memory.zig");
pub const core_relational_temporal_graph = @import("core_relational/temporal_graph.zig");
pub const core_relational_type_theory = @import("core_relational/type_theory.zig");
pub const core_relational_verified_inference_engine = @import("core_relational/verified_inference_engine.zig");
pub const core_relational_vpu = @import("core_relational/vpu.zig");
pub const core_relational_z_runtime = @import("core_relational/z_runtime.zig");
pub const core_relational_zk_verification = @import("core_relational/zk_verification.zig");
pub const distributed_checkpoint_envelope = @import("distributed/checkpoint_envelope.zig");
pub const distributed_dataset_partition = @import("distributed/dataset_partition.zig");
pub const distributed_gpu_coordinator = @import("distributed/gpu_coordinator.zig");
pub const distributed_mmap_token_dataset = @import("distributed/mmap_token_dataset.zig");
pub const distributed_modal_gpu = @import("distributed/modal_gpu.zig");
pub const distributed_nccl_bindings = @import("distributed/nccl_bindings.zig");
pub const distributed_trainer_futhark = @import("distributed/distributed_trainer_futhark.zig");
pub const hw_accel_accel_interface = @import("hw/accel/accel_interface.zig");
pub const hw_accel_compact_batch = @import("hw/accel/compact_batch.zig");
pub const hw_accel_cuda_bindings = @import("hw/accel/cuda_bindings.zig");
pub const hw_accel_fractal_lpu = @import("hw/accel/fractal_lpu.zig");
pub const hw_accel_futhark_bindings = @import("hw/accel/futhark_bindings.zig");
pub const hw_accel_gpu_memory = @import("hw/accel/gpu_memory.zig");
pub const hw_accel_tensor_core_matmul = @import("hw/accel/tensor_core_matmul.zig");
pub const index_ssi = @import("index/ssi.zig");
pub const optimizer_sfd = @import("optimizer/sfd.zig");
pub const processor_oftb = @import("processor/oftb.zig");
pub const processor_rsf = @import("processor/rsf.zig");
pub const ranker_ranker = @import("ranker/ranker.zig");
pub const tokenizer_mgt = @import("tokenizer/mgt.zig");
pub const tools_pretokenize = @import("tools/pretokenize.zig");

fn referenceNestedDeclarations(comptime T: type) void {
    @setEvalBranchQuota(10_000_000);
    inline for (comptime std.meta.declarations(T)) |declaration| {
        if (@TypeOf(@field(T, declaration.name)) == type) {
            const member = @field(T, declaration.name);
            switch (@typeInfo(member)) {
                .@"struct", .@"enum", .@"union", .@"opaque" => {
                    if (comptime std.mem.startsWith(u8, @typeName(member), @typeName(T) ++ ".") and
                        std.mem.indexOfScalar(u8, @typeName(member), '(') == null)
                    {
                        referenceNestedDeclarations(member);
                    }
                },
                else => {},
            }
        } else {
            _ = &@field(T, declaration.name);
        }
    }
}

export fn jaide_semantic_check_root() void {
    @setEvalBranchQuota(10_000_000);
    inline for (comptime std.meta.declarations(@This())) |declaration| {
        if (@TypeOf(@field(@This(), declaration.name)) == type) {
            const member = @field(@This(), declaration.name);
            switch (@typeInfo(member)) {
                .@"struct", .@"enum", .@"union", .@"opaque" => referenceNestedDeclarations(member),
                else => {},
            }
        }
    }
}
