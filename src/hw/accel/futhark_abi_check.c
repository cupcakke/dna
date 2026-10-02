#ifdef JAIDE_FUTHARK_CUDA
#include "main_gpu.h"
#else
#include "main_cpu.h"
#endif

#include <stdbool.h>
#include <stdint.h>

typedef struct futhark_context_config *(*context_config_new_type)(void);
typedef void (*context_config_free_type)(struct futhark_context_config *);
typedef struct futhark_context *(*context_new_type)(struct futhark_context_config *);
typedef void (*context_free_type)(struct futhark_context *);
typedef int (*context_sync_type)(struct futhark_context *);
typedef char *(*context_get_error_type)(struct futhark_context *);
typedef int (*context_clear_caches_type)(struct futhark_context *);

typedef struct futhark_f16_2d *(*new_f16_2d_type)(struct futhark_context *, const uint16_t *, int64_t, int64_t);
typedef struct futhark_f16_3d *(*new_f16_3d_type)(struct futhark_context *, const uint16_t *, int64_t, int64_t, int64_t);
typedef struct futhark_f32_1d *(*new_f32_1d_type)(struct futhark_context *, const float *, int64_t);
typedef struct futhark_f32_2d *(*new_f32_2d_type)(struct futhark_context *, const float *, int64_t, int64_t);
typedef struct futhark_f32_3d *(*new_f32_3d_type)(struct futhark_context *, const float *, int64_t, int64_t, int64_t);
typedef struct futhark_u64_1d *(*new_u64_1d_type)(struct futhark_context *, const uint64_t *, int64_t);
typedef struct futhark_i64_1d *(*new_i64_1d_type)(struct futhark_context *, const int64_t *, int64_t);

typedef int (*free_f16_2d_type)(struct futhark_context *, struct futhark_f16_2d *);
typedef int (*free_f16_3d_type)(struct futhark_context *, struct futhark_f16_3d *);
typedef int (*free_f32_1d_type)(struct futhark_context *, struct futhark_f32_1d *);
typedef int (*free_f32_2d_type)(struct futhark_context *, struct futhark_f32_2d *);
typedef int (*free_f32_3d_type)(struct futhark_context *, struct futhark_f32_3d *);
typedef int (*free_u64_1d_type)(struct futhark_context *, struct futhark_u64_1d *);
typedef int (*free_i64_1d_type)(struct futhark_context *, struct futhark_i64_1d *);

typedef int (*values_f16_2d_type)(struct futhark_context *, struct futhark_f16_2d *, uint16_t *);
typedef int (*values_f16_3d_type)(struct futhark_context *, struct futhark_f16_3d *, uint16_t *);
typedef int (*values_f32_1d_type)(struct futhark_context *, struct futhark_f32_1d *, float *);
typedef int (*values_f32_2d_type)(struct futhark_context *, struct futhark_f32_2d *, float *);
typedef int (*values_f32_3d_type)(struct futhark_context *, struct futhark_f32_3d *, float *);
typedef int (*values_u64_1d_type)(struct futhark_context *, struct futhark_u64_1d *, uint64_t *);
typedef int (*values_i64_1d_type)(struct futhark_context *, struct futhark_i64_1d *, int64_t *);

typedef int (*matmul_type)(
    struct futhark_context *,
    struct futhark_f32_2d **,
    const struct futhark_f32_2d *,
    const struct futhark_f32_2d *
);

typedef int (*rsf_forward_type)(
    struct futhark_context *,
    struct futhark_f16_2d **,
    const struct futhark_f16_2d *,
    const struct futhark_f16_2d *,
    const struct futhark_f16_2d *,
    uint16_t,
    uint16_t
);

typedef int (*scale_matrix_f32_type)(
    struct futhark_context *,
    struct futhark_f32_2d **,
    const struct futhark_f32_2d *,
    float
);

typedef int (*clip_matrix_type)(
    struct futhark_context *,
    struct futhark_f32_2d **,
    const struct futhark_f32_2d *,
    float
);

typedef int (*embedding_sum_squares_type)(
    struct futhark_context *,
    float *,
    const struct futhark_f16_2d *
);

typedef int (*master_weights_to_f16_3d_type)(
    struct futhark_context *,
    struct futhark_f16_3d **,
    const struct futhark_f32_3d *
);

typedef int (*master_weights_to_f16_2d_type)(
    struct futhark_context *,
    struct futhark_f16_2d **,
    const struct futhark_f32_2d *
);

typedef int (*embedding_update_sfd_master_type)(
    struct futhark_context *,
    struct futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32 **,
    const struct futhark_f32_2d *,
    const struct futhark_f32_2d *,
    const struct futhark_f32_2d *,
    const struct futhark_f32_2d *,
    float,
    float,
    float,
    int64_t,
    float,
    float,
    float
);

typedef int (*rsf_stack_forward_type)(
    struct futhark_context *,
    struct futhark_f16_3d **,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    uint16_t,
    uint16_t
);

typedef int (*rsf_stack_inverse_type)(
    struct futhark_context *,
    struct futhark_f16_3d **,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    uint16_t,
    uint16_t
);

typedef int (*backward_gradients_type)(
    struct futhark_context *,
    struct futhark_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32 **,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    const struct futhark_i64_1d *,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    bool,
    float,
    float,
    float,
    float,
    float,
    float
);

typedef int (*stack_sfd_type)(
    struct futhark_context *,
    struct futhark_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32 **,
    const struct futhark_f32_3d *,
    const struct futhark_f32_3d *,
    const struct futhark_f32_3d *,
    const struct futhark_f32_3d *,
    float,
    float,
    float,
    int64_t,
    float,
    float,
    float
);

typedef int (*embedding_forward_padded_type)(
    struct futhark_context *,
    struct futhark_f16_3d **,
    const struct futhark_i64_1d *,
    const struct futhark_i64_1d *,
    const struct futhark_i64_1d *,
    const struct futhark_f16_2d *
);

typedef int (*embedding_backward_padded_type)(
    struct futhark_context *,
    struct futhark_f32_2d **,
    const struct futhark_i64_1d *,
    const struct futhark_i64_1d *,
    const struct futhark_f16_3d *,
    const struct futhark_f32_2d *
);

typedef int (*stack_spectral_type)(
    struct futhark_context *,
    struct futhark_opaque_tup3_arr3d_f32_f32_f32 **,
    const struct futhark_f32_3d *,
    float,
    int64_t
);

typedef int (*embedding_spectral_type)(
    struct futhark_context *,
    struct futhark_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32 **,
    const struct futhark_f32_2d *,
    const struct futhark_f32_1d *,
    const struct futhark_f32_1d *,
    int64_t,
    float
);

typedef int (*graph_batch_encode_type)(
    struct futhark_context *,
    struct futhark_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64 **,
    const struct futhark_u64_1d *,
    uint64_t
);

_Static_assert(sizeof(int64_t) == 8, "int64_t must be 8 bytes");
_Static_assert(sizeof(uint64_t) == 8, "uint64_t must be 8 bytes");
_Static_assert(sizeof(uint16_t) == 2, "uint16_t must be 2 bytes");
_Static_assert(sizeof(float) == 4, "float must be 4 bytes");

_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_context_config_new), context_config_new_type), "futhark_context_config_new ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_context_config_free), context_config_free_type), "futhark_context_config_free ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_context_new), context_new_type), "futhark_context_new ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_context_free), context_free_type), "futhark_context_free ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_context_sync), context_sync_type), "futhark_context_sync ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_context_get_error), context_get_error_type), "futhark_context_get_error ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_context_clear_caches), context_clear_caches_type), "futhark_context_clear_caches ABI mismatch");

_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_new_f16_2d), new_f16_2d_type), "futhark_new_f16_2d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_new_f16_3d), new_f16_3d_type), "futhark_new_f16_3d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_new_f32_1d), new_f32_1d_type), "futhark_new_f32_1d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_new_f32_2d), new_f32_2d_type), "futhark_new_f32_2d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_new_f32_3d), new_f32_3d_type), "futhark_new_f32_3d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_new_u64_1d), new_u64_1d_type), "futhark_new_u64_1d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_new_i64_1d), new_i64_1d_type), "futhark_new_i64_1d ABI mismatch");

_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_free_f16_2d), free_f16_2d_type), "futhark_free_f16_2d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_free_f16_3d), free_f16_3d_type), "futhark_free_f16_3d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_free_f32_1d), free_f32_1d_type), "futhark_free_f32_1d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_free_f32_2d), free_f32_2d_type), "futhark_free_f32_2d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_free_f32_3d), free_f32_3d_type), "futhark_free_f32_3d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_free_u64_1d), free_u64_1d_type), "futhark_free_u64_1d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_free_i64_1d), free_i64_1d_type), "futhark_free_i64_1d ABI mismatch");

_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_values_f16_2d), values_f16_2d_type), "futhark_values_f16_2d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_values_f16_3d), values_f16_3d_type), "futhark_values_f16_3d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_values_f32_1d), values_f32_1d_type), "futhark_values_f32_1d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_values_f32_2d), values_f32_2d_type), "futhark_values_f32_2d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_values_f32_3d), values_f32_3d_type), "futhark_values_f32_3d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_values_u64_1d), values_u64_1d_type), "futhark_values_u64_1d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_values_i64_1d), values_i64_1d_type), "futhark_values_i64_1d ABI mismatch");

_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_matmul), matmul_type), "matmul ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_rsf_forward), rsf_forward_type), "rsf_forward ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_scale_matrix_f32), scale_matrix_f32_type), "scale_matrix_f32 ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_clip_matrix_global_norm_f32), clip_matrix_type), "clip_matrix_global_norm_f32 ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_embedding_sum_squares), embedding_sum_squares_type), "embedding_sum_squares ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_master_weights_to_f16_3d), master_weights_to_f16_3d_type), "master_weights_to_f16_3d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_master_weights_to_f16_2d), master_weights_to_f16_2d_type), "master_weights_to_f16_2d ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_embedding_update_sfd_master), embedding_update_sfd_master_type), "embedding_update_sfd_master ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_rsf_stack_forward), rsf_stack_forward_type), "rsf_stack_forward ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_rsf_stack_inverse), rsf_stack_inverse_type), "rsf_stack_inverse ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_rsf_stack_backward_gradients_fused), backward_gradients_type), "rsf_stack_backward_gradients_fused ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_stack_update_sfd_master), stack_sfd_type), "stack_update_sfd_master ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_embedding_forward_padded), embedding_forward_padded_type), "embedding_forward_padded ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_embedding_backward_padded), embedding_backward_padded_type), "embedding_backward_padded ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_stack_spectral_normalize), stack_spectral_type), "stack_spectral_normalize ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_embedding_spectral_normalize), embedding_spectral_type), "embedding_spectral_normalize ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_graph_batch_encode), graph_batch_encode_type), "graph_batch_encode ABI mismatch");

int jaide_futhark_abi_check_present(void) {
    return 1;
}
