#ifndef JAIDE_H
#define JAIDE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JAIDE_SUCCESS 0
#define JAIDE_ERROR_NULL_POINTER (-1)
#define JAIDE_ERROR_ALLOCATION (-2)
#define JAIDE_ERROR_NODE_NOT_FOUND (-3)
#define JAIDE_ERROR_EDGE_NOT_FOUND (-4)
#define JAIDE_ERROR_INVALID_QUALITY (-5)
#define JAIDE_ERROR_OPTIMIZATION_FAILED (-6)
#define JAIDE_ERROR_INVALID_STRING (-7)
#define JAIDE_ERROR_OPERATION_FAILED (-8)
#define JAIDE_ERROR_DUPLICATE_NODE (-9)
#define JAIDE_ERROR_DUPLICATE_EDGE (-10)
#define JAIDE_ERROR_INVALID_PARAMETER (-11)
#define JAIDE_ERROR_MATH_ERROR (-12)
#define JAIDE_ERROR_NOT_INITIALIZED (-13)
#define JAIDE_ERROR_SELF_REFERENCE (-14)
#define JAIDE_ERROR_INVALID_STATE (-15)
#define JAIDE_ERROR_THREADING (-16)
#define JAIDE_ERROR_UNKNOWN_GATE (-17)
#define JAIDE_ERROR_OUT_OF_MEMORY (-18)

#define JAIDE_GATE_IDENTITY 0
#define JAIDE_GATE_HADAMARD 1
#define JAIDE_GATE_PAULI_X 2
#define JAIDE_GATE_PAULI_Y 3
#define JAIDE_GATE_PAULI_Z 4

typedef struct JaideGraph JaideGraph;
typedef struct JaideOptimizer JaideOptimizer;

typedef struct JaideQuantumState {
    double real;
    double imag;
} JaideQuantumState;

const char *jaide_get_error_string(int code);

int jaide_version_major(void);
int jaide_version_minor(void);
int jaide_version_patch(void);

JaideGraph *jaide_create_graph(void);
void jaide_destroy_graph(JaideGraph *handle);
int jaide_clear_graph(JaideGraph *graph);

int jaide_add_node(JaideGraph *graph, const char *id, const char *type_name);
int jaide_remove_node(JaideGraph *graph, const char *id);
int jaide_has_node(JaideGraph *graph, const char *id);
int jaide_graph_node_count(JaideGraph *graph);

int jaide_set_node_quantum_state(JaideGraph *graph, const char *id, double real, double imag);
int jaide_get_node_quantum_state(JaideGraph *graph, const char *id, JaideQuantumState *out_state);
double jaide_get_node_probability(JaideGraph *graph, const char *id);
double jaide_get_node_magnitude(JaideGraph *graph, const char *id);
double jaide_measure_node(JaideGraph *graph, const char *node_id);
double jaide_get_node_phase(JaideGraph *graph, const char *id);
int jaide_set_node_phase(JaideGraph *graph, const char *id, double phase);
int jaide_get_node_data(JaideGraph *graph, const char *id, unsigned char *out_data, size_t max_len);

int jaide_apply_gate(JaideGraph *graph, const char *node_id, int gate_type);
int jaide_apply_identity_gate(JaideGraph *graph, const char *node_id);
int jaide_apply_hadamard(JaideGraph *graph, const char *node_id);
int jaide_apply_pauli_x(JaideGraph *graph, const char *node_id);
int jaide_apply_pauli_y(JaideGraph *graph, const char *node_id);
int jaide_apply_pauli_z(JaideGraph *graph, const char *node_id);

int jaide_add_edge(JaideGraph *graph, const char *source, const char *target, double weight);
int jaide_remove_edge(JaideGraph *graph, const char *source, const char *target);
int jaide_has_edge(JaideGraph *graph, const char *source, const char *target);
int jaide_graph_edge_count(JaideGraph *graph);
double jaide_get_edge_weight(JaideGraph *graph, const char *source, const char *target);
int jaide_set_edge_weight(JaideGraph *graph, const char *source, const char *target, double weight);
int jaide_get_edge_quality(JaideGraph *graph, const char *source, const char *target);
int jaide_set_edge_quality(JaideGraph *graph, const char *source, const char *target, int quality);
double jaide_get_edge_fractal_dimension(JaideGraph *graph, const char *source, const char *target);
int jaide_set_edge_fractal_dimension(JaideGraph *graph, const char *source, const char *target, double dimension);
double jaide_get_edge_correlation_magnitude(JaideGraph *graph, const char *source, const char *target);

int jaide_entangle_nodes(JaideGraph *graph, const char *node1, const char *node2);
double jaide_get_fractal_dimension(JaideGraph *graph);
int jaide_get_topology_hash(JaideGraph *graph, unsigned char *out_hash, size_t max_len);

int jaide_encode_information(JaideGraph *graph, const unsigned char *data, char *out_node_id, size_t max_len);
int jaide_decode_information(JaideGraph *graph, const char *node_id, unsigned char *out_data, size_t max_len);

JaideOptimizer *jaide_create_optimizer(double temp, double cooling, int max_iter);
void jaide_destroy_optimizer(JaideOptimizer *opt);
int jaide_optimize_graph(JaideOptimizer *opt, JaideGraph *graph);
int jaide_set_optimizer_config(JaideOptimizer *opt, const char *key, double value);
int jaide_get_optimizer_statistics(JaideOptimizer *opt, int *out_iterations, double *out_best_energy, double *out_acceptance_rate);

#ifdef __cplusplus
}
#endif

#endif
