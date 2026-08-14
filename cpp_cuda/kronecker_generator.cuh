#ifndef KRONECKER_GENERATOR_CUH
#define KRONECKER_GENERATOR_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>

struct GraphStats {
    uint64_t num_vertices;
    uint64_t active_vertices;
    uint64_t total_edges;
    uint64_t unique_edges;
    uint64_t self_loops;
    double matrix_density;
    double matrix_sparsity_pct;
    double avg_degree;
    uint64_t max_degree;
    double min_weight;
    double avg_weight;
    double max_weight;
    double generation_time_sec;
    double postprocess_time_sec;
    double total_time_sec;
};

struct CSRGraphGPU {
    uint64_t num_vertices;
    uint64_t num_edges;
    std::vector<uint64_t> h_indptr;
    std::vector<uint64_t> h_indices;
    std::vector<double> h_data;
    GraphStats stats;
};

struct CSRBinaryHeader {
    char magic[8];         // "GRAPH500"
    uint64_t num_vertices; // N
    uint64_t num_edges;    // M
    uint64_t weighted;     // 1 if weighted, 0 if unweighted
    uint64_t reserved;     // Reserved / alignment
};

CSRGraphGPU generate_kronecker_gpu(
    int scale,
    int edge_factor,
    const float initiator[4],
    uint64_t seed = 42,
    bool scramble = false,
    bool remove_self_loops = true,
    bool symmetrize = true,
    bool collapse_duplicates = true,
    bool enable_weights = true,
    bool compute_stats = false
);

void save_csr_text(const CSRGraphGPU& graph, const std::string& filepath, bool weighted = true);
void save_csr_binary(const CSRGraphGPU& graph, const std::string& filepath, bool weighted = true);
bool load_csr_binary(const std::string& filepath, CSRGraphGPU& graph);
void save_edgelist_text(const CSRGraphGPU& graph, const std::string& filepath, bool weighted = true);

#endif // KRONECKER_GENERATOR_CUH
