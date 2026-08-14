#ifndef SSSP_DELTA_CUH
#define SSSP_DELTA_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>

struct SSSPStats {
    int64_t root_vertex;
    uint64_t root_degree;
    uint64_t reachable_vertices;
    uint64_t total_vertices;
    uint64_t traversed_edges;
    uint64_t total_edges;
    double min_distance;
    double avg_distance;
    double max_distance;
    double total_time_sec;
    double teps;
    uint64_t total_bucket_steps;
    std::vector<uint64_t> distance_bins; // Optional distance distribution histogram
};

struct DeltaSteppingConfig {
    double delta;         // Delta bucket width (default: 0.1)
    int max_bucket_steps; // Safety limit on bucket steps
};

struct GraphCSRData {
    uint64_t num_vertices;
    uint64_t num_edges;
    uint64_t weighted;
    std::vector<uint64_t> h_indptr;
    std::vector<uint64_t> h_indices;
    std::vector<double> h_data;
};

// Function Declarations
bool load_graph_binary(const std::string& filepath, GraphCSRData& graph);

SSSPStats run_delta_stepping_sssp_gpu(
    const GraphCSRData& graph,
    int64_t requested_root = -1,
    double delta = 0.1,
    bool show_stats = false
);

#endif // SSSP_DELTA_CUH
