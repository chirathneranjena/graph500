#ifndef BFS_BEAMER_CUH
#define BFS_BEAMER_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>

struct CSRBinaryHeader {
    char magic[8];         // "GRAPH500"
    uint64_t num_vertices; // N
    uint64_t num_edges;    // M
    uint64_t weighted;     // 1 if weighted, 0 if unweighted
    uint64_t reserved;     // Reserved / alignment
};

struct GraphCSRData {
    uint64_t num_vertices;
    uint64_t num_edges;
    std::vector<uint64_t> h_indptr;
    std::vector<uint64_t> h_indices;
};

struct BFSStats {
    uint64_t root_vertex;
    uint64_t root_degree;
    uint64_t visited_vertices;
    uint64_t traversed_edges;
    int max_depth;
    double total_time_sec;
    double teps; // Traversed Edges Per Second
    std::vector<int> level_directions; // 0 = Top-Down, 1 = Bottom-Up
    std::vector<uint64_t> level_frontier_sizes;
    std::vector<double> level_times_ms;
};

bool load_graph_binary(const std::string& filepath, GraphCSRData& graph);

BFSStats run_beamer_bfs_gpu(
    const GraphCSRData& graph,
    int64_t requested_root = -1,
    float alpha = 14.0f,
    float beta = 24.0f,
    bool verbose = false
);

struct Graph500BenchmarkStats {
    int num_searches;
    std::vector<uint64_t> selected_roots;
    std::vector<BFSStats> search_results;
    std::vector<double> teps_values;
    
    double min_teps;
    double q1_teps;
    double median_teps;
    double q3_teps;
    double max_teps;
    double mean_teps;
    double stddev_teps;
    double harmonic_mean_teps;
};

Graph500BenchmarkStats run_graph500_benchmark_gpu(
    const GraphCSRData& graph,
    int num_searches = 64,
    float alpha = 14.0f,
    float beta = 24.0f,
    uint64_t seed = 42
);

#endif // BFS_BEAMER_CUH
