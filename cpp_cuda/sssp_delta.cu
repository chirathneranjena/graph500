#include "sssp_delta.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#ifndef CSR_BINARY_HEADER_DEFINED
#define CSR_BINARY_HEADER_DEFINED
struct CSRBinaryHeader {
    char magic[8];         // "GRAPH500"
    uint64_t num_vertices; // N
    uint64_t num_edges;    // M
    uint64_t weighted;     // 1 if weighted, 0 if unweighted
    uint64_t reserved[1];
};
#endif

// Lock-Free Double Precision Atomic Min on CUDA
__device__ inline double atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Binary CSR Loader
bool load_graph_binary(const std::string& filepath, GraphCSRData& graph) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open binary graph file '" << filepath << "'.\n";
        return false;
    }

    CSRBinaryHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (file.gcount() < (std::streamsize)sizeof(header)) {
        std::cerr << "Error: File header corrupted or incomplete.\n";
        return false;
    }

    if (std::string(header.magic, 8) != "GRAPH500") {
        std::cerr << "Error: Invalid binary file format (magic header mismatch).\n";
        return false;
    }

    graph.num_vertices = header.num_vertices;
    graph.num_edges = header.num_edges;
    graph.weighted = header.weighted;

    graph.h_indptr.resize(graph.num_vertices + 1);
    graph.h_indices.resize(graph.num_edges);
    if (graph.weighted) {
        graph.h_data.resize(graph.num_edges);
    } else {
        graph.h_data.assign(graph.num_edges, 1.0);
    }

    file.read(reinterpret_cast<char*>(graph.h_indptr.data()), (graph.num_vertices + 1) * sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(graph.h_indices.data()), graph.num_edges * sizeof(uint64_t));
    if (graph.weighted) {
        file.read(reinterpret_cast<char*>(graph.h_data.data()), graph.num_edges * sizeof(double));
    }

    if (!file) {
        std::cerr << "Error: Failed to read complete CSR arrays from file.\n";
        return false;
    }

    file.close();
    return true;
}

// CUDA Kernel: Relax edges originating from frontier vertices in the active bucket
__global__ void kernel_sssp_relax_edges(
    const uint64_t* __restrict__ d_frontier,
    int frontier_count,
    const uint64_t* __restrict__ d_indptr,
    const uint64_t* __restrict__ d_indices,
    const double* __restrict__ d_data,
    double* __restrict__ d_dist,
    bool relax_light,
    double delta,
    int* __restrict__ d_changed_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_count) return;

    uint64_t u = d_frontier[idx];
    double du = d_dist[u];

    uint64_t start_edge = d_indptr[u];
    uint64_t end_edge = d_indptr[u + 1];

    for (uint64_t e = start_edge; e < end_edge; ++e) {
        double w = d_data[e];
        bool is_light = (w <= delta);

        if ((relax_light && is_light) || (!relax_light && !is_light)) {
            uint64_t v = d_indices[e];
            double new_dist = du + w;

            // Fast pre-check to eliminate unnecessary atomic instructions
            if (d_dist[v] > new_dist) {
                double old_dist = atomicMinDouble(&d_dist[v], new_dist);
                if (new_dist < old_dist) {
                    *d_changed_flag = 1;
                }
            }
        }
    }
}

// CUDA Kernel: Build active frontier array for current bucket range [bucket_min, bucket_max)
__global__ void kernel_update_active_frontier(
    uint64_t num_vertices,
    const double* __restrict__ d_dist,
    uint64_t* __restrict__ d_frontier,
    double bucket_min,
    double bucket_max,
    int* __restrict__ d_frontier_count
) {
    uint64_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    double du = d_dist[u];
    if (du >= bucket_min && du < bucket_max) {
        int pos = atomicAdd(d_frontier_count, 1);
        d_frontier[pos] = u;
    }
}

// CUDA Kernel: Find minimum distance among remaining active vertices
__global__ void kernel_find_min_distance(
    uint64_t num_vertices,
    const double* __restrict__ d_dist,
    double current_bucket_max,
    double* __restrict__ d_global_min
) {
    uint64_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    double du = d_dist[u];
    if (du >= current_bucket_max && du < INFINITY) {
        atomicMinDouble(d_global_min, du);
    }
}

// High-degree root selection routine
static int64_t auto_select_root_gpu(const GraphCSRData& graph) {
    uint64_t N = graph.num_vertices;
    uint64_t best_root = 0;
    uint64_t max_degree = 0;

    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<uint64_t> dist(0, N - 1);

    int samples = std::min((uint64_t)1024, N);
    for (int i = 0; i < samples; ++i) {
        uint64_t v = dist(rng);
        uint64_t deg = graph.h_indptr[v + 1] - graph.h_indptr[v];
        if (deg > max_degree) {
            max_degree = deg;
            best_root = v;
        }
    }

    if (max_degree == 0) {
        for (uint64_t v = 0; v < N; ++v) {
            uint64_t deg = graph.h_indptr[v + 1] - graph.h_indptr[v];
            if (deg > max_degree) {
                max_degree = deg;
                best_root = v;
                break;
            }
        }
    }

    return (int64_t)best_root;
}

// Internal core SSSP solver using pre-allocated GPU CSR arrays
static SSSPStats run_delta_stepping_sssp_gpu_internal(
    const GraphCSRData& graph,
    const uint64_t* d_indptr,
    const uint64_t* d_indices,
    const double* d_data,
    int64_t root,
    double delta,
    bool show_stats
) {
    SSSPStats stats;
    stats.total_vertices = graph.num_vertices;
    stats.total_edges = graph.num_edges;
    stats.root_vertex = root;
    stats.root_degree = graph.h_indptr[root + 1] - graph.h_indptr[root];

    uint64_t N = graph.num_vertices;

    // Allocate GPU auxiliary buffers
    double* d_dist;
    uint64_t* d_frontier;
    int *d_flag, *d_frontier_count;
    double* d_global_min;

    CUDA_CHECK(cudaMalloc(&d_dist, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_frontier, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_global_min, sizeof(double)));

    // Initialize distances with INFINITY
    std::vector<double> h_dist(N, std::numeric_limits<double>::infinity());
    h_dist[root] = 0.0;
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size_N = (N + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    double current_bucket_min = 0.0;
    double current_bucket_max = delta;
    uint64_t step_count = 0;

    while (current_bucket_min < INFINITY) {
        step_count++;

        // 1. Build compact active frontier array for current bucket range [current_bucket_min, current_bucket_max)
        CUDA_CHECK(cudaMemset(d_frontier_count, 0, sizeof(int)));
        kernel_update_active_frontier<<<grid_size_N, block_size>>>(
            N, d_dist, d_frontier, current_bucket_min, current_bucket_max, d_frontier_count
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        int active_count = 0;
        CUDA_CHECK(cudaMemcpy(&active_count, d_frontier_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (active_count > 0) {
            int grid_size_active = (active_count + block_size - 1) / block_size;

            // 2. Inner Light Edge Relaxation Loop for active bucket vertices
            while (true) {
                int changed = 0;
                CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));

                kernel_sssp_relax_edges<<<grid_size_active, block_size>>>(
                    d_frontier, active_count, d_indptr, d_indices, d_data, d_dist,
                    true, delta, d_flag
                );
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(&changed, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
                if (!changed) break;

                // Re-build active frontier after light edge updates
                CUDA_CHECK(cudaMemset(d_frontier_count, 0, sizeof(int)));
                kernel_update_active_frontier<<<grid_size_N, block_size>>>(
                    N, d_dist, d_frontier, current_bucket_min, current_bucket_max, d_frontier_count
                );
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(&active_count, d_frontier_count, sizeof(int), cudaMemcpyDeviceToHost));
                if (active_count == 0) break;
                grid_size_active = (active_count + block_size - 1) / block_size;
            }

            // 3. Heavy Edge Relaxation for active bucket vertices
            if (active_count > 0) {
                CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
                kernel_sssp_relax_edges<<<grid_size_active, block_size>>>(
                    d_frontier, active_count, d_indptr, d_indices, d_data, d_dist,
                    false, delta, d_flag
                );
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        // 4. Advance to next non-empty bucket range
        double next_min = INFINITY;
        CUDA_CHECK(cudaMemcpy(d_global_min, &next_min, sizeof(double), cudaMemcpyHostToDevice));

        kernel_find_min_distance<<<grid_size_N, block_size>>>(
            N, d_dist, current_bucket_max, d_global_min
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&next_min, d_global_min, sizeof(double), cudaMemcpyDeviceToHost));

        if (next_min == INFINITY) {
            break;
        }

        // Set next bucket bounds
        current_bucket_min = std::floor(next_min / delta) * delta;
        current_bucket_max = current_bucket_min + delta;

        if (step_count > N * 10) break; // Safety cutoff
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    stats.total_time_sec = elapsed_ms / 1000.0;
    stats.total_bucket_steps = step_count;

    // Transfer final distance array back to host for statistics
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, N * sizeof(double), cudaMemcpyDeviceToHost));

    uint64_t reachable = 0;
    uint64_t traversed_edges = 0;
    double min_d = std::numeric_limits<double>::max();
    double max_d = 0.0;
    double sum_d = 0.0;

    std::vector<uint64_t> bins(10, 0);

    for (uint64_t i = 0; i < N; ++i) {
        double d = h_dist[i];
        if (d < std::numeric_limits<double>::infinity()) {
            reachable++;
            uint64_t deg = graph.h_indptr[i + 1] - graph.h_indptr[i];
            traversed_edges += deg;

            if (d > 0.0 && d < min_d) min_d = d;
            if (d > max_d) max_d = d;
            sum_d += d;
        }
    }

    if (min_d == std::numeric_limits<double>::max()) min_d = 0.0;

    stats.reachable_vertices = reachable;
    stats.traversed_edges = traversed_edges;
    stats.min_distance = min_d;
    stats.max_distance = max_d;
    stats.avg_distance = reachable > 0 ? (sum_d / reachable) : 0.0;
    stats.teps = stats.total_time_sec > 0 ? (double)traversed_edges / stats.total_time_sec : 0.0;

    if (show_stats && max_d > 0.0) {
        double bin_width = max_d / 10.0;
        for (uint64_t i = 0; i < N; ++i) {
            double d = h_dist[i];
            if (d < std::numeric_limits<double>::infinity()) {
                int bin_idx = std::min(9, (int)(d / bin_width));
                bins[bin_idx]++;
            }
        }
        stats.distance_bins = bins;
    }

    // Cleanup GPU Memory
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_flag));
    CUDA_CHECK(cudaFree(d_frontier_count));
    CUDA_CHECK(cudaFree(d_global_min));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return stats;
}

// Main Delta-Stepping GPU Driver (Single Search Entry Point)
SSSPStats run_delta_stepping_sssp_gpu(
    const GraphCSRData& graph,
    int64_t requested_root,
    double delta,
    bool show_stats
) {
    int64_t root = requested_root;
    if (root < 0 || (uint64_t)root >= graph.num_vertices) {
        root = auto_select_root_gpu(graph);
    }

    uint64_t N = graph.num_vertices;
    uint64_t M = graph.num_edges;

    // Allocate GPU arrays for graph CSR
    uint64_t *d_indptr, *d_indices;
    double *d_data;

    CUDA_CHECK(cudaMalloc(&d_indptr, (N + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_indices, M * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_data, M * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_indptr, graph.h_indptr.data(), (N + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, graph.h_indices.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, graph.h_data.data(), M * sizeof(double), cudaMemcpyHostToDevice));

    SSSPStats stats = run_delta_stepping_sssp_gpu_internal(
        graph, d_indptr, d_indices, d_data, root, delta, show_stats
    );

    CUDA_CHECK(cudaFree(d_indptr));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_data));

    return stats;
}

// Graph500 SSSP 64-Search Benchmark Routine
Graph500SSSPBenchmarkStats run_graph500_sssp_benchmark_gpu(
    const GraphCSRData& graph,
    int num_searches,
    double delta,
    uint64_t seed
) {
    Graph500SSSPBenchmarkStats bench_stats;
    bench_stats.num_searches = num_searches;

    uint64_t N = graph.num_vertices;
    uint64_t M = graph.num_edges;

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, N - 1);

    std::vector<uint64_t> selected_roots;
    selected_roots.reserve(num_searches);

    // 1. Sample 64 distinct root vertices with degree >= 1
    while ((int)selected_roots.size() < num_searches) {
        uint64_t candidate = dist(rng);
        uint64_t deg = graph.h_indptr[candidate + 1] - graph.h_indptr[candidate];
        if (deg >= 1) {
            bool duplicate = false;
            for (uint64_t existing : selected_roots) {
                if (existing == candidate) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                selected_roots.push_back(candidate);
            }
        }
    }

    bench_stats.selected_roots = selected_roots;

    // 2. Pre-allocate and transfer GPU Graph CSR arrays ONCE for all 64 SSSP runs
    uint64_t *d_indptr, *d_indices;
    double *d_data;

    CUDA_CHECK(cudaMalloc(&d_indptr, (N + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_indices, M * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_data, M * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_indptr, graph.h_indptr.data(), (N + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, graph.h_indices.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, graph.h_data.data(), M * sizeof(double), cudaMemcpyHostToDevice));

    // 3. Execute SSSP on all 64 roots
    bench_stats.search_results.reserve(num_searches);
    bench_stats.teps_values.reserve(num_searches);

    std::cout << "[Graph500 SSSP] Executing " << num_searches << " SSSP runs on GPU..." << std::flush;

    for (int i = 0; i < num_searches; ++i) {
        SSSPStats single_stats = run_delta_stepping_sssp_gpu_internal(
            graph, d_indptr, d_indices, d_data, (int64_t)selected_roots[i], delta, false
        );
        bench_stats.search_results.push_back(single_stats);
        bench_stats.teps_values.push_back(single_stats.teps);
        if ((i + 1) % 16 == 0 || i == num_searches - 1) {
            std::cout << " " << (i + 1) << "/" << num_searches << std::flush;
        }
    }
    std::cout << " Done.\n";

    CUDA_CHECK(cudaFree(d_indptr));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_data));

    // 4. Compute Graph500 SSSP Benchmark Statistics
    std::vector<double> sorted_teps = bench_stats.teps_values;
    std::sort(sorted_teps.begin(), sorted_teps.end());

    int n = num_searches;
    bench_stats.min_teps = sorted_teps[0];
    bench_stats.max_teps = sorted_teps[n - 1];
    
    // Percentiles
    bench_stats.q1_teps = sorted_teps[(int)(0.25 * (n - 1))];
    bench_stats.median_teps = (n % 2 == 0) ? 0.5 * (sorted_teps[n / 2 - 1] + sorted_teps[n / 2]) : sorted_teps[n / 2];
    bench_stats.q3_teps = sorted_teps[(int)(0.75 * (n - 1))];

    // Arithmetic Mean & StdDev
    double sum = 0.0;
    double inv_sum = 0.0;
    for (double teps : sorted_teps) {
        sum += teps;
        inv_sum += (teps > 0.0) ? (1.0 / teps) : 0.0;
    }
    bench_stats.mean_teps = sum / n;

    double variance_sum = 0.0;
    for (double teps : sorted_teps) {
        double diff = teps - bench_stats.mean_teps;
        variance_sum += diff * diff;
    }
    bench_stats.stddev_teps = std::sqrt(variance_sum / n);

    // Harmonic Mean
    bench_stats.harmonic_mean_teps = (inv_sum > 0.0) ? ((double)n / inv_sum) : 0.0;

    return bench_stats;
}
