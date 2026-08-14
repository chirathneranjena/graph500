#include "bfs_beamer.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Clear bitmap kernel
__global__ void kernel_clear_bitmap(uint32_t* bitmap, uint64_t size_words) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_words) {
        bitmap[idx] = 0;
    }
}

// Convert frontier array to bitmap
__global__ void kernel_frontier_to_bitmap(
    const uint64_t* d_frontier,
    uint64_t frontier_size,
    uint32_t* d_bitmap
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frontier_size) {
        uint64_t u = d_frontier[idx];
        uint64_t word_idx = u >> 5;
        uint32_t bit_mask = 1U << (u & 31);
        atomicOr(&d_bitmap[word_idx], bit_mask);
    }
}

// Compute frontier edge volume
__global__ void kernel_count_frontier_edges(
    const uint64_t* d_frontier,
    uint64_t frontier_size,
    const uint64_t* d_indptr,
    unsigned long long* d_edge_volume
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frontier_size) {
        uint64_t u = d_frontier[idx];
        uint64_t deg = d_indptr[u + 1] - d_indptr[u];
        atomicAdd(d_edge_volume, (unsigned long long)deg);
    }
}

// Top-Down BFS Kernel
__global__ void kernel_bfs_top_down(
    const uint64_t* d_indptr,
    const uint64_t* d_indices,
    const uint64_t* d_frontier,
    uint64_t frontier_size,
    int32_t current_depth,
    int32_t* d_depth,
    uint64_t* d_next_frontier,
    uint64_t* d_next_frontier_count,
    uint32_t* d_next_frontier_bitmap
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    uint64_t u = d_frontier[idx];
    uint64_t start = d_indptr[u];
    uint64_t end = d_indptr[u + 1];

    for (uint64_t e = start; e < end; ++e) {
        uint64_t v = d_indices[e];
        
        if (d_depth[v] == -1) {
            int32_t old = atomicCAS(&d_depth[v], -1, current_depth + 1);
            if (old == -1) {
                // Successfully visited v
                uint64_t pos = atomicAdd((unsigned long long*)d_next_frontier_count, 1ULL);
                d_next_frontier[pos] = v;
                
                if (d_next_frontier_bitmap) {
                    uint64_t word_idx = v >> 5;
                    uint32_t bit_mask = 1U << (v & 31);
                    atomicOr(&d_next_frontier_bitmap[word_idx], bit_mask);
                }
            }
        }
    }
}

// Bottom-Up BFS Kernel
__global__ void kernel_bfs_bottom_up(
    uint64_t num_vertices,
    const uint64_t* d_indptr,
    const uint64_t* d_indices,
    const uint32_t* d_frontier_bitmap,
    int32_t current_depth,
    int32_t* d_depth,
    uint64_t* d_next_frontier,
    uint64_t* d_next_frontier_count,
    uint32_t* d_next_frontier_bitmap
) {
    uint64_t v = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    if (d_depth[v] != -1) return; // Already visited

    uint64_t start = d_indptr[v];
    uint64_t end = d_indptr[v + 1];

    for (uint64_t e = start; e < end; ++e) {
        uint64_t u = d_indices[e];
        uint64_t word_idx = u >> 5;
        uint32_t bit_mask = 1U << (u & 31);

        if ((d_frontier_bitmap[word_idx] & bit_mask) != 0) {
            // Found a parent in current frontier!
            d_depth[v] = current_depth + 1;
            
            uint64_t pos = atomicAdd((unsigned long long*)d_next_frontier_count, 1ULL);
            d_next_frontier[pos] = v;

            if (d_next_frontier_bitmap) {
                uint64_t v_word = v >> 5;
                uint32_t v_mask = 1U << (v & 31);
                atomicOr(&d_next_frontier_bitmap[v_word], v_mask);
            }

            break; // Early exit on first parent hit
        }
    }
}

// Count visited vertices & traversed edges
__global__ void kernel_compute_bfs_stats(
    uint64_t num_vertices,
    const int32_t* d_depth,
    const uint64_t* d_indptr,
    unsigned long long* d_visited_count,
    unsigned long long* d_traversed_edges
) {
    uint64_t v = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    if (d_depth[v] >= 0) {
        atomicAdd(d_visited_count, 1ULL);
        uint64_t deg = d_indptr[v + 1] - d_indptr[v];
        atomicAdd(d_traversed_edges, (unsigned long long)deg);
    }
}

// Binary Graph Loader
bool load_graph_binary(const std::string& filepath, GraphCSRData& graph) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open binary graph file: " << filepath << std::endl;
        return false;
    }

    CSRBinaryHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (file.gcount() < (std::streamsize)sizeof(header)) {
        std::cerr << "Error: File truncated or corrupted header." << std::endl;
        return false;
    }

    if (std::string(header.magic, 8) != "GRAPH500") {
        std::cerr << "Error: Invalid binary magic header in " << filepath << std::endl;
        return false;
    }

    graph.num_vertices = header.num_vertices;
    graph.num_edges = header.num_edges;

    graph.h_indptr.resize(graph.num_vertices + 1);
    file.read(reinterpret_cast<char*>(graph.h_indptr.data()), (graph.num_vertices + 1) * sizeof(uint64_t));

    graph.h_indices.resize(graph.num_edges);
    file.read(reinterpret_cast<char*>(graph.h_indices.data()), graph.num_edges * sizeof(uint64_t));

    return true;
}

// Internal Beamer BFS Solver with pre-allocated GPU graph CSR pointers
static BFSStats run_beamer_bfs_gpu_internal(
    const GraphCSRData& graph,
    const uint64_t* d_indptr,
    const uint64_t* d_indices,
    int64_t requested_root,
    float alpha,
    float beta,
    bool verbose
) {
    BFSStats stats;
    uint64_t N = graph.num_vertices;
    uint64_t M = graph.num_edges;

    // 1. Root Selection Logic
    uint64_t root = 0;
    if (requested_root >= 0) {
        if ((uint64_t)requested_root >= N) {
            std::cerr << "Warning: Requested root " << requested_root << " >= N (" << N 
                      << "). Defaulting to auto-selection." << std::endl;
            requested_root = -1;
        } else {
            root = (uint64_t)requested_root;
        }
    }

    if (requested_root < 0) {
        // Auto-select root with highest degree from candidate sampling
        uint64_t max_deg = 0;
        uint64_t best_root = 0;
        std::mt19937_64 rng(1337);
        std::uniform_int_distribution<uint64_t> dist(0, N - 1);

        for (int sample = 0; sample < 1000; ++sample) {
            uint64_t candidate = dist(rng);
            uint64_t deg = graph.h_indptr[candidate + 1] - graph.h_indptr[candidate];
            if (deg > max_deg) {
                max_deg = deg;
                best_root = candidate;
            }
        }
        root = best_root;
    }

    stats.root_vertex = root;
    stats.root_degree = graph.h_indptr[root + 1] - graph.h_indptr[root];

    if (verbose) {
        std::cout << "[BFS] Selected Root Vertex: " << stats.root_vertex 
                  << " (Degree: " << stats.root_degree << ")" << std::endl;
    }

    // 2. GPU Auxiliary Memory Allocations
    int32_t* d_depth = nullptr;
    uint64_t* d_frontier = nullptr;
    uint64_t* d_next_frontier = nullptr;
    uint64_t* d_frontier_count = nullptr;
    uint64_t* d_next_frontier_count = nullptr;
    uint32_t* d_frontier_bitmap = nullptr;
    uint32_t* d_next_frontier_bitmap = nullptr;
    unsigned long long* d_edge_volume = nullptr;

    uint64_t bitmap_words = (N + 31) / 32;

    CUDA_CHECK(cudaMalloc(&d_depth, N * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_bitmap, bitmap_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_bitmap, bitmap_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_edge_volume, sizeof(unsigned long long)));

    // Initialize depth array (-1 for unvisited)
    CUDA_CHECK(cudaMemset(d_depth, -1, N * sizeof(int32_t)));

    // Initialize root in depth array & frontier
    int32_t root_depth = 0;
    CUDA_CHECK(cudaMemcpy(&d_depth[root], &root_depth, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier, &root, sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    uint64_t init_frontier_size = 1;
    CUDA_CHECK(cudaMemcpy(d_frontier_count, &init_frontier_size, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Clear Bitmaps
    CUDA_CHECK(cudaMemset(d_frontier_bitmap, 0, bitmap_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_next_frontier_bitmap, 0, bitmap_words * sizeof(uint32_t)));

    // Set bit for root in frontier bitmap
    uint32_t root_bit = 1U << (root & 31);
    uint64_t root_word = root >> 5;
    CUDA_CHECK(cudaMemcpy(&d_frontier_bitmap[root_word], &root_bit, sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Timers & State variables
    cudaEvent_t start_event, stop_event, level_start, level_stop;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventCreate(&level_start));
    CUDA_CHECK(cudaEventCreate(&level_stop));

    int current_depth = 0;
    uint64_t current_frontier_size = 1;
    bool is_bottom_up = false; // Start in Top-Down mode

    CUDA_CHECK(cudaEventRecord(start_event));

    while (current_frontier_size > 0) {
        CUDA_CHECK(cudaEventRecord(level_start));

        // 3. Compute Frontier Edge Volume
        CUDA_CHECK(cudaMemset(d_edge_volume, 0, sizeof(unsigned long long)));
        int threads = 256;
        int blocks = (current_frontier_size + threads - 1) / threads;
        kernel_count_frontier_edges<<<blocks, threads>>>(
            d_frontier, current_frontier_size, d_indptr, d_edge_volume
        );

        unsigned long long frontier_edges = 0;
        CUDA_CHECK(cudaMemcpy(&frontier_edges, d_edge_volume, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        // 4. Direction Switching Logic
        if (!is_bottom_up && (float)frontier_edges > (float)M / alpha) {
            is_bottom_up = true;
            if (verbose) {
                std::cout << " [Level " << current_depth << "] Switch -> BOTTOM-UP "
                          << "(Frontier Edges: " << frontier_edges << " > M/" << alpha << ")" << std::endl;
            }
        } else if (is_bottom_up && (float)current_frontier_size < (float)N / beta) {
            is_bottom_up = false;
            if (verbose) {
                std::cout << " [Level " << current_depth << "] Switch -> TOP-DOWN "
                          << "(Frontier Size: " << current_frontier_size << " < N/" << beta << ")" << std::endl;
            }
        }

        // Reset next frontier count
        CUDA_CHECK(cudaMemset(d_next_frontier_count, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_next_frontier_bitmap, 0, bitmap_words * sizeof(uint32_t)));

        // 5. Execute Kernel for Current Direction
        if (!is_bottom_up) {
            // TOP-DOWN Expansion
            int td_blocks = (current_frontier_size + threads - 1) / threads;
            kernel_bfs_top_down<<<td_blocks, threads>>>(
                d_indptr, d_indices, d_frontier, current_frontier_size,
                current_depth, d_depth, d_next_frontier, d_next_frontier_count,
                d_next_frontier_bitmap
            );
        } else {
            // BOTTOM-UP Expansion
            int bu_blocks = (N + threads - 1) / threads;
            kernel_bfs_bottom_up<<<bu_blocks, threads>>>(
                N, d_indptr, d_indices, d_frontier_bitmap,
                current_depth, d_depth, d_next_frontier, d_next_frontier_count,
                d_next_frontier_bitmap
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(level_stop));
        CUDA_CHECK(cudaEventSynchronize(level_stop));

        float level_time_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&level_time_ms, level_start, level_stop));

        // Retrieve next frontier count
        CUDA_CHECK(cudaMemcpy(&current_frontier_size, d_next_frontier_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));

        stats.level_directions.push_back(is_bottom_up ? 1 : 0);
        stats.level_frontier_sizes.push_back(current_frontier_size);
        stats.level_times_ms.push_back(level_time_ms);

        // Swap Frontiers & Bitmaps
        std::swap(d_frontier, d_next_frontier);
        std::swap(d_frontier_bitmap, d_next_frontier_bitmap);

        current_depth++;
    }

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float total_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start_event, stop_event));
    stats.total_time_sec = total_time_ms / 1000.0;
    stats.max_depth = current_depth - 1;

    // 6. Compute Final Statistics (Visited Vertices & Traversed Edges)
    unsigned long long* d_visited_count = nullptr;
    unsigned long long* d_traversed_edges = nullptr;
    CUDA_CHECK(cudaMalloc(&d_visited_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_traversed_edges, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_visited_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_traversed_edges, 0, sizeof(unsigned long long)));

    int stats_blocks = (N + 256 - 1) / 256;
    kernel_compute_bfs_stats<<<stats_blocks, 256>>>(
        N, d_depth, d_indptr, d_visited_count, d_traversed_edges
    );

    unsigned long long visited_count = 0;
    unsigned long long traversed_edges = 0;
    CUDA_CHECK(cudaMemcpy(&visited_count, d_visited_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&traversed_edges, d_traversed_edges, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    stats.visited_vertices = visited_count;
    stats.traversed_edges = traversed_edges;
    stats.teps = (stats.total_time_sec > 0.0) ? (double)traversed_edges / stats.total_time_sec : 0.0;

    // Free GPU auxiliary allocations
    cudaFree(d_depth);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_count);
    cudaFree(d_next_frontier_count);
    cudaFree(d_frontier_bitmap);
    cudaFree(d_next_frontier_bitmap);
    cudaFree(d_edge_volume);
    cudaFree(d_visited_count);
    cudaFree(d_traversed_edges);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(level_start);
    cudaEventDestroy(level_stop);

    return stats;
}

// Public Single BFS Entry Point
BFSStats run_beamer_bfs_gpu(
    const GraphCSRData& graph,
    int64_t requested_root,
    float alpha,
    float beta,
    bool verbose
) {
    uint64_t N = graph.num_vertices;
    uint64_t M = graph.num_edges;

    uint64_t* d_indptr = nullptr;
    uint64_t* d_indices = nullptr;

    CUDA_CHECK(cudaMalloc(&d_indptr, (N + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_indices, M * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_indptr, graph.h_indptr.data(), (N + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, graph.h_indices.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice));

    BFSStats stats = run_beamer_bfs_gpu_internal(graph, d_indptr, d_indices, requested_root, alpha, beta, verbose);

    cudaFree(d_indptr);
    cudaFree(d_indices);

    return stats;
}

// Graph500 64-Search Benchmark Routine
Graph500BenchmarkStats run_graph500_benchmark_gpu(
    const GraphCSRData& graph,
    int num_searches,
    float alpha,
    float beta,
    uint64_t seed
) {
    Graph500BenchmarkStats bench_stats;
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

    // 2. Pre-allocate and transfer GPU Graph CSR arrays ONCE for all 64 runs
    uint64_t* d_indptr = nullptr;
    uint64_t* d_indices = nullptr;

    CUDA_CHECK(cudaMalloc(&d_indptr, (N + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_indices, M * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_indptr, graph.h_indptr.data(), (N + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, graph.h_indices.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // 3. Execute BFS on all 64 roots
    bench_stats.search_results.reserve(num_searches);
    bench_stats.teps_values.reserve(num_searches);

    std::cout << "[Graph500] Executing " << num_searches << " BFS runs on GPU..." << std::flush;

    for (int i = 0; i < num_searches; ++i) {
        BFSStats single_stats = run_beamer_bfs_gpu_internal(
            graph, d_indptr, d_indices, (int64_t)selected_roots[i], alpha, beta, false
        );
        bench_stats.search_results.push_back(single_stats);
        bench_stats.teps_values.push_back(single_stats.teps);
        if ((i + 1) % 16 == 0 || i == num_searches - 1) {
            std::cout << " " << (i + 1) << "/" << num_searches << std::flush;
        }
    }
    std::cout << " Done.\n";

    cudaFree(d_indptr);
    cudaFree(d_indices);

    // 4. Compute Graph500 Statistics
    std::vector<double> sorted_teps = bench_stats.teps_values;
    std::sort(sorted_teps.begin(), sorted_teps.end());

    int n = num_searches;
    bench_stats.min_teps = sorted_teps[0];
    bench_stats.max_teps = sorted_teps[n - 1];
    
    // Percentile calculations
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
