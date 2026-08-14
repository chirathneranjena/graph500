#include "kronecker_generator.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <set>
#include <cstring>
#include <cstdio>
#include <iomanip>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define RADIX_BINS 256

// -------------------------------------------------------------
// Pure CUDA Device Helpers & Fused Kernels
// -------------------------------------------------------------

__device__ inline uint64_t scramble_device(uint64_t v, int scale, uint64_t seed) {
    uint64_t m = (1ULL << scale) - 1;
    uint64_t val = (v * 2654435761ULL + seed * 1013904223ULL) & 0xFFFFFFFFULL;
    return (val ^ (val >> scale)) & m;
}

// 1. Memory-Optimized Fused Kernel: Generate Edges + Filter Self-Loops + Symmetrize + Pack 64-bit Keys
__global__ void kernel_generate_and_symmetrize_kronecker(
    uint64_t num_edges_initial,
    int scale,
    float a, float ab, float abc,
    uint64_t seed,
    bool scramble,
    uint64_t* __restrict__ d_keys_out,
    double* __restrict__ d_w_out,
    uint64_t* __restrict__ d_out_count
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges_initial) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    uint64_t u = 0;
    uint64_t v = 0;

    for (int level = 0; level < scale; ++level) {
        float r = curand_uniform(&state);

        uint64_t bit_u = 1;
        uint64_t bit_v = 1;

        if (r < a) {
            bit_u = 0; bit_v = 0;
        } else if (r < ab) {
            bit_u = 0; bit_v = 1;
        } else if (r < abc) {
            bit_u = 1; bit_v = 0;
        }

        u = (u << 1) | bit_u;
        v = (v << 1) | bit_v;
    }

    if (scramble) {
        u = scramble_device(u, scale, seed);
        v = scramble_device(v, scale, seed);
    }

    bool is_valid = (u != v);

    unsigned int active = __ballot_sync(0xFFFFFFFF, is_valid);
    int lane = threadIdx.x & 31;

    int warp_item_count = __popc(active) * 2;
    int warp_item_offset = __popc(active & ((1u << lane) - 1)) * 2;

    __shared__ uint64_t smem_base;
    if (threadIdx.x == 0) smem_base = 0;
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    __shared__ uint64_t warp_bases[32];

    if (lane == 0 && warp_item_count > 0) {
        warp_bases[warp_id] = atomicAdd((unsigned long long*)&smem_base, (unsigned long long)warp_item_count);
    }
    __syncthreads();

    __shared__ uint64_t global_block_base;
    if (threadIdx.x == 0 && smem_base > 0) {
        global_block_base = atomicAdd((unsigned long long*)d_out_count, (unsigned long long)smem_base);
    }
    __syncthreads();

    if (is_valid) {
        uint64_t pos_fwd = global_block_base + warp_bases[warp_id] + warp_item_offset;
        uint64_t pos_rev = pos_fwd + 1;

        double weight = (double)curand_uniform(&state);

        uint64_t key_fwd = (scale <= 32) ? (((uint64_t)u << 32) | (uint64_t)v) : (((uint64_t)u << scale) | (uint64_t)v);
        uint64_t key_rev = (scale <= 32) ? (((uint64_t)v << 32) | (uint64_t)u) : (((uint64_t)v << scale) | (uint64_t)u);

        d_keys_out[pos_fwd] = key_fwd;
        d_w_out[pos_fwd]    = weight;

        d_keys_out[pos_rev] = key_rev;
        d_w_out[pos_rev]    = weight;
    }
}

// 2. Pure CUDA GPU Radix Sort - Count Pass
__global__ void kernel_radix_count(
    uint64_t num_elements,
    int shift,
    const uint64_t* __restrict__ d_keys_in,
    uint32_t* __restrict__ d_block_counts,
    uint32_t num_blocks
) {
    __shared__ uint32_t smem_counts[RADIX_BINS];

    int tid = threadIdx.x;
    if (tid < RADIX_BINS) smem_counts[tid] = 0;
    __syncthreads();

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    while (idx < num_elements) {
        uint64_t key = d_keys_in[idx];
        uint32_t digit = (key >> shift) & 0xFFULL;
        atomicAdd(&smem_counts[digit], 1u);
        idx += stride;
    }
    __syncthreads();

    if (tid < RADIX_BINS) {
        d_block_counts[tid * num_blocks + blockIdx.x] = smem_counts[tid];
    }
}

// 3. Pure CUDA GPU Radix Sort - Scatter Pass (Compact 2-Array Scatter)
__global__ void kernel_radix_scatter(
    uint64_t num_elements,
    int shift,
    const uint64_t* __restrict__ d_keys_in,
    const double* __restrict__ d_w_in,
    uint64_t* __restrict__ d_keys_out,
    double* __restrict__ d_w_out,
    const uint32_t* __restrict__ d_block_offsets,
    uint32_t num_blocks
) {
    __shared__ uint32_t smem_global_base[RADIX_BINS];
    __shared__ uint32_t smem_local_counts[RADIX_BINS];

    int tid = threadIdx.x;
    if (tid < RADIX_BINS) {
        smem_global_base[tid] = d_block_offsets[tid * num_blocks + blockIdx.x];
        smem_local_counts[tid] = 0;
    }
    __syncthreads();

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    uint64_t key = d_keys_in[idx];
    double w_val   = d_w_in[idx];

    uint32_t digit = (key >> shift) & 0xFFULL;

    uint32_t rank_in_block = atomicAdd(&smem_local_counts[digit], 1u);
    __syncthreads();

    uint32_t global_dest = smem_global_base[digit] + rank_in_block;

    d_keys_out[global_dest] = key;
    d_w_out[global_dest] = w_val;
}

// 4. FUSED KERNEL 2: Deduplicate (Min-Weight) + Direct CSR Indices & Degree Histogram
__global__ void kernel_fused_collapse_and_csr_histogram(
    uint64_t num_edges,
    int scale,
    const uint64_t* __restrict__ d_keys,
    const double* __restrict__ d_w_in,
    uint64_t* __restrict__ d_indices_out,
    double* __restrict__ d_w_out,
    uint64_t* __restrict__ d_indptr,
    uint64_t* __restrict__ d_out_count
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool is_unique_head = (idx < num_edges) && ((idx == 0) || (d_keys[idx] != d_keys[idx - 1]));

    double min_weight = 0.0;
    if (is_unique_head) {
        uint64_t my_key = d_keys[idx];
        min_weight = d_w_in[idx];

        uint64_t k = idx + 1;
        while (k < num_edges && d_keys[k] == my_key) {
            if (d_w_in[k] < min_weight) {
                min_weight = d_w_in[k];
            }
            k++;
        }
    }

    unsigned int active = __ballot_sync(0xFFFFFFFF, is_unique_head);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(active);
    int warp_offset = __popc(active & ((1u << lane) - 1));

    __shared__ uint64_t smem_base;
    if (threadIdx.x == 0) smem_base = 0;
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    __shared__ uint64_t warp_bases[32];

    if (lane == 0 && warp_count > 0) {
        warp_bases[warp_id] = atomicAdd((unsigned long long*)&smem_base, (unsigned long long)warp_count);
    }
    __syncthreads();

    __shared__ uint64_t global_block_base;
    if (threadIdx.x == 0 && smem_base > 0) {
        global_block_base = atomicAdd((unsigned long long*)d_out_count, (unsigned long long)smem_base);
    }
    __syncthreads();

    if (is_unique_head) {
        uint64_t write_pos = global_block_base + warp_bases[warp_id] + warp_offset;
        uint64_t key = d_keys[idx];

        uint64_t u = (scale <= 32) ? (key >> 32) : (key >> scale);
        uint64_t v = (scale <= 32) ? (key & 0xFFFFFFFFULL) : (key & ((1ULL << scale) - 1));

        d_indices_out[write_pos] = v;
        d_w_out[write_pos]       = min_weight;

        // Simultaneously increment CSR row degree histogram
        atomicAdd((unsigned long long*)&d_indptr[u + 1], 1ULL);
    }
}

// Helper: Pure CUDA GPU Radix Sort Launcher (2-Array Compact Radix Sort)
void gpu_radix_sort_64(
    uint64_t num_elements,
    uint64_t*& d_keys,
    double*& d_w
) {
    if (num_elements == 0) return;

    uint32_t block_size = 256;
    uint32_t num_blocks = (uint32_t)((num_elements + block_size - 1) / block_size);

    uint64_t *d_keys_tmp;
    double *d_w_tmp;
    uint32_t *d_block_counts, *d_block_offsets;

    CUDA_CHECK(cudaMalloc(&d_keys_tmp, num_elements * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_w_tmp, num_elements * sizeof(double)));

    size_t count_matrix_size = (size_t)RADIX_BINS * num_blocks;
    CUDA_CHECK(cudaMalloc(&d_block_counts, count_matrix_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, count_matrix_size * sizeof(uint32_t)));

    std::vector<uint32_t> h_counts(count_matrix_size);
    std::vector<uint32_t> h_offsets(count_matrix_size);

    uint64_t* p_keys_in = d_keys;
    double* p_w_in = d_w;

    uint64_t* p_keys_out = d_keys_tmp;
    double* p_w_out = d_w_tmp;

    for (int shift = 0; shift < 64; shift += 8) {
        CUDA_CHECK(cudaMemset(d_block_counts, 0, count_matrix_size * sizeof(uint32_t)));

        kernel_radix_count<<<num_blocks, block_size>>>(
            num_elements, shift, p_keys_in, d_block_counts, num_blocks
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_block_counts, count_matrix_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint32_t total = 0;
        for (int bin = 0; bin < RADIX_BINS; ++bin) {
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                size_t idx_pos = (size_t)bin * num_blocks + blk;
                h_offsets[idx_pos] = total;
                total += h_counts[idx_pos];
            }
        }

        CUDA_CHECK(cudaMemcpy(d_block_offsets, h_offsets.data(), count_matrix_size * sizeof(uint32_t), cudaMemcpyHostToDevice));

        kernel_radix_scatter<<<num_blocks, block_size>>>(
            num_elements, shift,
            p_keys_in, p_w_in,
            p_keys_out, p_w_out,
            d_block_offsets, num_blocks
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(p_keys_in, p_keys_out);
        std::swap(p_w_in, p_w_out);
    }

    if (p_keys_in != d_keys) {
        CUDA_CHECK(cudaMemcpy(d_keys, p_keys_in, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, p_w_in, num_elements * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_keys_tmp));
    CUDA_CHECK(cudaFree(d_w_tmp));
    CUDA_CHECK(cudaFree(d_block_counts));
    CUDA_CHECK(cudaFree(d_block_offsets));
}

// -------------------------------------------------------------
// Main Generator Entrypoint (Memory-Optimized Fused CUDA Pipeline)
// -------------------------------------------------------------

CSRGraphGPU generate_kronecker_gpu(
    int scale,
    int edge_factor,
    const float initiator[4],
    uint64_t seed,
    bool scramble,
    bool remove_self_loops,
    bool symmetrize,
    bool collapse_duplicates,
    bool enable_weights,
    bool compute_stats
) {
    cudaEvent_t ev_start, ev_gen, ev_post;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_gen));
    CUDA_CHECK(cudaEventCreate(&ev_post));

    CUDA_CHECK(cudaEventRecord(ev_start));

    uint64_t num_vertices = 1ULL << scale;
    uint64_t num_edges_initial = (uint64_t)edge_factor * num_vertices;

    float sum_init = initiator[0] + initiator[1] + initiator[2] + initiator[3];
    float a = initiator[0] / sum_init;
    float b = initiator[1] / sum_init;
    float c = initiator[2] / sum_init;
    float ab = a + b;
    float abc = a + b + c;

    uint32_t block_size = 256;
    uint32_t grid_size = (uint32_t)((num_edges_initial + block_size - 1) / block_size);

    std::cout << "\n[1/3] Generating & Symmetrizing Kronecker graph edges on GPU..." << std::flush;

    // Allocate buffers directly for Symmetrized & Packed Key Arrays (Max 2 * num_edges_initial)
    uint64_t max_sym_edges = num_edges_initial * 2;
    uint64_t *d_keys_sym, *d_sym_count;
    double *d_w_sym;

    CUDA_CHECK(cudaMalloc(&d_keys_sym, max_sym_edges * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_w_sym, max_sym_edges * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sym_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_sym_count, 0, sizeof(uint64_t)));

    // FUSED KERNEL 1: Sample Edges + Filter Self-Loops + Symmetrize + Pack 64-bit Keys in 1 Pass!
    kernel_generate_and_symmetrize_kronecker<<<grid_size, block_size>>>(
        num_edges_initial, scale, a, ab, abc, seed, scramble,
        d_keys_sym, d_w_sym, d_sym_count
    );
    CUDA_CHECK(cudaEventRecord(ev_gen));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << " Done.\n";

    uint64_t num_sym_edges = 0;
    CUDA_CHECK(cudaMemcpy(&num_sym_edges, d_sym_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sym_count));

    // -------------------------------------------------------------
    // FUSED POST-PROCESSING PIPELINE
    // -------------------------------------------------------------

    std::cout << "[2/3] Post-processing graph on GPU (Radix Sort, Deduplication, CSR Construction)..." << std::flush;

    // Sort Packed Keys
    gpu_radix_sort_64(num_sym_edges, d_keys_sym, d_w_sym);

    // Allocate Destination Arrays for Unique Edges + CSR indptr
    uint64_t *d_indices_final, *d_indptr, *d_final_count;
    double *d_w_final;

    CUDA_CHECK(cudaMalloc(&d_indices_final, num_sym_edges * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_w_final, num_sym_edges * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_indptr, (num_vertices + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_indptr, 0, (num_vertices + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_final_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_final_count, 0, sizeof(uint64_t)));

    // FUSED KERNEL 2: Deduplicate (Min-Weight) + Direct CSR Indices & Row Histogram in 1 Pass!
    uint32_t sort_grid = (uint32_t)((num_sym_edges + block_size - 1) / block_size);
    kernel_fused_collapse_and_csr_histogram<<<sort_grid, block_size>>>(
        num_sym_edges, scale, d_keys_sym, d_w_sym,
        d_indices_final, d_w_final, d_indptr, d_final_count
    );

    CUDA_CHECK(cudaEventRecord(ev_post));
    CUDA_CHECK(cudaEventSynchronize(ev_post));
    std::cout << " Done.\n";

    float ms_gen = 0.0f, ms_post = 0.0f, ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_gen, ev_start, ev_gen));
    CUDA_CHECK(cudaEventElapsedTime(&ms_post, ev_gen, ev_post));
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev_start, ev_post));

    uint64_t current_edges = 0;
    CUDA_CHECK(cudaMemcpy(&current_edges, d_final_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Free Input Sorted Keys BEFORE Transferring CSR to Host
    CUDA_CHECK(cudaFree(d_keys_sym));
    CUDA_CHECK(cudaFree(d_w_sym));
    CUDA_CHECK(cudaFree(d_final_count));

    std::cout << "[3/3] Transferring CSR graph data from GPU to Host memory..." << std::flush;

    // Copy CSR arrays to Host
    CSRGraphGPU result;
    result.num_vertices = num_vertices;
    result.num_edges = current_edges;
    result.h_indptr.resize(num_vertices + 1);
    result.h_indices.resize(current_edges);
    result.h_data.resize(current_edges);

    CUDA_CHECK(cudaMemcpy(result.h_indptr.data(), d_indptr, (num_vertices + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    // Compute Exclusive Prefix Sum on Host for CSR indptr
    uint64_t accum = 0;
    for (size_t i = 0; i <= num_vertices; ++i) {
        uint64_t count = result.h_indptr[i];
        result.h_indptr[i] = accum;
        accum += count;
    }

    if (current_edges > 0) {
        CUDA_CHECK(cudaMemcpy(result.h_indices.data(), d_indices_final, current_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result.h_data.data(), d_w_final, current_edges * sizeof(double), cudaMemcpyDeviceToHost));
    }

    std::cout << " Done.\n\n";

    CUDA_CHECK(cudaFree(d_indices_final));
    CUDA_CHECK(cudaFree(d_w_final));
    CUDA_CHECK(cudaFree(d_indptr));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_gen));
    CUDA_CHECK(cudaEventDestroy(ev_post));

    // Base Graph Statistics (O(1) calculation)
    double total_matrix_entries = (double)num_vertices * (double)num_vertices;
    double density = (double)current_edges / total_matrix_entries;
    double sparsity_pct = (1.0 - density) * 100.0;

    result.stats.num_vertices = num_vertices;
    result.stats.active_vertices = 0;
    result.stats.total_edges = current_edges;
    result.stats.unique_edges = current_edges;
    result.stats.self_loops = 0;
    result.stats.matrix_density = density;
    result.stats.matrix_sparsity_pct = sparsity_pct;
    result.stats.avg_degree = (double)current_edges / (double)num_vertices;
    result.stats.max_degree = 0;
    result.stats.min_weight = 0.0;
    result.stats.avg_weight = 0.0;
    result.stats.max_weight = 0.0;
    result.stats.generation_time_sec = (double)ms_gen / 1000.0;
    result.stats.postprocess_time_sec = (double)ms_post / 1000.0;
    result.stats.total_time_sec = (double)ms_total / 1000.0;

    // Detailed Host Statistics (Only if compute_stats is requested)
    if (compute_stats && current_edges > 0) {
        std::set<uint64_t> active_set;
        std::vector<uint64_t> degree_count(num_vertices, 0);

        double min_w = result.h_data[0];
        double max_w = result.h_data[0];
        double sum_w = 0.0;

        for (size_t i = 0; i < num_vertices; ++i) {
            uint64_t deg = result.h_indptr[i + 1] - result.h_indptr[i];
            degree_count[i] = deg;
            if (deg > 0) active_set.insert(i);
        }

        uint64_t max_deg = 0;
        for (size_t i = 0; i < current_edges; ++i) {
            uint64_t v_idx = result.h_indices[i];
            active_set.insert(v_idx);
            double w_val = result.h_data[i];
            if (w_val < min_w) min_w = w_val;
            if (w_val > max_w) max_w = w_val;
            sum_w += w_val;
        }

        for (uint64_t d_val : degree_count) {
            if (d_val > max_deg) max_deg = d_val;
        }

        result.stats.active_vertices = active_set.size();
        result.stats.max_degree = max_deg;
        result.stats.min_weight = min_w;
        result.stats.avg_weight = sum_w / current_edges;
        result.stats.max_weight = max_w;
    }

    return result;
}

// -------------------------------------------------------------
// I/O Helper Functions
// -------------------------------------------------------------

void save_csr_text(const CSRGraphGPU& graph, const std::string& filepath, bool weighted) {
    std::ofstream fout(filepath);
    if (!fout.is_open()) {
        std::cerr << "Error opening file for writing: " << filepath << std::endl;
        return;
    }

    fout << graph.num_vertices << " " << graph.num_edges << "\n";

    for (size_t i = 0; i <= graph.num_vertices; ++i) {
        fout << graph.h_indptr[i] << (i == graph.num_vertices ? "" : " ");
    }
    fout << "\n";

    for (size_t i = 0; i < graph.num_edges; ++i) {
        fout << graph.h_indices[i] << (i + 1 == graph.num_edges ? "" : " ");
    }
    fout << "\n";

    if (weighted) {
        fout << std::fixed;
        fout.precision(6);
        for (size_t i = 0; i < graph.num_edges; ++i) {
            fout << graph.h_data[i] << (i + 1 == graph.num_edges ? "" : " ");
        }
        fout << "\n";
    }
}

void save_csr_binary(const CSRGraphGPU& graph, const std::string& filepath, bool weighted) {
    auto t_start = std::chrono::high_resolution_clock::now();
    FILE* fp = fopen(filepath.c_str(), "wb");
    if (!fp) {
        std::cerr << "Error opening binary file for writing: " << filepath << std::endl;
        return;
    }

    CSRBinaryHeader header;
    std::memcpy(header.magic, "GRAPH500", 8);
    header.num_vertices = graph.num_vertices;
    header.num_edges = graph.num_edges;
    header.weighted = weighted ? 1 : 0;
    header.reserved = 0;

    fwrite(&header, sizeof(header), 1, fp);
    fwrite(graph.h_indptr.data(), sizeof(uint64_t), graph.num_vertices + 1, fp);
    fwrite(graph.h_indices.data(), sizeof(uint64_t), graph.num_edges, fp);
    if (weighted) {
        fwrite(graph.h_data.data(), sizeof(double), graph.num_edges, fp);
    }
    fclose(fp);

    auto t_end = std::chrono::high_resolution_clock::now();
    double io_sec = std::chrono::duration<double>(t_end - t_start).count();
    uint64_t total_bytes = sizeof(header) + (graph.num_vertices + 1) * sizeof(uint64_t) + graph.num_edges * sizeof(uint64_t) + (weighted ? graph.num_edges * sizeof(double) : 0);
    double mb_s = io_sec > 0 ? (total_bytes / (1024.0 * 1024.0)) / io_sec : 0.0;

    std::cout << "Saved binary CSR dump to '" << filepath << "' ("
              << std::fixed << std::setprecision(2) << (total_bytes / (1024.0 * 1024.0)) << " MB in "
              << std::setprecision(4) << io_sec << " s, " << std::setprecision(1) << mb_s << " MB/s).\n";
}

bool load_csr_binary(const std::string& filepath, CSRGraphGPU& graph) {
    auto t_start = std::chrono::high_resolution_clock::now();
    FILE* fp = fopen(filepath.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error opening binary file for reading: " << filepath << std::endl;
        return false;
    }

    CSRBinaryHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        std::cerr << "Error reading binary header from: " << filepath << std::endl;
        fclose(fp);
        return false;
    }

    if (std::memcmp(header.magic, "GRAPH500", 8) != 0) {
        std::cerr << "Invalid binary header magic in: " << filepath << std::endl;
        fclose(fp);
        return false;
    }

    graph.num_vertices = header.num_vertices;
    graph.num_edges = header.num_edges;
    graph.h_indptr.resize(header.num_vertices + 1);
    graph.h_indices.resize(header.num_edges);

    if (fread(graph.h_indptr.data(), sizeof(uint64_t), header.num_vertices + 1, fp) != header.num_vertices + 1) {
        std::cerr << "Error reading indptr array from: " << filepath << std::endl;
        fclose(fp);
        return false;
    }

    if (fread(graph.h_indices.data(), sizeof(uint64_t), header.num_edges, fp) != header.num_edges) {
        std::cerr << "Error reading indices array from: " << filepath << std::endl;
        fclose(fp);
        return false;
    }

    if (header.weighted) {
        graph.h_data.resize(header.num_edges);
        if (fread(graph.h_data.data(), sizeof(double), header.num_edges, fp) != header.num_edges) {
            std::cerr << "Error reading data array from: " << filepath << std::endl;
            fclose(fp);
            return false;
        }
    } else {
        graph.h_data.assign(header.num_edges, 1.0);
    }

    fclose(fp);

    auto t_end = std::chrono::high_resolution_clock::now();
    double io_sec = std::chrono::duration<double>(t_end - t_start).count();
    uint64_t total_bytes = sizeof(header) + (header.num_vertices + 1) * sizeof(uint64_t) + header.num_edges * sizeof(uint64_t) + (header.weighted ? header.num_edges * sizeof(double) : 0);
    double mb_s = io_sec > 0 ? (total_bytes / (1024.0 * 1024.0)) / io_sec : 0.0;

    std::cout << "Loaded binary CSR dump from '" << filepath << "' ("
              << std::fixed << std::setprecision(2) << (total_bytes / (1024.0 * 1024.0)) << " MB in "
              << std::setprecision(4) << io_sec << " s, " << std::setprecision(1) << mb_s << " MB/s).\n";

    return true;
}

void save_edgelist_text(const CSRGraphGPU& graph, const std::string& filepath, bool weighted) {
    std::ofstream fout(filepath);
    if (!fout.is_open()) {
        std::cerr << "Error opening file for writing: " << filepath << std::endl;
        return;
    }

    fout << std::fixed;
    fout.precision(6);

    for (uint64_t u = 0; u < graph.num_vertices; ++u) {
        uint64_t start = graph.h_indptr[u];
        uint64_t end   = graph.h_indptr[u + 1];
        for (uint64_t idx = start; idx < end; ++idx) {
            uint64_t v = graph.h_indices[idx];
            double w   = graph.h_data[idx];
            if (weighted) {
                fout << u << "\t" << v << "\t" << w << "\n";
            } else {
                fout << u << "\t" << v << "\n";
            }
        }
    }
}
