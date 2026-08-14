#include "kronecker_generator.cuh"

#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  -s, --scale <int>          Scale S of the graph (N = 2^S vertices, default: 10)\n"
              << "  -ef, --edge-factor <int>   Average edges per vertex (M = EF * 2^S initial samples, default: 16)\n"
              << "  -a, --initiator <A B C D>  Initiator matrix probabilities (default: 0.57 0.19 0.19 0.05)\n"
              << "  -o, --output <filepath>    Output file path to save graph\n"
              << "  --format <bin|csr|edgelist> File format ('bin' default high-performance dump, 'csr' text only if specified)\n"
              << "  --seed <int>               Random seed (if omitted, picks a random seed)\n"
              << "  --scramble                 Permute vertex IDs to eliminate zero index degree bias\n"
              << "  --unweighted               Omit edge weights from output\n"
              << "  --keep-self-loops          Do not remove self-loops (u == v)\n"
              << "  --no-symmetrize            Keep directed graph without symmetrizing\n"
              << "  --allow-duplicates         Do not collapse duplicate edges\n"
              << "  --stats                    Print detailed graph statistics\n"
              << "  -h, --help                 Show this help message\n";
}

int main(int argc, char** argv) {
    int scale = 10;
    int edge_factor = 16;
    float initiator[4] = {0.57f, 0.19f, 0.19f, 0.05f};
    std::string output_filepath = "";
    std::string format = "bin";
    uint64_t seed = 0;
    bool seed_specified = false;
    bool scramble = false;
    bool weighted = true;
    bool remove_self_loops = true;
    bool symmetrize = true;
    bool collapse_duplicates = true;
    bool print_stats = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-s" || arg == "--scale") {
            if (i + 1 < argc) scale = std::atoi(argv[++i]);
        } else if (arg == "-ef" || arg == "--edge-factor") {
            if (i + 1 < argc) edge_factor = std::atoi(argv[++i]);
        } else if (arg == "-a" || arg == "--initiator") {
            if (i + 4 < argc) {
                initiator[0] = std::atof(argv[++i]);
                initiator[1] = std::atof(argv[++i]);
                initiator[2] = std::atof(argv[++i]);
                initiator[3] = std::atof(argv[++i]);
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_filepath = argv[++i];
        } else if (arg == "--format") {
            if (i + 1 < argc) format = argv[++i];
        } else if (arg == "--seed") {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
                seed_specified = true;
            }
        } else if (arg == "--scramble") {
            scramble = true;
        } else if (arg == "--unweighted") {
            weighted = false;
        } else if (arg == "--keep-self-loops") {
            remove_self_loops = false;
        } else if (arg == "--no-symmetrize") {
            symmetrize = false;
        } else if (arg == "--allow-duplicates") {
            collapse_duplicates = false;
        } else if (arg == "--stats") {
            print_stats = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!seed_specified) {
        std::random_device rd;
        uint64_t r1 = rd();
        uint64_t r2 = rd();
        uint64_t now_ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        seed = (r1 << 32) ^ r2 ^ now_ticks;
    }

    uint64_t num_vertices = 1ULL << scale;
    uint64_t initial_edges = (uint64_t)edge_factor * num_vertices;

    std::cout << "==================================================\n"
              << "   CUDA Kronecker Graph Generator (RTX 5090)      \n"
              << "==================================================\n"
              << "Scale (S):             " << scale << " (" << num_vertices << " vertices)\n"
              << "Edges per Vertex (EF): " << edge_factor << "\n"
              << "Initial Edge Samples:  " << initial_edges << "\n"
              << "Initiator Matrix:      A=" << initiator[0] << ", B=" << initiator[1]
              << ", C=" << initiator[2] << ", D=" << initiator[3] << "\n"
              << "Random Seed:           " << seed << " (" << (seed_specified ? "User specified" : "Randomly generated") << ")\n"
              << "Edge Weights:          " << (weighted ? "Enabled [0, 1)" : "Disabled") << "\n"
              << "Vertex Scrambling:     " << (scramble ? "Enabled" : "Disabled") << "\n"
              << "Remove Self-Loops:     " << (remove_self_loops ? "Yes" : "No") << "\n"
              << "Symmetrize Graph:      " << (symmetrize ? "Yes (Undirected)" : "No (Directed)") << "\n"
              << "Collapse Duplicates:   " << (collapse_duplicates ? "Yes" : "No") << "\n"
              << "==================================================\n";

    CSRGraphGPU graph = generate_kronecker_gpu(
        scale, edge_factor, initiator, seed, scramble,
        remove_self_loops, symmetrize, collapse_duplicates, weighted, print_stats
    );

    double total_sec = graph.stats.total_time_sec;
    double edges_per_sec = total_sec > 0 ? (double)graph.stats.total_edges / total_sec : 0.0;

    std::cout << "GPU Generation completed in " << std::fixed << std::setprecision(4)
              << total_sec << " seconds ("
              << std::fixed << std::setprecision(0) << edges_per_sec << " edges/sec).\n"
              << "  - Sampling Kernel Time:     " << std::setprecision(4) << graph.stats.generation_time_sec << " s\n"
              << "  - GPU Post-Processing Time: " << std::setprecision(4) << graph.stats.postprocess_time_sec << " s\n";

    if (print_stats) {
        std::cout << "\n--- Graph Statistics ---\n"
                  << "Vertices (N):          " << graph.stats.num_vertices << "\n"
                  << "Active Vertices:       " << graph.stats.active_vertices << "\n"
                  << "Final Total Edges (M): " << graph.stats.total_edges << "\n"
                  << "Unique Edges:          " << graph.stats.unique_edges << "\n"
                  << "Matrix Sparsity:       " << std::setprecision(4) << graph.stats.matrix_sparsity_pct << "% (Density: " << graph.stats.matrix_density * 100.0 << "%)\n"
                  << "Self-Loops:            " << graph.stats.self_loops << "\n"
                  << "Avg Degree:            " << std::setprecision(2) << graph.stats.avg_degree << "\n"
                  << "Max Degree:            " << graph.stats.max_degree << "\n"
                  << "Min Edge Weight:       " << std::setprecision(6) << graph.stats.min_weight << "\n"
                  << "Avg Edge Weight:       " << std::setprecision(6) << graph.stats.avg_weight << "\n"
                  << "Max Edge Weight:       " << std::setprecision(6) << graph.stats.max_weight << "\n";
    }

    if (!output_filepath.empty()) {
        bool is_csr = (format == "csr") || (output_filepath.size() >= 4 && output_filepath.substr(output_filepath.size() - 4) == ".csr");
        bool is_edgelist = (format == "edgelist") || (output_filepath.find(".edgelist") != std::string::npos);

        if (is_csr) {
            save_csr_text(graph, output_filepath, weighted);
        } else if (is_edgelist) {
            save_edgelist_text(graph, output_filepath, weighted);
        } else {
            // Default: High-performance binary dump
            save_csr_binary(graph, output_filepath, weighted);
        }
    } else if (!print_stats) {
        std::cout << "\nFirst 10 sample edges (u, v, weight):\n";
        size_t count = 0;
        for (uint64_t u = 0; u < graph.num_vertices && count < 10; ++u) {
            uint64_t start = graph.h_indptr[u];
            uint64_t end   = graph.h_indptr[u + 1];
            for (uint64_t idx = start; idx < end && count < 10; ++idx, ++count) {
                std::cout << "  " << u << " -- " << graph.h_indices[idx]
                          << " : weight=" << std::setprecision(6) << graph.h_data[idx] << "\n";
            }
        }
    }

    return 0;
}
