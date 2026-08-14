#include "bfs_beamer.cuh"
#include <iostream>
#include <iomanip>
#include <string>
#include <getopt.h>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " --input <file.bin> [options]\n\n"
              << "Options:\n"
              << "  -i, --input <file>    Path to Graph500 binary CSR file (.bin) [Required]\n"
              << "  -r, --root <int>      Root vertex ID (default: auto-select highest degree vertex)\n"
              << "      --graph500        Run official Graph500 benchmark on 64 random valid roots\n"
              << "      --alpha <float>   Top-Down -> Bottom-Up switch threshold (default: 14.0)\n"
              << "      --beta <float>    Bottom-Up -> Top-Down switch threshold (default: 24.0)\n"
              << "      --stats           Show detailed level-by-level traversal statistics\n"
              << "  -h, --help            Show this help message\n";
}

int main(int argc, char** argv) {
    std::string input_file = "";
    int64_t root_vertex = -1;
    float alpha = 14.0f;
    float beta = 24.0f;
    bool show_stats = false;
    bool run_graph500 = false;

    static struct option long_options[] = {
        {"input",    required_argument, 0, 'i'},
        {"root",     required_argument, 0, 'r'},
        {"graph500", no_argument,       0, 'g'},
        {"alpha",    required_argument, 0, 'a'},
        {"beta",     required_argument, 0, 'b'},
        {"stats",    no_argument,       0, 's'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:r:gsh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'r':
                root_vertex = std::stoll(optarg);
                break;
            case 'g':
                run_graph500 = true;
                break;
            case 'a':
                alpha = std::stof(optarg);
                break;
            case 'b':
                beta = std::stof(optarg);
                break;
            case 's':
                show_stats = true;
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return 0;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: --input <file.bin> parameter is required.\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "====================================================\n";
    std::cout << "  GPU Beamer BFS (Direction-Optimized Traversal)\n";
    std::cout << "====================================================\n";
    std::cout << "Loading graph binary file: " << input_file << " ...\n";

    GraphCSRData graph;
    if (!load_graph_binary(input_file, graph)) {
        return 1;
    }

    std::cout << "Graph loaded successfully!\n"
              << "  Vertices (N): " << graph.num_vertices << "\n"
              << "  Edges (M):    " << graph.num_edges << "\n";

    if (run_graph500) {
        std::cout << "\nRunning Graph500 Benchmark (64 Random Starting Vertices with degree >= 1)...\n";
        Graph500BenchmarkStats bench = run_graph500_benchmark_gpu(graph, 64, alpha, beta);

        std::cout << "\n====================================================\n";
        std::cout << "          Official Graph500 Benchmark Stats         \n";
        std::cout << "====================================================\n";
        std::cout << "Number of BFS Searches: " << bench.num_searches << "\n\n";

        if (show_stats && !bench.search_results.empty()) {
            std::cout << "Per-Search Starting Vertices & TEPS Breakdown:\n";
            std::cout << std::left << std::setw(8) << "Search"
                      << std::setw(14) << "Root Vertex"
                      << std::setw(12) << "Degree"
                      << std::setw(16) << "Visited Verts"
                      << std::setw(18) << "Traversed Edges"
                      << std::setw(12) << "Time (ms)"
                      << std::setw(16) << "TEPS (raw)"
                      << std::setw(12) << "GTEPS" << "\n";
            std::cout << "----------------------------------------------------------------------------------------------------\n";
            for (int i = 0; i < bench.num_searches; ++i) {
                const auto& res = bench.search_results[i];
                std::cout << std::left << std::setw(8) << i
                          << std::setw(14) << bench.selected_roots[i]
                          << std::setw(12) << res.root_degree
                          << std::setw(16) << res.visited_vertices
                          << std::setw(18) << res.traversed_edges
                          << std::fixed << std::setprecision(2) << std::setw(12) << (res.total_time_sec * 1000.0)
                          << std::scientific << std::setprecision(3) << std::setw(16) << res.teps
                          << std::fixed << std::setprecision(3) << std::setw(12) << (res.teps / 1e9) << "\n";
            }
            std::cout << "----------------------------------------------------------------------------------------------------\n\n";
        }

        std::cout << std::left << std::setw(30) << "Metric" 
                  << std::setw(18) << "TEPS (raw)" 
                  << std::setw(15) << "GTEPS" << "\n";
        std::cout << "----------------------------------------------------\n";

        auto print_metric = [](const std::string& name, double teps) {
            std::cout << std::left << std::setw(30) << name
                      << std::scientific << std::setprecision(3) << std::setw(18) << teps
                      << std::fixed << std::setprecision(3) << (teps / 1e9) << " GTEPS\n";
        };

        print_metric("Minimum TEPS", bench.min_teps);
        print_metric("First Quartile TEPS", bench.q1_teps);
        print_metric("Median TEPS", bench.median_teps);
        print_metric("Third Quartile TEPS", bench.q3_teps);
        print_metric("Maximum TEPS", bench.max_teps);
        print_metric("Arithmetic Mean TEPS", bench.mean_teps);
        print_metric("Standard Deviation of TEPS", bench.stddev_teps);
        print_metric("Harmonic Mean TEPS", bench.harmonic_mean_teps);

        std::cout << "====================================================\n";
        std::cout << "Primary Graph500 Metric (Harmonic Mean): " 
                  << std::fixed << std::setprecision(3) << (bench.harmonic_mean_teps / 1e9) << " GTEPS\n";
        std::cout << "====================================================\n";
    } else {
        std::cout << "Running Single GPU Beamer BFS...\n";

        BFSStats stats = run_beamer_bfs_gpu(graph, root_vertex, alpha, beta, show_stats);

        std::cout << "\n====================================================\n";
        std::cout << "               BFS Traversal Results                \n";
        std::cout << "====================================================\n";
        std::cout << "Root Vertex:         " << stats.root_vertex;
        if (root_vertex < 0) {
            std::cout << " [Auto-selected High-Degree Root]";
        }
        std::cout << "\nRoot Vertex Degree:  " << stats.root_degree << "\n"
                  << "Visited Vertices:    " << stats.visited_vertices << " / " << graph.num_vertices 
                  << " (" << std::fixed << std::setprecision(2) << (100.0 * stats.visited_vertices / graph.num_vertices) << "%)\n"
                  << "Traversed Edges:     " << stats.traversed_edges << " / " << graph.num_edges << "\n"
                  << "Max Depth (Levels):  " << stats.max_depth << "\n"
                  << "BFS Execution Time:  " << std::setprecision(4) << stats.total_time_sec << " seconds\n"
                  << "Performance (TEPS):  " << std::scientific << std::setprecision(3) << stats.teps 
                  << " (" << std::fixed << std::setprecision(2) << (stats.teps / 1e9) << " GTEPS)\n";

        if (show_stats && !stats.level_directions.empty()) {
            std::cout << "\n---------------- Level Breakdown ----------------\n";
            std::cout << std::left << std::setw(8) << "Level" 
                      << std::setw(15) << "Direction" 
                      << std::setw(18) << "Frontier Size" 
                      << std::setw(15) << "Time (ms)" << "\n";
            std::cout << "-------------------------------------------------\n";
            for (size_t l = 0; l < stats.level_directions.size(); ++l) {
                std::cout << std::left << std::setw(8) << l 
                          << std::setw(15) << (stats.level_directions[l] == 1 ? "Bottom-Up" : "Top-Down")
                          << std::setw(18) << stats.level_frontier_sizes[l]
                          << std::setw(15) << std::fixed << std::setprecision(3) << stats.level_times_ms[l] << "\n";
            }
        }

        std::cout << "====================================================\n";
    }

    return 0;
}
