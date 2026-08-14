#include "sssp_delta.cuh"
#include <iostream>
#include <iomanip>
#include <string>
#include <getopt.h>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " --input <file.bin> [options]\n\n"
              << "Options:\n"
              << "  -i, --input <file>    Path to Graph500 binary CSR file (.bin) [Required]\n"
              << "  -r, --root <int>      Root vertex ID (default: auto-select highest degree vertex)\n"
              << "  -d, --delta <float>   Delta bucket width for relaxation (default: 0.1)\n"
              << "      --graph500        Run official Graph500 SSSP benchmark on 64 random valid roots\n"
              << "      --stats           Show per-root TEPS breakdown in --graph500 mode, or distance histogram in single-root mode\n"
              << "  -h, --help            Show this help message\n";
}

int main(int argc, char** argv) {
    std::string input_file = "";
    int64_t root_vertex = -1;
    double delta = 0.1;
    bool show_stats = false;
    bool run_graph500 = false;

    static struct option long_options[] = {
        {"input",    required_argument, 0, 'i'},
        {"root",     required_argument, 0, 'r'},
        {"delta",    required_argument, 0, 'd'},
        {"graph500", no_argument,       0, 'g'},
        {"stats",    no_argument,       0, 's'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:r:d:gsh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'r':
                root_vertex = std::stoll(optarg);
                break;
            case 'd':
                delta = std::stod(optarg);
                break;
            case 'g':
                run_graph500 = true;
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
    std::cout << "  GPU Delta-Stepping Single-Source Shortest Path\n";
    std::cout << "====================================================\n";
    std::cout << "Loading graph binary file: " << input_file << " ...\n";

    GraphCSRData graph;
    if (!load_graph_binary(input_file, graph)) {
        return 1;
    }

    std::cout << "Graph loaded successfully!\n"
              << "  Vertices (N): " << graph.num_vertices << "\n"
              << "  Edges (M):    " << graph.num_edges << "\n"
              << "  Weighted:     " << (graph.weighted ? "Yes" : "No (Default unit weight 1.0)") << "\n";

    if (run_graph500) {
        std::cout << "\nRunning Graph500 SSSP Benchmark (64 Random Starting Vertices with degree >= 1)...\n";
        Graph500SSSPBenchmarkStats bench = run_graph500_sssp_benchmark_gpu(graph, 64, delta);

        std::cout << "\n====================================================\n";
        std::cout << "        Official Graph500 SSSP Benchmark Stats      \n";
        std::cout << "====================================================\n";
        std::cout << "Number of SSSP Searches: " << bench.num_searches << "\n\n";

        if (show_stats && !bench.search_results.empty()) {
            std::cout << "Per-Search Starting Vertices & TEPS Breakdown:\n";
            std::cout << std::left << std::setw(8) << "Search"
                      << std::setw(14) << "Root Vertex"
                      << std::setw(12) << "Degree"
                      << std::setw(18) << "Reachable Verts"
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
                          << std::setw(18) << res.reachable_vertices
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
        std::cout << "Primary Graph500 SSSP Metric (Harmonic Mean): " 
                  << std::fixed << std::setprecision(3) << (bench.harmonic_mean_teps / 1e9) << " GTEPS\n";
        std::cout << "====================================================\n";
    } else {
        std::cout << "Running GPU Delta-Stepping SSSP (Delta = " << delta << ")...\n";

        SSSPStats stats = run_delta_stepping_sssp_gpu(graph, root_vertex, delta, show_stats);

        std::cout << "\n====================================================\n";
        std::cout << "               SSSP Traversal Results               \n";
        std::cout << "====================================================\n";
        std::cout << "Root Vertex:         " << stats.root_vertex;
        if (root_vertex < 0) {
            std::cout << " [Auto-selected High-Degree Root]";
        }
        std::cout << "\nRoot Vertex Degree:  " << stats.root_degree << "\n"
                  << "Reachable Vertices:  " << stats.reachable_vertices << " / " << graph.num_vertices 
                  << " (" << std::fixed << std::setprecision(2) << (100.0 * stats.reachable_vertices / graph.num_vertices) << "%)\n"
                  << "Traversed Edges:     " << stats.traversed_edges << " / " << graph.num_edges << "\n"
                  << "Bucket Step Iterations: " << stats.total_bucket_steps << "\n"
                  << "Minimum Distance:    " << std::fixed << std::setprecision(4) << stats.min_distance << "\n"
                  << "Average Distance:    " << std::fixed << std::setprecision(4) << stats.avg_distance << "\n"
                  << "Maximum Distance:    " << std::fixed << std::setprecision(4) << stats.max_distance << "\n"
                  << "SSSP Execution Time: " << std::setprecision(4) << stats.total_time_sec << " seconds\n"
                  << "Performance (TEPS):  " << std::scientific << std::setprecision(3) << stats.teps 
                  << " (" << std::fixed << std::setprecision(2) << (stats.teps / 1e9) << " GTEPS)\n";

        if (show_stats && !stats.distance_bins.empty()) {
            std::cout << "\n---------------- Distance Range Breakdown ----------------\n";
            double bin_width = stats.max_distance / 10.0;
            std::cout << std::left << std::setw(8) << "Bin" 
                      << std::setw(24) << "Distance Range" 
                      << std::setw(15) << "Vertex Count" << "\n";
            std::cout << "---------------------------------------------------------\n";
            for (size_t b = 0; b < stats.distance_bins.size(); ++b) {
                double low = b * bin_width;
                double high = (b + 1) * bin_width;
                std::string range_str = "[" + std::to_string(low).substr(0, 6) + ", " + std::to_string(high).substr(0, 6) + ")";
                std::cout << std::left << std::setw(8) << b 
                          << std::setw(24) << range_str
                          << std::setw(15) << stats.distance_bins[b] << "\n";
            }
        }

        std::cout << "====================================================\n";
    }

    return 0;
}
