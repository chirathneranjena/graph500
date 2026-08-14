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
              << "      --stats           Show distance distribution breakdown histogram\n"
              << "  -h, --help            Show this help message\n";
}

int main(int argc, char** argv) {
    std::string input_file = "";
    int64_t root_vertex = -1;
    double delta = 0.1;
    bool show_stats = false;

    static struct option long_options[] = {
        {"input",   required_argument, 0, 'i'},
        {"root",    required_argument, 0, 'r'},
        {"delta",   required_argument, 0, 'd'},
        {"stats",   no_argument,       0, 's'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:r:d:sh", long_options, &option_index)) != -1) {
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
    return 0;
}
