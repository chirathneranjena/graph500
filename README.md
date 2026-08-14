# Graph500 Kronecker Graph Generator, BFS & SSSP Benchmark Suite

A high-performance implementation of the **Graph500 Specifications** for synthetic **Stochastic Kronecker (R-MAT) Graph Generation**, **Direction-Optimized Breadth-First Search (BFS)**, and **Delta-Stepping Single-Source Shortest Path (SSSP)** graph traversals.

This repository features both a **100% Pure CUDA C++ GPU Engine** (`cpp_cuda/`) optimized for extreme throughput on NVIDIA GPUs, and a **Vectorized Python / NumPy Reference Suite** (`python/`).

---

## 📁 Repository Structure

```text
graph500/
├── README.md                  # Main repository overview and getting started guide
├── cpp_cuda/                  # 100% Pure CUDA C++ High-Performance GPU Engine
│   ├── kronecker_generator.cu # CUDA Kronecker Generator with Fused Post-Processing
│   ├── kronecker_generator.cuh# CUDA Generator headers & CSR binary dump structs
│   ├── bfs_beamer.cu          # Direction-Optimized Beamer BFS & Graph500 Benchmark
│   ├── bfs_beamer.cuh         # CUDA BFS headers & stats data structures
│   ├── sssp_delta.cu          # GPU Delta-Stepping Single-Source Shortest Path (SSSP)
│   ├── sssp_delta.cuh         # CUDA SSSP headers & distance stats structs
│   ├── main.cu                # CLI entry point for kronecker_cuda
│   ├── main_bfs.cu            # CLI entry point for bfs_beamer
│   ├── main_sssp.cu           # CLI entry point for sssp_delta
│   ├── Makefile               # GPU build script
│   └── README.md              # Detailed CUDA implementation documentation
└── python/                    # Vectorized Python / NumPy Reference Implementation
    ├── kronecker_generator.py # Vectorized Stochastic Kronecker Generator
    ├── bfs_csr.py             # CSR BFS Solver with Graph500 Tree Validation
    ├── sssp_csr.py            # Single-Source Shortest Path (Dijkstra) Solver
    ├── design_doc.md          # Reference Architecture Specifications
    └── README.md              # Python reference documentation
```

---

## ⚡ 1. Pure CUDA GPU Engine (`cpp_cuda/`)

Built with **zero external or helper library dependencies** (no Thrust, no CUB), a **Fused 2-Stage CUDA Post-Processing Pipeline**, real-time **console stage progress indicators**, and sub-microsecond **GPU event timers (`cudaEvent_t`)**.

### Key CUDA Components

1. **Pure CUDA Graph Generator (`kronecker_cuda`)**:
   - Philox PRNG counter-based stochastic sampling kernel.
   - Fused Kernel 1: Self-loop removal, symmetrization, and 64-bit key packing in a single VRAM write pass.
   - Custom 64-bit GPU Radix Sort using shared memory histograms and scatter kernels.
   - Fused Kernel 2: Duplicate edge collapsing and CSR row histogram accumulation in a single pass.
   - High-throughput binary dump writer (>1.2 GB/s stream speed).

2. **GPU Beamer BFS Solver (`bfs_beamer`)**:
   - Streams Graph500 binary CSR dumps (`.bin`) directly into GPU VRAM.
   - Hybrid **Top-Down / Bottom-Up Beamer Algorithm** with dynamic direction switching ($\alpha = 14.0$, $\beta = 24.0$).
   - Automatic high-degree root auto-selection when `--root` is omitted.
   - **Official Graph500 Benchmark (`--graph500`)**: Randomly samples 64 valid roots ($\text{degree} \ge 1$), runs optimized BFS traversals across all 64 roots on GPU, and outputs Min, Q1, Median, Q3, Max, Mean, StdDev, and Harmonic Mean TEPS.

3. **GPU Delta-Stepping SSSP Solver (`sssp_delta`)**:
   - Executes lock-free double-precision 64-bit atomic distance updates (`atomicMinDouble`) on GPU.
   - Bucketed Delta-Stepping algorithm with light edge ($\le \Delta$) inner loop and heavy edge ($> \Delta$) outer loop.
   - Automatic high-degree root selection when `--root` is omitted.
   - Distance range breakdown histogram when `--stats` is specified.

### Quick Start (CUDA Engine)

```bash
cd cpp_cuda

# Build CUDA binaries
make clean && make -j$(nproc)

# 1. Generate a Scale 18 Graph (262,144 vertices, ~8.3M edges)
./kronecker_cuda --scale 18 --edge-factor 16 -o graph_s18.bin

# 2. Run Single-Root GPU Beamer BFS
./bfs_beamer --input graph_s18.bin --stats

# 3. Run Official Graph500 64-Search Benchmark
./bfs_beamer --input graph_s18.bin --graph500 --stats

# 4. Run GPU Delta-Stepping SSSP
./sssp_delta --input graph_s18.bin --delta 0.1 --stats
```

---

## 🐍 2. Python Reference Suite (`python/`)

The Python implementation provides a fully vectorized NumPy reference engine capable of generating synthetic scale-free graphs, exporting to Compressed Sparse Row (CSR) formats (`.csr` and `.npz`), and running BFS/SSSP traversals with Graph500 tree validation.

### Quick Start (Python)

```bash
cd python

# Install dependencies
pip install numpy matplotlib

# 1. Generate Kronecker Graph
python3 kronecker_generator.py --scale 10 --edge-factor 16 --format npz --output graph_s10.npz --stats

# 2. Run BFS Traversal & Tree Validation
python3 bfs_csr.py --input graph_s10.npz --stats --validate

# 3. Run Single-Source Shortest Path (SSSP)
python3 sssp_csr.py --input graph_s10.npz --stats --plot sssp_hist.png --validate
```

---

## 🚀 GPU Benchmark Highlights (NVIDIA GeForce RTX 5090)

| Scale ($S$) | Vertices ($N$) | Total Edges ($M$) | GPU Generator Time | Generation Throughput | BFS Traversal Speed | SSSP Execution Time |
|---|---|---|---|---|---|---|
| **Scale 16** | $65,536$ | $2,079,468$ | $0.0355$ s | $58.62$M edges/sec | **$1.26$ GTEPS** | **$0.0712$ s** |
| **Scale 18** | $262,144$ | $8,332,682$ | $0.1317$ s | $63.26$M edges/sec | **$2.97$ GTEPS** | **$0.3744$ s** |
| **Scale 20** | $1,048,576$ | $33,383,025$ | $0.5080$ s | $65.71$M edges/sec | **$4.77$ GTEPS** | **$1.4281$ s** |

---

## 📄 Documentation

- [C++/CUDA Engine Documentation](file:///home/chirath/graph500/cpp_cuda/README.md) (`cpp_cuda/README.md`)
- [Python Suite Documentation](file:///home/chirath/graph500/python/README.md) (`python/README.md`)
