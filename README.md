# Graph500 Kronecker Graph Generator, BFS & SSSP Benchmark Suite

A high-performance implementation of the **Graph500 Specifications** for synthetic **Stochastic Kronecker (R-MAT) Graph Generation**, **Direction-Optimized Breadth-First Search (BFS)**, and **Delta-Stepping Single-Source Shortest Path (SSSP)** graph traversals.

This repository features both a **100% Pure CUDA C++ GPU Engine** (`cpp_cuda/`) optimized for extreme throughput and memory efficiency on NVIDIA GPUs, and a **Vectorized Python / NumPy Reference Suite** (`python/`).

---

## 📁 Repository Structure

```text
graph500/
├── README.md                  # Main repository overview and getting started guide
├── cpp_cuda/                  # 100% Pure CUDA C++ High-Performance GPU Engine
│   ├── kronecker_generator.cu # CUDA Kronecker Generator (Memory-Optimized Fused Pipeline)
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

## ⚡ 1. Memory-Optimized Pure CUDA GPU Engine (`cpp_cuda/`)

Built with **zero external or helper library dependencies** (no Thrust, no CUB), a **Fused CUDA Post-Processing Pipeline**, real-time **console stage progress indicators**, and sub-microsecond **GPU event timers (`cudaEvent_t`)**.

### Key Memory & Performance Features

1. **Memory-Optimized CUDA Generator (`kronecker_cuda`)**:
   - Direct sampling & symmetrization kernel eliminates intermediate raw edge arrays.
   - Compact 64-bit key packing `key = (u << 32) | v` eliminates separate source/destination arrays, saving $34.36\text{ GB}$ VRAM during sorting.
   - Halved Radix Sort memory overhead (sorts 2 arrays instead of 4).
   - Direct CSR index construction and degree histogram accumulation in a single pass.
   - **Scale 26 Support**: Generates $67.1\text{ Million}$ vertices and $2.14\text{ Billion}$ edges in **$68.72\text{ GB}$ VRAM** (down from $137.44\text{ GB}$), allowing Scale 26 to run cleanly on an $80\text{GB}$ GPU!

2. **GPU Beamer BFS Solver (`bfs_beamer`)**:
   - Streams Graph500 binary CSR dumps (`.bin`) directly into GPU VRAM.
   - Hybrid **Top-Down / Bottom-Up Beamer Algorithm** with dynamic direction switching ($\alpha = 14.0$, $\beta = 24.0$).
   - Automatic high-degree root auto-selection when `--root` is omitted.
   - **Official Graph500 Benchmark (`--graph500`)**: Randomly samples 64 valid roots ($\text{degree} \ge 1$), runs optimized BFS traversals across all 64 roots on GPU, and outputs Min, Q1, Median, Q3, Max, Mean, StdDev, and Harmonic Mean TEPS.

3. **GPU Delta-Stepping SSSP Solver (`sssp_delta`)**:
   - Executes lock-free double-precision 64-bit atomic distance updates (`atomicMinDouble`) on GPU.
   - Bucketed Delta-Stepping algorithm with active frontier compaction and fast pre-atomic distance checking.
   - Automatic high-degree root selection when `--root` is omitted.
   - **Official Graph500 SSSP Benchmark (`--graph500`)**: Randomly samples 64 valid roots ($\text{degree} \ge 1$), runs optimized SSSP across all 64 roots with pre-allocated GPU CSR memory, and outputs Min, Q1, Median, Q3, Max, Mean, StdDev, and Harmonic Mean TEPS.

### Quick Start (CUDA Engine)

```bash
cd cpp_cuda

# Build CUDA binaries
make clean && make -j$(nproc)

# 1. Generate a Scale 20 Graph (1,048,576 vertices, ~33.4M edges)
./kronecker_cuda --scale 20 --edge-factor 16 -o graph_s20.bin

# 2. Run Single-Root GPU Beamer BFS
./bfs_beamer --input graph_s20.bin --stats

# 3. Run Official Graph500 64-Search BFS Benchmark
./bfs_beamer --input graph_s20.bin --graph500 --stats

# 4. Run Single-Root GPU Delta-Stepping SSSP
./sssp_delta --input graph_s20.bin --delta 0.1 --stats

# 5. Run Official Graph500 64-Search SSSP Benchmark
./sssp_delta --input graph_s20.bin --graph500 --stats
```

---

## 🚀 Peak GPU Memory Footprint Across Graph Scales

| Scale ($S$) | Vertices ($N$) | Initial Edges ($M_{raw}$) | Symmetrized Edges ($M_{sym}$) | Old Peak VRAM | **New Peak VRAM** | Fits in 80GB GPU? |
|---|---|---|---|---|---|---|
| **Scale 20** | $1,048,576$ | $16.78$ M | $33.55$ M | $2.14$ GB | **$1.07$ GB** | Yes |
| **Scale 22** | $4,194,304$ | $67.11$ M | $134.22$ M | $8.58$ GB | **$4.29$ GB** | Yes |
| **Scale 24** | $16,777,216$ | $268.44$ M | $536.87$ M | $34.36$ GB | **$17.18$ GB** | Yes |
| **Scale 25** | $33,554,432$ | $536.87$ M | $1,073.74$ M | $68.72$ GB | **$34.36$ GB** | Yes |
| **Scale 26** | $67,108,864$ | $1,073.74$ M | $2,147.48$ M | $137.44$ GB | **$68.72$ GB** | **YES!** |

---

## 📄 Documentation

- [C++/CUDA Engine Documentation](file:///home/chirath/graph500/cpp_cuda/README.md) (`cpp_cuda/README.md`)
- [Python Suite Documentation](file:///home/chirath/graph500/python/README.md) (`python/README.md`)
