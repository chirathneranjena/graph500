# Graph500 Kronecker Graph Generator & Benchmark Suite

A high-performance benchmark suite implementing the **Graph500 Specifications** for synthetic **Kronecker Graph Generation**, **Breadth-First Search (BFS)** traversal, and **Single-Source Shortest Path (SSSP)** Dijkstra search.

---

## Project Structure

```
graph500/
├── design_doc.md          # Architecture & Design Specifications
├── README.md              # Project overview and getting started guide
├── python/                # Python / NumPy implementation
│   ├── kronecker_generator.py # Stochastic Kronecker (R-MAT) Graph Generator
│   ├── bfs_csr.py         # BFS Traversal & Graph500 Tree Validation
│   ├── sssp_csr.py        # Dijkstra SSSP Solver & Distance Histogram
│   ├── test_kronecker.py  # Unit tests for Graph Generator
│   ├── test_bfs.py        # Unit tests for BFS Solver
│   └── test_sssp.py       # Unit tests for SSSP Solver
└── cpp_cuda/              # (Planned) C++/CUDA GPU Acceleration
```

---

## 🐍 Python Implementation (`python/`)

The Python implementation provides a fully vectorized NumPy reference engine capable of generating millions of edges per second, exporting to Compressed Sparse Row (CSR) formats (`.csr` and `.npz`), and running BFS/SSSP traversals with Graph500 validation.

### Prerequisites
```bash
pip install numpy matplotlib
```

### Running the Python Suite

1. **Generate a Kronecker Graph**:
   ```bash
   cd python
   python3 kronecker_generator.py --scale 10 --edge-factor 16 --seed 42 --format npz --output graph_s10.npz --stats
   ```

2. **Execute Breadth-First Search (BFS)**:
   ```bash
   python3 bfs_csr.py --input graph_s10.npz --stats --validate
   ```

3. **Execute Single-Source Shortest Path (SSSP)**:
   ```bash
   python3 sssp_csr.py --input graph_s10.npz --stats --plot sssp_hist.png --validate
   ```

4. **Run All Unit Tests**:
   ```bash
   python3 -m unittest discover -p "test_*.py"
   ```

For detailed documentation on the Python implementation, see [python/README.md](file:///home/chirath/graph500/python/README.md) (or `python/` directory).

---

## ⚡ C++ / CUDA GPU Implementation (`cpp_cuda/` - Planned)

Next phase: High-throughput GPU implementation in C++/CUDA targeting massively parallel edge generation, parallel CSR construction, and GPU-accelerated BFS / SSSP graph traversals.
