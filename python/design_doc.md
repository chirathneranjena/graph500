# System Design Document: Graph500 Kronecker Graph Generator & Benchmark Suite

## 1. System Overview & Objectives

This system provides a high-performance Python framework for generating, storing, analyzing, and benchmarking **Kronecker / Stochastic Kronecker (R-MAT) graphs** in accordance with the **Graph500 Benchmark Specification**.

The suite comprises three core components:
1. **Kronecker Graph Generator** (`kronecker_generator.py`): Synthetic power-law graph generator with automated post-processing and CSR export.
2. **Breadth-First Search (BFS) Solver** (`bfs_csr.py`): Level-synchronous unweighted graph traversal and validation.
3. **Single-Source Shortest Path (SSSP) Solver** (`sssp_csr.py`): Weighted shortest-path Dijkstra solver and validation.

---

## 2. High-Level Architecture

```
                               +-----------------------------+
                               |     Initiator Matrix        |
                               |    [A=0.57, B=0.19, ...]    |
                               +--------------+--------------+
                                              |
                                              v
+------------------------+     +-----------------------------+     +------------------------+
| Scale (S), Edge Factor | --> | kronecker_generator.py      | --> | Output File (.npz/.csr)|
+------------------------+     | - Vectorized R-MAT Sampling |     +-----------+------------+
                               | - Symmetrize & Deduplicate  |                 |
                               +-----------------------------+                 |
                                                                               v
                               +-----------------------------+     +-----------+------------+
                               | bfs_csr.py (BFS Solver)     | <-- | Compressed Sparse Row  |
                               | - Traversed Edges / TEPS    |     | CSR Arrays:            |
                               | - Graph500 Tree Validation  |     | - indptr  (N + 1)      |
                               +-----------------------------+     | - indices (M)          |
                                                                   | - data    (M)          |
                               +-----------------------------+     +-----------+------------+
                               | sssp_csr.py (SSSP Solver)   |                 |
                               | - Dijkstra Min-Heap Search  | <---------------+
                               | - Distance Quantiles & Hist |
                               +-----------------------------+
```

---

## 3. Detailed Component Architecture

### 3.1. Kronecker Graph Generator (`kronecker_generator.py`)

#### Algorithm & Mathematical Foundation
Kronecker graph generation is based on recursive tensor powers of an initiator matrix $K_1$:
$$K_1 = \begin{bmatrix} A & B \\ C & D \end{bmatrix}, \quad A + B + C + D = 1.0$$

The $S$-th Kronecker power $K_S = \bigotimes_{i=1}^S K_1$ produces a $2^S \times 2^S$ probability matrix.

For large graphs, storing $2^S \times 2^S$ probabilities is intractable. The generator implements **Stochastic Kronecker / R-MAT sampling**:
- For each edge, $S$ coin flips select quadrant $Q \in \{(0,0), (0,1), (1,0), (1,1)\}$ with probabilities $(A, B, C, D)$.
- Bitwise accumulation builds row index $u$ and column index $v$:
  $$u = \sum_{k=0}^{S-1} b_u^{(k)} \cdot 2^{S-1-k}, \quad v = \sum_{k=0}^{S-1} b_v^{(k)} \cdot 2^{S-1-k}$$

#### Vectorized Implementation
Using NumPy, random uniform samples $r \in \mathbb{R}^{M \times S}$ are evaluated concurrently across all edges and recursive levels:
```python
r = np.random.random((num_edges, scale))
for level in range(scale):
    # Vectorized bit assignment via boolean quadrant masks
    bit_u = ... ; bit_v = ...
    u = (u << 1) | bit_u
    v = (v << 1) | bit_v
```

#### Post-Processing Pipeline
Raw stochastic edges undergo three mandatory post-processing transformations:
1. **Self-Loop Removal**: Filters out edges where $u = v$.
2. **Graph Symmetrization**: Concatenates $(u, v, w)$ and $(v, u, w)$ to construct undirected graphs.
3. **Duplicate Edge Collapsing**: Sorts edge triples via `np.lexsort((w, v, u))` and selects unique $(u, v)$ pairs, keeping the minimum edge weight $\min(w)$.

---

### 3.2. Data Storage & CSR Representation

To support high-memory efficiency and fast neighbor access during traversal, graphs are stored in **Compressed Sparse Row (CSR)** format:

- **`indptr`**: Array of size $N + 1$. `indptr[i]` specifies the starting offset of row $i$'s neighbors in `indices` and `data`.
- **`indices`**: Array of size $M$ containing target column vertex IDs.
- **`data`**: Array of size $M$ containing edge weights $w \in [0, 1)$.

$$\text{Matrix Sparsity} = \left(1 - \frac{M}{N^2}\right) \times 100\%$$

---

### 3.3. Breadth-First Search Solver (`bfs_csr.py`)

#### Algorithm
Level-synchronous frontier expansion:
- **State Arrays**: `distances` (size $N$, initialized to $-1$), `parents` (size $N$, initialized to $-1$).
- **Frontier Loop**: In each hop level $d$, processes vertices $u \in \text{Frontier}_d$. Unvisited neighbors $v$ (`distances[v] == -1`) receive `distances[v] = d + 1` and `parents[v] = u`, forming $\text{Frontier}_{d+1}$.

#### Benchmark Metrics
- **TEPS (Traversed Edges Per Second)**: $\text{TEPS} = \frac{\text{Traversed Edges}}{\text{Execution Time (seconds)}}$.

#### Graph500 Tree Validation
Validates:
1. `parents[source] == source` and `distances[source] == 0`.
2. Every tree edge $(p, v)$ exists in CSR arrays (`indices[indptr[p] : indptr[p+1]]`).
3. Level invariant: $\text{distances}[v] == \text{distances}[p] + 1$.

---

### 3.4. Single-Source Shortest Path Solver (`sssp_csr.py`)

#### Algorithm
Dijkstra's Shortest Path algorithm using a min-priority queue (`heapq`):
- **Distance Array**: `distances[v]` initialized to $\infty$, `distances[source] = 0.0`.
- **Priority Queue**: Stores `(distance, vertex)` pairs.

#### Validation
Verifies shortest path tree invariants and global **Triangle Inequality**:
$$\text{distances}[v] \le \text{distances}[u] + w(u, v) + \epsilon$$

---

## 4. Verification & Testing Strategy

The test suite consists of 3 dedicated unit test modules:

1. **`test_kronecker.py`**: Tests generator probability normalization, vertex ID bounds, random seed determinism, post-processing (self-loops, symmetrization, duplicate collapsing), matrix sparsity calculations, and file export.
2. **`test_bfs.py`**: Tests BFS distance array correctness, random source selection, CSR loading, and tree validation error detection.
3. **`test_sssp.py`**: Tests Dijkstra shortest path calculation, edge weight relaxation, triangle inequality validation, and histogram generation.

To run the complete test suite:
```bash
python3 -m unittest discover -p "test_*.py"
```
