#!/usr/bin/env python3
"""
Kronecker Graph Generator
=========================
Generates Kronecker Graphs (Stochastic / R-MAT and Exact Kronecker Power)
using NumPy vectorization, compliant with Graph500 benchmark principles.

Automatically assigns random edge weights in [0, 1), removes self-loops,
collapses duplicate edges, symmetrizes the graph, and exports to Compressed Sparse Row (CSR) format.

Usage example:
    python3 kronecker_generator.py --scale 6 --edge-factor 16 --output graph.csr --stats
"""

import argparse
import random
import sys
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for all environments
import matplotlib.pyplot as plt


def normalize_initiator(initiator):
    """Ensure initiator matrix [A, B, C, D] sums to 1.0."""
    total = sum(initiator)
    if total <= 0:
        raise ValueError("Sum of initiator matrix parameters must be positive.")
    return [x / total for x in initiator]


def scramble_vertex_id(v, scale, seed=None):
    """
    Vectorized hash-based pseudo-random permutation of vertex IDs to eliminate
    index bias (e.g. vertex 0 having the highest degree).
    """
    s_seed = seed if seed is not None else 42
    m = (1 << scale) - 1
    val = (v * 2654435761 + s_seed * 1013904223) & 0xFFFFFFFF
    return (val ^ (val >> scale)) & m


def post_process_graph(u, v, w, remove_self_loops=True, symmetrize=True, collapse_duplicates=True):
    """
    Post-processes edge arrays with weights:
    1. Removes self-loops (u == v)
    2. Symmetrizes graph by adding reverse edges (v, u) with identical weight w
    3. Collapses duplicate edges keeping the minimum weight
    """
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)

    # 1. Remove self loops
    if remove_self_loops:
        non_loop_mask = (u != v)
        u, v, w = u[non_loop_mask], v[non_loop_mask], w[non_loop_mask]

    # 2. Symmetrize graph
    if symmetrize:
        u_sym = np.concatenate([u, v])
        v_sym = np.concatenate([v, u])
        w_sym = np.concatenate([w, w])
        u, v, w = u_sym, v_sym, w_sym

    # 3. Collapse duplicate edges (keeping minimum weight for each (u, v) pair)
    if collapse_duplicates and len(u) > 0:
        sort_idx = np.lexsort((w, v, u))
        u_sorted, v_sorted, w_sorted = u[sort_idx], v[sort_idx], w[sort_idx]

        edges_2d = np.column_stack((u_sorted, v_sorted))
        _, unique_indices = np.unique(edges_2d, axis=0, return_index=True)

        u = u_sorted[unique_indices]
        v = v_sorted[unique_indices]
        w = w_sorted[unique_indices]

    return list(zip(u.tolist(), v.tolist(), w.tolist()))


def generate_stochastic_kronecker(
    scale,
    edge_factor,
    initiator,
    seed=None,
    scramble=False,
    remove_self_loops=True,
    symmetrize=True,
    collapse_duplicates=True,
):
    """
    Fast vectorized generation of Stochastic Kronecker (R-MAT) edges with random weights using NumPy.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    a, b, c, d = normalize_initiator(initiator)
    ab = a + b
    abc = a + b + c

    num_vertices = 1 << scale
    num_edges = edge_factor * num_vertices

    r = np.random.random((num_edges, scale))

    u = np.zeros(num_edges, dtype=np.int64)
    v = np.zeros(num_edges, dtype=np.int64)

    for level in range(scale):
        r_level = r[:, level]

        is_q0 = r_level < a
        is_q1 = (r_level >= a) & (r_level < ab)
        is_q2 = (r_level >= ab) & (r_level < abc)

        bit_u = np.ones(num_edges, dtype=np.int64)
        bit_v = np.ones(num_edges, dtype=np.int64)

        bit_u[is_q0 | is_q1] = 0
        bit_v[is_q0 | is_q2] = 0

        u = (u << 1) | bit_u
        v = (v << 1) | bit_v

    if scramble:
        u = scramble_vertex_id(u, scale, seed)
        v = scramble_vertex_id(v, scale, seed)

    w = np.random.random(num_edges)

    return post_process_graph(
        u,
        v,
        w,
        remove_self_loops=remove_self_loops,
        symmetrize=symmetrize,
        collapse_duplicates=collapse_duplicates,
    )


def generate_exact_kronecker(
    scale,
    initiator,
    seed=None,
    remove_self_loops=True,
    symmetrize=True,
    collapse_duplicates=True,
):
    """
    Generates exact Kronecker matrix power P = K_1 (X) K_1 (X) ... (X) K_1
    using np.kron and samples weighted edges.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    a, b, c, d = normalize_initiator(initiator)
    k1 = np.array([[a, b], [c, d]], dtype=np.float64)

    matrix = k1
    for _ in range(scale - 1):
        matrix = np.kron(matrix, k1)

    rand_vals = np.random.random(matrix.shape)
    edges_mask = rand_vals < matrix
    rows, cols = np.where(edges_mask)

    w = np.random.random(len(rows))

    return post_process_graph(
        rows,
        cols,
        w,
        remove_self_loops=remove_self_loops,
        symmetrize=symmetrize,
        collapse_duplicates=collapse_duplicates,
    )


def edges_to_csr(edges, num_vertices):
    """
    Converts edge list [(u, v, w), ...] into Compressed Sparse Row (CSR) format:
    Returns (indptr, indices, data) arrays.
    """
    if not edges:
        return (
            np.zeros(num_vertices + 1, dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    edge_arr = np.array(edges, dtype=np.float64)
    u = edge_arr[:, 0].astype(np.int64)
    v = edge_arr[:, 1].astype(np.int64)
    w = edge_arr[:, 2]

    # Sort primarily by u, then v
    sort_idx = np.lexsort((v, u))
    u_sorted, v_sorted, w_sorted = u[sort_idx], v[sort_idx], w[sort_idx]

    # Compute indptr array
    counts = np.bincount(u_sorted, minlength=num_vertices)
    indptr = np.zeros(num_vertices + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)

    indices = v_sorted
    data = w_sorted

    return indptr, indices, data


def build_adjacency_matrix(edges, num_vertices, binary=False):
    """
    Constructs a 2D NumPy array adjacency matrix from weighted edge list.
    """
    adj = np.zeros((num_vertices, num_vertices), dtype=np.float64)
    if edges:
        edge_arr = np.array(edges, dtype=np.float64)
        u_nodes = edge_arr[:, 0].astype(np.int64)
        v_nodes = edge_arr[:, 1].astype(np.int64)
        weights = edge_arr[:, 2]
        adj[u_nodes, v_nodes] = weights
    if binary:
        adj = (adj > 0).astype(np.float64)
    return adj


def plot_adjacency_matrix(edges, num_vertices, output_filepath="adjacency_matrix.png", title=None):
    """
    Plots the weighted adjacency matrix using Matplotlib spy/imshow and saves as PNG.
    """
    adj = build_adjacency_matrix(edges, num_vertices, binary=False)

    plt.figure(figsize=(8, 8), dpi=150)

    if num_vertices <= 256:
        plt.imshow(adj, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Edge Weight")
    else:
        plt.spy(adj > 0, markersize=0.8, color="black")

    if title is None:
        title = f"Weighted Symmetric Kronecker Graph Adjacency Matrix (N={num_vertices:,}, Edges={len(edges):,})"

    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Target Vertex (v)", fontsize=10)
    plt.ylabel("Source Vertex (u)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()

    print(f"\nSaved Adjacency Matrix plot to '{output_filepath}'.")


def compute_graph_stats(edges, num_vertices):
    """Calculates basic graph statistics including matrix sparsity and weight distribution."""
    total_matrix_entries = num_vertices * num_vertices if num_vertices > 0 else 1
    num_edges = len(edges)

    density = num_edges / total_matrix_entries
    sparsity = 1.0 - density
    sparsity_pct = sparsity * 100.0

    if not edges:
        return {
            "num_vertices": num_vertices,
            "active_vertices": 0,
            "total_edges": 0,
            "unique_edges": 0,
            "self_loops": 0,
            "matrix_density": density,
            "matrix_sparsity": sparsity,
            "matrix_sparsity_pct": sparsity_pct,
            "avg_degree": 0.0,
            "max_degree": 0,
            "min_weight": 0.0,
            "avg_weight": 0.0,
            "max_weight": 0.0,
        }

    edge_arr = np.array(edges, dtype=np.float64)
    u_arr = edge_arr[:, 0].astype(np.int64)
    v_arr = edge_arr[:, 1].astype(np.int64)
    weights = edge_arr[:, 2]

    unique_edges = len(set((u, v) for u, v, _ in edges))
    self_loops = int(np.sum(u_arr == v_arr))

    u_uniq, u_counts = np.unique(u_arr, return_counts=True)
    v_uniq, v_counts = np.unique(v_arr, return_counts=True)

    active_vertices = len(set(u_uniq.tolist()).union(set(v_uniq.tolist())))
    max_degree = int(np.max(u_counts)) if len(u_counts) > 0 else 0
    avg_degree = num_edges / num_vertices if num_vertices > 0 else 0

    return {
        "num_vertices": num_vertices,
        "active_vertices": active_vertices,
        "total_edges": num_edges,
        "unique_edges": unique_edges,
        "self_loops": self_loops,
        "matrix_density": density,
        "matrix_sparsity": sparsity,
        "matrix_sparsity_pct": sparsity_pct,
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "min_weight": float(np.min(weights)),
        "avg_weight": float(np.mean(weights)),
        "max_weight": float(np.max(weights)),
    }


def save_graph(edges, num_vertices, filepath, fmt="csr", weighted=True):
    """
    Saves generated graph to file in the specified format (default: CSR).
    
    Supported Formats:
    - 'csr': Plaintext CSR file (Header N M, indptr line, indices line, data line)
    - 'npz': NumPy binary archive containing indptr, indices, data, shape
    - 'edgelist': Standard edge list (u v w or u v)
    """
    indptr, indices, data = edges_to_csr(edges, num_vertices)

    if fmt == "npz" or filepath.endswith(".npz"):
        np.savez(
            filepath,
            indptr=indptr,
            indices=indices,
            data=data if weighted else np.ones_like(indices, dtype=np.float64),
            shape=np.array([num_vertices, num_vertices], dtype=np.int64),
        )
    elif fmt == "edgelist" or filepath.endswith(".edgelist"):
        sep = "\t"
        with open(filepath, "w") as f:
            if weighted:
                for u, v, w in edges:
                    f.write(f"{int(u)}{sep}{int(v)}{sep}{w:.6f}\n")
            else:
                for u, v, _ in edges:
                    f.write(f"{int(u)}{sep}{int(v)}\n")
    else:  # Default 'csr' format
        with open(filepath, "w") as f:
            f.write(f"{num_vertices} {len(indices)}\n")
            f.write(" ".join(map(str, indptr.tolist())) + "\n")
            f.write(" ".join(map(str, indices.tolist())) + "\n")
            if weighted:
                f.write(" ".join([f"{w:.6f}" for w in data.tolist()]) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kronecker Graph Generator (NumPy Vectorized, CSR Output, Graph500 compliant)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=10, help="Scale S of the graph (N = 2^S vertices)"
    )
    parser.add_argument(
        "-ef",
        "--edge-factor",
        "--edges-per-vertex",
        type=int,
        default=16,
        help="Average edges per vertex (M = EF * 2^S total initial edges sampled)",
    )
    parser.add_argument(
        "-a",
        "--initiator",
        nargs=4,
        type=float,
        default=[0.57, 0.19, 0.19, 0.05],
        help="Initiator matrix probabilities [A, B, C, D]",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path for CSR / edge list graph data")
    parser.add_argument(
        "--format",
        type=str,
        choices=["csr", "npz", "edgelist"],
        default="csr",
        help="Output file format ('csr' for text CSR, 'npz' for binary numpy CSR, 'edgelist' for TSV edge list)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (if omitted, picks a random seed)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stochastic", "exact"],
        default="stochastic",
        help="Generation mode ('stochastic' for R-MAT sampling, 'exact' for matrix power sampling)",
    )
    parser.add_argument(
        "--scramble", action="store_true", help="Permute vertex IDs to eliminate index zero degree bias"
    )
    parser.add_argument(
        "--unweighted", action="store_true", help="Omit edge weights from output"
    )
    parser.add_argument(
        "--keep-self-loops", action="store_true", help="Do not remove self-loops (u == v)"
    )
    parser.add_argument(
        "--no-symmetrize", action="store_true", help="Do not symmetrize graph (keep directed edges)"
    )
    parser.add_argument(
        "--allow-duplicates", action="store_true", help="Allow duplicate edges"
    )
    parser.add_argument("--stats", action="store_true", help="Print detailed graph statistics")
    parser.add_argument(
        "--plot-matrix",
        "--plot",
        nargs="?",
        const="adjacency_matrix.png",
        default=None,
        metavar="FILEPATH",
        help="Save plot image of the Adjacency Matrix using Matplotlib (default image: 'adjacency_matrix.png')",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    num_vertices = 1 << args.scale
    expected_edges = args.edge_factor * num_vertices

    remove_self_loops = not args.keep_self_loops
    symmetrize = not args.no_symmetrize
    collapse_duplicates = not args.allow_duplicates
    weighted = not args.unweighted

    seed_specified = (args.seed is not None)
    if not seed_specified:
        args.seed = random.randint(0, 2**32 - 1)

    print("==================================================")
    print("      Kronecker Graph Generator (NumPy CSR)       ")
    print("==================================================")
    print(f"Scale (S):             {args.scale} ({num_vertices:,} vertices)")
    print(f"Edges per Vertex (EF): {args.edge_factor}")
    print(f"Initial Edge Samples:  {expected_edges:,}")
    print(f"Initiator Matrix:      A={args.initiator[0]}, B={args.initiator[1]}, C={args.initiator[2]}, D={args.initiator[3]}")
    print(f"Generation Mode:       {args.mode}")
    print(f"Random Seed:           {args.seed} ({'User specified' if seed_specified else 'Randomly generated'})")
    print(f"Edge Weights:          {'Enabled [0, 1)' if weighted else 'Disabled (Unweighted)'}")
    print(f"Vertex Scrambling:     {'Enabled' if args.scramble else 'Disabled'}")
    print(f"Remove Self-Loops:     {'Yes' if remove_self_loops else 'No'}")
    print(f"Symmetrize Graph:      {'Yes (Undirected)' if symmetrize else 'No (Directed)'}")
    print(f"Collapse Duplicates:   {'Yes' if collapse_duplicates else 'No'}")
    print("==================================================")

    start_time = time.time()

    if args.mode == "stochastic":
        edges = generate_stochastic_kronecker(
            scale=args.scale,
            edge_factor=args.edge_factor,
            initiator=args.initiator,
            seed=args.seed,
            scramble=args.scramble,
            remove_self_loops=remove_self_loops,
            symmetrize=symmetrize,
            collapse_duplicates=collapse_duplicates,
        )
    else:
        edges = generate_exact_kronecker(
            scale=args.scale,
            initiator=args.initiator,
            seed=args.seed,
            remove_self_loops=remove_self_loops,
            symmetrize=symmetrize,
            collapse_duplicates=collapse_duplicates,
        )

    elapsed_time = time.time() - start_time

    print(f"Generation & Post-processing completed in {elapsed_time:.4f} seconds ({len(edges) / elapsed_time if elapsed_time > 0 else 0:,.0f} edges/sec).")

    if args.stats or args.output is None:
        stats = compute_graph_stats(edges, num_vertices)
        print("\n--- Graph Statistics ---")
        print(f"Vertices (N):          {stats['num_vertices']:,}")
        print(f"Active Vertices:       {stats['active_vertices']:,}")
        print(f"Final Total Edges (M): {stats['total_edges']:,}")
        print(f"Unique Edges:          {stats['unique_edges']:,}")
        print(f"Matrix Sparsity:       {stats['matrix_sparsity_pct']:.4f}% (Density: {stats['matrix_density']*100:.4f}%)")
        print(f"Self-Loops:            {stats['self_loops']:,}")
        print(f"Avg Degree:            {stats['avg_degree']:.2f}")
        print(f"Max Degree:            {stats['max_degree']:,}")
        print(f"Min Edge Weight:       {stats['min_weight']:.6f}")
        print(f"Avg Edge Weight:       {stats['avg_weight']:.6f}")
        print(f"Max Edge Weight:       {stats['max_weight']:.6f}")

    if args.plot_matrix is not None:
        plot_adjacency_matrix(edges, num_vertices, output_filepath=args.plot_matrix)

    if args.output:
        save_graph(edges, num_vertices, args.output, fmt=args.format, weighted=weighted)
        print(f"\nSaved graph to '{args.output}' in '{args.format}' format (Vertices: {num_vertices:,}, Edges: {len(edges):,}).")
    elif not args.stats and args.plot_matrix is None:
        indptr, indices, data = edges_to_csr(edges, num_vertices)
        print("\nCSR Format Summary:")
        print(f"  indptr  (length {len(indptr)}):  {indptr[:10].tolist()} ... {indptr[-1]}")
        print(f"  indices (length {len(indices)}): {indices[:10].tolist()} ...")
        print(f"  data    (length {len(data)}):    {[round(x, 4) for x in data[:10].tolist()]} ...")


if __name__ == "__main__":
    main()
