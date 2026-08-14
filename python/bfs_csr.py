#!/usr/bin/env python3
"""
Breadth-First Search (BFS) for CSR Graphs
=========================================
Loads a Kronecker Graph saved in Compressed Sparse Row (.npz) format,
executes Breadth-First Search (Graph500 compliant), validates results,
and outputs performance statistics (TEPS, search depth, level distribution).

If no source vertex is specified, a vertex is selected randomly from all available vertices.

Usage example:
    python3 bfs_csr.py --input graph_sample.npz --stats --validate
"""

import argparse
import random
import sys
import time
import numpy as np


def load_csr_npz(filepath):
    """
    Loads CSR sparse matrix arrays from a .npz file.
    Expects 'indptr', 'indices', and optional 'data' / 'shape'.
    """
    try:
        archive = np.load(filepath)
    except Exception as e:
        raise IOError(f"Failed to load NPZ file '{filepath}': {e}")

    if "indptr" not in archive or "indices" not in archive:
        raise ValueError(f"File '{filepath}' is missing 'indptr' or 'indices' arrays required for CSR format.")

    indptr = archive["indptr"]
    indices = archive["indices"]
    data = archive["data"] if "data" in archive else None

    if "shape" in archive:
        num_vertices = int(archive["shape"][0])
    else:
        num_vertices = len(indptr) - 1

    return indptr, indices, data, num_vertices


def run_bfs(indptr, indices, num_vertices, source=None, seed=None):
    """
    Executes Breadth-First Search from source vertex using CSR graph.
    If source is None, picks a random source vertex from 0 to num_vertices - 1.

    Returns:
        dict containing distances, parents, traversal time, TEPS, and level metrics.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if source is None:
        source = int(np.random.randint(0, num_vertices))

    if source < 0 or source >= num_vertices:
        raise ValueError(f"Source vertex {source} is out of bounds for graph with {num_vertices} vertices.")

    distances = np.full(num_vertices, -1, dtype=np.int64)
    parents = np.full(num_vertices, -1, dtype=np.int64)

    start_time = time.time()

    distances[source] = 0
    parents[source] = source

    frontier = [source]
    level_sizes = []
    traversed_edges = 0

    while frontier:
        level_sizes.append(len(frontier))
        next_frontier = []

        for u in frontier:
            start_idx = indptr[u]
            end_idx = indptr[u + 1]
            neighbors = indices[start_idx:end_idx]
            traversed_edges += len(neighbors)

            for v in neighbors:
                if distances[v] == -1:
                    distances[v] = distances[u] + 1
                    parents[v] = u
                    next_frontier.append(v)

        frontier = next_frontier

    elapsed_time = time.time() - start_time
    teps = traversed_edges / elapsed_time if elapsed_time > 0 else 0.0

    visited_count = int(np.sum(distances != -1))
    max_depth = len(level_sizes) - 1 if level_sizes else 0

    return {
        "source": source,
        "distances": distances,
        "parents": parents,
        "visited_vertices": visited_count,
        "total_vertices": num_vertices,
        "traversed_edges": traversed_edges,
        "max_depth": max_depth,
        "level_sizes": level_sizes,
        "elapsed_time": elapsed_time,
        "teps": teps,
    }


def validate_bfs_tree(indptr, indices, source, distances, parents):
    """
    Validates the BFS tree according to Graph500 criteria:
    1. Parent of source is source.
    2. For all v != source where visited, (parent[v], v) is a valid edge in CSR.
    3. distance[v] == distance[parent[v]] + 1.
    """
    errors = []

    if parents[source] != source:
        errors.append(f"Source parent validation failed: parents[{source}] = {parents[source]}, expected {source}.")

    if distances[source] != 0:
        errors.append(f"Source distance validation failed: distances[{source}] = {distances[source]}, expected 0.")

    visited_indices = np.where(distances != -1)[0]

    for v in visited_indices:
        if v == source:
            continue

        p = parents[v]
        if p == -1:
            errors.append(f"Vertex {v} is marked visited with distance {distances[v]} but has parent -1.")
            continue

        # Check edge (p, v) exists in CSR
        p_neighbors = indices[indptr[p]:indptr[p + 1]]
        if v not in p_neighbors:
            errors.append(f"Invalid BFS tree edge: ({p}, {v}) does not exist in graph.")

        # Check level relationship
        if distances[v] != distances[p] + 1:
            errors.append(
                f"Invalid level relation: distance[{v}]={distances[v]} != distance[{p}]({distances[p]}) + 1."
            )

    return len(errors) == 0, errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSR Graph Breadth-First Search (BFS) & Graph500 Benchmark Traversal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input CSR graph file in .npz format"
    )
    parser.add_argument(
        "-src",
        "--source",
        type=int,
        default=None,
        help="Source vertex ID for BFS traversal (if omitted/unspecified, picks a random vertex; if -1, picks highest degree vertex)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for random source vertex selection"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate BFS tree correctness against Graph500 criteria"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Display detailed BFS traversal statistics and level distribution"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print("        CSR Graph Breadth-First Search            ")
    print("==================================================")
    print(f"Input NPZ File: {args.input}")

    indptr, indices, data, num_vertices = load_csr_npz(args.input)
    num_edges = len(indices)

    # Determine source vertex
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    source = args.source
    if source is None:
        source = int(np.random.randint(0, num_vertices))
        print(f"Source Vertex Selection: Randomly chosen vertex -> {source}")
    elif source < 0:
        degrees = indptr[1:] - indptr[:-1]
        source = int(np.argmax(degrees))
        print(f"Source Vertex Selection: Highest degree vertex -> {source} (degree: {degrees[source]:,})")
    else:
        print(f"Source Vertex Selection: User specified -> {source}")

    print(f"Graph Vertices: {num_vertices:,}")
    print(f"Graph Edges:    {num_edges:,}")
    print("==================================================")

    bfs_results = run_bfs(indptr, indices, num_vertices, source=source)

    print(f"BFS completed in {bfs_results['elapsed_time']:.6f} seconds.")
    print(f"Performance: {bfs_results['teps']:,.0f} TEPS (Traversed Edges Per Second)")

    if args.validate:
        print("\nValidating BFS Tree Structure...")
        valid, errors = validate_bfs_tree(
            indptr,
            indices,
            source=source,
            distances=bfs_results["distances"],
            parents=bfs_results["parents"],
        )
        if valid:
            print("Validation PASSED: All parent-child edges and distance levels are correct.")
        else:
            print(f"Validation FAILED with {len(errors)} error(s):")
            for err in errors[:5]:
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more error(s).")

    if args.stats or not args.validate:
        visited = bfs_results["visited_vertices"]
        pct_visited = (visited / num_vertices) * 100.0 if num_vertices > 0 else 0.0

        print("\n--- BFS Traversal Statistics ---")
        print(f"Source Vertex:         {source}")
        print(f"Reachable Vertices:    {visited:,} / {num_vertices:,} ({pct_visited:.2f}% component coverage)")
        print(f"Edges Traversed:       {bfs_results['traversed_edges']:,}")
        print(f"Max Search Depth:      {bfs_results['max_depth']} hops")
        print(f"Execution Time:        {bfs_results['elapsed_time']*1000:.3f} ms")
        print(f"TEPS Metric:           {bfs_results['teps']:,.0f} TEPS")

        print("\n--- Level-by-Level Discovery Distribution ---")
        for depth, count in enumerate(bfs_results["level_sizes"]):
            bar = "#" * min(40, int((count / visited) * 40)) if visited > 0 else ""
            print(f"  Depth {depth:2d}: {count:8,} vertices  {bar}")


if __name__ == "__main__":
    main()
