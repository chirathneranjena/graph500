#!/usr/bin/env python3
"""
Single-Source Shortest Path (SSSP) for CSR Weighted Graphs
===========================================================
Loads a Kronecker Graph saved in Compressed Sparse Row (.npz) format,
executes Dijkstra's SSSP algorithm, validates the shortest-path tree,
and outputs performance metrics, console ASCII distance histogram, and plot image.

If no source vertex is specified, a vertex is selected randomly from all available vertices.

Usage example:
    python3 sssp_csr.py --input graph_sample.npz --stats --plot histogram.png --validate
"""

import argparse
import heapq
import random
import sys
import time
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend suitable for all environments
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_csr_npz(filepath):
    """
    Loads CSR sparse matrix arrays from a .npz file.
    Expects 'indptr', 'indices', and 'data' / 'shape'.
    """
    try:
        archive = np.load(filepath)
    except Exception as e:
        raise IOError(f"Failed to load NPZ file '{filepath}': {e}")

    if "indptr" not in archive or "indices" not in archive:
        raise ValueError(f"File '{filepath}' is missing 'indptr' or 'indices' arrays required for CSR format.")

    indptr = archive["indptr"]
    indices = archive["indices"]
    data = archive["data"] if "data" in archive else np.ones_like(indices, dtype=np.float64)

    if "shape" in archive:
        num_vertices = int(archive["shape"][0])
    else:
        num_vertices = len(indptr) - 1

    return indptr, indices, data, num_vertices


def run_sssp(indptr, indices, data, num_vertices, source=None, seed=None):
    """
    Executes Dijkstra's Single-Source Shortest Path (SSSP) algorithm from source vertex.
    If source is None, picks a random source vertex from 0 to num_vertices - 1.

    Returns:
        dict containing distances, parents, traversal time, TEPS, and distance metrics.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if source is None:
        source = int(np.random.randint(0, num_vertices))

    if source < 0 or source >= num_vertices:
        raise ValueError(f"Source vertex {source} is out of bounds for graph with {num_vertices} vertices.")

    distances = np.full(num_vertices, np.inf, dtype=np.float64)
    parents = np.full(num_vertices, -1, dtype=np.int64)

    start_time = time.time()

    distances[source] = 0.0
    parents[source] = source

    pq = [(0.0, source)]
    edges_traversed = 0

    while pq:
        d, u = heapq.heappop(pq)

        if d > distances[u]:
            continue

        start_idx = indptr[u]
        end_idx = indptr[u + 1]
        neighbors = indices[start_idx:end_idx]
        weights = data[start_idx:end_idx]
        edges_traversed += len(neighbors)

        for i in range(len(neighbors)):
            v = neighbors[i]
            w = weights[i]
            new_dist = d + w

            if new_dist < distances[v]:
                distances[v] = new_dist
                parents[v] = u
                heapq.heappush(pq, (new_dist, v))

    elapsed_time = time.time() - start_time
    teps = edges_traversed / elapsed_time if elapsed_time > 0 else 0.0

    reachable_mask = np.isfinite(distances)
    reachable_count = int(np.sum(reachable_mask))
    reachable_distances = distances[reachable_mask]

    max_dist = float(np.max(reachable_distances)) if len(reachable_distances) > 0 else 0.0
    avg_dist = float(np.mean(reachable_distances)) if len(reachable_distances) > 0 else 0.0

    return {
        "source": source,
        "distances": distances,
        "parents": parents,
        "visited_vertices": reachable_count,
        "total_vertices": num_vertices,
        "traversed_edges": edges_traversed,
        "max_distance": max_dist,
        "avg_distance": avg_dist,
        "elapsed_time": elapsed_time,
        "teps": teps,
    }


def validate_sssp_tree(indptr, indices, data, source, distances, parents):
    """
    Validates SSSP tree correctness:
    1. parent[source] == source and distance[source] == 0.0.
    2. For all reachable v != source: (parent[v], v) is an edge in CSR with weight w, and distance[v] == distance[parent[v]] + w.
    3. Triangle inequality: for all edges (u, v, w), distance[v] <= distance[u] + w + 1e-7.
    """
    errors = []

    if parents[source] != source:
        errors.append(f"Source parent validation failed: parents[{source}] = {parents[source]}, expected {source}.")

    if abs(distances[source]) > 1e-9:
        errors.append(f"Source distance validation failed: distances[{source}] = {distances[source]}, expected 0.0.")

    reachable_indices = np.where(np.isfinite(distances))[0]

    for v in reachable_indices:
        if v == source:
            continue

        p = parents[v]
        if p == -1:
            errors.append(f"Vertex {v} is marked reachable with distance {distances[v]} but has parent -1.")
            continue

        p_start = indptr[p]
        p_end = indptr[p + 1]
        p_neighbors = indices[p_start:p_end]
        p_weights = data[p_start:p_end]

        match_mask = (p_neighbors == v)
        if not np.any(match_mask):
            errors.append(f"Invalid SSSP tree edge: ({p}, {v}) does not exist in graph.")
            continue

        w = p_weights[match_mask][0]
        expected_dist = distances[p] + w

        if abs(distances[v] - expected_dist) > 1e-7:
            errors.append(
                f"Distance mismatch for vertex {v}: distances[{v}]={distances[v]:.6f} != distances[{p}]({distances[p]:.6f}) + {w:.6f}."
            )

    for u in reachable_indices:
        u_start = indptr[u]
        u_end = indptr[u + 1]
        u_neighbors = indices[u_start:u_end]
        u_weights = data[u_start:u_end]

        for v, w in zip(u_neighbors, u_weights):
            if distances[v] > distances[u] + w + 1e-7:
                errors.append(
                    f"Triangle inequality violated for edge ({u}, {v}): distance[{v}]={distances[v]:.6f} > distance[{u}]({distances[u]:.6f}) + {w:.6f}."
                )

    return len(errors) == 0, errors


def print_ascii_distance_histogram(reachable_distances, num_bins=10):
    """
    Prints a console ASCII histogram of shortest path distances.
    """
    if len(reachable_distances) == 0:
        return

    counts, bin_edges = np.histogram(reachable_distances, bins=num_bins)
    max_count = np.max(counts) if len(counts) > 0 else 1

    print("\n--- Shortest Path Distance Distribution (Histogram) ---")
    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        cnt = counts[i]
        bar = "#" * min(40, int((cnt / max_count) * 40)) if max_count > 0 else ""
        print(f"  [{low:6.4f} - {high:6.4f}] : {cnt:8,} vertices  {bar}")


def plot_distance_histogram(reachable_distances, source, num_vertices, output_filepath="sssp_distance_histogram.png"):
    """
    Plots a histogram of shortest path distances using Matplotlib and saves as PNG.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: Matplotlib is not installed. Skipping plot histogram generation.")
        return

    plt.figure(figsize=(8, 5), dpi=150)
    plt.hist(reachable_distances, bins=30, color="#2b5c8f", edgecolor="black", alpha=0.85)

    title = f"SSSP Distance Distribution from Source {source} (N={num_vertices:,}, Reachable={len(reachable_distances):,})"
    plt.title(title, fontsize=11, pad=10)
    plt.xlabel("Shortest Path Distance", fontsize=10)
    plt.ylabel("Number of Vertices", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()

    print(f"\nSaved SSSP Distance Histogram plot to '{output_filepath}'.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSR Graph Single-Source Shortest Path (SSSP) Solver (Dijkstra)",
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
        help="Source vertex ID for SSSP traversal (if omitted/unspecified, picks a random vertex; if -1, picks highest degree vertex)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for random source vertex selection"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate SSSP tree correctness against shortest path invariants"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Display detailed SSSP traversal statistics and distance distribution"
    )
    parser.add_argument(
        "--plot-histogram",
        "--plot",
        nargs="?",
        const="sssp_distance_histogram.png",
        default=None,
        metavar="FILEPATH",
        help="Save plot image histogram of shortest path distances (default image: 'sssp_distance_histogram.png')",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print("    CSR Graph Single-Source Shortest Path (SSSP)  ")
    print("==================================================")
    print(f"Input NPZ File: {args.input}")

    indptr, indices, data, num_vertices = load_csr_npz(args.input)
    num_edges = len(indices)

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

    sssp_results = run_sssp(indptr, indices, data, num_vertices, source=source)

    print(f"SSSP completed in {sssp_results['elapsed_time']:.6f} seconds.")
    print(f"Performance: {sssp_results['teps']:,.0f} TEPS (Traversed Edges Per Second)")

    if args.validate:
        print("\nValidating SSSP Shortest Path Tree...")
        valid, errors = validate_sssp_tree(
            indptr,
            indices,
            data,
            source=source,
            distances=sssp_results["distances"],
            parents=sssp_results["parents"],
        )
        if valid:
            print("Validation PASSED: All parent-child edges and shortest path distances are valid.")
        else:
            print(f"Validation FAILED with {len(errors)} error(s):")
            for err in errors[:5]:
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more error(s).")

    reachable_dists = sssp_results["distances"][np.isfinite(sssp_results["distances"])]

    if args.stats or not args.validate:
        visited = sssp_results["visited_vertices"]
        pct_visited = (visited / num_vertices) * 100.0 if num_vertices > 0 else 0.0

        print("\n--- SSSP Traversal Statistics ---")
        print(f"Source Vertex:         {source}")
        print(f"Reachable Vertices:    {visited:,} / {num_vertices:,} ({pct_visited:.2f}% component coverage)")
        print(f"Edges Traversed:       {sssp_results['traversed_edges']:,}")
        print(f"Max Path Distance:     {sssp_results['max_distance']:.6f}")
        print(f"Avg Path Distance:     {sssp_results['avg_distance']:.6f}")
        print(f"Execution Time:        {sssp_results['elapsed_time']*1000:.3f} ms")
        print(f"TEPS Metric:           {sssp_results['teps']:,.0f} TEPS")

        if len(reachable_dists) > 0:
            p25, p50, p75 = np.percentile(reachable_dists, [25, 50, 75])
            print("\n--- Shortest Path Distance Quantiles ---")
            print(f"  Min Distance: {np.min(reachable_dists):.6f}")
            print(f"  25th %ile:    {p25:.6f}")
            print(f"  50th %ile:    {p50:.6f} (Median)")
            print(f"  75th %ile:    {p75:.6f}")
            print(f"  Max Distance: {np.max(reachable_dists):.6f}")

            # Console Distance Histogram
            print_ascii_distance_histogram(reachable_dists, num_bins=10)

    if args.plot_histogram is not None and len(reachable_dists) > 0:
        plot_distance_histogram(
            reachable_dists,
            source=source,
            num_vertices=num_vertices,
            output_filepath=args.plot_histogram,
        )


if __name__ == "__main__":
    main()
