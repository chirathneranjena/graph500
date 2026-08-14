#!/usr/bin/env python3
"""
Unit tests for NumPy and Matplotlib-based Kronecker Graph Generator with CSR file export and matrix sparsity stats
"""

import os
import tempfile
import unittest
import numpy as np

from kronecker_generator import (
    generate_stochastic_kronecker,
    generate_exact_kronecker,
    post_process_graph,
    edges_to_csr,
    build_adjacency_matrix,
    compute_graph_stats,
    save_graph,
    normalize_initiator,
    scramble_vertex_id,
    plot_adjacency_matrix,
)


class TestKroneckerGenerator(unittest.TestCase):

    def test_normalize_initiator(self):
        init = [57, 19, 19, 5]
        norm = normalize_initiator(init)
        self.assertAlmostEqual(sum(norm), 1.0)
        self.assertAlmostEqual(norm[0], 0.57)

    def test_edges_to_csr(self):
        edges = [(0, 1, 0.5), (0, 2, 0.3), (1, 2, 0.8), (2, 0, 0.1)]
        num_vertices = 3

        indptr, indices, data = edges_to_csr(edges, num_vertices)

        self.assertEqual(len(indptr), num_vertices + 1)
        self.assertEqual(indptr.tolist(), [0, 2, 3, 4])
        self.assertEqual(indices.tolist(), [1, 2, 2, 0])
        self.assertAlmostEqual(data[0], 0.5)

    def test_compute_stats_sparsity(self):
        edges = [(0, 1, 0.2), (1, 0, 0.2), (2, 0, 0.6), (0, 2, 0.6)]
        num_vertices = 4  # Total possible entries = 16

        stats = compute_graph_stats(edges, num_vertices)
        self.assertEqual(stats["total_edges"], 4)
        self.assertAlmostEqual(stats["matrix_density"], 4 / 16)  # 0.25
        self.assertAlmostEqual(stats["matrix_sparsity"], 12 / 16)  # 0.75
        self.assertAlmostEqual(stats["matrix_sparsity_pct"], 75.0)

    def test_post_process_graph_with_weights(self):
        u = [0, 0, 1, 2, 2]
        v = [0, 1, 2, 0, 2]
        w = [0.1, 0.5, 0.3, 0.8, 0.9]

        edges = post_process_graph(
            u, v, w, remove_self_loops=True, symmetrize=True, collapse_duplicates=True
        )

        for a, b, weight in edges:
            self.assertNotEqual(a, b)
            self.assertTrue(0.0 <= weight < 1.0)

    def test_stochastic_generator_csr(self):
        scale = 5
        edge_factor = 10
        num_vertices = 1 << scale

        edges = generate_stochastic_kronecker(
            scale=scale,
            edge_factor=edge_factor,
            initiator=[0.57, 0.19, 0.19, 0.05],
            seed=123,
        )

        indptr, indices, data = edges_to_csr(edges, num_vertices)
        self.assertEqual(len(indptr), num_vertices + 1)
        self.assertEqual(indptr[-1], len(edges))

    def test_save_graph_csr_text_and_npz(self):
        edges = [(0, 1, 0.5), (1, 0, 0.5), (1, 2, 0.8), (2, 1, 0.8)]
        num_vertices = 3

        with tempfile.NamedTemporaryFile("w+", suffix=".csr", delete=False) as tmp_csr:
            csr_path = tmp_csr.name

        with tempfile.NamedTemporaryFile("w+", suffix=".npz", delete=False) as tmp_npz:
            npz_path = tmp_npz.name

        try:
            save_graph(edges, num_vertices, csr_path, fmt="csr", weighted=True)
            with open(csr_path, "r") as f:
                lines = f.readlines()
            self.assertEqual(lines[0].strip(), "3 4")

            save_graph(edges, num_vertices, npz_path, fmt="npz", weighted=True)
            loaded = np.load(npz_path)
            self.assertEqual(loaded["indptr"].tolist(), [0, 1, 3, 4])
        finally:
            if os.path.exists(csr_path):
                os.remove(csr_path)
            if os.path.exists(npz_path):
                os.remove(npz_path)


if __name__ == "__main__":
    unittest.main()
