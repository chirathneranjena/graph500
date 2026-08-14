#!/usr/bin/env python3
"""
Unit tests for CSR Graph BFS Solver (bfs_csr.py)
"""

import os
import tempfile
import unittest
import numpy as np

from bfs_csr import load_csr_npz, run_bfs, validate_bfs_tree


class TestBFSCSR(unittest.TestCase):

    def setUp(self):
        # Create a simple test graph: 0 - 1 - 2, 0 - 3
        self.indptr = np.array([0, 2, 4, 5, 6], dtype=np.int64)
        self.indices = np.array([1, 3, 0, 2, 1, 0], dtype=np.int64)
        self.data = np.ones(6, dtype=np.float64)
        self.num_vertices = 4

    def test_run_bfs_explicit_source(self):
        results = run_bfs(self.indptr, self.indices, self.num_vertices, source=0)

        self.assertEqual(results["source"], 0)
        self.assertEqual(results["visited_vertices"], 4)
        self.assertEqual(results["max_depth"], 2)
        self.assertEqual(results["level_sizes"], [1, 2, 1])

        distances = results["distances"]
        self.assertEqual(distances[0], 0)
        self.assertEqual(distances[1], 1)
        self.assertEqual(distances[3], 1)
        self.assertEqual(distances[2], 2)

    def test_run_bfs_random_source(self):
        # When source is None, it should pick a random source between 0 and num_vertices - 1
        results = run_bfs(self.indptr, self.indices, self.num_vertices, source=None, seed=42)

        source = results["source"]
        self.assertTrue(0 <= source < self.num_vertices)
        self.assertEqual(results["distances"][source], 0)
        self.assertEqual(results["parents"][source], source)

    def test_validate_bfs_tree_valid(self):
        results = run_bfs(self.indptr, self.indices, self.num_vertices, source=0)
        valid, errors = validate_bfs_tree(
            self.indptr,
            self.indices,
            source=results["source"],
            distances=results["distances"],
            parents=results["parents"],
        )

        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_validate_bfs_tree_invalid(self):
        results = run_bfs(self.indptr, self.indices, self.num_vertices, source=0)
        parents = results["parents"].copy()
        distances = results["distances"].copy()

        parents[2] = 0  # Invalid edge (0, 2) does not exist!

        valid, errors = validate_bfs_tree(
            self.indptr, self.indices, source=0, distances=distances, parents=parents
        )
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)

    def test_load_csr_npz_integration(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            np.savez(
                tmp_path,
                indptr=self.indptr,
                indices=self.indices,
                data=self.data,
                shape=np.array([4, 4], dtype=np.int64),
            )

            indptr_loaded, indices_loaded, data_loaded, num_v_loaded = load_csr_npz(tmp_path)

            self.assertEqual(num_v_loaded, 4)
            self.assertEqual(indptr_loaded.tolist(), self.indptr.tolist())
            self.assertEqual(indices_loaded.tolist(), self.indices.tolist())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
