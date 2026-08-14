#!/usr/bin/env python3
"""
Unit tests for CSR Graph SSSP Solver (sssp_csr.py)
"""

import os
import tempfile
import unittest
import numpy as np

from sssp_csr import (
    load_csr_npz,
    run_sssp,
    validate_sssp_tree,
    print_ascii_distance_histogram,
    plot_distance_histogram,
    HAS_MATPLOTLIB,
)


class TestSSSPCSR(unittest.TestCase):

    def setUp(self):
        self.indptr = np.array([0, 2, 3, 4, 4], dtype=np.int64)
        self.indices = np.array([1, 2, 2, 3], dtype=np.int64)
        self.data = np.array([0.2, 0.8, 0.3, 0.1], dtype=np.float64)
        self.num_vertices = 4

    def test_run_sssp_explicit_source(self):
        results = run_sssp(self.indptr, self.indices, self.data, self.num_vertices, source=0)

        self.assertEqual(results["source"], 0)
        self.assertEqual(results["visited_vertices"], 4)

        distances = results["distances"]
        self.assertAlmostEqual(distances[0], 0.0)
        self.assertAlmostEqual(distances[1], 0.2)
        self.assertAlmostEqual(distances[2], 0.5)
        self.assertAlmostEqual(distances[3], 0.6)

    def test_run_sssp_random_source(self):
        results = run_sssp(self.indptr, self.indices, self.data, self.num_vertices, source=None, seed=42)

        source = results["source"]
        self.assertTrue(0 <= source < self.num_vertices)
        self.assertAlmostEqual(results["distances"][source], 0.0)

    def test_validate_sssp_tree_valid(self):
        results = run_sssp(self.indptr, self.indices, self.data, self.num_vertices, source=0)
        valid, errors = validate_sssp_tree(
            self.indptr,
            self.indices,
            self.data,
            source=0,
            distances=results["distances"],
            parents=results["parents"],
        )

        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_print_ascii_distance_histogram(self):
        dists = np.array([0.0, 0.2, 0.5, 0.6])
        # Verify it runs without error
        print_ascii_distance_histogram(dists, num_bins=5)

    @unittest.skipUnless(HAS_MATPLOTLIB, "Matplotlib not installed")
    def test_plot_distance_histogram(self):
        dists = np.array([0.0, 0.2, 0.5, 0.6])
        with tempfile.NamedTemporaryFile("w+", suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plot_distance_histogram(dists, source=0, num_vertices=4, output_filepath=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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
