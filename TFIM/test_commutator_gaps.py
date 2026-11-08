"""Regression checks for the TFIM commutator gaps plotted in S1a and S4b."""
import sys
import unittest
from functools import reduce
from pathlib import Path

import numpy as np

from AH_commutator_scaling import TFIM_commutator_AH_scaling as static

sys.path.insert(0, str(Path(__file__).resolve().parent / "Smoothed gap"))
import smoothed_gap_tfim as smoothed


def pauli_gram(labels, coefficients):
    single = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Z": np.diag([1, -1])}
    basis = np.array([reduce(np.kron, (single[p] for p in label)) for label in labels])
    H = np.einsum("a,aij->ij", coefficients, basis)
    commutators = (basis @ H - H @ basis).reshape(len(basis), -1)
    return commutators @ commutators.T / H.shape[0]


class CommutatorGapTest(unittest.TestCase):
    def test_signed_static_gap(self):
        for n in (2, 4):
            with self.subTest(n=n):
                rng = np.random.default_rng(42)
                bonds = rng.normal(size=n - 1)
                fields = rng.normal(size=n)
                coefficients = np.r_[bonds, fields]
                labels, _ = smoothed.build_search_basis(n)  # fields first
                gram = pauli_gram(labels[n:] + labels[:n], coefficients)
                np.testing.assert_allclose(static.compute_Q(n, bonds, fields), gram, atol=1e-13)
                np.testing.assert_allclose(gram @ coefficients, 0, atol=1e-13)
                perp = np.linalg.svd(coefficients[None, :], full_matrices=True)[2][1:].T
                expected = np.linalg.eigvalsh(perp.T @ gram @ perp)[0]
                self.assertAlmostEqual(static.commutator_gap(bonds, fields), expected, places=12)

    def test_extra_conserved_direction(self):
        self.assertAlmostEqual(static.commutator_gap(np.ones(3), np.zeros(4)), 0, places=14)

    def test_smoothed_gap_scale(self):
        n = 4
        labels, basis = smoothed.build_search_basis(n)
        coefficients = np.r_[[-0.4, 0.6, 0.2, -0.1], np.ones(n - 1)]
        expected = pauli_gram(labels, coefficients)
        np.testing.assert_allclose(smoothed.gamma_matrix(basis, coefficients), expected, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
