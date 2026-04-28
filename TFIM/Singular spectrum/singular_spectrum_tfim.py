"""Figure: Singular spectrum of the learning matrix X for an open-boundary TFIM.

Model (NOT rescaled):
    H = sum_i g_i X_i  +  sum_i J_i Z_i Z_{i+1},   g_i, J_i ~ N(0, 1).

Search family:
    V = {X_i}_{i=1..n}  union  {Z_i Z_{i+1}}_{i=1..n-1}.

Construction of X (noiseless):
    For each product probe rho_l (six-state ensemble),
        X_{l,v} = tr(P_v rho_l) - tr(P_v U rho_l U^dagger)
    with U = exp(-i H t).

We compute SVD(X) and plot the (sorted-increasing) singular spectrum.

Outputs (next to this script):
    tfim_singular_spectrum.csv       columns: j, sigma
    tfim_singular_spectrum_meta.csv  scalar diagnostics
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
sys.path.insert(0, str(REPO / "Hamiltonian reconstruction"))

import recon_common as rc  # noqa: E402


def family_labels(n):
    labels, names = [], []
    for i in range(n):
        labels.append(rc.single_site_label(n, i, "X"))
        names.append(f"X_{i}")
    for i in range(n - 1):
        labels.append(rc.two_site_label(n, i, i + 1, "Z", "Z"))
        names.append(f"Z{i}Z{i+1}")
    return labels, names


def main():
    n = 10
    t = 400.0
    n_probes = 1000
    seed_inst, seed_probes = 123, 456

    rng_i = np.random.default_rng(seed_inst)
    rng_p = np.random.default_rng(seed_probes)

    labels, names = family_labels(n)
    coeffs = rng_i.standard_normal(len(labels))   # g_i, J_i ~ N(0,1), unnormalized

    H_true, true_h, descriptors = rc.build_hamiltonian(labels, coeffs)
    U = expm(-1j * t * H_true)

    probes = rc.sample_product_probes(n, n_probes, rng_p)
    X = rc.build_feature_matrix_exact(U, probes, labels, descriptors,
                                      n_jobs=rc.N_JOBS)

    # SVD; sort singular values increasing for the plot.
    _, svals, Vh = np.linalg.svd(X, full_matrices=False)
    sigma_sorted = np.sort(svals)            # ascending
    sigma1 = float(sigma_sorted[0])
    sigma2 = float(sigma_sorted[1])
    h_hat = Vh[-1]
    h_hat = h_hat / np.linalg.norm(h_hat)
    if np.dot(h_hat, true_h) < 0:
        h_hat = -h_hat
    overlap = float(abs(np.dot(h_hat, true_h)))

    residual = float(np.linalg.norm(X @ true_h))
    ratio = sigma2 / sigma1 if sigma1 > 0 else float("inf")

    # ────── outputs ──────
    out_dir = THIS.parent
    csv_path = out_dir / "tfim_singular_spectrum.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["j", "sigma"])
        for j, s in enumerate(sigma_sorted, start=1):
            w.writerow([j, f"{s:.10e}"])
    print(f"[csv] {csv_path}")

    meta_path = out_dir / "tfim_singular_spectrum_meta.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in [
            ("n", n), ("t", t), ("n_probes", n_probes),
            ("|V|", len(labels)),
            ("sigma_1", sigma1), ("sigma_2", sigma2),
            ("ratio_sigma2_sigma1", ratio),
            ("residual_X_h", residual),
            ("overlap_hhat_h", overlap),
            ("seed_instance", seed_inst), ("seed_probes", seed_probes),
        ]:
            w.writerow([k, v])
    print(f"[csv] {meta_path}")

    print()
    print("=" * 56)
    print(f" TFIM singular spectrum  n={n}  t={t}  N_S={n_probes}  |V|={len(labels)}")
    print("=" * 56)
    print(f" sigma_1               = {sigma1:.6e}")
    print(f" sigma_2               = {sigma2:.6e}")
    print(f" sigma_2/sigma_1       = {ratio:.6e}")
    print(f" ||X h||  (h raw, unit) = {residual:.6e}")
    print(f" overlap |<hhat, h>|    = {overlap:.6f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
