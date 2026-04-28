"""Figure: smallest two singular values of the learning matrix X
versus the number of product-state probes N_S, for an open-boundary TFIM.

Model (NOT rescaled):
    H = sum_i g_i X_i  +  sum_i J_i Z_i Z_{i+1},   g_i, J_i ~ N(0, 1).

Search family:
    V = {X_i}_{i=1..n}  union  {Z_i Z_{i+1}}_{i=1..n-1}.

For a fixed instance and a single product-probe pool of size N_max,
we sweep N_S and, for each prefix of N_S probes, build the noiseless
feature matrix and report sigma_1 (smallest) and sigma_2 (second
smallest) of X.

Outputs (next to this script):
    tfim_singular_spectrum.csv       columns: N_S, sigma_1, sigma_2
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
    seed_inst, seed_probes = 123, 456

    # Sweep grid for N_S: log-spaced, starting at |V| so X has full column rank.
    Ns_grid = [20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]
    N_max = max(Ns_grid)

    rng_i = np.random.default_rng(seed_inst)
    rng_p = np.random.default_rng(seed_probes)

    labels, names = family_labels(n)
    coeffs = rng_i.standard_normal(len(labels))   # raw, unnormalized

    H_true, true_h, descriptors = rc.build_hamiltonian(labels, coeffs)
    U = expm(-1j * t * H_true)

    # Single big pool of probes; subsample prefixes for each N_S.
    probes = rc.sample_product_probes(n, N_max, rng_p)
    X_full = rc.build_feature_matrix_exact(U, probes, labels, descriptors,
                                           n_jobs=rc.N_JOBS)
    print(f"[ok] full feature matrix built, shape {X_full.shape}")

    rows = []
    for NS in Ns_grid:
        X = X_full[:NS]
        s = np.linalg.svd(X, compute_uv=False)
        s_asc = np.sort(s)
        sigma_1 = float(s_asc[0])
        sigma_2 = float(s_asc[1])
        rows.append((NS, sigma_1, sigma_2))
        print(f"  N_S={NS:>4d}   sigma_1={sigma_1:.4e}   "
              f"sigma_2={sigma_2:.4e}   ratio={sigma_2/sigma_1:.3e}"
              if sigma_1 > 0 else f"  N_S={NS}   sigma_1=0")

    # Final-instance overlap diagnostic at N_max.
    _, svals_full, Vh_full = np.linalg.svd(X_full, full_matrices=False)
    h_hat = Vh_full[-1] / np.linalg.norm(Vh_full[-1])
    if np.dot(h_hat, true_h) < 0:
        h_hat = -h_hat
    overlap = float(abs(np.dot(h_hat, true_h)))
    residual = float(np.linalg.norm(X_full @ true_h))

    out_dir = THIS.parent
    csv_path = out_dir / "tfim_singular_spectrum.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N_S", "sigma_1", "sigma_2"])
        for NS, s1, s2 in rows:
            w.writerow([NS, f"{s1:.10e}", f"{s2:.10e}"])
    print(f"[csv] {csv_path}")

    meta_path = out_dir / "tfim_singular_spectrum_meta.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in [
            ("n", n), ("t", t),
            ("|V|", len(labels)),
            ("N_max", N_max),
            ("residual_X_h_at_Nmax", residual),
            ("overlap_hhat_h_at_Nmax", overlap),
            ("seed_instance", seed_inst), ("seed_probes", seed_probes),
        ]:
            w.writerow([k, v])
    print(f"[csv] {meta_path}")

    print("\n" + "=" * 56)
    print(f" overlap |<hhat,h>|  at N_S={N_max}: {overlap:.6f}")
    print(f" ||X h||              at N_S={N_max}: {residual:.3e}")
    print("=" * 56)


if __name__ == "__main__":
    main()
