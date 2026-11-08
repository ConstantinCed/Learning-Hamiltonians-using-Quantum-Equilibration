"""Compute the open-chain TFIM static gap for Supplementary Fig. S4b.

Run with --distribution gaussian to reproduce the figure CSV and its fit.
The bond and field coefficients are independent draws without rescaling.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal


OUTDIR = Path(__file__).resolve().parent


def commutator_columns(J, h):
    """Return the two signed entries in each column of the commutator map.

    In the ordering X_1, Z_1 Z_2, X_2, ..., X_n, each neighboring pair
    contributes one orthogonal Pauli string. For coefficients c of H and a
    of A, its amplitude, up to an overall phase, is
    2 * (c[i+1] * a[i] - c[i] * a[i+1]). Hence V.T @ c = 0.
    """
    coefficients = np.empty(len(J) + len(h))
    coefficients[::2] = h
    coefficients[1::2] = J
    return 2 * coefficients[1:], -2 * coefficients[:-1]


def compute_Q(n, J, h):
    """Build the normalized Hilbert--Schmidt Gram matrix, bonds before fields."""
    top, bottom = commutator_columns(J, h)
    columns = np.arange(2 * n - 2)
    V = np.zeros((2 * n - 1, len(columns)))
    V[columns, columns] = top
    V[columns + 1, columns] = bottom
    V = np.concatenate((V[1::2], V[::2]))
    return V @ V.T


def constrained_min_eig(Q, g):
    """Restrict a dense Gram matrix to the orthogonal complement of g."""
    if not np.any(g):
        return float(np.linalg.eigvalsh(Q)[0])
    basis = np.linalg.qr(np.asarray(g)[:, None], mode="complete")[0][:, 1:]
    return float(np.linalg.eigvalsh(basis.T @ Q @ basis)[0])


def commutator_gap(J, h):
    """Compute min_{A perpendicular to H, ||A||_HS=1} ||[A,H]||_HS^2."""
    if len(h) < 2 or len(J) != len(h) - 1:
        raise ValueError("An open chain needs n >= 2 fields and n - 1 bonds.")
    top, bottom = commutator_columns(J, h)
    # V has one more row than column and V.T annihilates H. Thus the
    # restricted gap is its smallest singular value squared, including zero
    # if there is another conserved direction. The symmetric block matrix
    # [0, V; V.T, 0] is tridiagonal after interleaving its rows and columns.
    # Its eigenvalues are the signed singular values and one central zero.
    off_diagonal = np.empty(2 * len(top))
    off_diagonal[::2] = top
    off_diagonal[1::2] = bottom
    index = len(top) + 1
    smallest_singular_value = eigvalsh_tridiagonal(
        np.zeros(len(off_diagonal) + 1), off_diagonal,
        select="i", select_range=(index, index), check_finite=False,
    )[0]
    return float(smallest_singular_value ** 2)


def sample_inf_value_unnormalized(n, rng, distribution):
    if distribution == "gaussian":
        J = rng.normal(size=n - 1)
        h = rng.normal(size=n)
    elif distribution == "uniform":
        J = rng.uniform(-1.0, 1.0, size=n - 1)
        h = rng.uniform(-1.0, 1.0, size=n)
    else:
        raise ValueError("distribution must be either 'gaussian' or 'uniform'")
    return commutator_gap(J, h)


def run_experiment(ns, num_samples, seed, distribution, outdir=OUTDIR):
    """Save the mean, sample standard deviation and SEM at each chain size."""
    rng = np.random.default_rng(seed)
    ns = list(ns)
    means = np.empty(len(ns))
    stds = np.empty(len(ns))
    for i, n in enumerate(ns):
        gaps = [sample_inf_value_unnormalized(n, rng, distribution)
                for _ in range(num_samples)]
        means[i] = np.mean(gaps)
        stds[i] = np.std(gaps, ddof=1)
        print(f"n={n}, {distribution}: mean gap {means[i]:.8g}", flush=True)
    sems = stds / np.sqrt(num_samples)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"tfim_data_unnormalized_{distribution}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["n", "m=2n-1", "distribution", "mean", "std", "sem", "num_samples"])
        for n, mean, std, sem in zip(ns, means, stds, sems):
            writer.writerow([n, 2*n-1, distribution, f"{mean:.10e}",
                             f"{std:.10e}", f"{sem:.10e}", num_samples])
    return csv_path, means, stds, sems


def write_scaling_fit(ns, means, path):
    """Fit log(mean gap) = log(C) + alpha log(n) + beta log(log(n))."""
    log_n = np.log(ns)
    design = np.column_stack((np.ones(len(ns)), log_n, np.log(log_n)))
    log_c, alpha, beta = np.linalg.lstsq(design, np.log(means), rcond=None)[0]
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["C", "alpha", "beta"])
        writer.writerow([np.exp(log_c), alpha, beta])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=["gaussian", "uniform", "both"], default="both")
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()
    sizes = range(4, 201)
    for law, seed in (("gaussian", 42), ("uniform", 43)):
        if args.distribution in (law, "both"):
            csv_path, means, _, _ = run_experiment(sizes, 1000, seed, law, args.outdir)
            write_scaling_fit(sizes, means, csv_path.with_name(f"tfim_gap_fit_{law}.csv"))
