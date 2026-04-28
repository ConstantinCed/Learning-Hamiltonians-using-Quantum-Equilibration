#!/usr/bin/env python3
"""
Weak equilibration decay: self-contained numerics for random dense
nearest-neighbor 2-local Hamiltonians.

Computes and plots the weak self-correlation (autocorrelation) bound

  M_self(t) = sup_{|x|=1, x real} |x^T K(t) x|

where K_{ab}(t) = (1/d) tr( E_a(t) E_b ) and {E_a} is an HS-orthonormal
basis of V = W \cap H^perp. W is the space of 1-local + nearest-neighbor
2-local Hermitian Pauli strings on n qubits.

This single script is fully self-contained: no external module
dependencies beyond numpy, scipy, matplotlib, pandas. Outputs land in
the script's own folder.

Usage:
    python weak_equilibration_decay.py
"""

# CRITICAL: prevent BLAS thread oversubscription when using multiprocessing.
# Must be set BEFORE importing numpy.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as iprod
from scipy.linalg import eigh
from multiprocessing import Pool

# ======================================================================
# Configuration
# ======================================================================
N_VALUES     = [6, 7, 8, 9, 10, 11, 12]
REALIZATIONS = {6: 30, 7: 30, 8: 20, 9: 15, 10: 10, 11: 5, 12: 3}
WORKERS      = {6: 14, 7: 14, 8: 12, 9: 8,  10: 4,  11: 2, 12: 1}

T_MAX     = 400.0
DT        = 2.0
BASE_SEED = 20240101
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# Pauli string descriptors (lightweight, no full d x d matrices)
# ======================================================================

def _make_pauli_descriptor(n, site_pauli_pairs):
    """
    Create a Pauli string descriptor: (flip_mask, phases).

    For a Pauli string P, acting on basis state |s>:
      P|s> = phases[s] * |s XOR flip_mask>

    This allows O(d) application without building the d x d matrix.
    """
    d = 2 ** n
    flip_mask = 0
    phases = np.ones(d, dtype=np.complex128)
    states = np.arange(d)

    for site, pauli in site_pauli_pairs:
        bit_pos = n - 1 - site
        bit_mask = 1 << bit_pos
        bit_vals = (states >> bit_pos) & 1

        if pauli == 'X':
            flip_mask ^= bit_mask
        elif pauli == 'Y':
            flip_mask ^= bit_mask
            phases *= np.where(bit_vals == 0, 1j, -1j)
        elif pauli == 'Z':
            phases *= 1 - 2 * bit_vals  # +1 for 0, -1 for 1

    return flip_mask, phases


def build_pauli_descriptors(n):
    """
    Build lightweight descriptors for all W basis Pauli strings
    (1-local + nearest-neighbor 2-local on an open chain of n qubits).

    Each descriptor enables O(d) matrix-vector application instead of
    storing the full d x d matrix -- critical for n >= 11.
    """
    descriptors = []
    labels = []
    # 1-local terms
    for i in range(n):
        for mu in ['X', 'Y', 'Z']:
            descriptors.append(_make_pauli_descriptor(n, [(i, mu)]))
            labels.append(f"{mu}_{i}")
    # 2-local nearest-neighbor terms
    for i in range(n - 1):
        for mu, nu in iprod(['X', 'Y', 'Z'], repeat=2):
            descriptors.append(_make_pauli_descriptor(n, [(i, mu), (i + 1, nu)]))
            labels.append(f"{mu}{nu}_{i},{i+1}")
    return descriptors, labels


def apply_pauli_vec(v, descriptor):
    """Apply Pauli string to vector v: result = P @ v, in O(d)."""
    flip_mask, phases = descriptor
    result = np.empty_like(v)
    indices = np.arange(len(v)) ^ flip_mask
    result[indices] = phases * v
    return result


# ======================================================================
# Hamiltonian and V-basis construction (memory-efficient)
# ======================================================================

def build_hamiltonian(n, rng):
    """
    Build a random dense nearest-neighbor 2-local Hamiltonian:
        H = sum_{i,mu} h_{i,mu} sigma_i^mu
          + sum_{i,mu,nu} J_{i,mu,nu} sigma_i^mu sigma_{i+1}^nu
    with all coefficients i.i.d. N(0,1).

    Uses O(d) per basis element via Pauli descriptors instead of
    storing full d x d basis matrices.
    """
    descriptors, labels = build_pauli_descriptors(n)
    d = 2 ** n
    m = len(descriptors)
    coeffs = rng.standard_normal(m)
    H = np.zeros((d, d), dtype=np.complex128)
    arange_d = np.arange(d)
    for a in range(m):
        flip_mask, phases = descriptors[a]
        targets = arange_d ^ flip_mask
        H[targets, arange_d] += coeffs[a] * phases
    assert np.max(np.abs(H - H.conj().T)) < 1e-12, "H is not Hermitian"
    return H, coeffs, descriptors, labels


def build_V_coeffs(coeffs):
    """
    Compute V_coeffs: orthonormal basis for V = W \cap H^perp in
    coefficient space. V_coeffs has shape (m, m-1).
    """
    m = len(coeffs)
    c = np.array(coeffs, dtype=np.float64)
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-14:
        raise ValueError("Hamiltonian has near-zero norm in W")
    h_hat = c / c_norm
    P = np.eye(m) - np.outer(h_hat, h_hat)
    U_svd, S, _ = np.linalg.svd(P, full_matrices=False)
    keep = S > 1e-10
    V_coeffs = U_svd[:, keep]  # (m, m-1)
    dim_V = V_coeffs.shape[1]
    assert dim_V == m - 1, f"Expected dim(V)={m-1}, got {dim_V}"
    return V_coeffs


# ======================================================================
# Core computation
# ======================================================================

def diagonalize_hamiltonian(H):
    """Diagonalize Hermitian H = U diag(eigvals) U^dagger."""
    eigvals, U = eigh(H)
    return eigvals, U


def compute_all_K(descriptors, V_coeffs, eigvals, U, t_array, d):
    """
    Compute K(t) for ALL time points simultaneously.
    K_{ab}(t) = (1/d) sum_j e^{i lam_j t}
                       sum_k e^{-i lam_k t} \tilde{E}_a_{jk} conj(\tilde{E}_b_{jk})

    Row j of each \tilde{W}_a is computed on-the-fly via Pauli string
    application:
      Z[:, a] = W_a v_j         (O(d) per basis element)
      w_j     = Z^dagger U      (one matmul per j)
      V_j     = V^T w_j         (project to V subspace)
    then accumulate K.
    """
    m = len(descriptors)
    dim_V = V_coeffs.shape[1]
    n_times = len(t_array)
    VT = V_coeffs.T  # (dim_V, m)

    u_all      = np.exp(1j * np.outer(t_array, eigvals))  # (n_times, d)
    conj_u_all = np.conj(u_all)                            # (n_times, d)

    K_all = np.zeros((n_times, dim_V, dim_V), dtype=np.complex128)
    P_buf = np.empty((n_times, dim_V, d), dtype=np.complex128)
    Z = np.empty((d, m), dtype=np.complex128)

    for j in range(d):
        v_j = U[:, j]
        for a in range(m):
            Z[:, a] = apply_pauli_vec(v_j, descriptors[a])

        w_j = Z.conj().T @ U  # (m, d)
        V_j = VT @ w_j        # (dim_V, d)

        np.multiply(V_j, conj_u_all[:, None, :], out=P_buf)
        V_j_conj_T = V_j.conj().T                 # (d, dim_V)
        C = np.matmul(P_buf, V_j_conj_T)          # (n_times, dim_V, dim_V)
        K_all += u_all[:, j:j+1, None] * C

    K_all /= d
    return K_all


def compute_M_self(K):
    """
    M_self(t) = sup_{|x|=1, x real} |x^T K(t) x|.

    Equals spectral radius of Re(K_S) where K_S = (K + K^T)/2,
    since Im(K_S) = 0 to machine precision for this Hamiltonian class.
    """
    K_S = (K + K.T) / 2
    K_S_re = np.real(K_S)
    K_S_im = np.imag(K_S)

    im_norm = np.max(np.abs(K_S_im))
    re_norm = np.max(np.abs(K_S_re))

    if im_norm < 1e-10 * max(re_norm, 1e-15):
        eigvals_re = np.linalg.eigvalsh(K_S_re)
        return max(abs(eigvals_re[0]), abs(eigvals_re[-1]))

    import warnings
    warnings.warn(
        f"Im(K_S) non-negligible: |Im|/|Re| = {im_norm/max(re_norm,1e-15):.2e}. "
        f"Using eigenvalue fallback."
    )
    eigvals_re = np.linalg.eigvalsh(K_S_re)
    return max(abs(eigvals_re[0]), abs(eigvals_re[-1]))


def compute_single_realization(n, t_array, seed):
    """
    Full pipeline for one Hamiltonian realization:
      1. Build H, Pauli descriptors, V coefficients
      2. Diagonalize H
      3. Compute K(t) on the time grid, extract M_gen(t), M_self(t)
    """
    rng = np.random.default_rng(seed)
    d = 2 ** n

    H, coeffs, descriptors, labels = build_hamiltonian(n, rng)
    m = len(descriptors)
    V_coeffs = build_V_coeffs(coeffs)
    dim_V = V_coeffs.shape[1]

    VTV = V_coeffs.T @ V_coeffs
    v_orth_err = np.max(np.abs(VTV - np.eye(dim_V)))

    h_proj_err = np.max(np.abs(V_coeffs.T @ coeffs))
    if h_proj_err > 1e-8:
        raise ValueError(f"H not projected out: max overlap = {h_proj_err:.2e}")

    eigvals, U = diagonalize_hamiltonian(H)
    del H

    K_all = compute_all_K(descriptors, V_coeffs, eigvals, U, t_array, d)
    del U

    n_times = len(t_array)
    M_self_array = np.empty(n_times)
    for ti in range(n_times):
        M_self_array[ti] = compute_M_self(K_all[ti])

    checks = {
        'w_orth_err': 0.0,
        'v_orth_err': v_orth_err,
        'h_proj_err': h_proj_err,
        'M_self_0':   M_self_array[0],
        'M_self_max': np.max(M_self_array),
        'dim_V':      dim_V,
    }
    return M_self_array, checks


# ======================================================================
# Runner
# ======================================================================

def make_time_grid():
    return np.arange(0.0, T_MAX + DT / 2, DT)


def _worker(args):
    n, t_array, seed = args
    try:
        M_self, checks = compute_single_realization(n, t_array, seed)
        return n, seed, M_self, checks, None
    except Exception:
        import traceback
        return n, seed, None, None, traceback.format_exc()


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def run_all():
    t_array = make_time_grid()
    all_results = {}

    flush_print(f"Time points: {len(t_array)}")
    flush_print(f"N_VALUES: {N_VALUES}")
    flush_print(f"REALIZATIONS: {REALIZATIONS}\n")

    for n in N_VALUES:
        n_real = REALIZATIONS[n]
        tasks = [(n, t_array, BASE_SEED + n * 10000 + r) for r in range(n_real)]
        n_workers = min(WORKERS[n], n_real)

        flush_print(f"=== n={n}, d={2**n}, realizations={n_real}, "
                    f"workers={n_workers} ===")
        t0 = time.time()

        M_self_all = []

        with Pool(n_workers) as pool:
            for idx, result in enumerate(pool.imap_unordered(_worker, tasks)):
                _, seed, M_self, checks, err = result
                elapsed = time.time() - t0
                if err is not None:
                    flush_print(f"  [{idx+1}/{n_real}] seed={seed} "
                                f"ERROR ({elapsed:.1f}s):\n{err}")
                    continue

                ok = True
                if abs(checks["M_self_0"] - 1.0) > 1e-6:
                    flush_print(f"  WARNING: M_self(0)={checks['M_self_0']:.8f} != 1")
                    ok = False
                if checks["M_self_max"] > 1.0 + 1e-6:
                    flush_print(f"  WARNING: M_self max={checks['M_self_max']:.8f} > 1")
                    ok = False

                status = "OK" if ok else "WARN"
                flush_print(f"  [{idx+1}/{n_real}] seed={seed} {status} "
                            f"({elapsed:.1f}s)")

                M_self_all.append(M_self)

        wall = time.time() - t0
        all_results[n] = {
            "M_self": np.array(M_self_all),
        }
        flush_print(f"  n={n} done: {len(M_self_all)} realizations "
                    f"in {wall:.1f}s\n")

    return t_array, all_results


def save_csv(t_array, all_results):
    rows = []
    for n in N_VALUES:
        if n not in all_results or all_results[n]["M_self"].size == 0:
            continue
        B = all_results[n]["M_self"]
        n_real = B.shape[0]

        B_mean = np.mean(B, axis=0)
        B_std  = np.std(B, axis=0, ddof=1) if n_real > 1 else np.zeros_like(B_mean)
        B_med  = np.median(B, axis=0)
        B_q25  = np.percentile(B, 25, axis=0)
        B_q75  = np.percentile(B, 75, axis=0)

        for i, t in enumerate(t_array):
            rows.append({
                "n": n, "t": t, "observable": "autocorrelation",
                "mean": B_mean[i], "std": B_std[i],
                "median": B_med[i], "q25": B_q25[i], "q75": B_q75[i],
                "n_realizations": n_real,
            })

    path = os.path.join(OUTPUT_DIR, "autocorrelation.csv")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10g")
    flush_print(f"Saved {path}")


def make_plots(t_array, all_results):
    colors = {
        6: "#1f77b4", 7: "#ff7f0e", 8: "#2ca02c", 9: "#d62728",
        10: "#9467bd", 11: "#8c564b", 12: "#e377c2",
    }

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'legend.fontsize': 9,
        'mathtext.fontset': 'cm',
        'figure.dpi': 150,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in N_VALUES:
        if n not in all_results or all_results[n]["M_self"].size == 0:
            continue
        data = all_results[n]["M_self"]
        mean = np.mean(data, axis=0)
        ax.plot(t_array, mean, color=colors[n], lw=1.4, label=f"$n={n}$")
        if data.shape[0] > 1:
            std = np.std(data, axis=0, ddof=1)
            ax.fill_between(t_array, mean - std, mean + std,
                            color=colors[n], alpha=0.15, linewidth=0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"autocorrelation $M_{\mathrm{self}}(t)$")
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "autocorrelation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    flush_print(f"Saved {path}")
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    flush_print("=" * 70)
    flush_print("Weak equilibration decay -- self-contained runner")
    flush_print("=" * 70)
    flush_print(f"Python {sys.version}")
    flush_print(f"NumPy  {np.__version__}")
    flush_print(f"Output -> {OUTPUT_DIR}/")
    flush_print()

    t0 = time.time()
    t_array, all_results = run_all()
    flush_print(f"Total compute: {time.time() - t0:.1f}s\n")

    save_csv(t_array, all_results)
    flush_print()
    make_plots(t_array, all_results)
    flush_print(f"\nAll done in {time.time() - t0:.1f}s total.")
