"""Compute the smoothed TFIM commutator gap with symbolic Pauli arithmetic."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
from pathlib import Path

import numpy as np


def label_to_xzc(label):
    n = len(label)
    x = 0
    z = 0
    c = 1.0 + 0.0j
    for i, ch in enumerate(label):
        bit = 1 << (n - 1 - i)
        if ch == "X":
            x |= bit
        elif ch == "Z":
            z |= bit
        elif ch == "Y":
            x |= bit
            z |= bit
            c *= 1j
        elif ch == "I":
            pass
        else:
            raise ValueError(ch)
    return x, z, c


def popcount(b):
    return bin(b).count("1")


def commutator_pauli(P_a, P_c):
    """Return [P_a, P_c] as a list of (x, z, complex_coef) terms (0 or 1)."""
    xa, za, ca = P_a
    xc, zc, cc = P_c
    sab = (-1) ** popcount(za & xc)
    sba = (-1) ** popcount(zc & xa)
    if sab == sba:
        return []
    x = xa ^ xc
    z = za ^ zc
    coef = (sab - sba) * ca * cc        # = 2 * sab * ca*cc
    return [(x, z, coef)]


def add_term(d, key, val):
    if key in d:
        d[key] += val
        if abs(d[key]) < 1e-15:
            del d[key]
    else:
        if abs(val) >= 1e-15:
            d[key] = val


def commutator_with_H(P_a, H_terms):
    """[P_a, H] for H = sum_c h_c P_c (h_c can be complex)."""
    out = {}
    for h_c, P_c in H_terms:
        if h_c == 0.0:
            continue
        for x, z, coef in commutator_pauli(P_a, P_c):
            add_term(out, (x, z), h_c * coef)
    return out


def hs_inner(D1, D2):
    """<L1, L2>_HS = 2^{-n} tr(L1^dagger L2) = sum_k conj(c1_k) * c2_k."""
    if len(D1) > len(D2):
        D1, D2 = D2, D1
        conj_first = False
    else:
        conj_first = True
    s = 0.0 + 0.0j
    for k, c1 in D1.items():
        c2 = D2.get(k)
        if c2 is None:
            continue
        if conj_first:
            s += np.conj(c1) * c2
        else:
            s += c1 * np.conj(c2)
    return s


def build_search_basis(n):
    """Return the X_i and Z_i Z_{i+1} search basis."""
    labels = []
    for i in range(n):
        s = ["I"] * n
        s[i] = "X"
        labels.append("".join(s))
    for i in range(n - 1):
        s = ["I"] * n
        s[i] = "Z"
        s[i + 1] = "Z"
        labels.append("".join(s))
    P = [label_to_xzc(lbl) for lbl in labels]
    return labels, P


def gamma_matrix(P_basis, h_vec):
    """Build the commutator Gram matrix for H."""
    H_terms = [(complex(h_vec[i]), P_basis[i]) for i in range(len(P_basis))]
    Ls = [commutator_with_H(P_a, H_terms) for P_a in P_basis]
    m = len(P_basis)
    G = np.zeros((m, m), dtype=float)
    for a in range(m):
        for b in range(a, m):
            v = hs_inner(Ls[a], Ls[b])
            G[a, b] = 0.25 * float(np.real(v))
            G[b, a] = G[a, b]
    return G


def projected_min_eig(G, h_vec):
    h = np.asarray(h_vec, dtype=float)
    nh = np.linalg.norm(h)
    m = G.shape[0]
    if nh < 1e-15:
        return float(np.linalg.eigvalsh(G)[0])
    _, _, Vh = np.linalg.svd(h.reshape(1, -1), full_matrices=True)
    Q = Vh[1:].T
    Gp = Q.T @ G @ Q
    return float(np.linalg.eigvalsh(Gp)[0])


def main():
    n_list = [10, 20, 30, 40, 50]
    eps_grid = np.round(np.linspace(-1.0, 1.0, 41), 6).tolist()
    n_samples = 30          # for eps != 0
    seed = 20250428

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "tfim_smoothed_gap.csv"
    qc_path = out_dir / "tfim_smoothed_gap_qc.txt"

    rows = []
    qc_lines = []
    for n in n_list:
        labels, P_basis = build_search_basis(n)
        m = len(P_basis)
        idx_X = list(range(0, n))
        idx_ZZ = list(range(n, m))
        h_zz = np.zeros(m); h_zz[idx_ZZ] = 1.0
        G0 = gamma_matrix(P_basis, h_zz)
        qc_lines.append(
            f"n={n}: ||Gamma(H0) h0||_inf = {np.max(np.abs(G0 @ h_zz)):.3e}"
            f"   pi_proj(H0) = {projected_min_eig(G0, h_zz):.3e}"
        )

        rng = np.random.default_rng(seed + n)
        print(f"\n[n={n}] |V_search|={m}")
        for eps in eps_grid:
            if eps == 0.0:
                pi = projected_min_eig(G0, h_zz)
                rows.append([n, eps, pi, 1])
                print(f"  eps={eps:>+8.4f}  pi_mean={pi:+.3e}  N=1 (det.)")
                continue
            vals = []
            for s in range(n_samples):
                g = rng.standard_normal(n)
                h_vec = np.zeros(m)
                h_vec[idx_X] = eps * g
                h_vec[idx_ZZ] = 1.0
                G = gamma_matrix(P_basis, h_vec)
                vals.append(projected_min_eig(G, h_vec))
            vals = np.asarray(vals)
            mean = float(np.mean(vals))
            rows.append([n, eps, mean, n_samples])
            print(f"  eps={eps:>+8.4f}  pi_mean={mean:.3e}  N={n_samples}")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "eps", "mean", "n_samples"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.6e}", f"{r[2]:.10e}", r[3]])
    print(f"\n[csv] {csv_path}")

    with open(qc_path, "w") as f:
        f.write("\n".join(qc_lines) + "\n")
    print(f"[qc]  {qc_path}")
    print("\n".join(qc_lines))


if __name__ == "__main__":
    main()
