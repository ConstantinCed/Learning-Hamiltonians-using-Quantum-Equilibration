"""XYZ (Heisenberg) Hamiltonian reconstruction at n=10.

Family:    H = sum_{i, P in {X,Y,Z}} c_{i,P} P_i P_{i+1}    (|V| = 3(n-1))
Sampling:  c_{i,P} ~ N(0, 1)  (H is NOT normalized)
Shadow:    3 measurement settings (global X, Y, Z bases; one third of the
           shots each).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import recon_common as rc


# ──────────────────────────────────────────────────
#  Model definition
# ──────────────────────────────────────────────────
def family_labels(n):
    labels, names = [], []
    for i in range(n - 1):
        for P in ["X", "Y", "Z"]:
            labels.append(rc.two_site_label(n, i, i + 1, P, P))
            names.append(f"{P}{i}{P}{i+1}")
    return labels, names


def random_instance(n, rng):
    labels, names = family_labels(n)
    coeffs = rng.normal(size=len(labels))
    return labels, names, coeffs


def estimate_shadow(psi_t, n, n_shots, rng):
    """Three fixed bases (X, Y, Z), each gets ~ n_shots / 3 shots.
    For each bond i, basis k in {0,1,2} ↔ XX/YY/ZZ correlator."""
    d = 2 ** n
    bit_shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    m = 3 * (n - 1)
    est = np.zeros(m)

    base = n_shots // 3
    rem = n_shots - 3 * base
    counts = [base + (1 if k < rem else 0) for k in range(3)]

    for k in range(3):  # 0=X, 1=Y, 2=Z
        U_rot = rc.rotation_unitary(tuple([k] * n))
        phi = U_rot @ psi_t
        prob = phi.real ** 2 + phi.imag ** 2
        prob /= prob.sum()

        out = rng.choice(d, size=counts[k], p=prob)
        bits = ((out[:, None] >> bit_shifts[None, :]) & 1).astype(np.float64)
        s = 1.0 - 2.0 * bits
        for i in range(n - 1):
            est[3 * i + k] = np.mean(s[:, i] * s[:, i + 1])

    return est


# ──────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────
def run_trial(n, t, n_probes, shots_per_probe,
              seed_instance=0, seed_probes=1, seed_shadows=2,
              n_jobs=rc.N_JOBS):
    return rc.run_trial(
        n=n, t=t,
        random_instance_fn=random_instance,
        estimate_shadow_fn=estimate_shadow,
        n_probes=n_probes, shots_per_probe=shots_per_probe,
        seed_instance=seed_instance, seed_probes=seed_probes,
        seed_shadows=seed_shadows, n_jobs=n_jobs,
    )


# ──────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    n = 10
    seed_inst = 123

    rng_t = np.random.default_rng(seed_inst + 7)
    t = float(rng_t.uniform(1000.0, 2000.0))

    n_probes = 1000
    shots = 50_000

    print(f"\n{'='*56}")
    print(f" XYZ reconstruction  n={n}  |V|={3*(n-1)}")
    print(f" N_S={n_probes}  nu={shots}  t={t:.4f}")
    print(f" N_JOBS={rc.N_JOBS}")
    print(f"{'='*56}\n")

    res = run_trial(n=n, t=t, n_probes=n_probes, shots_per_probe=shots,
                    seed_instance=seed_inst, seed_probes=456, seed_shadows=789)
    rc.print_summary(res, shots)

    out_csv = Path(__file__).resolve().parent / f"xyz_n{n}_reconstruction.csv"
    rc.save_reconstruction_csv(res, out_csv)
