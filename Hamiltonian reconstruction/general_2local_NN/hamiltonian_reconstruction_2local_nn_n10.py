"""General 2-local nearest-neighbor Hamiltonian reconstruction at n=10.

Family:    H = sum_{i, P,Q in {X,Y,Z}} c_{i,P,Q} P_i Q_{i+1}   (|V| = 9(n-1))
Sampling:  c_{i,P,Q} ~ N(0, 1)  (H is NOT normalized)
Shadow:    9 measurement settings (one per (P,Q) pair). In setting (P,Q),
           even-indexed qubits are measured in basis P, odd-indexed in Q;
           together the 9 settings cover all 9*(n-1) correlators.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import recon_common as rc

PAULI_NAMES = ["X", "Y", "Z"]


# ──────────────────────────────────────────────────
#  Model definition
# ──────────────────────────────────────────────────
def family_labels(n):
    labels, names = [], []
    for i in range(n - 1):
        for P in PAULI_NAMES:
            for Q in PAULI_NAMES:
                labels.append(rc.two_site_label(n, i, i + 1, P, Q))
                names.append(f"{P}{i}{Q}{i+1}")
    return labels, names


def random_instance(n, rng):
    labels, names = family_labels(n)
    coeffs = rng.normal(size=len(labels))
    return labels, names, coeffs


def _precompute_rotation_matrices(n):
    rots = {}
    for p in range(3):
        for q in range(3):
            bases = tuple(p if k % 2 == 0 else q for k in range(n))
            rots[(p, q)] = rc.rotation_unitary(bases)
    return rots


def estimate_shadow(psi_t, n, n_shots, rng):
    """9 fixed measurement settings (one per (P,Q) pair); ~n_shots/9 shots each."""
    d = 2 ** n
    bit_shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    m = 9 * (n - 1)
    est = np.zeros(m)

    rots = _precompute_rotation_matrices(n)

    base = n_shots // 9
    rem = n_shots - 9 * base

    setting_idx = 0
    for p in range(3):
        for q in range(3):
            n_s = base + (1 if setting_idx < rem else 0)
            setting_idx += 1

            phi = rots[(p, q)] @ psi_t
            prob = phi.real ** 2 + phi.imag ** 2
            prob /= prob.sum()

            out = rng.choice(d, size=n_s, p=prob)
            bits = ((out[:, None] >> bit_shifts[None, :]) & 1).astype(np.float64)
            s = 1.0 - 2.0 * bits

            for i in range(n - 1):
                # In setting (p,q): even-i bond measures P_iQ_{i+1};
                # odd-i bond measures Q_iP_{i+1}.
                if i % 2 == 0:
                    cp, cq = p, q
                else:
                    cp, cq = q, p
                est[9 * i + cp * 3 + cq] = np.mean(s[:, i] * s[:, i + 1])

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
    shots = 20_000

    print(f"\n{'='*56}")
    print(f" General 2-local NN reconstruction  n={n}  |V|={9*(n-1)}")
    print(f" N_S={n_probes}  nu={shots}  t={t:.4f}")
    print(f" N_JOBS={rc.N_JOBS}")
    print(f"{'='*56}\n")

    res = run_trial(n=n, t=t, n_probes=n_probes, shots_per_probe=shots,
                    seed_instance=seed_inst, seed_probes=456, seed_shadows=789)
    rc.print_summary(res, shots)

    out_csv = (Path(__file__).resolve().parent
               / f"general_2local_nn_n{n}_reconstruction.csv")
    rc.save_reconstruction_csv(res, out_csv)
