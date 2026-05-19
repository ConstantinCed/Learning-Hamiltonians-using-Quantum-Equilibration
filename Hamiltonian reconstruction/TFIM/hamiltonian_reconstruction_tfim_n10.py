"""Run n=10 TFIM Hamiltonian reconstruction with two fixed shadow bases."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import recon_common as rc


def family_labels(n):
    labels, names = [], []
    for i in range(n):
        labels.append(rc.single_site_label(n, i, "X"))
        names.append(f"X_{i}")
    for i in range(n - 1):
        labels.append(rc.two_site_label(n, i, i + 1, "Z", "Z"))
        names.append(f"Z{i}Z{i+1}")
    return labels, names


def random_instance(n, rng):
    labels, names = family_labels(n)
    coeffs = rng.uniform(-1.0, 1.0, size=2 * n - 1)
    return labels, names, coeffs


def estimate_shadow(psi_t, n, n_shots, rng):
    """Two fixed bases: half the shots in the global X basis, half in Z."""
    d = 2 ** n
    bit_shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    m = 2 * n - 1
    est = np.zeros(m)

    n_x = n_shots // 2
    n_z = n_shots - n_x

    U_x = rc.rotation_unitary(tuple([0] * n))
    phi = U_x @ psi_t
    p_x = phi.real ** 2 + phi.imag ** 2
    p_x /= p_x.sum()
    out = rng.choice(d, size=n_x, p=p_x)
    bits = ((out[:, None] >> bit_shifts[None, :]) & 1).astype(np.float64)
    s_x = 1.0 - 2.0 * bits
    for i in range(n):
        est[i] = np.mean(s_x[:, i])

    p_z = psi_t.real ** 2 + psi_t.imag ** 2
    p_z /= p_z.sum()
    out = rng.choice(d, size=n_z, p=p_z)
    bits = ((out[:, None] >> bit_shifts[None, :]) & 1).astype(np.float64)
    s_z = 1.0 - 2.0 * bits
    for i in range(n - 1):
        est[n + i] = np.mean(s_z[:, i] * s_z[:, i + 1])

    return est


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


if __name__ == "__main__":
    n = 10
    seed_inst = 123

    rng_t = np.random.default_rng(seed_inst + 7)
    t = float(rng_t.uniform(1000.0, 2000.0))

    n_probes = 1000
    shots = 20_000

    print(f"\n{'='*56}")
    print(f" TFIM reconstruction  n={n}  |V|={2*n-1}")
    print(f" N_S={n_probes}  nu={shots}  t={t:.4f}")
    print(f" N_JOBS={rc.N_JOBS}")
    print(f"{'='*56}\n")

    res = run_trial(n=n, t=t, n_probes=n_probes, shots_per_probe=shots,
                    seed_instance=seed_inst, seed_probes=456, seed_shadows=789)
    rc.print_summary(res, shots)

    out_csv = Path(__file__).resolve().parent / f"tfim_n{n}_reconstruction.csv"
    rc.save_reconstruction_csv(res, out_csv)
