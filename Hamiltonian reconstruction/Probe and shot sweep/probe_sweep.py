"""TFIM n=10 probe-count scaling.

Fixed shots per probe = 50 000. Vary the number of probes.
For each grid point, repeat over several independent probe/shadow seeds
(the Hamiltonian instance is fixed) and report mean +- std of

    - operator-norm error   ||H_hat - H||_op
    - mean absolute coefficient error   <|h_hat_i - h_i|>

Outputs next to this script:
    tfim_probe_sweep.csv
    tfim_probe_sweep.pdf
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from _tfim_loader import tfim

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kwargs): return it


def _trial(n, t, n_probes, shots, seed_instance, seed_probes, seed_shadows):
    res = tfim.run_trial(
        n=n, t=t, n_probes=n_probes, shots_per_probe=shots,
        seed_instance=seed_instance,
        seed_probes=seed_probes,
        seed_shadows=seed_shadows,
        n_jobs=1,
    )
    return {
        "op_norm": res["op_norm_noisy"],
        "avg_coeff_err": res["avg_coeff_err_noisy"],
    }


def probe_sweep(n, t, probe_grid, fixed_shots, seed_instance,
                n_trials=32, master_seed=42, n_jobs=None):
    if n_jobs is None:
        from joblib import cpu_count
        n_jobs = cpu_count()
    rng = np.random.default_rng(master_seed)
    records = []
    for npv in tqdm(probe_grid, desc="probe sweep"):
        seeds = rng.integers(10**9, size=(n_trials, 2))
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_trial)(n, t, npv, fixed_shots, seed_instance,
                            int(s[0]), int(s[1])) for s in seeds
        )
        for key in ("op_norm", "avg_coeff_err"):
            vals = np.array([r[key] for r in results])
            records.append({
                "param": npv, "metric": key,
                "mean": float(np.nanmean(vals)),
                "std":  float(np.nanstd(vals)),
            })
    return records


def _extract(records, metric):
    sub = [r for r in records if r["metric"] == metric]
    x = np.array([r["param"] for r in sub], dtype=float)
    m = np.array([r["mean"]  for r in sub], dtype=float)
    s = np.array([r["std"]   for r in sub], dtype=float)
    return x, m, s


def _fit_inv_sqrt(x, y):
    mask = np.isfinite(y) & (y > 0) & (x > 0)
    if mask.sum() < 2:
        return None, None
    C = float(np.median(y[mask] * np.sqrt(x[mask])))
    return C, C / np.sqrt(x)


def save_csv(records, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "metric", "mean", "std"])
        for r in records:
            w.writerow([r["param"], r["metric"], r["mean"], r["std"]])
    print(f"[csv] {path}")


def plot_sweep(records, fixed_shots, n, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x, m, s = _extract(records, "op_norm")
    axes[0].errorbar(x, m, yerr=s, fmt="o-", capsize=4,
                     color="tab:blue", label="data")
    C, fit = _fit_inv_sqrt(x, m)
    if fit is not None:
        axes[0].plot(x, fit, "--", color="grey", alpha=0.7,
                     label=rf"$C/\sqrt{{N_S}}$  (C={C:.3f})")
    axes[0].set_xlabel("number of probes $N_S$")
    axes[0].set_ylabel(r"$\|\hat H - H\|_{\mathrm{op}}$")
    axes[0].set_title(f"TFIM, n={n}, shots={fixed_shots}")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25)

    x, m, s = _extract(records, "avg_coeff_err")
    axes[1].errorbar(x, m, yerr=s, fmt="s-", capsize=4,
                     color="tab:orange", label="data")
    C, fit = _fit_inv_sqrt(x, m)
    if fit is not None:
        axes[1].plot(x, fit, "--", color="grey", alpha=0.7,
                     label=rf"$C/\sqrt{{N_S}}$  (C={C:.3f})")
    axes[1].set_xlabel("number of probes $N_S$")
    axes[1].set_ylabel(r"mean $|\hat h_i - h_i|$")
    axes[1].set_title(f"TFIM, n={n}, shots={fixed_shots}")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


if __name__ == "__main__":
    n = 10
    seed_inst = 123

    rng_t = np.random.default_rng(seed_inst + 7)
    t = float(rng_t.uniform(1.0, 8.0))
    print(f"[t] t = {t:.4f}")

    fixed_shots = 20_000
    probe_grid = [int(x) for x in np.linspace(50, 2000, 20)]
    n_trials = 32

    print(f"[sweep] {len(probe_grid)} probe values in "
          f"[{probe_grid[0]}, {probe_grid[-1]}], "
          f"fixed shots = {fixed_shots}, n_trials = {n_trials}")

    records = probe_sweep(n, t, probe_grid, fixed_shots,
                          seed_instance=seed_inst, n_trials=n_trials)

    here = Path(__file__).resolve().parent
    save_csv(records, here / "tfim_probe_sweep.csv")
    plot_sweep(records, fixed_shots=fixed_shots, n=n,
               out_path=here / "tfim_probe_sweep.pdf")
