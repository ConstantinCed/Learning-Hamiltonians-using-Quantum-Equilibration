"""Estimate TFIM commutator-gap scaling over random coefficient draws."""

import os
import csv
import numpy as np
from google.colab import files

OUTDIR = "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)


def compute_Q(n, J, h):
    """Build the positive semidefinite quadratic form for TFIM coefficients."""
    m = 2 * n - 1
    V = np.zeros((m, 2 * (n - 1)))

    ks = np.arange(n - 1)

    V[ks, ks] = 2.0 * h[:-1]
    V[ks, n - 1 + ks] = 2.0 * h[1:]

    if n >= 2:
        V[n - 1, 0] = 2.0 * J[0]
        V[2 * n - 2, 2 * n - 3] = 2.0 * J[n - 2]

    if n >= 3:
        js = np.arange(1, n - 1)
        V[n - 1 + js, js] = 2.0 * J[js]
        V[n - 1 + js, n - 2 + js] = 2.0 * J[js - 1]

    return V @ V.T


def constrained_min_eig(Q, g):
    """Approximate the minimum eigenvalue of Q on g-perp."""
    g2 = np.dot(g, g)

    if g2 < 1e-28:
        return float(np.linalg.eigvalsh(Q)[0])

    mu = float(np.trace(Q)) + 1.0
    Qdef = Q + (mu / g2) * np.outer(g, g)
    return float(np.linalg.eigvalsh(Qdef)[0])


def sample_inf_value_unnormalized(n, rng, distribution):
    """Sample unnormalized TFIM coefficients and evaluate the constrained gap."""
    if distribution == "gaussian":
        J = rng.normal(loc=0.0, scale=1.0, size=n - 1)
        h = rng.normal(loc=0.0, scale=1.0, size=n)

    elif distribution == "uniform":
        J = rng.uniform(low=-1.0, high=1.0, size=n - 1)
        h = rng.uniform(low=-1.0, high=1.0, size=n)

    else:
        raise ValueError("distribution must be either 'gaussian' or 'uniform'")

    g = np.concatenate([J, h])
    Q = compute_Q(n, J, h)
    return constrained_min_eig(Q, g)


def run_experiment(
    ns,
    num_samples,
    seed,
    distribution,
    outdir=OUTDIR,
):
    """Run the Monte Carlo sweep and save one CSV for a coefficient law."""
    rng = np.random.default_rng(seed)

    vals = {n: [] for n in ns}

    for n in ns:
        print(f"Running n = {n}, distribution = {distribution}")

        for _ in range(num_samples):
            val = sample_inf_value_unnormalized(
                n=n,
                rng=rng,
                distribution=distribution,
            )
            vals[n].append(val)

    means = np.array([np.mean(vals[n]) for n in ns])
    stds = np.array([np.std(vals[n], ddof=1) for n in ns])
    sems = stds / np.sqrt(num_samples)

    csv_path = os.path.join(
        outdir,
        f"tfim_data_unnormalized_{distribution}.csv"
    )

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow([
            "n",
            "m=2n-1",
            "distribution",
            "mean",
            "std",
            "sem",
            "num_samples",
        ])

        for i, n in enumerate(ns):
            w.writerow([
                n,
                int(2 * n - 1),
                distribution,
                f"{means[i]:.10e}",
                f"{stds[i]:.10e}",
                f"{sems[i]:.10e}",
                num_samples,
            ])

    return csv_path, means, stds, sems


seed = 42

ns = list(range(4, 201))
num_samples = 1000

csv_gaussian, means_gaussian, stds_gaussian, sems_gaussian = run_experiment(
    ns=ns,
    num_samples=num_samples,
    seed=seed,
    distribution="gaussian",
)

csv_uniform, means_uniform, stds_uniform, sems_uniform = run_experiment(
    ns=ns,
    num_samples=num_samples,
    seed=seed + 1,
    distribution="uniform",
)

print()
print("Saved files:")
print(csv_gaussian)
print(csv_uniform)

files.download(csv_gaussian)
files.download(csv_uniform)
