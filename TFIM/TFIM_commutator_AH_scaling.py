import os
import csv
import numpy as np

OUTDIR = "/mnt/user-data/outputs"
os.makedirs(OUTDIR, exist_ok=True)

def compute_Q(n, J, h):
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
    g2 = np.dot(g, g)
    if g2 < 1e-28:
        return float(np.linalg.eigvalsh(Q)[0])
    mu = float(np.trace(Q)) + 1.0
    Qdef = Q + (mu / g2) * np.outer(g, g)
    return float(np.linalg.eigvalsh(Qdef)[0])

def sample_inf_value_normalized(n, rng):
    J_raw = rng.normal(size=n - 1)
    h_raw = rng.normal(size=n)
    g_raw = np.concatenate([J_raw, h_raw])
    norm_g = np.linalg.norm(g_raw)
    if norm_g < 1e-28:
        return 0.0
    J = J_raw / norm_g
    h = h_raw / norm_g
    g = np.concatenate([J, h])
    Q = compute_Q(n, J, h)
    return constrained_min_eig(Q, g)

seed = 42
rng = np.random.default_rng(seed)

ns = list(range(4, 201))
num_samples = 1000

vals = {n: [] for n in ns}

for n in ns:
    for _ in range(num_samples):
        vals[n].append(sample_inf_value_normalized(n, rng))

means = np.array([np.mean(vals[n]) for n in ns])
stds = np.array([np.std(vals[n], ddof=1) for n in ns])
sems = stds / np.sqrt(num_samples)

csv_path = os.path.join(OUTDIR, "tfim_data_normalized.csv")

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "m=2n-1", "mean", "std", "sem"])
    for i, n in enumerate(ns):
        w.writerow([
            n,
            int(2 * n - 1),
            f"{means[i]:.10e}",
            f"{stds[i]:.10e}",
            f"{sems[i]:.10e}",
        ])
        