import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
from functools import lru_cache
from pathlib import Path
import csv, time

from joblib import Parallel, delayed, cpu_count

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kwargs): return it

N_JOBS = cpu_count()
print(f"[config] N_JOBS = {N_JOBS}")

plt.rcParams.update({
    "figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.25,
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "lines.linewidth": 2.0,
})

I2 = np.array([[1,0],[0,1]], dtype=complex)
X2 = np.array([[0,1],[1,0]], dtype=complex)
Y2 = np.array([[0,-1j],[1j,0]], dtype=complex)
Z2 = np.array([[1,0],[0,-1]], dtype=complex)

Hgate = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
Sdg = np.array([[1,0],[0,-1j]], dtype=complex)

BASIS_ROT_LIST = [Hgate, Hgate @ Sdg, I2]
PAULI_1Q = {'I': I2, 'X': X2, 'Y': Y2, 'Z': Z2}

def kron_all(mats):
    out = np.array([[1]], dtype=complex)
    for M in mats:
        out = np.kron(out, M)
    return out

@lru_cache(maxsize=None)
def pauli_label_to_matrix(label):
    return kron_all([PAULI_1Q[c] for c in label])

def rotation_unitary(basis_tuple):
    return kron_all([BASIS_ROT_LIST[b] for b in basis_tuple])

def hs_norm(A):
    d = A.shape[0]
    return np.sqrt(np.real(np.trace(A.conj().T @ A) / d))

def single_site_label(n, i, p):
    s = ["I"] * n
    s[i] = p
    return "".join(s)

def two_site_label(n, i, j, p, q):
    s = ["I"] * n
    s[i] = p
    s[j] = q
    return "".join(s)

def tfim_family_labels(n):
    labels, names = [], []
    for i in range(n):
        labels.append(single_site_label(n, i, "X"))
        names.append(f"X_{i}")
    for i in range(n - 1):
        labels.append(two_site_label(n, i, i + 1, "Z", "Z"))
        names.append(f"Z{i}Z{i+1}")
    return labels, names

def random_tfim_instance(n, rng):
    labels, names = tfim_family_labels(n)
    coeffs = np.concatenate([rng.normal(size=n), rng.normal(size=n - 1)])
    return labels, names, coeffs

def normalize_hamiltonian(labels, coeffs):
    mats = [pauli_label_to_matrix(lbl) for lbl in labels]
    H = sum(c * P for c, P in zip(coeffs, mats))
    norm = hs_norm(H)
    return H / norm, coeffs / norm, mats

ket0 = np.array([1,0], dtype=complex)
ket1 = np.array([0,1], dtype=complex)
ketp = np.array([1,1], dtype=complex) / np.sqrt(2)
ketm = np.array([1,-1], dtype=complex) / np.sqrt(2)
kety_p = np.array([1,1j], dtype=complex) / np.sqrt(2)
kety_m = np.array([1,-1j], dtype=complex) / np.sqrt(2)

PROBE_OPTIONS = [
    (2, +1., ket0), (2, -1., ket1),
    (0, +1., ketp), (0, -1., ketm),
    (1, +1., kety_p), (1, -1., kety_m),
]

def kron_vec(vecs):
    out = np.array([1.], dtype=complex)
    for v in vecs:
        out = np.kron(out, v)
    return out

def sample_product_probes(n, n_probes, rng):
    probes = []
    for _ in range(n_probes):
        bases = np.empty(n, dtype=np.int64)
        signs = np.empty(n)
        vecs = []
        for j in range(n):
            b, s, ket = PROBE_OPTIONS[rng.integers(len(PROBE_OPTIONS))]
            bases[j] = b
            signs[j] = s
            vecs.append(ket)
        probes.append({"psi": kron_vec(vecs), "basis": bases, "sign": signs})
    return probes

def exact_input_expectations_tfim(probe):
    basis = probe["basis"]
    sign = probe["sign"]
    n = len(basis)
    vals = np.zeros(2 * n - 1)
    for i in range(n):
        vals[i] = sign[i] if basis[i] == 0 else 0.
    for i in range(n - 1):
        vals[n + i] = sign[i] * sign[i + 1] if (basis[i] == 2 and basis[i + 1] == 2) else 0.
    return vals

def _sv_probs(psi, n):
    return (psi.real**2 + psi.imag**2).reshape((2,) * n)

def expect_X_i(psi, i, n):
    t = psi.reshape((2,) * n)
    t_x = np.flip(t, axis=i)
    return float(np.real(np.dot(psi.conj(), t_x.reshape(-1))))

def expect_ZiZj(psi, i, j, n):
    prob = _sv_probs(psi, n)
    sz = np.array([1., -1.])
    si = sz.reshape([2 if k == i else 1 for k in range(n)])
    sj = sz.reshape([2 if k == j else 1 for k in range(n)])
    return float(np.sum(prob * si * sj))

def exact_output_expectations_tfim(psi_t, n):
    vals = np.zeros(2 * n - 1)
    for i in range(n):
        vals[i] = expect_X_i(psi_t, i, n)
    for i in range(n - 1):
        vals[n + i] = expect_ZiZj(psi_t, i, i + 1, n)
    return vals

def estimate_tfim_two_basis(psi_t, n, n_shots, rng, U_x=None):
    d = 2**n
    bit_shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    m = 2 * n - 1
    est = np.zeros(m)

    n_x = n_shots // 2
    n_z = n_shots - n_x

    if U_x is None:
        U_x = rotation_unitary(tuple([0] * n))
    phi_x = U_x @ psi_t
    prob_x = phi_x.real**2 + phi_x.imag**2
    prob_x /= prob_x.sum()

    out_x = rng.choice(d, size=n_x, p=prob_x)
    bits_x = ((out_x[:, None] >> bit_shifts[None, :]) & 1).astype(np.float64)
    s_x = 1.0 - 2.0 * bits_x

    for i in range(n):
        est[i] = np.mean(s_x[:, i])

    prob_z = psi_t.real**2 + psi_t.imag**2
    prob_z /= prob_z.sum()

    out_z = rng.choice(d, size=n_z, p=prob_z)
    bits_z = ((out_z[:, None] >> bit_shifts[None, :]) & 1).astype(np.float64)
    s_z = 1.0 - 2.0 * bits_z

    for i in range(n - 1):
        est[n + i] = np.mean(s_z[:, i] * s_z[:, i + 1])

    return est

def _exact_row(probe, U, n):
    psi_t = U @ probe["psi"]
    exact_in = exact_input_expectations_tfim(probe)
    exact_out = exact_output_expectations_tfim(psi_t, n)
    d = 2**n
    return (exact_in - exact_out) / d

def _shadow_row(probe, U, n_shots, seed, n, U_x):
    rng = np.random.default_rng(seed)
    psi_t = U @ probe["psi"]
    exact_in = exact_input_expectations_tfim(probe)
    est_out = estimate_tfim_two_basis(psi_t, n, n_shots, rng, U_x=U_x)
    d = 2**n
    return (exact_in - est_out) / d

def build_feature_matrix_exact(U, probes, n_jobs=N_JOBS):
    d = U.shape[0]
    n = int(np.log2(d))
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_exact_row)(p, U, n) for p in probes
    )
    return np.array(rows, dtype=float)

def build_feature_matrix_shadow(U, probes, shots_per_probe, rng, n_jobs=N_JOBS):
    d = U.shape[0]
    n = int(np.log2(d))
    seeds = rng.integers(0, 2**31, size=len(probes))
    U_x = rotation_unitary(tuple([0] * n))
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_shadow_row)(p, U, shots_per_probe, int(s), n, U_x)
        for p, s in zip(probes, seeds)
    )
    return np.array(rows, dtype=float)

def reconstruct(X, true_h):
    _, svals, Vh = np.linalg.svd(X, full_matrices=False)
    h = Vh[-1]
    h /= np.linalg.norm(h)
    if np.dot(h, true_h) < 0:
        h = -h
    return {
        "h": h,
        "svals": svals,
        "overlap": float(np.dot(h, true_h)),
        "coeff_err": float(np.linalg.norm(h - true_h))
    }

def run_trial(n, t, n_probes, shots_per_probe, seed_instance=0, seed_probes=1, seed_shadows=2, n_jobs=N_JOBS):
    rng_i = np.random.default_rng(seed_instance)
    rng_p = np.random.default_rng(seed_probes)
    rng_s = np.random.default_rng(seed_shadows)

    labels, names, coeffs = random_tfim_instance(n, rng_i)
    H_true, true_h, _ = normalize_hamiltonian(labels, coeffs)
    U = expm(-1j * t * H_true)

    probes = sample_product_probes(n, n_probes, rng_p)

    t0 = time.perf_counter()
    X = build_feature_matrix_exact(U, probes, n_jobs=n_jobs)
    t1 = time.perf_counter()
    Xhat = build_feature_matrix_shadow(U, probes, shots_per_probe, rng_s, n_jobs=n_jobs)
    t2 = time.perf_counter()

    rec_e = reconstruct(X, true_h)
    rec_n = reconstruct(Xhat, true_h)
    Delta = Xhat - X

    return {
        "n": n, "d": 2**n, "labels": labels, "names": names,
        "coeffs_raw": coeffs, "true_h": true_h, "H_true": H_true, "U": U,
        "probes": probes, "X": X, "Xhat": Xhat, "Delta": Delta,
        "residual_exact": float(np.linalg.norm(X @ true_h)),
        "noise_op": float(np.linalg.norm(Delta, 2)),
        "noise_fro": float(np.linalg.norm(Delta, "fro")),
        "exact": rec_e, "noisy": rec_n,
        "time_exact_s": t1 - t0,
        "time_shadow_s": t2 - t1,
    }

def print_summary(res, shots_per_probe=None):
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    ge = se[-2] - se[-1] if len(se) >= 2 else float("nan")
    gn = sn[-2] - sn[-1] if len(sn) >= 2 else float("nan")
    ratio = res["noise_op"] / ge if ge > 0 else float("nan")

    print("\n" + "=" * 52)
    print(f" n={res['n']}, d={res['d']}, |V|={len(res['true_h'])}")
    if shots_per_probe:
        print(f" N_S={res['X'].shape[0]}, nu={shots_per_probe}")
    print("=" * 52)
    print(f" ||X h_true||        = {res['residual_exact']:.2e}   (should be ~0)")
    print(f" sigma_min(X)        = {se[-1]:.4e}")
    print(f" gap(X)              = {ge:.4e}")
    print(f" exact overlap       = {res['exact']['overlap']:.6f}")
    print(f" exact coeff err     = {res['exact']['coeff_err']:.4e}")
    print()
    print(f" sigma_min(Xhat)     = {sn[-1]:.4e}")
    print(f" gap(Xhat)           = {gn:.4e}")
    print(f" noisy overlap       = {res['noisy']['overlap']:.6f}  ← TARGET > 0.99")
    print(f" noisy coeff err     = {res['noisy']['coeff_err']:.4e}")
    print()
    print(f" ||Xhat-X||_2        = {res['noise_op']:.4e}")
    print(f" ||Xhat-X||_F        = {res['noise_fro']:.4e}")
    print(f" noise/gap ratio     = {ratio:.4e}  ← must be << 1")
    print(f" wall time (exact)   = {res['time_exact_s']:.1f}s")
    print(f" wall time (shadow)  = {res['time_shadow_s']:.1f}s")
    print("=" * 52)

def _mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def save_dict_csv(fp, d):
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in d.items():
            w.writerow([k, v])

def save_rows_csv(fp, header, rows):
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def save_trial_csvs(res, t, n_probes, shots_per_probe, out_dir, prefix=None):
    out_dir = _mkdir(out_dir)
    n = res["n"]
    if prefix is None:
        prefix = f"tfim_v2_n{n}_t{t}_NS{n_probes}_nu{shots_per_probe}"
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    ge = se[-2] - se[-1]
    gn = sn[-2] - sn[-1]
    ratio = res["noise_op"] / ge if ge > 0 else float("nan")

    save_dict_csv(out_dir / f"{prefix}_summary.csv", {
        "n": n, "d": res["d"], "t": t, "n_probes": n_probes,
        "shots_per_probe": shots_per_probe,
        "residual_exact": res["residual_exact"],
        "sigma_min_X": se[-1], "sigma_2_X": se[-2], "gap_X": ge,
        "exact_overlap": res["exact"]["overlap"],
        "exact_coeff_err": res["exact"]["coeff_err"],
        "sigma_min_Xhat": sn[-1], "sigma_2_Xhat": sn[-2], "gap_Xhat": gn,
        "noisy_overlap": res["noisy"]["overlap"],
        "noisy_coeff_err": res["noisy"]["coeff_err"],
        "noise_op": res["noise_op"], "noise_fro": res["noise_fro"],
        "noise_gap_ratio": ratio,
        "time_exact_s": res["time_exact_s"],
        "time_shadow_s": res["time_shadow_s"],
    })
    save_rows_csv(
        out_dir / f"{prefix}_coefficients.csv",
        ["idx", "name", "label", "raw", "true_h", "exact_h", "noisy_h", "err_exact", "err_noisy"],
        [[i, nm, lb, rc, tc, ec, nc, ec - tc, nc - tc]
         for i, (nm, lb, rc, tc, ec, nc) in enumerate(zip(
             res["names"], res["labels"], res["coeffs_raw"], res["true_h"],
             res["exact"]["h"], res["noisy"]["h"]))]
    )
    kmax = max(len(se), len(sn))
    save_rows_csv(
        out_dir / f"{prefix}_svals.csv",
        ["k", "sval_X", "sval_Xhat"],
        [[k, se[k] if k < len(se) else "", sn[k] if k < len(sn) else ""]
         for k in range(kmax)]
    )
    np.savetxt(out_dir / f"{prefix}_X_exact.csv", res["X"], delimiter=",")
    np.savetxt(out_dir / f"{prefix}_Xhat.csv", res["Xhat"], delimiter=",")
    np.savetxt(out_dir / f"{prefix}_Delta.csv", res["Delta"], delimiter=",")
    print(f"[csv] saved to {out_dir}/{prefix}_*.csv")

def plot_scatter(true_h, est_h, title, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(true_h, est_h, s=28, alpha=0.85)
    lim = 1.1 * max(np.max(np.abs(true_h)), np.max(np.abs(est_h)))
    ax.plot([-lim, lim], [-lim, lim], "--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("true h")
    ax.set_ylabel("reconstructed h")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")

def plot_svals(res, out_path, k_show=20):
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    k = min(k_show, len(se))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(range(1, k + 1), se[-k:], "o-", label="exact X")
    ax.plot(range(1, k + 1), sn[-k:], "s--", label="shadow Xhat (two-basis)")
    ax.set_xlabel("index among smallest singular values")
    ax.set_ylabel("singular value")
    ax.set_title("Bottom singular values: exact vs two-basis")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")

def _bench_trial(n, t, n_probes, shots, seed0):
    res = run_trial(
        n=n, t=t, n_probes=n_probes, shots_per_probe=shots,
        seed_instance=seed0, seed_probes=seed0 + 1, seed_shadows=seed0 + 2,
        n_jobs=1
    )
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    ge = se[-2] - se[-1]
    gn = sn[-2] - sn[-1]
    return {
        "gap_ex": ge, "sigma1_no": sn[-1], "gap_no": gn,
        "overlap": res["noisy"]["overlap"],
        "coeff_err": res["noisy"]["coeff_err"],
        "ratio": res["noise_op"] / ge if ge > 0 else float("nan"),
    }

def shot_sweep(n, t, n_probes, shot_grid, n_trials=5, master_seed=1234, n_jobs=N_JOBS):
    rng = np.random.default_rng(master_seed)
    stats = {k: [] for k in [
        "shots",
        "exact_gap_mean", "exact_gap_std", "sigma_min_hat_mean", "sigma_min_hat_std",
        "gap_hat_mean", "gap_hat_std", "overlap_mean", "overlap_std",
        "coeff_err_mean", "coeff_err_std", "noise_gap_ratio_mean", "noise_gap_ratio_std"
    ]}

    for shots in tqdm(shot_grid, desc="shot sweep"):
        seeds = rng.integers(10**9, size=n_trials).tolist()
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bench_trial)(n, t, n_probes, shots, int(s)) for s in seeds
        )
        stats["shots"].append(shots)
        for key, sk in [
            ("gap_ex", "exact_gap"), ("sigma1_no", "sigma_min_hat"),
            ("gap_no", "gap_hat"), ("overlap", "overlap"),
            ("coeff_err", "coeff_err"), ("ratio", "noise_gap_ratio")
        ]:
            vals = np.array([r[key] for r in results])
            stats[f"{sk}_mean"].append(np.nanmean(vals))
            stats[f"{sk}_std"].append(np.nanstd(vals))
    return {k: np.array(v) for k, v in stats.items()}

def plot_shot_sweep(stats, n, t, n_probes, out_path):
    shots = stats["shots"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].errorbar(shots, stats["coeff_err_mean"], yerr=stats["coeff_err_std"], fmt="o-", capsize=4)
    axes[0].set_title(f"n={n} t={t} NS={n_probes}: coeff error")
    axes[0].set_xlabel("shots/probe")
    axes[0].set_ylabel("||ĥ-h||")

    axes[1].errorbar(shots, stats["overlap_mean"], yerr=stats["overlap_std"], fmt="o-", capsize=4)
    axes[1].set_title(f"n={n} t={t} NS={n_probes}: overlap")
    axes[1].set_xlabel("shots/probe")
    axes[1].set_ylabel("overlap")
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].errorbar(
        shots, stats["noise_gap_ratio_mean"], yerr=stats["noise_gap_ratio_std"],
        fmt="o-", capsize=4, label="noise/gap"
    )
    axes[2].axhline(1.0, ls="--", color="red", alpha=0.7, label="noise=gap threshold")
    axes[2].set_title(f"n={n} t={t} NS={n_probes}: noise/gap ratio")
    axes[2].set_xlabel("shots/probe")
    axes[2].set_ylabel("||Δ||₂ / gap")
    axes[2].legend(frameon=False)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")

if __name__ == "__main__":
    out_dir = Path("./results_v2")
    out_dir.mkdir(exist_ok=True)

    n = 12
    t = 10
    n_probes = 60_000
    shots_per_probe = 250_000

    print(f"\n{'='*52}")
    print(f" TFIM v2  n={n}  |V|={2*n-1}  N_S={n_probes}  nu={shots_per_probe}")
    print(f" N_JOBS={N_JOBS}   estimator=two-basis (no shadow overhead)")
    print(f"{'='*52}\n")

    res = run_trial(
        n=n, t=t, n_probes=n_probes, shots_per_probe=shots_per_probe,
        seed_instance=123, seed_probes=456, seed_shadows=789,
    )

    print_summary(res, shots_per_probe)

    save_trial_csvs(
        res, t=t, n_probes=n_probes, shots_per_probe=shots_per_probe,
        out_dir=out_dir
    )

    plot_scatter(
        res["true_h"], res["exact"]["h"],
        f"TFIM exact, n={n}", out_dir / f"scatter_exact_n{n}.png"
    )
    plot_scatter(
        res["true_h"], res["noisy"]["h"],
        f"TFIM two-basis, n={n}, nu={shots_per_probe}",
        out_dir / f"scatter_twobasis_n{n}.png"
    )
    plot_svals(res, out_dir / f"svals_n{n}.png")