import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
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

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

Hgate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
HSdg = Hgate @ Sdg

_PAULI_NAMES = {0: "X", 1: "Y", 2: "Z"}

PERIOD3_PATTERNS = list(itertools.product([0, 1, 2], repeat=3))

def hs_norm(A):
    d = A.shape[0]
    return np.sqrt(np.real(np.trace(A.conj().T @ A) / d))

def enumerate_dense_operators(n, k=3, R=2):
    ops = []
    names = []
    TYPES = [0, 1, 2]

    for size in range(1, k + 1):
        for sites in itertools.combinations(range(n), size):
            if max(sites) - min(sites) <= R:
                for types in itertools.product(TYPES, repeat=size):
                    ops.append((sites, types))
                    lbl = "".join(
                        f"{_PAULI_NAMES[t]}{s}" for s, t in zip(sites, types)
                    )
                    names.append(lbl)

    return ops, names

def operator_hs_norms(ops):
    return np.ones(len(ops))

def _add_pauli_term(H, n, support, pauli_types, coeff):
    d = H.shape[0]
    idx = np.arange(d, dtype=np.int64)

    flip_mask = 0
    for j, t in zip(support, pauli_types):
        if t in (0, 1):
            flip_mask |= (1 << (n - 1 - j))

    phase = np.ones(d, dtype=complex)
    for j, t in zip(support, pauli_types):
        bp = n - 1 - j
        bit_j = (idx >> bp) & 1
        if t == 1:
            phase *= np.where(bit_j == 0, -1j, +1j)
        elif t == 2:
            phase *= np.where(bit_j == 0, +1.0, -1.0)

    col = idx ^ flip_mask
    H[idx, col] += coeff * phase

def build_dense_hamiltonian(n, ops, coeffs):
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    for (supp, types), c in zip(ops, coeffs):
        _add_pauli_term(H, n, supp, types, c)
    return H

def random_dense_instance(n, ops, rng):
    h_raw = rng.standard_normal(len(ops))
    H_raw = build_dense_hamiltonian(n, ops, h_raw)
    norm = hs_norm(H_raw)
    h_true = h_raw / norm
    H_true = H_raw / norm
    return h_true, H_true

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ketp = np.array([1, 1], dtype=complex) / np.sqrt(2)
ketm = np.array([1, -1], dtype=complex) / np.sqrt(2)
kety_p = np.array([1, 1j], dtype=complex) / np.sqrt(2)
kety_m = np.array([1, -1j], dtype=complex) / np.sqrt(2)

PROBE_OPTIONS = [
    (2, +1., ket0), (2, -1., ket1),
    (0, +1., ketp), (0, -1., ketm),
    (1, +1., kety_p), (1, -1., kety_m),
]

def _kron_vec(vecs):
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
        probes.append({"psi": _kron_vec(vecs), "basis": bases, "sign": signs})
    return probes

def exact_input_expectations_dense(probe, ops):
    basis = probe["basis"]
    sign = probe["sign"]
    vals = np.zeros(len(ops))
    for k, (supp, types) in enumerate(ops):
        if all(basis[supp[i]] == types[i] for i in range(len(supp))):
            vals[k] = np.prod([sign[j] for j in supp])
    return vals

def _expect_pauli(psi, support, pauli_types, n):
    d = 2**n
    idx = np.arange(d, dtype=np.int64)

    flip_mask = 0
    for j, t in zip(support, pauli_types):
        if t in (0, 1):
            flip_mask |= (1 << (n - 1 - j))

    phase = np.ones(d, dtype=complex)
    for j, t in zip(support, pauli_types):
        bp = n - 1 - j
        bit_j = (idx >> bp) & 1
        if t == 1:
            phase *= np.where(bit_j == 0, -1j, +1j)
        elif t == 2:
            phase *= np.where(bit_j == 0, +1.0, -1.0)

    col = idx ^ flip_mask
    return float(np.real(np.dot(psi.conj(), psi[col] * phase)))

def exact_output_expectations_dense(psi_t, ops, n):
    return np.array([_expect_pauli(psi_t, supp, types, n) for supp, types in ops])

def apply_single_qubit_gate(phi, gate, site, n):
    psi_t = phi.reshape([2] * n)
    result = np.tensordot(gate, psi_t, axes=([1], [site]))
    return np.moveaxis(result, 0, site).reshape(-1)

def apply_basis_rotation(psi, basis_list, n):
    phi = psi.copy()
    for j, b in enumerate(basis_list):
        if b == 0:
            phi = apply_single_qubit_gate(phi, Hgate, j, n)
        elif b == 1:
            phi = apply_single_qubit_gate(phi, HSdg, j, n)
    return phi

def precompute_shadow_structure(ops, n):
    n_ops = len(ops)
    match_matrix = np.zeros((27, n_ops), dtype=bool)

    for p_idx, pat in enumerate(PERIOD3_PATTERNS):
        site_bases = [pat[j % 3] for j in range(n)]
        for k, (supp, types) in enumerate(ops):
            match_matrix[p_idx, k] = all(
                site_bases[supp[i]] == types[i] for i in range(len(supp))
            )

    scales = np.array([3.0 ** len(op[0]) for op in ops])
    return match_matrix, scales

def estimate_shadow_period3(psi_t, ops, n, nu, rng, match_matrix, scales):
    d = 2**n
    n_ops = len(ops)
    bit_shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    shots_pp = nu // 27
    weight = shots_pp / nu

    est = np.zeros(n_ops)

    for p_idx, pat in enumerate(PERIOD3_PATTERNS):
        basis_list = [pat[j % 3] for j in range(n)]
        phi = apply_basis_rotation(psi_t, basis_list, n)
        prob = phi.real**2 + phi.imag**2
        prob /= prob.sum()
        out = rng.choice(d, size=shots_pp, p=prob)
        bits = ((out[:, None] >> bit_shifts[None, :]) & 1).astype(np.float64)
        sgns = 1.0 - 2.0 * bits

        matching = np.where(match_matrix[p_idx])[0]
        for k in matching:
            supp = ops[k][0]
            s = len(supp)
            if s == 1:
                prod = sgns[:, supp[0]]
            elif s == 2:
                prod = sgns[:, supp[0]] * sgns[:, supp[1]]
            else:
                prod = sgns[:, supp[0]] * sgns[:, supp[1]] * sgns[:, supp[2]]
            est[k] += scales[k] * np.mean(prod) * weight

    return est

def _exact_row(probe, U, ops, n):
    psi_t = U @ probe["psi"]
    inp = exact_input_expectations_dense(probe, ops)
    out = exact_output_expectations_dense(psi_t, ops, n)
    return (inp - out) / (2**n)

def _shadow_row(probe, U, ops, n, nu, seed, match_matrix, scales):
    rng = np.random.default_rng(seed)
    psi_t = U @ probe["psi"]
    inp = exact_input_expectations_dense(probe, ops)
    est = estimate_shadow_period3(psi_t, ops, n, nu, rng, match_matrix, scales)
    return (inp - est) / (2**n)

def build_feature_matrix_exact(U, probes, ops, n, n_jobs=N_JOBS):
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_exact_row)(p, U, ops, n) for p in probes)
    return np.array(rows, dtype=float)

def build_feature_matrix_shadow(U, probes, ops, n, nu, rng, match_matrix, scales, n_jobs=N_JOBS):
    seeds = rng.integers(0, 2**31, size=len(probes))
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_shadow_row)(p, U, ops, n, nu, int(s), match_matrix, scales)
        for p, s in zip(probes, seeds))
    return np.array(rows, dtype=float)

def reconstruct(X, true_h):
    _, svals, Vh = np.linalg.svd(X, full_matrices=False)
    h = Vh[-1]
    h /= np.linalg.norm(h)
    if np.dot(h, true_h) < 0:
        h = -h
    overlap = float(np.dot(h, true_h))
    coeff_err = float(np.linalg.norm(h - true_h))
    return {"h": h, "svals": svals, "overlap": overlap, "coeff_err": coeff_err}

def run_trial(n, ops, t, n_probes, shots_per_probe, seed_instance=0, seed_probes=1, seed_shadows=2, n_jobs=N_JOBS):
    rng_i = np.random.default_rng(seed_instance)
    rng_p = np.random.default_rng(seed_probes)
    rng_s = np.random.default_rng(seed_shadows)

    n_ops = len(ops)

    print(f"[check] |V| = {n_ops}  (operators in V_dense, n={n}, k=3, R=2)")

    print(f"[setup] Building H ({2**n}x{2**n}, {n_ops} terms) ...", end=" ", flush=True)
    t0 = time.perf_counter()
    h_true, H_true = random_dense_instance(n, ops, rng_i)
    print(f"done ({time.perf_counter()-t0:.1f}s)")
    print(f"[check] ||h_true||_2 = {np.linalg.norm(h_true):.6f}  (must be 1.000000)")

    print(f"[setup] Computing U = exp(-itH), n={n}, d={2**n} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    U = expm(-1j * t * H_true)
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    print(f"[setup] Precomputing period-3 shadow structure ({27} patterns) ...", end=" ", flush=True)
    t0 = time.perf_counter()
    match_matrix, scales = precompute_shadow_structure(ops, n)
    print(f"done ({time.perf_counter()-t0:.2f}s)")
    avg_matching = match_matrix.sum(axis=1).mean()
    print(f"[check] avg matching ops per pattern: {avg_matching:.1f} / {n_ops}")

    probes = sample_product_probes(n, n_probes, rng_p)

    print(f"[run]   Exact feature matrix  ({n_probes} x {n_ops}) ...")
    t1 = time.perf_counter()
    X = build_feature_matrix_exact(U, probes, ops, n, n_jobs=n_jobs)
    t2 = time.perf_counter()
    print(f"        done ({t2-t1:.1f}s)")

    print(f"[run]   Shadow feature matrix ({n_probes} x {n_ops}, nu={shots_per_probe}) ...")
    Xhat = build_feature_matrix_shadow(
        U, probes, ops, n, shots_per_probe, rng_s,
        match_matrix, scales, n_jobs=n_jobs)
    t3 = time.perf_counter()
    print(f"        done ({t3-t2:.1f}s)")

    rec_e = reconstruct(X, h_true)
    rec_n = reconstruct(Xhat, h_true)
    Delta = Xhat - X

    return {
        "n": n, "d": 2**n, "n_ops": n_ops,
        "ops": ops, "h_true": h_true, "H_true": H_true,
        "U": U, "probes": probes,
        "X": X, "Xhat": Xhat, "Delta": Delta,
        "residual_exact": float(np.linalg.norm(X @ h_true)),
        "noise_op": float(np.linalg.norm(Delta, 2)),
        "noise_fro": float(np.linalg.norm(Delta, "fro")),
        "exact": rec_e, "noisy": rec_n,
        "time_exact_s": t2 - t1,
        "time_shadow_s": t3 - t2,
    }

def print_summary(res, shots_per_probe=None):
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    ge = se[-2] - se[-1] if len(se) >= 2 else float("nan")
    gn = sn[-2] - sn[-1] if len(sn) >= 2 else float("nan")
    ratio = res["noise_op"] / ge if ge > 0 else float("nan")
    n = res["n"]

    print("\n" + "=" * 66)
    print(f" Dense 3-local  n={n}  d={res['d']}  |V|={res['n_ops']}  k=3  R=2  OBC")
    if shots_per_probe:
        print(f" N_S={res['X'].shape[0]}  nu={shots_per_probe}  estimator=period-3 shadow")
    print(" reconstruct: plain SVD (HS=Euclidean, all w_k=1)")
    print("=" * 66)
    print(f" ||X h_true||        = {res['residual_exact']:.2e}")
    print(f" sigma_min(X)        = {se[-1]:.4e}")
    print(f" gap(X)              = {ge:.4e}")
    print(f" exact overlap       = {res['exact']['overlap']:.6f}")
    print(f" exact coeff err     = {res['exact']['coeff_err']:.4e}")
    print()
    print(f" sigma_min(Xhat)     = {sn[-1]:.4e}")
    print(f" gap(Xhat)           = {gn:.4e}")
    print(f" noisy overlap       = {res['noisy']['overlap']:.6f}")
    print(f" noisy coeff err     = {res['noisy']['coeff_err']:.4e}")
    print()
    print(f" ||Xhat-X||_2        = {res['noise_op']:.4e}")
    print(f" ||Xhat-X||_F        = {res['noise_fro']:.4e}")
    print(f" noise/gap ratio     = {ratio:.4e}")
    print(f" wall time (exact)   = {res['time_exact_s']:.1f}s")
    print(f" wall time (shadow)  = {res['time_shadow_s']:.1f}s")
    print("=" * 66)
    print()

    h_t = res["h_true"]
    h_e = res["noisy"]["h"]
    ops = res["ops"]
    for size in (1, 2, 3):
        idx_s = [k for k, (supp, _) in enumerate(ops) if len(supp) == size]
        if not idx_s:
            continue
        errs = np.abs(h_e[idx_s] - h_t[idx_s])
        print(f" {size}-body ops ({len(idx_s):3d}):  "
              f"mean|err|={np.mean(errs):.4f}  "
              f"max|err|={np.max(errs):.4f}  "
              f"||h_true||={np.linalg.norm(h_t[idx_s]):.4f}")
    print()

def _mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def _save_dict_csv(fp, d):
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in d.items():
            w.writerow([k, v])

def _save_rows_csv(fp, header, rows):
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def save_trial_csvs(res, t, n_probes, shots_per_probe, out_dir, prefix=None):
    out_dir = _mkdir(out_dir)
    n = res["n"]
    if prefix is None:
        prefix = f"dense3_n{n}_t{t}_NS{n_probes}_nu{shots_per_probe}"

    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    ge = se[-2] - se[-1]
    gn = sn[-2] - sn[-1]
    ratio = res["noise_op"] / ge if ge > 0 else float("nan")

    _save_dict_csv(out_dir / f"{prefix}_summary.csv", {
        "n": n, "d": res["d"], "t": t, "n_ops": res["n_ops"],
        "k": 3, "R": 2, "n_probes": n_probes, "shots_per_probe": shots_per_probe,
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

    ops = res["ops"]
    _save_rows_csv(
        out_dir / f"{prefix}_coefficients.csv",
        ["idx", "name", "size", "true_h", "exact_h", "noisy_h", "err_exact", "err_noisy"],
        [[i, f"{supp}_{types}", len(supp), tc, ec, nc, ec - tc, nc - tc]
         for i, ((supp, types), tc, ec, nc) in enumerate(zip(
             ops, res["h_true"], res["exact"]["h"], res["noisy"]["h"]))]
    )

    kmax = max(len(se), len(sn))
    _save_rows_csv(
        out_dir / f"{prefix}_svals.csv",
        ["k", "sval_exact", "sval_shadow"],
        [[k, se[k] if k < len(se) else "", sn[k] if k < len(sn) else ""]
         for k in range(kmax)]
    )

    np.savetxt(out_dir / f"{prefix}_X_exact.csv", res["X"], delimiter=",")
    np.savetxt(out_dir / f"{prefix}_Xhat.csv", res["Xhat"], delimiter=",")
    np.savetxt(out_dir / f"{prefix}_Delta.csv", res["Delta"], delimiter=",")
    print(f"[csv] saved -> {out_dir}/{prefix}_*.csv")

def plot_scatter_by_size(true_h, est_h, ops, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    colours = {1: "steelblue", 2: "darkorange", 3: "seagreen"}
    markers = {1: "o", 2: "^", 3: "s"}
    for size in (1, 2, 3):
        idx = [k for k, (supp, _) in enumerate(ops) if len(supp) == size]
        if not idx:
            continue
        ax.scatter(true_h[idx], est_h[idx],
                   s=30, alpha=0.7, c=colours[size], marker=markers[size],
                   label=f"{size}-body ({len(idx)} ops)", zorder=3)
    lim = 1.15 * max(np.max(np.abs(true_h)), np.max(np.abs(est_h)))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("true h")
    ax.set_ylabel("reconstructed h")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")

def plot_error_by_size(res, out_path):
    ops = res["ops"]
    h_t = res["h_true"]
    h_e = res["noisy"]["h"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data_by_size = {}
    for size in (1, 2, 3):
        idx = [k for k, (supp, _) in enumerate(ops) if len(supp) == size]
        data_by_size[size] = np.abs(h_e[idx] - h_t[idx])

    ax.boxplot([data_by_size[s] for s in (1, 2, 3)],
               labels=["1-body\n(36 ops)", "2-body\n(189 ops)", "3-body\n(270 ops)"],
               patch_artist=True,
               boxprops=dict(facecolor="lightyellow"),
               medianprops=dict(color="red", lw=2))
    ax.set_ylabel("|h_est - h_true|")
    ax.set_title(f"Coefficient error by operator size  (n={res['n']})")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")

def plot_svals(res, out_path, k_show=40):
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    k = min(k_show, len(se))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, k + 1), se[-k:], "o-", label="exact X")
    ax.plot(range(1, k + 1), sn[-k:], "s--", label="shadow Xhat (period-3)")
    ax.set_xlabel("index (smallest singular values)")
    ax.set_ylabel("singular value")
    ax.set_title(f"Bottom singular values  —  dense 3-local n={res['n']}")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")

def _bench_trial(n, ops, t, n_probes, shots, seed0, match_matrix, scales):
    res = run_trial(n=n, ops=ops, t=t, n_probes=n_probes, shots_per_probe=shots,
                    seed_instance=seed0, seed_probes=seed0 + 1,
                    seed_shadows=seed0 + 2, n_jobs=1)
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    ge = se[-2] - se[-1]
    return {
        "gap_ex": ge,
        "sigma1_no": sn[-1],
        "gap_no": sn[-2] - sn[-1],
        "overlap": res["noisy"]["overlap"],
        "coeff_err": res["noisy"]["coeff_err"],
        "ratio": res["noise_op"] / ge if ge > 0 else float("nan"),
    }

def shot_sweep(n, ops, t, n_probes, shot_grid, n_trials=3, master_seed=1234, n_jobs=N_JOBS):
    match_matrix, scales = precompute_shadow_structure(ops, n)
    rng = np.random.default_rng(master_seed)
    stats = {k: [] for k in [
        "shots",
        "exact_gap_mean", "exact_gap_std",
        "sigma_min_hat_mean", "sigma_min_hat_std",
        "gap_hat_mean", "gap_hat_std",
        "overlap_mean", "overlap_std",
        "coeff_err_mean", "coeff_err_std",
        "noise_gap_ratio_mean", "noise_gap_ratio_std",
    ]}
    for shots in tqdm(shot_grid, desc="shot sweep"):
        seeds = rng.integers(10**9, size=n_trials).tolist()
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bench_trial)(n, ops, t, n_probes, shots, int(s), match_matrix, scales)
            for s in seeds)
        stats["shots"].append(shots)
        for key, sk in [
            ("gap_ex", "exact_gap"), ("sigma1_no", "sigma_min_hat"),
            ("gap_no", "gap_hat"), ("overlap", "overlap"),
            ("coeff_err", "coeff_err"), ("ratio", "noise_gap_ratio"),
        ]:
            vals = np.array([r[key] for r in results])
            stats[f"{sk}_mean"].append(np.nanmean(vals))
            stats[f"{sk}_std"].append(np.nanstd(vals))
    return {k: np.array(v) for k, v in stats.items()}

if __name__ == "__main__":
    out_dir = Path("./results_dense3local")
    out_dir.mkdir(exist_ok=True)

    n = 12
    k = 3
    R = 2
    t = 10
    n_probes = 30_000
    shots_per_probe = 1_000_000

    print(f"[setup] Enumerating V_dense (n={n}, k={k}, R={R}) ...", end=" ", flush=True)
    ops, names = enumerate_dense_operators(n, k=k, R=R)
    print(f"done: |V| = {len(ops)}")
    for size in (1, 2, 3):
        cnt = sum(1 for supp, _ in ops if len(supp) == size)
        print(f"        {size}-body: {cnt} operators")

    print(f"\n{'='*66}")
    print(f" Dense 3-local  n={n}  |V|={len(ops)}  k={k}  R={R}  OBC")
    print(f" N_S={n_probes}  nu={shots_per_probe}  N_JOBS={N_JOBS}")
    print(f" H = sum_{{P in V_dense}} h_P * P,  h_P ~ N(0,1)")
    print(f" estimator: period-3 Pauli shadow (27 patterns)")
    print(f"{'='*66}\n")

    res = run_trial(
        n=n, ops=ops, t=t,
        n_probes=n_probes,
        shots_per_probe=shots_per_probe,
        seed_instance=123,
        seed_probes=456,
        seed_shadows=789,
    )

    print_summary(res, shots_per_probe)

    save_trial_csvs(res, t=t, n_probes=n_probes, shots_per_probe=shots_per_probe,
                    out_dir=out_dir)

    plot_scatter_by_size(res["h_true"], res["exact"]["h"], ops,
                         f"Dense 3-local exact, n={n}",
                         out_dir / f"scatter_exact_n{n}.png")
    plot_scatter_by_size(res["h_true"], res["noisy"]["h"], ops,
                         f"Dense 3-local period-3 shadow, n={n}, nu={shots_per_probe}",
                         out_dir / f"scatter_shadow_n{n}.png")
    plot_error_by_size(res, out_dir / f"error_by_size_n{n}.png")
    plot_svals(res, out_dir / f"svals_n{n}.png")