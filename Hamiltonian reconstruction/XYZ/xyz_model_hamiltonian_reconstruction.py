import os
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("MKL_NUM_THREADS",     "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

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

I2 = np.array([[1,  0],  [0,  1]],  dtype=complex)
X2 = np.array([[0,  1],  [1,  0]],  dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]],  dtype=complex)
Z2 = np.array([[1,  0],  [0, -1]],  dtype=complex)

Hgate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Sdg   = np.array([[1, 0], [0, -1j]], dtype=complex)

_BASIS_ROT = [Hgate, Hgate @ Sdg, I2]

def kron_all(mats):
    out = np.array([[1]], dtype=complex)
    for M in mats:
        out = np.kron(out, M)
    return out

def rotation_unitary(basis_tuple):
    return kron_all([_BASIS_ROT[b] for b in basis_tuple])

def hs_norm(A):
    d = A.shape[0]
    return np.sqrt(np.real(np.trace(A.conj().T @ A) / d))

def operator_hs_norms(n):
    return np.ones(3 * (n - 1))

def _add_XX(H, n, i, j, coeff):
    d = H.shape[0]
    bpi = n - 1 - i
    bpj = n - 1 - j
    idx = np.arange(d, dtype=np.int64)
    H[idx, idx ^ ((1 << bpi) | (1 << bpj))] += coeff

def _add_YY(H, n, i, j, coeff):
    d = H.shape[0]
    bpi = n - 1 - i
    bpj = n - 1 - j
    idx = np.arange(d, dtype=np.int64)
    bi  = (idx >> bpi) & 1
    bj  = (idx >> bpj) & 1
    phase = np.where(bi == bj, -1.0, +1.0)
    H[idx, idx ^ ((1 << bpi) | (1 << bpj))] += coeff * phase

def _add_ZZ(H, n, i, j, coeff):
    d = H.shape[0]
    bpi = n - 1 - i
    bpj = n - 1 - j
    idx = np.arange(d, dtype=np.int64)
    zi  = 1 - 2 * ((idx >> bpi) & 1).astype(float)
    zj  = 1 - 2 * ((idx >> bpj) & 1).astype(float)
    H[idx, idx] += coeff * zi * zj

def build_xyz_hamiltonian(n, a_coeffs, b_coeffs, c_coeffs):
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        _add_XX(H, n, i, i + 1, a_coeffs[i])
        _add_YY(H, n, i, i + 1, b_coeffs[i])
        _add_ZZ(H, n, i, i + 1, c_coeffs[i])
    return H

def random_xyz_instance(n, rng):
    a_raw = rng.standard_normal(n - 1)
    b_raw = rng.standard_normal(n - 1)
    c_raw = rng.standard_normal(n - 1)
    H_raw = build_xyz_hamiltonian(n, a_raw, b_raw, c_raw)
    norm  = hs_norm(H_raw)
    a, b, c = a_raw / norm, b_raw / norm, c_raw / norm
    H       = H_raw / norm
    h_true  = np.concatenate([a, b, c])
    names   = (
        [f"XX_{i}{i+1}" for i in range(n - 1)] +
        [f"YY_{i}{i+1}" for i in range(n - 1)] +
        [f"ZZ_{i}{i+1}" for i in range(n - 1)]
    )
    return a, b, c, names, h_true, H

ket0   = np.array([1,  0  ], dtype=complex)
ket1   = np.array([0,  1  ], dtype=complex)
ketp   = np.array([1,  1  ], dtype=complex) / np.sqrt(2)
ketm   = np.array([1, -1  ], dtype=complex) / np.sqrt(2)
kety_p = np.array([1,  1j ], dtype=complex) / np.sqrt(2)
kety_m = np.array([1, -1j ], dtype=complex) / np.sqrt(2)

PROBE_OPTIONS = [
    (2, +1., ket0  ), (2, -1., ket1  ),
    (0, +1., ketp  ), (0, -1., ketm  ),
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
        vecs  = []
        for j in range(n):
            b, s, ket = PROBE_OPTIONS[rng.integers(len(PROBE_OPTIONS))]
            bases[j] = b
            signs[j] = s
            vecs.append(ket)
        probes.append({"psi": _kron_vec(vecs), "basis": bases, "sign": signs})
    return probes

def exact_input_expectations_xyz(probe):
    basis = probe["basis"]
    sign  = probe["sign"]
    n     = len(basis)
    vals  = np.zeros(3 * (n - 1))
    m     = n - 1
    for i in range(m):
        if basis[i] == 0 and basis[i+1] == 0:
            vals[i]       = sign[i] * sign[i+1]
        if basis[i] == 1 and basis[i+1] == 1:
            vals[m + i]   = sign[i] * sign[i+1]
        if basis[i] == 2 and basis[i+1] == 2:
            vals[2*m + i] = sign[i] * sign[i+1]
    return vals

def _sv_probs_tensor(psi, n):
    return (psi.real**2 + psi.imag**2).reshape((2,) * n)

def _expect_XX(psi, i, j, n):
    t = psi.reshape((2,) * n)
    return float(np.real(np.dot(
        psi.conj(), np.flip(np.flip(t, axis=i), axis=j).reshape(-1))))

def _expect_YY(psi, i, j, n):
    t      = psi.reshape((2,) * n)
    t_flip = np.flip(np.flip(t, axis=i), axis=j)
    d   = 2**n
    idx = np.arange(d, dtype=np.int64)
    bi  = (idx >> (n - 1 - i)) & 1
    bj  = (idx >> (n - 1 - j)) & 1
    phase = np.where(bi == bj, -1.0, +1.0)
    return float(np.real(np.dot(psi.conj(), t_flip.reshape(-1) * phase)))

def _expect_ZZ(psi, i, j, n):
    prob = _sv_probs_tensor(psi, n)
    sz   = np.array([1., -1.])
    si   = sz.reshape([2 if k == i else 1 for k in range(n)])
    sj   = sz.reshape([2 if k == j else 1 for k in range(n)])
    return float(np.sum(prob * si * sj))

def exact_output_expectations_xyz(psi_t, n):
    m    = n - 1
    vals = np.zeros(3 * m)
    for i in range(m):
        vals[i]       = _expect_XX(psi_t, i, i+1, n)
        vals[m + i]   = _expect_YY(psi_t, i, i+1, n)
        vals[2*m + i] = _expect_ZZ(psi_t, i, i+1, n)
    return vals

def estimate_xyz_three_basis(psi_t, n, n_shots, rng, U_x=None, U_y=None):
    d          = 2**n
    bit_shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    m          = n - 1
    est        = np.zeros(3 * m)

    n_each = n_shots // 3
    n_x    = n_each
    n_y    = n_each
    n_z    = n_shots - 2 * n_each

    def _sample(phi, n_meas):
        prob = phi.real**2 + phi.imag**2
        prob /= prob.sum()
        out  = rng.choice(d, size=n_meas, p=prob)
        bits = ((out[:, None] >> bit_shifts[None, :]) & 1).astype(np.float64)
        return 1.0 - 2.0 * bits

    if U_x is None:
        U_x = rotation_unitary(tuple([0] * n))
    s_x = _sample(U_x @ psi_t, n_x)
    for i in range(m):
        est[i] = np.mean(s_x[:, i] * s_x[:, i+1])

    if U_y is None:
        U_y = rotation_unitary(tuple([1] * n))
    s_y = _sample(U_y @ psi_t, n_y)
    for i in range(m):
        est[m + i] = np.mean(s_y[:, i] * s_y[:, i+1])

    s_z = _sample(psi_t, n_z)
    for i in range(m):
        est[2*m + i] = np.mean(s_z[:, i] * s_z[:, i+1])

    return est

def _exact_row(probe, U, n):
    psi_t = U @ probe["psi"]
    return (exact_input_expectations_xyz(probe)
            - exact_output_expectations_xyz(psi_t, n)) / (2**n)

def _shadow_row(probe, U, n_shots, seed, n, U_x, U_y):
    rng   = np.random.default_rng(seed)
    psi_t = U @ probe["psi"]
    est   = estimate_xyz_three_basis(psi_t, n, n_shots, rng, U_x=U_x, U_y=U_y)
    return (exact_input_expectations_xyz(probe) - est) / (2**n)

def build_feature_matrix_exact(U, probes, n_jobs=N_JOBS):
    n    = int(np.log2(U.shape[0]))
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_exact_row)(p, U, n) for p in probes)
    return np.array(rows, dtype=float)

def build_feature_matrix_shadow(U, probes, shots_per_probe, rng, n_jobs=N_JOBS):
    n     = int(np.log2(U.shape[0]))
    seeds = rng.integers(0, 2**31, size=len(probes))
    U_x   = rotation_unitary(tuple([0] * n))
    U_y   = rotation_unitary(tuple([1] * n))
    rows  = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_shadow_row)(p, U, shots_per_probe, int(s), n, U_x, U_y)
        for p, s in zip(probes, seeds))
    return np.array(rows, dtype=float)

def reconstruct(X, true_h, n):
    w = operator_hs_norms(n)

    _, svals, Vh = np.linalg.svd(X, full_matrices=False)
    h_svd = Vh[-1]
    h_svd /= np.linalg.norm(h_svd)

    if np.dot(h_svd, true_h) < 0:
        h_svd = -h_svd

    h_phys = h_svd / np.linalg.norm(w * h_svd)

    overlap   = float(np.dot(w * h_phys, w * true_h))
    coeff_err = float(np.linalg.norm(h_phys - true_h))

    return {"h": h_phys, "svals": svals, "overlap": overlap, "coeff_err": coeff_err}

def run_trial(n, t, n_probes, shots_per_probe,
              seed_instance=0, seed_probes=1, seed_shadows=2,
              n_jobs=N_JOBS):

    rng_i = np.random.default_rng(seed_instance)
    rng_p = np.random.default_rng(seed_probes)
    rng_s = np.random.default_rng(seed_shadows)

    a_coeffs, b_coeffs, c_coeffs, names, true_h, H_true = random_xyz_instance(n, rng_i)

    w    = operator_hs_norms(n)
    h_hs = np.linalg.norm(w * true_h)
    print(f"[check] ||H_true||_HS = {h_hs:.6f}   (must be 1.000000)")
    print(f"[check] ||h_true||_2  = {np.linalg.norm(true_h):.6f}  (= 1 since all w_k=1)")

    print(f"[setup] Building H ({2**n}x{2**n}, {3*(n-1)} terms) ... ", end="", flush=True)
    t0 = time.perf_counter()
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    print(f"[setup] Computing U = exp(-itH), n={n}, d={2**n} ... ", end="", flush=True)
    t0 = time.perf_counter()
    U  = expm(-1j * t * H_true)
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    probes = sample_product_probes(n, n_probes, rng_p)

    print(f"[run]   Exact feature matrix  ({n_probes} x {3*(n-1)}) ...")
    t1 = time.perf_counter()
    X    = build_feature_matrix_exact(U, probes, n_jobs=n_jobs)
    t2   = time.perf_counter()
    print(f"        done ({t2-t1:.1f}s)")

    print(f"[run]   Shadow feature matrix ({n_probes} x {3*(n-1)}, nu={shots_per_probe}) ...")
    Xhat = build_feature_matrix_shadow(U, probes, shots_per_probe, rng_s, n_jobs=n_jobs)
    t3   = time.perf_counter()
    print(f"        done ({t3-t2:.1f}s)")

    rec_e = reconstruct(X,    true_h, n)
    rec_n = reconstruct(Xhat, true_h, n)
    Delta = Xhat - X

    return {
        "n": n, "d": 2**n,
        "a_coeffs": a_coeffs, "b_coeffs": b_coeffs, "c_coeffs": c_coeffs,
        "names": names, "true_h": true_h,
        "H_true": H_true, "U": U, "probes": probes,
        "X": X, "Xhat": Xhat, "Delta": Delta,
        "residual_exact":  float(np.linalg.norm(X @ true_h)),
        "noise_op":        float(np.linalg.norm(Delta, 2)),
        "noise_fro":       float(np.linalg.norm(Delta, "fro")),
        "exact":           rec_e,
        "noisy":           rec_n,
        "time_exact_s":    t2 - t1,
        "time_shadow_s":   t3 - t2,
    }

def print_summary(res, shots_per_probe=None):
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    ge = se[-2] - se[-1] if len(se) >= 2 else float("nan")
    gn = sn[-2] - sn[-1] if len(sn) >= 2 else float("nan")
    ratio = res["noise_op"] / ge if ge > 0 else float("nan")
    n = res["n"]
    m = n - 1

    print("\n" + "=" * 64)
    print(f" Full XYZ chain  n={n}  d={res['d']}  |V|={len(res['true_h'])}  OBC")
    if shots_per_probe:
        print(f" N_S={res['X'].shape[0]}  nu={shots_per_probe}  estimator=three-basis")
    print(" reconstruct: plain SVD + unit rescaling  (HS=Euclidean since all w_k=1)")
    print("=" * 64)
    print(f" ||X h_true||        = {res['residual_exact']:.2e}   (-> 0)")
    print(f" sigma_min(X)        = {se[-1]:.4e}")
    print(f" gap(X)              = {ge:.4e}")
    print(f" exact overlap       = {res['exact']['overlap']:.6f}  <- must be ~1.0000")
    print(f" exact coeff err     = {res['exact']['coeff_err']:.4e}")
    print()
    print(f" sigma_min(Xhat)     = {sn[-1]:.4e}")
    print(f" gap(Xhat)           = {gn:.4e}")
    print(f" noisy overlap       = {res['noisy']['overlap']:.6f}  <- TARGET > 0.99")
    print(f" noisy coeff err     = {res['noisy']['coeff_err']:.4e}")
    print()
    print(f" ||Xhat-X||_2        = {res['noise_op']:.4e}")
    print(f" ||Xhat-X||_F        = {res['noise_fro']:.4e}")
    print(f" noise/gap ratio     = {ratio:.4e}  <- must be << 1")
    print(f" wall time (exact)   = {res['time_exact_s']:.1f}s")
    print(f" wall time (shadow)  = {res['time_shadow_s']:.1f}s")
    print("=" * 64)
    print()

    h_est = res["noisy"]["h"]
    print(f"  {'bond':<6} {'a_true':>9} {'a_est':>9} {'err_a':>8}  "
          f"{'b_true':>9} {'b_est':>9} {'err_b':>8}  "
          f"{'c_true':>9} {'c_est':>9} {'err_c':>8}")
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*8}  "
          f"{'-'*9} {'-'*9} {'-'*8}  "
          f"{'-'*9} {'-'*9} {'-'*8}")
    for i in range(m):
        a_t = res["true_h"][i]
        a_e = h_est[i]
        b_t = res["true_h"][m + i]
        b_e = h_est[m + i]
        c_t = res["true_h"][2*m+i]
        c_e = h_est[2*m+i]
        print(f"  {i:>2}-{i+1:<3} "
              f"{a_t:>+9.4f} {a_e:>+9.4f} {a_e-a_t:>+8.4f}  "
              f"{b_t:>+9.4f} {b_e:>+9.4f} {b_e-b_t:>+8.4f}  "
              f"{c_t:>+9.4f} {c_e:>+9.4f} {c_e-c_t:>+8.4f}")
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
        prefix = f"xyz_full_n{n}_t{t}_NS{n_probes}_nu{shots_per_probe}"

    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    ge = se[-2] - se[-1]
    gn = sn[-2] - sn[-1]
    ratio = res["noise_op"] / ge if ge > 0 else float("nan")

    _save_dict_csv(out_dir / f"{prefix}_summary.csv", {
        "n": n, "d": res["d"], "t": t, "n_probes": n_probes,
        "shots_per_probe": shots_per_probe,
        "residual_exact":    res["residual_exact"],
        "sigma_min_X":       se[-1], "sigma_2_X": se[-2], "gap_X": ge,
        "exact_overlap":     res["exact"]["overlap"],
        "exact_coeff_err":   res["exact"]["coeff_err"],
        "sigma_min_Xhat":    sn[-1], "sigma_2_Xhat": sn[-2], "gap_Xhat": gn,
        "noisy_overlap":     res["noisy"]["overlap"],
        "noisy_coeff_err":   res["noisy"]["coeff_err"],
        "noise_op":          res["noise_op"], "noise_fro": res["noise_fro"],
        "noise_gap_ratio":   ratio,
        "time_exact_s":      res["time_exact_s"],
        "time_shadow_s":     res["time_shadow_s"],
    })

    _save_rows_csv(
        out_dir / f"{prefix}_coefficients.csv",
        ["bond", "name", "true_h", "exact_h", "noisy_h", "err_exact", "err_noisy"],
        [[i, nm, tc, ec, nc, ec-tc, nc-tc]
         for i, (nm, tc, ec, nc) in enumerate(zip(
             res["names"], res["true_h"],
             res["exact"]["h"], res["noisy"]["h"]))]
    )

    kmax = max(len(se), len(sn))
    _save_rows_csv(
        out_dir / f"{prefix}_svals.csv",
        ["k", "sval_exact", "sval_shadow"],
        [[k, se[k] if k < len(se) else "", sn[k] if k < len(sn) else ""]
         for k in range(kmax)]
    )

    np.savetxt(out_dir / f"{prefix}_X_exact.csv",  res["X"],     delimiter=",")
    np.savetxt(out_dir / f"{prefix}_Xhat.csv",     res["Xhat"],  delimiter=",")
    np.savetxt(out_dir / f"{prefix}_Delta.csv",    res["Delta"], delimiter=",")
    print(f"[csv] saved -> {out_dir}/{prefix}_*.csv")

def plot_scatter(true_h, est_h, title, out_path):
    m = len(true_h) // 3
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(true_h[:m],    est_h[:m],    s=55, alpha=0.85, label="XX bonds (a_i)", zorder=3)
    ax.scatter(true_h[m:2*m], est_h[m:2*m], s=55, alpha=0.85, marker="^", label="YY bonds (b_i)", zorder=3)
    ax.scatter(true_h[2*m:],  est_h[2*m:],  s=55, alpha=0.85, marker="s", label="ZZ bonds (c_i)", zorder=3)
    lim = 1.15 * max(np.max(np.abs(true_h)), np.max(np.abs(est_h)))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("true h")
    ax.set_ylabel("reconstructed h")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")

def plot_couplings_per_bond(res, out_path):
    n   = res["n"]
    m = n - 1
    h_t = res["true_h"]
    h_e = res["noisy"]["h"]
    x   = np.arange(m)
    bw = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, offset, ct, ce, title in [
        (axes[0], 0,   "steelblue",    "tomato",     "XX coupling a_i per bond"),
        (axes[1], m,   "mediumpurple", "goldenrod",  "YY coupling b_i per bond"),
        (axes[2], 2*m, "seagreen",     "darkorange", "ZZ coupling c_i per bond"),
    ]:
        ax.bar(x - bw/2, h_t[offset:offset+m], bw, label="true",    color=ct, alpha=0.8)
        ax.bar(x + bw/2, h_e[offset:offset+m], bw, label="learned", color=ce, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}-{i+1}" for i in range(m)], fontsize=9)
        ax.set_xlabel("bond")
        ax.set_title(title)
        ax.legend(frameon=False)

    overlap = res["noisy"]["overlap"]
    plt.suptitle(f"Full XYZ chain  n={n}  coupling recovery  (overlap={overlap:.4f})", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")

def plot_svals(res, out_path, k_show=30):
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    k  = min(k_show, len(se))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, k+1), se[-k:], "o-",  label="exact X")
    ax.plot(range(1, k+1), sn[-k:], "s--", label="shadow Xhat (three-basis)")
    ax.set_xlabel("index (smallest singular values)")
    ax.set_ylabel("singular value")
    ax.set_title(f"Bottom singular values  —  full XYZ n={res['n']}")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")

def _bench_trial(n, t, n_probes, shots, seed0):
    res = run_trial(n=n, t=t, n_probes=n_probes, shots_per_probe=shots,
                    seed_instance=seed0, seed_probes=seed0+1,
                    seed_shadows=seed0+2, n_jobs=1)
    se = res["exact"]["svals"]
    sn = res["noisy"]["svals"]
    ge = se[-2] - se[-1]
    return {
        "gap_ex":    ge,
        "sigma1_no": sn[-1],
        "gap_no":    sn[-2] - sn[-1],
        "overlap":   res["noisy"]["overlap"],
        "coeff_err": res["noisy"]["coeff_err"],
        "ratio":     res["noise_op"] / ge if ge > 0 else float("nan"),
    }

def shot_sweep(n, t, n_probes, shot_grid, n_trials=5, master_seed=1234, n_jobs=N_JOBS):
    rng   = np.random.default_rng(master_seed)
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
        seeds   = rng.integers(10**9, size=n_trials).tolist()
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bench_trial)(n, t, n_probes, shots, int(s)) for s in seeds)
        stats["shots"].append(shots)
        for key, sk in [
            ("gap_ex",    "exact_gap"),   ("sigma1_no", "sigma_min_hat"),
            ("gap_no",    "gap_hat"),     ("overlap",   "overlap"),
            ("coeff_err", "coeff_err"),   ("ratio",     "noise_gap_ratio"),
        ]:
            vals = np.array([r[key] for r in results])
            stats[f"{sk}_mean"].append(np.nanmean(vals))
            stats[f"{sk}_std"].append(np.nanstd(vals))
    return {k: np.array(v) for k, v in stats.items()}

if __name__ == "__main__":

    out_dir = Path("./results_xyz_full")
    out_dir.mkdir(exist_ok=True)

    n               = 12
    t               = 10
    n_probes        = 60_000
    shots_per_probe = 300_000

    print(f"\n{'='*64}")
    print(f" Full XYZ chain  n={n}  |V|={3*(n-1)}  OBC  Gaussian couplings")
    print(f" N_S={n_probes}  nu={shots_per_probe}  N_JOBS={N_JOBS}")
    print(f" H = sum_i [ a_i XiXi+1 + b_i YiYi+1 + c_i ZiZi+1 ]")
    print(f" reconstruct: plain SVD (HS=Euclidean metric)")
    print(f"{'='*64}\n")

    res = run_trial(
        n=n, t=t,
        n_probes=n_probes,
        shots_per_probe=shots_per_probe,
        seed_instance=123,
        seed_probes=456,
        seed_shadows=789,
    )

    print_summary(res, shots_per_probe)

    save_trial_csvs(res, t=t, n_probes=n_probes, shots_per_probe=shots_per_probe,
                    out_dir=out_dir)

    plot_scatter(res["true_h"], res["exact"]["h"],
                 f"Full XYZ exact, n={n}",
                 out_dir / f"scatter_exact_n{n}.png")
    plot_scatter(res["true_h"], res["noisy"]["h"],
                 f"Full XYZ three-basis, n={n}, nu={shots_per_probe}",
                 out_dir / f"scatter_threebasis_n{n}.png")
    plot_couplings_per_bond(res, out_dir / f"couplings_n{n}.png")
    plot_svals(res, out_dir / f"svals_n{n}.png")

    # shot_grid = [20_000, 50_000, 100_000, 200_000, 300_000, 500_000]
    # stats = shot_sweep(n=n, t=t, n_probes=n_probes, shot_grid=shot_grid, n_trials=3)
    # _save_rows_csv(out_dir / f"shot_sweep_n{n}.csv",
    #     list(stats.keys()), list(zip(*stats.values())))