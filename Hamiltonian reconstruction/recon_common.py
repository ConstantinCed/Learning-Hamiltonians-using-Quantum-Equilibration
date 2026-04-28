"""Shared utilities for Hamiltonian reconstruction (TFIM / XYZ / 2-local NN).

Provides:
  - Pauli algebra helpers (single- and two-site labels, kron, rotation gates)
  - Pauli-string descriptors for O(d) matvec  (P|psi>) without storing P
  - Product-state probe sampling
  - Generic exact input/output expectation routines
  - Generic Hamiltonian builder (NOT HS-normalized; coeffs left raw)
  - Feature matrix builders (exact & shadow), reconstruction, error metrics
  - A generic run_trial that takes model-specific callbacks

Each model script (TFIM/XYZ/2-local-NN) only needs to provide:
  - family_labels(n)        -> (labels, names)
  - random_instance(n, rng) -> (labels, names, coeffs)
  - estimate_shadow(psi_t, n, n_shots, rng) -> estimated expectations
        aligned with family_labels(n).

The recovered direction `true_h = coeffs / ||coeffs||_2` is the unit vector
the SVD null-space reconstruction can recover (H itself is left UNNORMALIZED).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import time

import numpy as np
from scipy.linalg import expm
from joblib import Parallel, delayed, cpu_count

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

N_JOBS = cpu_count()

# ──────────────────────────────────────────────────
#  Pauli algebra
# ──────────────────────────────────────────────────
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_1Q = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

Hgate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Sdg   = np.array([[1, 0], [0, -1j]], dtype=complex)
# Rotation that diagonalises X, Y, Z respectively (0->X, 1->Y, 2->Z).
BASIS_ROT_LIST = [Hgate, Hgate @ Sdg, I2]

PAULI_TO_BASIS = {"X": 0, "Y": 1, "Z": 2}


def kron_all(mats):
    out = np.array([[1]], dtype=complex)
    for M in mats:
        out = np.kron(out, M)
    return out


def pauli_label_to_matrix(label):
    return kron_all([PAULI_1Q[c] for c in label])


def rotation_unitary(basis_tuple):
    return kron_all([BASIS_ROT_LIST[b] for b in basis_tuple])


def single_site_label(n, i, p):
    s = ["I"] * n
    s[i] = p
    return "".join(s)


def two_site_label(n, i, j, p, q):
    s = ["I"] * n
    s[i] = p
    s[j] = q
    return "".join(s)


# ──────────────────────────────────────────────────
#  Pauli descriptor: P|s> = phases[s] |s XOR flip>   (O(d) memory & matvec)
# ──────────────────────────────────────────────────
def pauli_descriptor(n, label):
    d = 2 ** n
    flip = 0
    phases = np.ones(d, dtype=complex)
    states = np.arange(d)
    for site, p in enumerate(label):
        if p == "I":
            continue
        bit = n - 1 - site
        mask = 1 << bit
        bit_vals = (states >> bit) & 1
        if p == "X":
            flip ^= mask
        elif p == "Y":
            flip ^= mask
            phases *= np.where(bit_vals == 0, 1j, -1j)
        elif p == "Z":
            phases *= 1 - 2 * bit_vals
    return flip, phases


def pauli_expectation(psi, descriptor):
    flip, phases = descriptor
    # <psi| P |psi> = sum_s conj(psi[s XOR flip]) * phases[s] * psi[s]
    idx = np.arange(len(psi)) ^ flip
    return float(np.real(np.sum(np.conj(psi[idx]) * phases * psi)))


# ──────────────────────────────────────────────────
#  Hamiltonian (unnormalized)
# ──────────────────────────────────────────────────
def build_hamiltonian(labels, coeffs):
    """Return (H, true_h, descriptors) with H = sum_i c_i P_i (raw, unnormalized)
    and true_h = coeffs / ||coeffs||_2 (unit-norm direction)."""
    n = len(labels[0])
    d = 2 ** n
    descriptors = [pauli_descriptor(n, lbl) for lbl in labels]
    H = np.zeros((d, d), dtype=complex)
    arange_d = np.arange(d)
    for c, (flip, phases) in zip(coeffs, descriptors):
        H[arange_d ^ flip, arange_d] += c * phases
    true_h = coeffs / np.linalg.norm(coeffs)
    return H, true_h, descriptors


# ──────────────────────────────────────────────────
#  Product-state probes
# ──────────────────────────────────────────────────
ket0   = np.array([1, 0], dtype=complex)
ket1   = np.array([0, 1], dtype=complex)
ketp   = np.array([1, 1], dtype=complex) / np.sqrt(2)
ketm   = np.array([1, -1], dtype=complex) / np.sqrt(2)
kety_p = np.array([1, 1j], dtype=complex) / np.sqrt(2)
kety_m = np.array([1, -1j], dtype=complex) / np.sqrt(2)

# (basis_index, sign, ket)  with basis_index 0=X, 1=Y, 2=Z
PROBE_OPTIONS = [
    (0, +1., ketp),  (0, -1., ketm),
    (1, +1., kety_p), (1, -1., kety_m),
    (2, +1., ket0),  (2, -1., ket1),
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
        for q in range(n):
            b, s, v = PROBE_OPTIONS[rng.integers(len(PROBE_OPTIONS))]
            bases[q] = b
            signs[q] = s
            vecs.append(v)
        probes.append({"basis": bases, "sign": signs, "psi": kron_vec(vecs)})
    return probes


# ──────────────────────────────────────────────────
#  Generic exact expectations
# ──────────────────────────────────────────────────
def exact_input_expectations(labels, probe):
    """For a product-state probe, <psi|P|psi> for each Pauli string label.
    Nonzero only when every non-I site of the label matches the probe's
    eigenbasis on that qubit; value is the product of the corresponding signs."""
    basis = probe["basis"]
    sign = probe["sign"]
    out = np.zeros(len(labels))
    for k, lbl in enumerate(labels):
        v = 1.0
        ok = True
        for q, p in enumerate(lbl):
            if p == "I":
                continue
            if basis[q] != PAULI_TO_BASIS[p]:
                ok = False
                break
            v *= sign[q]
        out[k] = v if ok else 0.0
    return out


def exact_output_expectations(psi_t, descriptors):
    return np.array([pauli_expectation(psi_t, D) for D in descriptors])


# ──────────────────────────────────────────────────
#  Feature matrix builders
# ──────────────────────────────────────────────────
def _exact_row(probe, U, labels, descriptors):
    psi_t = U @ probe["psi"]
    ein = exact_input_expectations(labels, probe)
    eout = exact_output_expectations(psi_t, descriptors)
    return (ein - eout) / U.shape[0]


def _shadow_row(probe, U, labels, n, n_shots, seed, estimate_shadow_fn):
    rng = np.random.default_rng(seed)
    psi_t = U @ probe["psi"]
    ein = exact_input_expectations(labels, probe)
    eout = estimate_shadow_fn(psi_t, n, n_shots, rng)
    return (ein - eout) / U.shape[0]


def build_feature_matrix_exact(U, probes, labels, descriptors, n_jobs=N_JOBS):
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_exact_row)(p, U, labels, descriptors) for p in probes
    )
    return np.array(rows, dtype=float)


def build_feature_matrix_shadow(U, probes, labels, shots_per_probe, rng,
                                estimate_shadow_fn, n_jobs=N_JOBS):
    n = int(np.log2(U.shape[0]))
    seeds = rng.integers(0, 2 ** 31, size=len(probes))
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_shadow_row)(p, U, labels, n, shots_per_probe, int(s),
                             estimate_shadow_fn)
        for p, s in zip(probes, seeds)
    )
    return np.array(rows, dtype=float)


# ──────────────────────────────────────────────────
#  Reconstruction & error metrics
# ──────────────────────────────────────────────────
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
        "coeff_err": float(np.linalg.norm(h - true_h)),
    }


def operator_norm_error(h_est, true_h, labels):
    diff = h_est - true_h
    n = len(labels[0])
    d = 2 ** n
    H_diff = np.zeros((d, d), dtype=complex)
    arange_d = np.arange(d)
    for c, lbl in zip(diff, labels):
        flip, phases = pauli_descriptor(n, lbl)
        H_diff[arange_d ^ flip, arange_d] += c * phases
    return float(np.linalg.norm(H_diff, 2))


def avg_coeff_error(h_est, true_h):
    return float(np.mean(np.abs(h_est - true_h)))


def max_coeff_error(h_est, true_h):
    return float(np.max(np.abs(h_est - true_h)))


# ──────────────────────────────────────────────────
#  Trial runner
# ──────────────────────────────────────────────────
def run_trial(n, t, random_instance_fn, estimate_shadow_fn,
              n_probes, shots_per_probe,
              seed_instance=0, seed_probes=1, seed_shadows=2, n_jobs=N_JOBS):
    rng_i = np.random.default_rng(seed_instance)
    rng_p = np.random.default_rng(seed_probes)
    rng_s = np.random.default_rng(seed_shadows)

    labels, names, coeffs = random_instance_fn(n, rng_i)
    H_true, true_h, descriptors = build_hamiltonian(labels, coeffs)
    U = expm(-1j * t * H_true)

    probes = sample_product_probes(n, n_probes, rng_p)

    t0 = time.perf_counter()
    X = build_feature_matrix_exact(U, probes, labels, descriptors, n_jobs=n_jobs)
    t1 = time.perf_counter()
    Xhat = build_feature_matrix_shadow(U, probes, labels, shots_per_probe,
                                       rng_s, estimate_shadow_fn, n_jobs=n_jobs)
    t2 = time.perf_counter()

    rec_e = reconstruct(X, true_h)
    rec_n = reconstruct(Xhat, true_h)
    Delta = Xhat - X

    op_e   = operator_norm_error(rec_e["h"], true_h, labels)
    op_n   = operator_norm_error(rec_n["h"], true_h, labels)
    avg_e  = avg_coeff_error(rec_e["h"], true_h)
    avg_n  = avg_coeff_error(rec_n["h"], true_h)
    max_n  = max_coeff_error(rec_n["h"], true_h)

    se = rec_e["svals"]
    ge = se[-2] - se[-1] if len(se) >= 2 else float("nan")

    return {
        "n": n, "d": 2 ** n, "labels": labels, "names": names,
        "coeffs_raw": coeffs, "true_h": true_h, "H_true": H_true, "U": U,
        "probes": probes, "X": X, "Xhat": Xhat, "Delta": Delta,
        "residual_exact": float(np.linalg.norm(X @ true_h)),
        "noise_op":   float(np.linalg.norm(Delta, 2)),
        "noise_fro":  float(np.linalg.norm(Delta, "fro")),
        "exact": rec_e, "noisy": rec_n,
        "op_norm_exact": op_e, "op_norm_noisy": op_n,
        "avg_coeff_err_exact": avg_e,
        "avg_coeff_err_noisy": avg_n,
        "max_coeff_err_noisy": max_n,
        "gap_exact": ge,
        "noise_gap_ratio": float(np.linalg.norm(Delta, 2)) / ge if ge > 0
                           else float("nan"),
        "time_exact_s":  t1 - t0,
        "time_shadow_s": t2 - t1,
    }


# ──────────────────────────────────────────────────
#  Output helpers
# ──────────────────────────────────────────────────
def print_summary(res, shots_per_probe=None):
    se = res["exact"]["svals"]
    ge = res["gap_exact"]
    ratio = res["noise_gap_ratio"]
    print("\n" + "=" * 56)
    print(f" n={res['n']}, d={res['d']}, |V|={len(res['true_h'])}")
    if shots_per_probe is not None:
        print(f" N_S={res['X'].shape[0]}, nu={shots_per_probe}")
    print("=" * 56)
    print(f" ||X h_true||            = {res['residual_exact']:.2e}")
    print(f" exact overlap           = {res['exact']['overlap']:.6f}")
    print(f" exact ||Ĥ-H||_op       = {res['op_norm_exact']:.4e}")
    print()
    print(f" noisy overlap           = {res['noisy']['overlap']:.6f}")
    print(f" noisy ||Ĥ-H||_op       = {res['op_norm_noisy']:.4e}")
    print(f" noisy avg |Δh_i|        = {res['avg_coeff_err_noisy']:.4e}")
    print(f" noisy max |Δh_i|        = {res['max_coeff_err_noisy']:.4e}")
    print()
    print(f" ||Xhat-X||_2            = {res['noise_op']:.4e}")
    print(f" noise/gap ratio         = {ratio:.4e}")
    print(f" wall time (exact)       = {res['time_exact_s']:.1f}s")
    print(f" wall time (shadow)      = {res['time_shadow_s']:.1f}s")
    print("=" * 56)


def save_reconstruction_csv(res, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "name", "label", "true_h", "exact_h", "noisy_h",
                    "err_exact", "err_noisy"])
        for i, (nm, lb, th, eh, nh) in enumerate(zip(
                res["names"], res["labels"], res["true_h"],
                res["exact"]["h"], res["noisy"]["h"])):
            w.writerow([i, nm, lb, th, eh, nh, eh - th, nh - th])
    print(f"[csv] {out_path}")
