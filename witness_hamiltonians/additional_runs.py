"""Sparse backend: rank of C(h) via the spectrum of the dense Gram C^H C."""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from witness_structured import (  # noqa: E402
    Job,
    build_local_family_for_job,
)


def _bit_count(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    arr = np.asarray(x, dtype=np.uint64)
    if arr.shape == ():
        return np.array(int(arr).bit_count(), dtype=np.int64)
    byte_view = arr.view(np.uint8).reshape(arr.shape + (8,))
    return np.unpackbits(byte_view, axis=-1).sum(axis=-1).astype(np.int64)


def pauli_to_symplectic_words(
    U_ops: List[Tuple[str, ...]],
) -> Tuple[np.ndarray, np.ndarray, int]:
    m = len(U_ops)
    n_q = len(U_ops[0]) if m > 0 else 0
    n_words = max(1, (n_q + 63) // 64)
    xs = np.zeros((m, n_words), dtype=np.uint64)
    zs = np.zeros((m, n_words), dtype=np.uint64)
    for i, op in enumerate(U_ops):
        for j, p in enumerate(op):
            word = j // 64
            bit = np.uint64(1) << np.uint64(j % 64)
            if p == "X":
                xs[i, word] |= bit
            elif p == "Z":
                zs[i, word] |= bit
            elif p == "Y":
                xs[i, word] |= bit
                zs[i, word] |= bit
    return xs, zs, n_q


def _bit_count_rows(arr: np.ndarray) -> np.ndarray:
    return _bit_count(arr).sum(axis=1).astype(np.int64)


def precompute_sparse_structure(
    U_ops: List[Tuple[str, ...]],
    root_patch: int,
) -> Dict[str, Any]:
    m = len(U_ops)
    xs, zs, n_q = pauli_to_symplectic_words(U_ops)
    n_words = xs.shape[1]
    Y_count = _bit_count_rows(xs & zs)

    root_word = root_patch // 64
    root_mask = np.uint64(1) << np.uint64(root_patch % 64)
    i_powers = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)

    chunk_keys: List[np.ndarray] = []
    chunk_iu: List[np.ndarray] = []
    chunk_iv: List[np.ndarray] = []
    chunk_phase: List[np.ndarray] = []

    for iu in range(m):
        xu = xs[iu]
        zu = zs[iu]
        Yu = int(Y_count[iu])

        p1 = _bit_count_rows(xu & zs)
        p2 = _bit_count_rows(zu & xs)
        anticomm = ((p1 + p2) & 1).astype(bool)
        anticomm[iu] = False

        w_x_all = xs ^ xu
        w_z_all = zs ^ zu
        nontrivial = ((w_x_all[:, root_word] | w_z_all[:, root_word]) & root_mask) != 0
        mask = anticomm & nontrivial

        ivs = np.nonzero(mask)[0]
        if ivs.size == 0:
            continue

        w_xa = w_x_all[ivs]
        w_za = w_z_all[ivs]

        Yw = _bit_count_rows(w_xa & w_za)
        Yv = Y_count[ivs]
        zu_xv_parity = _bit_count_rows(zu & xs[ivs]) & 1
        sign = (1 - 2 * zu_xv_parity).astype(np.int64)

        k_mod_4 = (Yu + Yv - Yw) & 3
        i_pow = i_powers[k_mod_4]
        ph_uv = sign.astype(np.complex128) * i_pow
        coeff = 2.0 * ph_uv

        keys = np.concatenate(
            [w_xa.astype(np.uint64), w_za.astype(np.uint64)],
            axis=1,
        )
        chunk_keys.append(keys)
        chunk_iu.append(np.full(ivs.size, iu, dtype=np.int64))
        chunk_iv.append(ivs.astype(np.int64))
        chunk_phase.append(coeff)

    if not chunk_keys:
        return {
            "rows": np.empty(0, dtype=np.int64),
            "cols": np.empty(0, dtype=np.int64),
            "iv_idx": np.empty(0, dtype=np.int64),
            "phases": np.empty(0, dtype=np.complex128),
            "n_W": 0,
            "n_U": m,
        }

    all_keys = np.vstack(chunk_keys).reshape(-1, 2 * n_words)
    all_iu = np.concatenate(chunk_iu)
    all_iv = np.concatenate(chunk_iv)
    all_phase = np.concatenate(chunk_phase)

    _unique_keys, inverse = np.unique(all_keys, axis=0, return_inverse=True)
    n_W = int(_unique_keys.shape[0])

    return {
        "rows": inverse.astype(np.int64),
        "cols": all_iu,
        "iv_idx": all_iv,
        "phases": all_phase,
        "n_W": n_W,
        "n_U": m,
        "n_patch_sites": int(n_q),
    }


def _op_weights(U_ops: List[Tuple[str, ...]]) -> np.ndarray:
    return np.array(
        [sum(p != "I" for p in op) for op in U_ops],
        dtype=np.int64,
    )


def precompute_sparse_structure_active_h(
    U_ops: List[Tuple[str, ...]],
    root_patch: int,
    active_iv: np.ndarray,
    progress_every: int = 0,
) -> Dict[str, Any]:
    """Precompute C(h) structure when h is supported only on active terms.

    This is the practical path for large dense k=3,4 formal OBC certificates:
    the certificate vector may be sparse, so we only enumerate commutators
    [P_u, P_v] with v in supp(h), instead of all U x U pairs.
    """
    m = len(U_ops)
    xs, zs, n_q = pauli_to_symplectic_words(U_ops)
    n_words = xs.shape[1]
    Y_count = _bit_count_rows(xs & zs)

    root_word = root_patch // 64
    root_mask = np.uint64(1) << np.uint64(root_patch % 64)
    i_powers = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)

    chunk_keys: List[np.ndarray] = []
    chunk_iu: List[np.ndarray] = []
    chunk_iv: List[np.ndarray] = []
    chunk_phase: List[np.ndarray] = []

    t0 = time.time()
    candidate_count = 0
    for count, iv in enumerate(active_iv, 1):
        xv = xs[iv]
        zv = zs[iv]
        Yv = int(Y_count[iv])

        p1 = _bit_count_rows(xs & zv)
        p2 = _bit_count_rows(zs & xv)
        mask = ((p1 + p2) & 1).astype(bool)
        mask[iv] = False

        w_x_all = xs ^ xv
        w_z_all = zs ^ zv
        nontrivial = ((w_x_all[:, root_word] | w_z_all[:, root_word]) & root_mask) != 0
        mask &= nontrivial

        ius = np.nonzero(mask)[0]
        if ius.size == 0:
            continue

        candidate_count += int(ius.size)
        w_xa = w_x_all[ius]
        w_za = w_z_all[ius]

        Yw = _bit_count_rows(w_xa & w_za)
        zu_xv_parity = _bit_count_rows(zs[ius] & xv) & 1
        sign = (1 - 2 * zu_xv_parity).astype(np.int64)

        k_mod_4 = (Y_count[ius] + Yv - Yw) & 3
        ph_uv = sign.astype(np.complex128) * i_powers[k_mod_4]
        coeff = 2.0 * ph_uv

        keys = np.concatenate(
            [w_xa.astype(np.uint64), w_za.astype(np.uint64)],
            axis=1,
        )
        chunk_keys.append(keys)
        chunk_iu.append(ius.astype(np.int64))
        chunk_iv.append(np.full(ius.size, iv, dtype=np.int64))
        chunk_phase.append(coeff)

        if progress_every and count % progress_every == 0:
            print(
                f"  active precompute {count}/{len(active_iv)} "
                f"pairs={candidate_count} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    if not chunk_keys:
        return {
            "rows": np.empty(0, dtype=np.int64),
            "cols": np.empty(0, dtype=np.int64),
            "iv_idx": np.empty(0, dtype=np.int64),
            "phases": np.empty(0, dtype=np.complex128),
            "n_W": 0,
            "n_U": m,
            "n_active_h": int(len(active_iv)),
            "n_candidate_pairs": 0,
            "n_patch_sites": int(n_q),
        }

    all_keys = np.vstack(chunk_keys).reshape(-1, 2 * n_words)
    all_iu = np.concatenate(chunk_iu)
    all_iv = np.concatenate(chunk_iv)
    all_phase = np.concatenate(chunk_phase)

    _unique_keys, inverse = np.unique(all_keys, axis=0, return_inverse=True)
    n_W = int(_unique_keys.shape[0])

    return {
        "rows": inverse.astype(np.int64),
        "cols": all_iu,
        "iv_idx": all_iv,
        "phases": all_phase,
        "n_W": n_W,
        "n_U": m,
        "n_active_h": int(len(active_iv)),
        "n_candidate_pairs": int(candidate_count),
        "n_patch_sites": int(n_q),
    }


def commutator_rank_for_h(
    precomp: Dict[str, Any],
    h: np.ndarray,
    tol_rel: float = 1e-9,
    iterative_gram_threshold_gb: float = 0.1,
) -> Tuple[int, float, Dict[str, float]]:
    rows = precomp["rows"]
    cols = precomp["cols"]
    iv_idx = precomp["iv_idx"]
    phases = precomp["phases"]
    n_W = precomp["n_W"]
    m = precomp["n_U"]
    gram_gb = (m * m * 16) / (1024 ** 3)

    timings: Dict[str, float] = {}

    t = time.time()
    data = h[iv_idx] * phases
    C = sp.coo_matrix((data, (rows, cols)), shape=(n_W, m)).tocsr()
    timings["build_C"] = time.time() - t

    if gram_gb > iterative_gram_threshold_gb:
        rank, max_eig, iterative_timings = certify_rank_iterative(
            C, h, tol_rel=tol_rel
        )
        timings.update(iterative_timings)
        return rank, max_eig, timings

    t = time.time()
    G_sp = (C.conj().T @ C)
    timings["gram_sparse"] = time.time() - t

    t = time.time()
    G = G_sp.toarray()
    G = 0.5 * (G + G.conj().T)
    timings["gram_dense"] = time.time() - t

    t = time.time()
    eigs = np.linalg.eigvalsh(G)
    timings["eigvalsh"] = time.time() - t

    max_eig = float(eigs[-1]) if eigs[-1] > 0 else 1.0
    tol = tol_rel * max_eig
    rank = int(np.sum(eigs > tol))
    return rank, max_eig, timings


def certify_rank_iterative(
    C: sp.csr_matrix,
    h: np.ndarray,
    tol_rel: float = 1e-9,
    eigsh_tol: float = 1e-6,
    maxiter: Optional[int] = None,
) -> Tuple[int, float, Dict[str, float]]:
    timings: Dict[str, float] = {"rank_method_iterative": 1.0}
    m = C.shape[1]
    target = m - 1
    h = np.asarray(h, dtype=np.complex128)
    h_norm = float(np.linalg.norm(h))
    if h_norm == 0.0:
        h = h.copy()
        h[0] = 1.0
        h_norm = 1.0
    hhat = h / h_norm

    def gram_mv(x):
        y = C.conj().T @ (C @ x)
        return np.asarray(y).reshape(-1)

    Gop = spla.LinearOperator((m, m), matvec=gram_mv, dtype=np.complex128)

    t = time.time()
    max_eig = float(
        spla.eigsh(
            Gop,
            k=1,
            which="LA",
            tol=eigsh_tol,
            maxiter=maxiter,
            return_eigenvectors=False,
        )[0].real
    )
    timings["eigsh_max"] = time.time() - t

    alpha = 10.0 * max(max_eig, 1.0)

    def augmented_mv(x):
        return gram_mv(x) + alpha * hhat * np.vdot(hhat, x)

    Aop = spla.LinearOperator((m, m), matvec=augmented_mv, dtype=np.complex128)

    t = time.time()
    min_aug = float(
        spla.eigsh(
            Aop,
            k=1,
            which="SA",
            tol=eigsh_tol,
            maxiter=maxiter,
            return_eigenvectors=False,
        )[0].real
    )
    timings["eigsh_min_h_perp"] = time.time() - t
    timings["lambda_min_h_perp"] = min_aug
    timings["lambda_max"] = max_eig

    tol = tol_rel * max(max_eig, 1.0)
    rank = target if min_aug > tol else target - 1
    return rank, max_eig, timings


def search_witness(
    U_ops: List[Tuple[str, ...]],
    root_patch: int,
    trials: int,
    seed: int,
    coeff_bound: int = 3,
    gram_cap_gb: float = 8.0,
    tol_rel: float = 1e-9,
    verbose: bool = True,
    witness_weight_cap: Optional[int] = None,
) -> Dict[str, Any]:
    m = len(U_ops)
    target = m - 1
    gram_gb = (m * m * 16) / (1024 ** 3)
    active_iv: Optional[np.ndarray] = None
    if witness_weight_cap is not None:
        weights = _op_weights(U_ops)
        active_iv = np.nonzero(weights <= witness_weight_cap)[0].astype(np.int64)
        if active_iv.size == 0:
            return {
                "status": "no_active_witness_terms",
                "found_witness": False,
                "best_rank": None,
                "target_rank": int(target),
                "Uc_size": int(m),
                "Wc_size": None,
                "nnz": None,
                "estimated_gram_gb": float(gram_gb),
                "precompute_sec": 0.0,
                "trials_used": 0,
                "best_h_real": None,
                "witness_weight_cap": int(witness_weight_cap),
                "active_h_size": 0,
            }

    if active_iv is None and gram_gb > gram_cap_gb:
        return {
            "status": "skipped_gram_cap",
            "found_witness": False,
            "best_rank": None,
            "target_rank": int(target),
            "Uc_size": int(m),
            "Wc_size": None,
            "nnz": None,
            "estimated_gram_gb": float(gram_gb),
            "precompute_sec": 0.0,
            "trials_used": 0,
            "best_h_real": None,
        }

    t = time.time()
    if active_iv is None:
        precomp = precompute_sparse_structure(U_ops, root_patch)
    else:
        precomp = precompute_sparse_structure_active_h(
            U_ops,
            root_patch,
            active_iv,
            progress_every=500 if verbose else 0,
        )
    t_pre = time.time() - t
    n_W = precomp["n_W"]
    nnz = int(precomp["phases"].size)

    if verbose:
        active_note = ""
        if active_iv is not None:
            active_note = (
                f" active_h={len(active_iv)} "
                f"weight<={witness_weight_cap}"
            )
        print(
            f"  precompute: |U|={m} |W|={n_W} nnz={nnz} "
            f"Gram={gram_gb:.2f} GB{active_note} took={t_pre:.1f}s",
            flush=True,
        )

    rng = np.random.default_rng(seed)
    best_rank = -1
    best_h = None
    last_timings: Dict[str, float] = {}

    for trial in range(trials):
        t_trial = time.time()
        h = np.zeros(m, dtype=np.complex128)
        if active_iv is None:
            h = rng.integers(-coeff_bound, coeff_bound + 1, size=m).astype(
                np.complex128
            )
        else:
            vals = rng.integers(
                -coeff_bound,
                coeff_bound + 1,
                size=len(active_iv),
            )
            vals[vals == 0] = 1
            h[active_iv] = vals.astype(np.complex128)
        if np.all(np.abs(h) < 1e-12):
            h[0] = 1.0
        rank, max_eig, timings = commutator_rank_for_h(
            precomp, h, tol_rel=tol_rel
        )
        last_timings = timings
        dt = time.time() - t_trial
        if verbose:
            def fmt_timing(item):
                k, v = item
                if k.startswith("lambda") or k == "rank_method_iterative":
                    return f"{k}={v:.2e}"
                return f"{k}={v:.1f}s"

            tstr = " ".join(fmt_timing(item) for item in timings.items())
            print(
                f"  trial {trial+1}/{trials}: rank={rank}/{target} "
                f"max_eig={max_eig:.2e} elapsed={dt:.1f}s [{tstr}]",
                flush=True,
            )
        if rank > best_rank:
            best_rank = rank
            best_h = h.real.astype(np.float64).copy()
        if rank == target:
            return {
                "status": "ok",
                "found_witness": True,
                "best_rank": int(rank),
                "target_rank": int(target),
                "Uc_size": int(m),
                "Wc_size": int(n_W),
                "nnz": int(nnz),
                "estimated_gram_gb": float(gram_gb),
                "precompute_sec": float(t_pre),
                "trials_used": int(trial + 1),
                "rank_diagnostic": {k: float(v) for k, v in timings.items()},
                "best_h_real": [float(x) for x in best_h],
                "witness_weight_cap": (
                    None if witness_weight_cap is None else int(witness_weight_cap)
                ),
                "active_h_size": (
                    None if active_iv is None else int(len(active_iv))
                ),
                "active_candidate_pairs": precomp.get("n_candidate_pairs"),
            }

    return {
        "status": "ok",
        "found_witness": False,
        "best_rank": int(best_rank),
        "target_rank": int(target),
        "Uc_size": int(m),
        "Wc_size": int(n_W),
        "nnz": int(nnz),
        "estimated_gram_gb": float(gram_gb),
        "precompute_sec": float(t_pre),
        "trials_used": int(trials),
        "rank_diagnostic": {k: float(v) for k, v in last_timings.items()},
        "best_h_real": None if best_h is None else [float(x) for x in best_h],
        "witness_weight_cap": (
            None if witness_weight_cap is None else int(witness_weight_cap)
        ),
        "active_h_size": None if active_iv is None else int(len(active_iv)),
        "active_candidate_pairs": precomp.get("n_candidate_pairs"),
    }


def run_job_sparse(job: Job, gram_cap_gb: float = 8.0, verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print(
            f"\n=== {job.tag}: graph={job.graph_kind}{job.graph_args} "
            f"k={job.k} R={job.R_geom} Rpatch={job.R_patch} trials={job.trials} ===",
            flush=True,
        )

    t0 = time.time()
    U_ops, patch_nodes, root_patch = build_local_family_for_job(job)
    t_build = time.time() - t0
    if verbose:
        print(
            f"  family: |U|={len(U_ops)} patch_sites={len(patch_nodes)} "
            f"({t_build:.1f}s)",
            flush=True,
        )

    out = search_witness(
        U_ops=U_ops,
        root_patch=root_patch,
        trials=job.trials,
        seed=job.seed,
        coeff_bound=job.coeff_bound,
        gram_cap_gb=gram_cap_gb,
        verbose=verbose,
        witness_weight_cap=job.witness_weight_cap,
    )

    return {
        "tag": job.tag,
        "family": job.family,
        "graph_kind": job.graph_kind,
        "graph_args": list(job.graph_args),
        "k": job.k,
        "R_geom": job.R_geom,
        "R_patch": job.R_patch,
        "root": job.root,
        "root_label": job.root_label,
        "boundary": job.boundary,
        "local_mode": job.local_mode,
        "root_coord": job.root_coord,
        "covered_root_count": job.covered_root_count,
        "covered_root_sample": job.covered_root_sample,
        "coverage_note": job.coverage_note,
        "witness_weight_cap": job.witness_weight_cap,
        "patch_sites": int(len(patch_nodes)),
        "family_build_sec": float(t_build),
        "elapsed_sec": float(time.time() - t0),
        **out,
    }
