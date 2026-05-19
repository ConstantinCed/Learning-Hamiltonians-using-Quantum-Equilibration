"""
Fill the three ``skipped_memory_cap`` gaps from structured_push_50.json.

Those three jobs have a dense |W| x |U| commutator matrix larger than the
8 GB cap used in the original sweep:

  * push_tri_dense, triangular_torus (5, 5), k=3, R=2   |U|=9939, |W|=695793
  * push_tri_dense, triangular_torus (6, 6), k=3, R=2   |U|=7239, |W|=535899
  * push_d3_dense,  cubic_periodic   (3, 3, 3), k=3, R=2 |U|=10911, |W|=743340

Strategy
--------
For each job:
  1. Enumerate all anticommuting (u_i, u_j) pairs of the local family U_c
     with nontrivial Pauli at the root and store the contributions as
     sparse (row, col, h-index, phase) triples.
  2. For a random integer h, materialise the commutator matrix
     C(h) (shape |W| x |U|) as a scipy sparse matrix.
  3. Compute G(h) = C(h)^H @ C(h), a dense |U| x |U| Hermitian PSD matrix
     whose rank equals rank(C(h)).
  4. Use numpy.linalg.eigvalsh to obtain the spectrum of G and read off
     the rank.

The bottleneck is the complex Hermitian eigendecomposition.  Apple's
Accelerate framework runs ``zheevd`` at roughly 50 GFLOPS on this M-class
chip, so for |U| ~ 7000-11000 each trial takes 10-40 minutes.  Witnesses
are almost always found on the first trial, so we run a small number of
trials per job and save results after each one.

Results land in ``additional_results.json`` in the same directory.
"""

import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from witness_structured import (  # noqa: E402
    Job,
    local_dense_family_direct,
    make_graph,
)


# --------------------------------------------------------------------------
# Symplectic preprocessing
# --------------------------------------------------------------------------


def pauli_to_symplectic(U_ops: List[Tuple[str, ...]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pack each U operator into (x, z) bitmasks; return arrays and n_qubits."""
    m = len(U_ops)
    n_q = len(U_ops[0]) if m > 0 else 0
    xs = np.zeros(m, dtype=np.int64)
    zs = np.zeros(m, dtype=np.int64)
    for i, op in enumerate(U_ops):
        x = 0
        z = 0
        for j, p in enumerate(op):
            if p == "X":
                x |= 1 << j
            elif p == "Z":
                z |= 1 << j
            elif p == "Y":
                x |= 1 << j
                z |= 1 << j
        xs[i] = x
        zs[i] = z
    return xs, zs, n_q


def precompute_sparse_structure(
    U_ops: List[Tuple[str, ...]],
    root_patch: int,
) -> Dict[str, Any]:
    """Enumerate (iu, iv, row=index_of_W, phase) for every anticommuting pair.

    The phase stored is ``2 * ph_uv`` because for anticommuting Paulis
    ``ph_uv - ph_vu = 2 ph_uv``.

    Rows of W are deduplicated by packing (wx, wz) into a single uint64 key
    (requires n_qubits <= 32, which holds for every patch we touch here).
    """
    m = len(U_ops)
    xs, zs, n_q = pauli_to_symplectic(U_ops)
    assert n_q <= 32, f"patch has {n_q} sites; this fast packing assumes <=32"
    Y_count = np.bitwise_count(xs & zs).astype(np.int64)

    root_mask = np.int64(1 << root_patch)
    i_powers = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)

    chunk_keys: List[np.ndarray] = []   # uint64 packed (wx, wz)
    chunk_iu: List[np.ndarray] = []
    chunk_iv: List[np.ndarray] = []
    chunk_phase: List[np.ndarray] = []

    for iu in range(m):
        xu = int(xs[iu])
        zu = int(zs[iu])
        Yu = int(Y_count[iu])

        p1 = np.bitwise_count(np.int64(xu) & zs).astype(np.int64)
        p2 = np.bitwise_count(np.int64(zu) & xs).astype(np.int64)
        anticomm = ((p1 + p2) & 1).astype(bool)
        anticomm[iu] = False

        w_x_all = xu ^ xs
        w_z_all = zu ^ zs
        nontrivial = ((w_x_all | w_z_all) & root_mask) != 0
        mask = anticomm & nontrivial

        ivs = np.nonzero(mask)[0]
        if ivs.size == 0:
            continue

        w_xa = w_x_all[ivs]
        w_za = w_z_all[ivs]

        Yw = np.bitwise_count(w_xa & w_za).astype(np.int64)
        Yv = Y_count[ivs]
        zu_xv_parity = (
            np.bitwise_count(np.int64(zu) & xs[ivs]).astype(np.int64) & 1
        )
        sign = (1 - 2 * zu_xv_parity).astype(np.int64)

        k_mod_4 = (Yu + Yv - Yw) & 3
        i_pow = i_powers[k_mod_4]
        ph_uv = sign.astype(np.complex128) * i_pow
        coeff = 2.0 * ph_uv

        # Pack (wx, wz) into a single uint64 key
        keys = (w_xa.astype(np.uint64) << np.uint64(32)) | w_za.astype(np.uint64)

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

    all_keys = np.concatenate(chunk_keys)
    all_iu = np.concatenate(chunk_iu)
    all_iv = np.concatenate(chunk_iv)
    all_phase = np.concatenate(chunk_phase)

    _unique_keys, inverse = np.unique(all_keys, return_inverse=True)
    n_W = int(_unique_keys.shape[0])

    return {
        "rows": inverse.astype(np.int64),
        "cols": all_iu,
        "iv_idx": all_iv,
        "phases": all_phase,
        "n_W": n_W,
        "n_U": m,
    }


# --------------------------------------------------------------------------
# Rank check via Gram matrix
# --------------------------------------------------------------------------


def commutator_rank_for_h(
    precomp: Dict[str, Any],
    h: np.ndarray,
    tol_rel: float = 1e-9,
) -> Tuple[int, float, Dict[str, float]]:
    rows = precomp["rows"]
    cols = precomp["cols"]
    iv_idx = precomp["iv_idx"]
    phases = precomp["phases"]
    n_W = precomp["n_W"]
    m = precomp["n_U"]

    timings: Dict[str, float] = {}

    t = time.time()
    data = h[iv_idx] * phases
    C = sp.coo_matrix((data, (rows, cols)), shape=(n_W, m)).tocsr()
    timings["build_C"] = time.time() - t

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


def search_witness(
    U_ops: List[Tuple[str, ...]],
    root_patch: int,
    trials: int,
    seed: int,
    coeff_bound: int = 3,
    gram_cap_gb: float = 8.0,
    tol_rel: float = 1e-9,
    verbose: bool = True,
) -> Dict[str, Any]:
    m = len(U_ops)
    target = m - 1

    t = time.time()
    precomp = precompute_sparse_structure(U_ops, root_patch)
    t_pre = time.time() - t
    n_W = precomp["n_W"]
    nnz = int(precomp["phases"].size)
    gram_gb = (m * m * 16) / (1024 ** 3)

    if verbose:
        print(
            f"  precompute: |U|={m} |W|={n_W} nnz={nnz} "
            f"Gram={gram_gb:.2f} GB took={t_pre:.1f}s",
            flush=True,
        )

    if gram_gb > gram_cap_gb:
        return {
            "status": "skipped_gram_cap",
            "found_witness": False,
            "best_rank": None,
            "target_rank": int(target),
            "Uc_size": int(m),
            "Wc_size": int(n_W),
            "nnz": int(nnz),
            "estimated_gram_gb": float(gram_gb),
            "precompute_sec": float(t_pre),
            "trials_used": 0,
            "best_h_real": None,
        }

    rng = np.random.default_rng(seed)
    best_rank = -1
    best_h = None

    for trial in range(trials):
        t_trial = time.time()
        h = rng.integers(-coeff_bound, coeff_bound + 1, size=m).astype(np.complex128)
        if np.all(np.abs(h) < 1e-12):
            h[0] = 1.0
        rank, max_eig, timings = commutator_rank_for_h(
            precomp, h, tol_rel=tol_rel
        )
        dt = time.time() - t_trial
        if verbose:
            tstr = " ".join(f"{k}={v:.1f}s" for k, v in timings.items())
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
                "best_h_real": [float(x) for x in best_h],
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
        "best_h_real": None if best_h is None else [float(x) for x in best_h],
    }


def run_job_sparse(job: Job, gram_cap_gb: float = 8.0, verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print(
            f"\n=== {job.tag}: graph={job.graph_kind}{job.graph_args} "
            f"k={job.k} R={job.R_geom} Rpatch={job.R_patch} trials={job.trials} ===",
            flush=True,
        )

    t0 = time.time()
    G = make_graph(job.graph_kind, job.graph_args)
    U_ops, patch_nodes, root_patch = local_dense_family_direct(
        G, job.root, job.R_patch, job.k, job.R_geom
    )
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
        "patch_sites": int(len(patch_nodes)),
        "family_build_sec": float(t_build),
        "elapsed_sec": float(time.time() - t0),
        **out,
    }


# --------------------------------------------------------------------------
# Jobs to run
# --------------------------------------------------------------------------


def missing_jobs() -> List[Job]:
    """The three skipped_memory_cap entries in structured_push_50.json."""
    return [
        # smallest |U| first so we finish *something* quickly
        Job(
            tag="push_tri_dense_L6_k3_R2",
            family="dense",
            graph_kind="triangular_torus",
            graph_args=(6, 6),
            root=0,
            R_patch=2,
            trials=2,
            seed=40001,
            k=3,
            R_geom=2,
        ),
        Job(
            tag="push_tri_dense_L5_k3_R2",
            family="dense",
            graph_kind="triangular_torus",
            graph_args=(5, 5),
            root=0,
            R_patch=2,
            trials=2,
            seed=40002,
            k=3,
            R_geom=2,
        ),
        Job(
            tag="push_d3_dense_L3_k3_R2",
            family="dense",
            graph_kind="cubic_periodic",
            graph_args=(3, 3, 3),
            root=0,
            R_patch=2,
            trials=2,
            seed=40003,
            k=3,
            R_geom=2,
        ),
    ]


def main() -> None:
    jobs = missing_jobs()
    out_path = os.path.join(HERE, "additional_results.json")
    results: List[Dict[str, Any]] = []
    print(
        f"Running {len(jobs)} additional jobs; writing -> {out_path}",
        flush=True,
    )
    for i, job in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] starting {job.tag}", flush=True)
        try:
            info = run_job_sparse(job, gram_cap_gb=8.0, verbose=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            info = {"tag": job.tag, "error": str(exc)}
        results.append(info)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        status = info.get("status", "?")
        found = info.get("found_witness")
        br = info.get("best_rank")
        tr = info.get("target_rank")
        uc = info.get("Uc_size")
        wc = info.get("Wc_size")
        el = info.get("elapsed_sec")
        print(
            f"[{i}/{len(jobs)}] DONE {info.get('tag', '?')} "
            f"status={status} found={found} rank={br}/{tr} "
            f"Uc={uc} Wc={wc} elapsed={el}",
            flush=True,
        )

    print("\n=== FINAL SUMMARY ===", flush=True)
    for info in results:
        if "error" in info:
            print(f"  {info['tag']}: ERROR {info['error']}")
        else:
            print(
                f"  {info['tag']}: status={info.get('status')} "
                f"found={info.get('found_witness')} "
                f"rank={info.get('best_rank')}/{info.get('target_rank')} "
                f"Uc={info.get('Uc_size')} Wc={info.get('Wc_size')} "
                f"({info.get('elapsed_sec'):.1f}s)"
            )


if __name__ == "__main__":
    main()
