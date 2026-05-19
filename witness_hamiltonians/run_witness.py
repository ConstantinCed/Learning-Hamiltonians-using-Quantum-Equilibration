"""Witness-Hamiltonian non-degeneracy: unified driver.

Single entry point that defines every job, dispatches to the dense or
sparse rank backend depending on the problem size, and writes results
to ``<family>/<lattice>.json``.

Usage::

    python3 run_witness.py                          # run every job (skip already-done)
    python3 run_witness.py --families dense         # restrict to a family
    python3 run_witness.py --lattices cycle cubic_periodic
    python3 run_witness.py --force                  # re-run even if entry exists
    python3 run_witness.py --dry-run                # list jobs only
    python3 run_witness.py --memory-cap-gb 16       # raise the dense backend cap

Dispatch rule:
    * ``|U_c| <= 2000`` -> dense backend (``witness_structured.run_job``)
    * ``|U_c| >  2000`` -> sparse Gram backend (``additional_runs.run_job_sparse``)

Results from each job are merged into the per-family JSON, deduplicating
on ``(graph_args, k, R_geom, R_patch, root_label)`` and preferring the
entry that found a witness.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from witness_structured import (  # noqa: E402
    Job,
    local_dense_family_direct,
    run_job,
)
from additional_runs import run_job_sparse  # noqa: E402

DENSE_TO_SPARSE_UC_THRESHOLD = 2000


# ---------------------------------------------------------------------------
# Job catalogue (organised by family)
# ---------------------------------------------------------------------------


def _dense_cycle_jobs() -> List[Job]:
    """Generic local family on the 1D periodic chain."""
    specs = [
        # (k, R_geom, L, seed)
        (2, 1,  9,    119),
        (2, 2, 11,    231),
        (2, 3, 13,    343),
        (2, 4, 17,    457),
        (2, 5, 21,    571),
        (3, 1,  9,    139),
        (3, 2, 11,    251),
        (3, 3, 15,    385),
        (3, 4, 17,    487),
        (3, 5, 21,    601),
        (4, 2, 13,    263),
        (4, 3, 17,    387),
        (4, 4, 21,    511),
    ]
    return [
        Job(
            tag=f"dense_cycle_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="cycle",
            graph_args=(L,),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
        )
        for (k, R, L, seed) in specs
    ]


def _dense_grid_jobs() -> List[Job]:
    """Generic local family on the 2D square (periodic) lattice."""
    specs = [
        # (k, R_geom, L, seed) -- multiple L per (k,R) are redundancy
        # checks (identical by patch isomorphism for L >= 2R+2).
        (2, 1, 4,  4124),
        (2, 2, 5,  5225),
        (2, 2, 6,  6226),
        (2, 3, 6,  6326),
        (2, 3, 7,  7327),
        (3, 1, 4,  4134),
        (3, 2, 5,  5235),
        (3, 2, 6,  6236),
    ]
    return [
        Job(
            tag=f"dense_grid_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="grid_periodic",
            graph_args=(L, L),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
        )
        for (k, R, L, seed) in specs
    ]


def _dense_triangular_jobs() -> List[Job]:
    """Generic local family on the 2D triangular torus."""
    specs = [
        (2, 1, 4,  4124),
        (2, 2, 5,  5225),
        (2, 2, 6,  6226),
        (2, 3, 6,  6326),
        (3, 1, 4,  4134),
        (3, 2, 5,  5235),
        (3, 2, 6,  6236),
    ]
    return [
        Job(
            tag=f"dense_tri_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="triangular_torus",
            graph_args=(L, L),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
        )
        for (k, R, L, seed) in specs
    ]


def _dense_honeycomb_jobs() -> List[Job]:
    """Generic local family on the 2D honeycomb torus."""
    specs = [
        (2, 1, 3,  3123),
        (2, 2, 4,  4224),
        (2, 3, 5,  5325),
        (2, 3, 6,  6326),
        (3, 1, 3,  3133),
        (3, 2, 4,  4234),
        (3, 2, 5,  5235),
    ]
    return [
        Job(
            tag=f"dense_honey_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="honeycomb_torus",
            graph_args=(L, L),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
        )
        for (k, R, L, seed) in specs
    ]


def _dense_cubic_jobs() -> List[Job]:
    """Generic local family on the 3D cubic (periodic) lattice."""
    specs = [
        # (k, R_geom, L, seed)
        (2, 1, 3,  3123),
        (2, 2, 4,  4224),
        (3, 1, 3,  3133),
        (3, 1, 4,  4134),
        (3, 2, 3,  3233),
    ]
    return [
        Job(
            tag=f"dense_cubic_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="cubic_periodic",
            graph_args=(L, L, L),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
        )
        for (k, R, L, seed) in specs
    ]


def _xyz_cycle_jobs() -> List[Job]:
    """XYZ chain with on-site X,Y,Z fields, 1D periodic."""
    return [
        Job(
            tag=f"xyz_cycle_L{L}",
            family="xyz",
            graph_kind="cycle",
            graph_args=(L,),
            root=0,
            R_patch=1,
            trials=500,
            seed=1000 + L,
        )
        for L in [9, 11, 13, 15, 17, 21, 25]
    ]


def _full_nn_cycle_jobs() -> List[Job]:
    """Full nearest-neighbour 2-body family with all on-site fields, 1D periodic."""
    return [
        Job(
            tag=f"fullnn_cycle_L{L}",
            family="full_nn_2body_all_fields",
            graph_kind="cycle",
            graph_args=(L,),
            root=0,
            R_patch=1,
            trials=500,
            seed=2000 + L,
        )
        for L in [9, 11, 13, 15, 17, 21, 25]
    ]


def _kitaev_honeycomb_jobs() -> List[Job]:
    """Kitaev honeycomb model with on-site X,Y,Z fields, honeycomb torus."""
    return [
        Job(
            tag=f"kitaev_honey_L{L}",
            family="kitaev_honey_2d",
            graph_kind="honeycomb_torus",
            graph_args=(L, L),
            root=0,
            R_patch=1,
            trials=500,
            seed=3000 + L,
        )
        for L in [3, 4, 5, 6]
    ]


# Single registry: family -> list of jobs
ALL_JOBS_BY_FAMILY: Dict[str, List[Job]] = {
    "dense": (
        _dense_cycle_jobs()
        + _dense_grid_jobs()
        + _dense_triangular_jobs()
        + _dense_honeycomb_jobs()
        + _dense_cubic_jobs()
    ),
    "xyz": _xyz_cycle_jobs(),
    "full_nn_2body_all_fields": _full_nn_cycle_jobs(),
    "kitaev_honey_2d": _kitaev_honeycomb_jobs(),
}


# ---------------------------------------------------------------------------
# Result I/O (per-family JSON files)
# ---------------------------------------------------------------------------


def _result_path(family: str, lattice: str) -> str:
    return os.path.join(HERE, family, f"{lattice}.json")


def _load_results(family: str, lattice: str) -> List[Dict[str, Any]]:
    path = _result_path(family, lattice)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _save_results(family: str, lattice: str, rows: List[Dict[str, Any]]) -> None:
    path = _result_path(family, lattice)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def sort_key(r):
        return (
            tuple(r["graph_args"]),
            r.get("k") if r.get("k") is not None else -1,
            r.get("R_geom") if r.get("R_geom") is not None else -1,
            r["R_patch"],
            r.get("root_label", "-") or "-",
        )

    rows = sorted(rows, key=sort_key)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)


def _entry_key(r: Dict[str, Any]) -> Tuple:
    root_lbl = r.get("root_label")
    if root_lbl in (None, "", "bulk"):
        root_lbl = "-"
    return (
        tuple(r["graph_args"]),
        r.get("k"),
        r.get("R_geom"),
        r["R_patch"],
        root_lbl,
    )


def _merge_entry(rows: List[Dict[str, Any]], new: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Upsert ``new`` into ``rows`` keyed by ``_entry_key``; prefer found_witness=True."""
    new_key = _entry_key(new)
    out = []
    replaced = False
    for r in rows:
        if _entry_key(r) == new_key:
            if not r.get("found_witness", False) and new.get("found_witness", False):
                out.append(new)
            elif r.get("found_witness", False) and not new.get("found_witness", False):
                out.append(r)
            else:
                out.append(new)
            replaced = True
        else:
            out.append(r)
    if not replaced:
        out.append(new)
    return out


# ---------------------------------------------------------------------------
# Dispatch + driver
# ---------------------------------------------------------------------------


def _build_family_size(job: Job) -> int:
    """Cheap upfront estimate of |U_c| for dispatch (re-built inside backends)."""
    from witness_structured import (  # local import; avoids cycles in tests
        make_graph,
        xyz_fields_family,
        full_nn_2body_all_fields_family,
        kitaev_honeycomb_fields_fixed,
    )
    G = make_graph(job.graph_kind, job.graph_args)
    if job.family == "dense":
        U_ops, _, _ = local_dense_family_direct(
            G, job.root, job.R_patch, job.k, job.R_geom
        )
    elif job.family == "xyz":
        U_ops, _, _ = xyz_fields_family(G, job.root, job.R_patch)
    elif job.family == "full_nn_2body_all_fields":
        U_ops, _, _ = full_nn_2body_all_fields_family(G, job.root, job.R_patch)
    elif job.family == "kitaev_honey_2d":
        U_ops, _, _ = kitaev_honeycomb_fields_fixed(G, job.root, job.R_patch)
    else:
        raise ValueError(job.family)
    return len(U_ops)


def run_one(job: Job, memory_cap_gb: float = 8.0, verbose: bool = True) -> Dict[str, Any]:
    """Run ``job`` with dense or sparse backend depending on |U_c|."""
    Uc_size = _build_family_size(job)
    if Uc_size > DENSE_TO_SPARSE_UC_THRESHOLD:
        if verbose:
            print(f"  [sparse backend, |U_c|={Uc_size}>{DENSE_TO_SPARSE_UC_THRESHOLD}]")
        return run_job_sparse(job, gram_cap_gb=memory_cap_gb, verbose=verbose)
    if verbose:
        print(f"  [dense backend, |U_c|={Uc_size}]")
    info = run_job(job, memory_cap_gb=memory_cap_gb)
    if info.get("status") == "skipped_memory_cap":
        if verbose:
            print(f"  [dense skipped; falling back to sparse]")
        info = run_job_sparse(job, gram_cap_gb=memory_cap_gb, verbose=verbose)
    return info


def _job_lattice_label(job: Job) -> str:
    return job.graph_kind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families", nargs="+", default=list(ALL_JOBS_BY_FAMILY.keys()),
        help="Restrict to a subset of families.",
    )
    parser.add_argument(
        "--lattices", nargs="+", default=None,
        help="Restrict to a subset of lattice kinds.",
    )
    parser.add_argument(
        "--memory-cap-gb", type=float, default=8.0,
        help="Dense backend memory cap; sparse backend uses it as Gram cap.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run jobs even if a successful entry already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the jobs that would run without executing.",
    )
    args = parser.parse_args()

    # Flatten and filter the job catalogue
    jobs: List[Job] = []
    for fam, fam_jobs in ALL_JOBS_BY_FAMILY.items():
        if fam not in args.families:
            continue
        for j in fam_jobs:
            if args.lattices and _job_lattice_label(j) not in args.lattices:
                continue
            jobs.append(j)

    print(f"Catalogue: {len(jobs)} jobs across "
          f"{len(set(j.family for j in jobs))} families")
    if args.dry_run:
        for j in jobs:
            print(f"  {j.tag}  ({j.family} / {j.graph_kind}{j.graph_args}, "
                  f"k={j.k} R={j.R_geom} Rpatch={j.R_patch})")
        return

    # Cache (family, lattice) -> current rows
    cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def get_rows(fam, lat):
        if (fam, lat) not in cache:
            cache[(fam, lat)] = _load_results(fam, lat)
        return cache[(fam, lat)]

    t_total = time.time()
    for idx, job in enumerate(jobs, 1):
        lat = _job_lattice_label(job)
        rows = get_rows(job.family, lat)
        existing = next(
            (r for r in rows if _entry_key(r) == _entry_key({
                "graph_args": list(job.graph_args),
                "k": job.k, "R_geom": job.R_geom, "R_patch": job.R_patch,
                "root_label": job.root_label,
            })), None,
        )
        if existing is not None and existing.get("found_witness") and not args.force:
            print(f"[{idx}/{len(jobs)}] SKIP {job.tag}  (witness already on file)")
            continue

        print(f"\n[{idx}/{len(jobs)}] {job.tag}  "
              f"({job.family} / {job.graph_kind}{job.graph_args}, "
              f"k={job.k} R={job.R_geom} Rpatch={job.R_patch})")
        t0 = time.time()
        info = run_one(job, memory_cap_gb=args.memory_cap_gb, verbose=True)
        info.setdefault("tag", job.tag)
        info.setdefault("family", job.family)
        info.setdefault("graph_kind", job.graph_kind)
        info.setdefault("graph_args", list(job.graph_args))
        info.setdefault("k", job.k)
        info.setdefault("R_geom", job.R_geom)
        info.setdefault("R_patch", job.R_patch)
        info.setdefault("root_label", job.root_label)
        cache[(job.family, lat)] = _merge_entry(rows, info)
        _save_results(job.family, lat, cache[(job.family, lat)])

        dt = time.time() - t0
        print(f"   -> status={info.get('status')} found={info.get('found_witness')} "
              f"rank={info.get('best_rank')}/{info.get('target_rank')} "
              f"({dt:.1f}s)")

    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
