"""Run remaining skipped formal-OBC dense triangular/cubic certificates."""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_witness  # noqa: E402


def row_key(r: Dict[str, Any]) -> Tuple:
    return (
        tuple(r["graph_args"]),
        r.get("k"),
        r.get("R_geom"),
        r["R_patch"],
        r.get("local_mode", "rooted_ball"),
        r.get("root_label", "bulk") or "bulk",
    )


def job_key(job: run_witness.Job) -> Tuple:
    return (
        tuple(job.graph_args),
        job.k,
        job.R_geom,
        job.R_patch,
        job.local_mode,
        job.root_label,
    )


def pending_jobs(lattices: List[str]) -> List[Tuple[int, run_witness.Job]]:
    jobs = run_witness.all_jobs_by_family("open_boundary")["dense"]
    by_key = {job_key(job): job for job in jobs if job.graph_kind in lattices}
    pending: List[Tuple[int, run_witness.Job]] = []

    for lattice in lattices:
        rows = run_witness._load_results("open_boundary", "dense", lattice)
        for row in rows:
            if row.get("found_witness"):
                continue
            if row.get("status") not in {"skipped_gram_cap", "ok"}:
                continue
            job = by_key.get(row_key(row))
            if job is None:
                print(f"[WARN] no catalogue job for row {row.get('tag')}", flush=True)
                continue
            pending.append((int(row.get("Uc_size") or 0), job))

    pending.sort(key=lambda item: (item[0], item[1].tag))
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lattices",
        nargs="+",
        default=["triangular_open", "cubic_open"],
        help="Open-boundary dense lattice files to continue.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Sparse witness trials per job. Each trial can be expensive.",
    )
    parser.add_argument(
        "--memory-cap-gb",
        type=float,
        default=8.0,
        help="Kept for backend compatibility; sparse-witness jobs bypass Gram caps.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Optional limit for testing/resuming in chunks.",
    )
    args = parser.parse_args()

    todo = pending_jobs(args.lattices)
    if args.max_jobs is not None:
        todo = todo[: args.max_jobs]

    print(
        f"[START] {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"pending={len(todo)} lattices={args.lattices} trials={args.trials}",
        flush=True,
    )

    t_all = time.time()
    for idx, (size, job) in enumerate(todo, 1):
        job.trials = args.trials
        print(
            f"\n[{idx}/{len(todo)}] {job.tag} |U_c|={size} "
            f"witness_weight_cap={job.witness_weight_cap}",
            flush=True,
        )

        rows = run_witness._load_results(job.boundary, job.family, job.graph_kind)
        try:
            info = run_witness.run_one(
                job,
                memory_cap_gb=args.memory_cap_gb,
                verbose=True,
            )
        except Exception as exc:  # keep the overnight batch moving
            print(f"[ERROR] {job.tag}: {type(exc).__name__}: {exc}", flush=True)
            continue

        rows = run_witness._merge_entry(rows, info)
        run_witness._save_results(job.boundary, job.family, job.graph_kind, rows)
        print(
            f"[SAVED] {job.tag} status={info.get('status')} "
            f"found={info.get('found_witness')} "
            f"rank={info.get('best_rank')}/{info.get('target_rank')} "
            f"elapsed={info.get('elapsed_sec'):.1f}s",
            flush=True,
        )

    print(
        f"\n[DONE] {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"elapsed={time.time() - t_all:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
