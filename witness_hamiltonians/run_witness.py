"""Driver: define certification jobs and dispatch to dense/sparse backends."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from witness_structured import (  # noqa: E402
    Job,
    ball_nodes,
    build_local_family_for_job,
    make_graph,
    run_job,
)

DENSE_TO_SPARSE_UC_THRESHOLD = 1000
BOUNDARIES = ("periodic", "open_boundary")


def _periodic_dense_cycle_jobs() -> List[Job]:
    specs = [
        (2, 1, 9, 119),
        (2, 2, 11, 231),
        (2, 3, 13, 343),
        (2, 4, 17, 457),
        (2, 5, 21, 571),
        (3, 1, 9, 139),
        (3, 2, 11, 251),
        (3, 3, 15, 385),
        (3, 4, 17, 487),
        (3, 5, 21, 601),
        (4, 2, 13, 263),
        (4, 3, 17, 387),
        (4, 4, 21, 511),
    ]
    return [
        Job(
            tag=f"periodic_dense_cycle_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="cycle",
            graph_args=(L,),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
            boundary="periodic",
        )
        for (k, R, L, seed) in specs
    ]


def _periodic_dense_grid_jobs() -> List[Job]:
    specs = [
        (2, 1, 4, 4124),
        (2, 2, 5, 5225),
        (2, 2, 6, 6226),
        (2, 3, 6, 6326),
        (2, 3, 7, 7327),
        (3, 1, 4, 4134),
        (3, 2, 5, 5235),
        (3, 2, 6, 6236),
        (4, 1, 4, 4144),
        (4, 2, 6, 6246),
    ]
    return [
        Job(
            tag=f"periodic_dense_grid_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="grid_periodic",
            graph_args=(L, L),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
            boundary="periodic",
        )
        for (k, R, L, seed) in specs
    ]


def _periodic_dense_triangular_jobs() -> List[Job]:
    specs = [
        (2, 1, 4, 4124),
        (2, 2, 5, 5225),
        (2, 2, 6, 6226),
        (2, 3, 6, 6326),
        (3, 1, 4, 4134),
        (3, 2, 5, 5235),
        (3, 2, 6, 6236),
        (4, 1, 4, 4244),
        (4, 2, 6, 6346),
    ]
    return [
        Job(
            tag=f"periodic_dense_tri_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="triangular_torus",
            graph_args=(L, L),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
            boundary="periodic",
        )
        for (k, R, L, seed) in specs
    ]


def _periodic_dense_honeycomb_jobs() -> List[Job]:
    specs = [
        (2, 1, 3, 3123),
        (2, 2, 4, 4224),
        (2, 3, 5, 5325),
        (2, 3, 6, 6326),
        (3, 1, 3, 3133),
        (3, 2, 4, 4234),
        (3, 2, 5, 5235),
        (4, 1, 4, 4344),
        (4, 2, 6, 6446),
    ]
    return [
        Job(
            tag=f"periodic_dense_honey_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="honeycomb_torus",
            graph_args=(L, L),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
            boundary="periodic",
        )
        for (k, R, L, seed) in specs
    ]


def _periodic_dense_cubic_jobs() -> List[Job]:
    specs = [
        (2, 1, 3, 3123),
        (2, 2, 4, 4224),
        (3, 1, 3, 3133),
        (3, 1, 4, 4134),
        (3, 2, 3, 3233),
        (4, 1, 4, 4444),
        (4, 2, 6, 6546),
    ]
    return [
        Job(
            tag=f"periodic_dense_cubic_L{L}_k{k}_R{R}",
            family="dense",
            graph_kind="cubic_periodic",
            graph_args=(L, L, L),
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=k,
            R_geom=R,
            boundary="periodic",
        )
        for (k, R, L, seed) in specs
    ]


def _periodic_xyz_cycle_jobs() -> List[Job]:
    return [
        Job(
            tag=f"periodic_xyz_cycle_L{L}",
            family="xyz",
            graph_kind="cycle",
            graph_args=(L,),
            root=0,
            R_patch=1,
            trials=500,
            seed=1000 + L,
            boundary="periodic",
        )
        for L in [9, 11, 13, 15, 17, 21, 25]
    ]


def _periodic_full_nn_cycle_jobs() -> List[Job]:
    # Redundant with dense k=2, R=1 on the 1D chain; retained to reproduce
    # legacy JSON artifacts explicitly when requested.
    return [
        Job(
            tag=f"periodic_fullnn_cycle_L{L}",
            family="full_nn_2body_all_fields",
            graph_kind="cycle",
            graph_args=(L,),
            root=0,
            R_patch=1,
            trials=500,
            seed=2000 + L,
            boundary="periodic",
        )
        for L in [9, 11, 13, 15, 17, 21, 25]
    ]


def _periodic_full_2body_no_fields_jobs() -> List[Job]:
    specs = [
        ("cycle", (9,), 1, 80_101),
        ("cycle", (12,), 2, 80_102),
        ("grid_periodic", (9, 9), 1, 81_101),
        ("grid_periodic", (9, 9), 2, 81_102),
        ("triangular_torus", (9, 9), 1, 82_101),
        ("triangular_torus", (9, 9), 2, 82_102),
        ("honeycomb_torus", (9, 9), 1, 83_101),
        ("honeycomb_torus", (9, 9), 2, 83_102),
        ("cubic_periodic", (9, 9, 9), 1, 84_101),
        ("cubic_periodic", (9, 9, 9), 2, 84_102),
    ]
    return [
        Job(
            tag=f"periodic_full2body_no_fields_{graph_kind}_R{R}",
            family="full_2body_no_fields",
            graph_kind=graph_kind,
            graph_args=graph_args,
            root=0,
            R_patch=R,
            trials=500,
            seed=seed,
            k=2,
            R_geom=R,
            boundary="periodic",
            local_mode="formal_uc",
            coverage_note=(
                "formal proof U_c: all exact two-body terms of range R whose "
                "support intersects B(root,R); no one-body fields"
            ),
        )
        for graph_kind, graph_args, R, seed in specs
    ]


def _node_coord(G: nx.Graph, root: int) -> Any:
    return G.nodes[root].get("coord", root)


def _rooted_patch(G: nx.Graph, root: int, R_patch: int) -> nx.Graph:
    H = G.subgraph(ball_nodes(G, root, R_patch)).copy()
    for node in H.nodes():
        H.nodes[node]["is_root"] = node == root
    return H


def _rooted_patch_representatives(
    G: nx.Graph,
    R_patch: int,
    *,
    edge_attrs: Tuple[str, ...] = (),
) -> List[Tuple[int, List[int]]]:
    node_match = lambda a, b: a.get("is_root") == b.get("is_root")
    edge_match = None
    if edge_attrs:
        edge_match = nx.algorithms.isomorphism.categorical_edge_match(
            list(edge_attrs),
            [None] * len(edge_attrs),
        )
    reps: List[Tuple[int, nx.Graph, List[int]]] = []
    for root in sorted(G.nodes()):
        H = _rooted_patch(G, root, R_patch)
        for _rep_root, rep_H, roots in reps:
            if nx.is_isomorphic(
                H,
                rep_H,
                node_match=node_match,
                edge_match=edge_match,
            ):
                roots.append(root)
                break
        else:
            reps.append((root, H, [root]))
    return [(root, roots) for root, _H, roots in reps]


def _coord_label(coord: Any) -> str:
    if isinstance(coord, tuple):
        return "(" + ",".join(str(x) for x in coord) + ")"
    return str(coord)


def _open_job_for_rep(
    *,
    family: str,
    graph_kind: str,
    graph_args: Tuple[int, ...],
    root: int,
    roots: List[int],
    idx: int,
    R_patch: int,
    seed: int,
    k: Optional[int] = None,
    R_geom: Optional[int] = None,
    trials: int = 500,
    G: Optional[nx.Graph] = None,
    coverage_note: Optional[str] = None,
    local_mode: str = "formal_uc",
    witness_weight_cap: Optional[int] = None,
) -> Job:
    G = G if G is not None else make_graph(graph_kind, graph_args)
    coord = _node_coord(G, root)
    sample = [_node_coord(G, r) for r in roots[:12]]
    label = f"env{idx:02d}_root{_coord_label(coord)}"
    dense_suffix = f"_k{k}_R{R_geom}" if k is not None else ""
    tag = f"open_{family}_{graph_kind}{dense_suffix}_{label}"
    return Job(
        tag=tag,
        family=family,
        graph_kind=graph_kind,
        graph_args=graph_args,
        root=root,
        root_label=label,
        R_patch=R_patch,
        trials=trials,
        seed=seed + idx,
        k=k,
        R_geom=R_geom,
        boundary="open_boundary",
        local_mode=local_mode,
        root_coord=coord,
        covered_root_count=len(roots),
        covered_root_sample=sample,
        coverage_note=(
            coverage_note
            or "one representative for each exact rooted radius-2R proof-local patch"
        ),
        witness_weight_cap=witness_weight_cap,
    )


def _open_dense_jobs_for_lattice(
    graph_kind: str,
    graph_args: Tuple[int, ...],
    seed_base: int,
) -> List[Job]:
    G = make_graph(graph_kind, graph_args)
    jobs: List[Job] = []
    for R in [1, 2]:
        reps = _rooted_patch_representatives(G, 2 * R)
        for k in [2, 3, 4]:
            for idx, (root, roots) in enumerate(reps):
                jobs.append(
                    _open_job_for_rep(
                        family="dense",
                        graph_kind=graph_kind,
                        graph_args=graph_args,
                        root=root,
                        roots=roots,
                        idx=idx,
                        R_patch=R,
                        k=k,
                        R_geom=R,
                        seed=seed_base + 100 * k + 10 * R,
                        G=G,
                        coverage_note=(
                            "formal proof U_c: all diameter-R terms whose "
                            "support intersects B(root,R); representatives "
                            "are rooted radius-2R environments"
                        ),
                        witness_weight_cap=2,
                    )
                )
    return jobs


def _open_dense_jobs() -> List[Job]:
    specs = [
        ("path", (9,), 10_000),
        ("grid_open", (9, 9), 20_000),
        ("triangular_open", (9, 9), 30_000),
        ("honeycomb_open", (9, 9), 40_000),
        ("cubic_open", (9, 9, 9), 50_000),
    ]
    jobs: List[Job] = []
    for graph_kind, graph_args, seed_base in specs:
        jobs.extend(_open_dense_jobs_for_lattice(graph_kind, graph_args, seed_base))
    return jobs


def _open_structured_path_jobs(family: str, seed_base: int) -> List[Job]:
    graph_kind = "path"
    graph_args = (9,)
    R_patch = 1
    G = make_graph(graph_kind, graph_args)
    reps = _rooted_patch_representatives(G, 2 * R_patch)
    return [
        _open_job_for_rep(
            family=family,
            graph_kind=graph_kind,
            graph_args=graph_args,
            root=root,
            roots=roots,
            idx=idx,
            R_patch=R_patch,
            seed=seed_base,
            G=G,
            coverage_note=(
                "formal proof U_c: all nearest-neighbour terms whose support "
                "intersects B(root,1); representatives are rooted radius-2 "
                "environments"
            ),
        )
        for idx, (root, roots) in enumerate(reps)
    ]


def _open_full_2body_no_fields_jobs() -> List[Job]:
    specs = [
        ("path", (9,), 80_000),
        ("grid_open", (9, 9), 81_000),
        ("triangular_open", (9, 9), 82_000),
        ("honeycomb_open", (9, 9), 83_000),
        ("cubic_open", (9, 9, 9), 84_000),
    ]
    jobs: List[Job] = []
    for graph_kind, graph_args, seed_base in specs:
        G = make_graph(graph_kind, graph_args)
        for R in [1, 2]:
            reps = _rooted_patch_representatives(G, 2 * R)
            for idx, (root, roots) in enumerate(reps):
                jobs.append(
                    _open_job_for_rep(
                        family="full_2body_no_fields",
                        graph_kind=graph_kind,
                        graph_args=graph_args,
                        root=root,
                        roots=roots,
                        idx=idx,
                        R_patch=R,
                        k=2,
                        R_geom=R,
                        seed=seed_base + 10 * R,
                        trials=500,
                        G=G,
                        coverage_note=(
                            "formal proof U_c: all exact two-body terms of "
                            "range R whose support intersects B(root,R); "
                            "no one-body fields; representatives are rooted "
                            "radius-2R environments"
                        ),
                    )
                )
    return jobs


def all_jobs_by_family(boundary: str) -> Dict[str, List[Job]]:
    if boundary == "periodic":
        return {
            "dense": (
                _periodic_dense_cycle_jobs()
                + _periodic_dense_grid_jobs()
                + _periodic_dense_triangular_jobs()
                + _periodic_dense_honeycomb_jobs()
                + _periodic_dense_cubic_jobs()
            ),
            "xyz": _periodic_xyz_cycle_jobs(),
            "full_nn_2body_all_fields": _periodic_full_nn_cycle_jobs(),
            "full_2body_no_fields": _periodic_full_2body_no_fields_jobs(),
        }
    if boundary == "open_boundary":
        return {
            "dense": _open_dense_jobs(),
            "xyz": _open_structured_path_jobs("xyz", 60_000),
            "full_nn_2body_all_fields": _open_structured_path_jobs(
                "full_nn_2body_all_fields", 70_000
            ),
            "full_2body_no_fields": _open_full_2body_no_fields_jobs(),
        }
    raise ValueError(boundary)


def _result_path(boundary: str, family: str, lattice: str) -> str:
    return os.path.join(HERE, boundary, family, f"{lattice}.json")


def _legacy_result_path(family: str, lattice: str) -> str:
    return os.path.join(HERE, family, f"{lattice}.json")


def _load_results(boundary: str, family: str, lattice: str) -> List[Dict[str, Any]]:
    path = _result_path(boundary, family, lattice)
    if not os.path.exists(path) and boundary == "periodic":
        path = _legacy_result_path(family, lattice)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _save_results(
    boundary: str,
    family: str,
    lattice: str,
    rows: List[Dict[str, Any]],
) -> None:
    path = _result_path(boundary, family, lattice)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def sort_key(r: Dict[str, Any]) -> Tuple:
        return (
            tuple(r["graph_args"]),
            r.get("k") if r.get("k") is not None else -1,
            r.get("R_geom") if r.get("R_geom") is not None else -1,
            r["R_patch"],
            r.get("local_mode", "rooted_ball"),
            r.get("root_label", "bulk") or "bulk",
        )

    rows = sorted(rows, key=sort_key)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)


def _entry_key(r: Dict[str, Any]) -> Tuple:
    return (
        tuple(r["graph_args"]),
        r.get("k"),
        r.get("R_geom"),
        r["R_patch"],
        r.get("local_mode", "rooted_ball"),
        r.get("root_label", "bulk") or "bulk",
    )


def _job_entry_key(job: Job) -> Tuple:
    return (
        tuple(job.graph_args),
        job.k,
        job.R_geom,
        job.R_patch,
        job.local_mode,
        job.root_label,
    )


def _merge_entry(rows: List[Dict[str, Any]], new: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def _family_size(job: Job) -> int:
    U_ops, _patch_nodes, _root_patch = build_local_family_for_job(job)
    return len(U_ops)


def run_one(job: Job, memory_cap_gb: float = 8.0, verbose: bool = True) -> Dict[str, Any]:
    Uc_size = _family_size(job)
    if Uc_size > DENSE_TO_SPARSE_UC_THRESHOLD:
        from additional_runs import run_job_sparse  # noqa: WPS433

        if verbose:
            print(f"  [sparse backend, |U_c|={Uc_size}>{DENSE_TO_SPARSE_UC_THRESHOLD}]")
        return run_job_sparse(job, gram_cap_gb=memory_cap_gb, verbose=verbose)
    if verbose:
        print(f"  [dense backend, |U_c|={Uc_size}]")
    info = run_job(job, memory_cap_gb=memory_cap_gb)
    if info.get("status") == "skipped_memory_cap":
        from additional_runs import run_job_sparse  # noqa: WPS433

        if verbose:
            print("  [dense skipped; falling back to sparse]")
        info = run_job_sparse(job, gram_cap_gb=memory_cap_gb, verbose=verbose)
    return info


def _job_lattice_label(job: Job) -> str:
    return job.graph_kind


def _iter_selected_jobs(args: argparse.Namespace) -> List[Job]:
    selected_boundaries = BOUNDARIES if args.boundary == "all" else (args.boundary,)
    jobs: List[Job] = []
    for boundary in selected_boundaries:
        by_family = all_jobs_by_family(boundary)
        for fam, fam_jobs in by_family.items():
            if fam not in args.families:
                continue
            for job in fam_jobs:
                if args.lattices and _job_lattice_label(job) not in args.lattices:
                    continue
                jobs.append(job)
    return jobs


def _print_estimate(jobs: Iterable[Job], memory_cap_gb: float) -> None:
    rows = []
    for job in jobs:
        Uc_size = _family_size(job)
        backend = "sparse" if Uc_size > DENSE_TO_SPARSE_UC_THRESHOLD else "dense"
        gram_gb = (Uc_size * Uc_size * 16) / (1024 ** 3)
        rows.append((job, Uc_size, backend, gram_gb))

    by_backend = defaultdict(int)
    for _job, _size, backend, _gb in rows:
        by_backend[backend] += 1

    print(f"Estimated catalogue: {len(rows)} jobs")
    print(f"  dense backend jobs:  {by_backend['dense']}")
    print(f"  sparse backend jobs: {by_backend['sparse']}")
    if rows:
        print(f"  max |U_c|: {max(size for _job, size, _backend, _gb in rows)}")
        print(f"  max Gram size: {max(gb for _job, _size, _backend, gb in rows):.2f} GB")
        over_cap = [r for r in rows if r[3] > memory_cap_gb]
        if over_cap:
            print(f"  over --memory-cap-gb={memory_cap_gb}: {len(over_cap)} jobs")
    for job, size, backend, gram_gb in rows:
        print(
            f"  {job.boundary:13s} {job.tag:58s} "
            f"mode={job.local_mode:11s} |U_c|={size:6d} "
            f"backend={backend:6s} Gram={gram_gb:6.2f} GB "
            f"witness_weight_cap={job.witness_weight_cap}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary",
        choices=("periodic", "open_boundary", "all"),
        default="periodic",
        help="Choose periodic, open-boundary, or both catalogues.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["dense", "xyz", "full_2body_no_fields"],
        help="Restrict to a subset of families.",
    )
    parser.add_argument(
        "--lattices",
        nargs="+",
        default=None,
        help="Restrict to a subset of lattice kinds.",
    )
    parser.add_argument(
        "--memory-cap-gb",
        type=float,
        default=8.0,
        help="Dense backend memory cap; sparse backend uses it as Gram cap.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run jobs even if a successful entry already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List jobs without executing.",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Also build each local family and print backend/memory estimates.",
    )
    args = parser.parse_args()

    jobs = _iter_selected_jobs(args)
    boundaries = sorted(set(j.boundary for j in jobs))
    print(
        f"Catalogue: {len(jobs)} jobs across "
        f"{len(set(j.family for j in jobs))} families; boundaries={boundaries}"
    )

    if args.dry_run:
        for job in jobs:
            print(
                f"  {job.boundary:13s} {job.tag}  "
                f"({job.family} / {job.graph_kind}{job.graph_args}, "
                f"k={job.k} R={job.R_geom} Rpatch={job.R_patch}, "
                f"mode={job.local_mode}, root={job.root_label}, "
                f"covers={job.covered_root_count}, "
                f"witness_weight_cap={job.witness_weight_cap})"
            )
        if args.estimate:
            print()
            _print_estimate(jobs, args.memory_cap_gb)
        return

    if args.estimate:
        _print_estimate(jobs, args.memory_cap_gb)
        print()

    cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

    def get_rows(boundary: str, fam: str, lat: str) -> List[Dict[str, Any]]:
        key = (boundary, fam, lat)
        if key not in cache:
            cache[key] = _load_results(boundary, fam, lat)
        return cache[key]

    t_total = time.time()
    for idx, job in enumerate(jobs, 1):
        lat = _job_lattice_label(job)
        rows = get_rows(job.boundary, job.family, lat)
        existing = next((r for r in rows if _entry_key(r) == _job_entry_key(job)), None)
        if existing is not None and existing.get("found_witness") and not args.force:
            print(f"[{idx}/{len(jobs)}] SKIP {job.tag}  (witness already on file)")
            continue

        print(
            f"\n[{idx}/{len(jobs)}] {job.tag}  "
            f"({job.boundary} / {job.family} / {job.graph_kind}{job.graph_args}, "
            f"k={job.k} R={job.R_geom} Rpatch={job.R_patch}, "
            f"mode={job.local_mode}, root={job.root_label}, "
            f"covers={job.covered_root_count})"
        )
        t0 = time.time()
        info = run_one(job, memory_cap_gb=args.memory_cap_gb, verbose=True)
        info.setdefault("tag", job.tag)
        info.setdefault("family", job.family)
        info.setdefault("graph_kind", job.graph_kind)
        info.setdefault("graph_args", list(job.graph_args))
        info.setdefault("k", job.k)
        info.setdefault("R_geom", job.R_geom)
        info.setdefault("R_patch", job.R_patch)
        info.setdefault("root", job.root)
        info.setdefault("root_label", job.root_label)
        info.setdefault("boundary", job.boundary)
        info.setdefault("local_mode", job.local_mode)
        info.setdefault("root_coord", job.root_coord)
        info.setdefault("covered_root_count", job.covered_root_count)
        info.setdefault("covered_root_sample", job.covered_root_sample)
        info.setdefault("coverage_note", job.coverage_note)
        cache[(job.boundary, job.family, lat)] = _merge_entry(rows, info)
        _save_results(job.boundary, job.family, lat, cache[(job.boundary, job.family, lat)])

        dt = time.time() - t0
        print(
            f"   -> status={info.get('status')} found={info.get('found_witness')} "
            f"rank={info.get('best_rank')}/{info.get('target_rank')} "
            f"({dt:.1f}s)"
        )

    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
