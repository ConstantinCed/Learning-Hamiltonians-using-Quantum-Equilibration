#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from certifier.active import active_witness_vector, build_active_matrix
from certifier.core import CaseParams, hash_witness, jsonable_coord, write_certificate
from certifier.cpp_rank import cpp_modular_rank
from certifier.families import LocalPauliLimitExceeded, generate_local_paulis, normalize_family
from certifier.graphs import normalize_boundary, normalize_lattice
from certifier.matrix import check_integer_null
from certifier.types import enumerate_rooted_types


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Exact active-witness certifier")
    parser.add_argument("--family", required=True)
    parser.add_argument("--lattice", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prime", type=int, default=2147483647)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-witnesses", type=int, default=20)
    parser.add_argument("--max-pair-checks", type=int, default=500_000_000)
    parser.add_argument("--witness-weight-cap", type=int, default=2)
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--open-checks", type=int, default=2)
    args = parser.parse_args(argv)

    params = CaseParams(
        family=args.family,
        lattice=args.lattice,
        boundary=args.boundary,
        R=args.R,
        k=args.k,
        mode="theorem",
        prime=args.prime,
        max_witnesses=args.max_witnesses,
        max_pair_checks=args.max_pair_checks,
        seed=args.seed,
        L=args.L,
        open_checks=args.open_checks,
        backend="cpp_active_sparse",
    )
    cert = certify_case_active(params, args.witness_weight_cap)
    write_certificate(cert, args.out)
    print(f"{args.out}: {cert['overall_status']} in {cert['runtime_seconds']:.2f}s")
    return 0 if cert["overall_status"] == "exactly_certified" else 2


def certify_case_active(params: CaseParams, witness_weight_cap: int) -> Dict[str, object]:
    family = normalize_family(params.family)
    lattice = normalize_lattice(params.lattice)
    boundary = normalize_boundary(params.boundary)
    t0 = time.perf_counter()
    enum = enumerate_rooted_types(lattice, boundary, params.R, params.L, params.open_checks)
    local_results = []
    for idx, typ in enumerate(enum.types, 1):
        print(
            f"  local type {idx}/{len(enum.types)} id={typ.type_id} "
            f"root={jsonable_coord(enum.graph.coords[typ.root])}",
            flush=True,
        )
        result = certify_local_type_active(params, enum, typ.type_id, typ.root, witness_weight_cap)
        print(
            f"    {result['status']} d_U={result.get('d_U')} "
            f"W={result.get('W_count')} nnz={result.get('nnz')} "
            f"rank={result.get('rank_mod_p', result.get('last_rank_mod_p'))}",
            flush=True,
        )
        local_results.append(result)
    exact = enum.stabilized and local_results and all(
        result["status"] == "exactly_certified" for result in local_results
    )
    if exact:
        overall = "exactly_certified"
    elif any(result["status"] == "failed_resource_limit" for result in local_results):
        overall = "failed_resource_limit"
    elif any(result["status"] == "failed_math_obstruction" for result in local_results):
        overall = "failed_math_obstruction"
    else:
        overall = "inconclusive"
    if not enum.stabilized:
        overall = "inconclusive"

    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "software_versions": software_versions(),
        "parameters": {
            "family": family,
            "lattice": lattice,
            "boundary": boundary,
            "R": params.R,
            "k": params.k,
            "mode": params.mode,
            "prime": params.prime,
            "max_witnesses": params.max_witnesses,
            "max_pair_checks": params.max_pair_checks,
            "max_rank_seconds": params.max_rank_seconds,
            "max_dimension": params.max_dimension,
            "seed": params.seed,
            "backend": params.backend,
            "witness_weight_cap": witness_weight_cap,
        },
        "type_enumeration": {
            "radius": enum.radius,
            "stabilized": enum.stabilized,
            "stabilization_Ls": enum.stabilization_Ls,
            "previous_type_count": enum.previous_type_count,
            "type_count": len(enum.types),
            "iso_checks": enum.iso_checks,
            "graph_L": enum.graph.L,
        },
        "overall_status": overall,
        "local_types": local_results,
        "runtime_seconds": time.perf_counter() - t0,
        "peak_memory_mb": peak_memory_mb(),
    }


def certify_local_type_active(
    params: CaseParams,
    enum,
    type_id: int,
    root: int,
    witness_weight_cap: int,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    graph = enum.graph
    try:
        local = generate_local_paulis(
            graph,
            root,
            params.family,
            params.R,
            params.k,
            params.mode,
            max_paulis=params.max_dimension,
        )
    except LocalPauliLimitExceeded as exc:
        return {
            "local_type_id": type_id,
            "root_vertex": root,
            "root_coord": jsonable_coord(graph.coords[root]),
            "multiplicity": next((typ.multiplicity for typ in enum.types if typ.type_id == type_id), 1),
            "family": normalize_family(params.family),
            "lattice": graph.lattice,
            "boundary": graph.boundary,
            "graph_L": graph.L,
            "R": params.R,
            "k": params.k,
            "mode": params.mode,
            "d_U": exc.pauli_count,
            "support_count": exc.support_count,
            "target_rank": None,
            "prime": params.prime,
            "backend": "cpp_active_sparse",
            "status": "failed_resource_limit",
            "message": str(exc),
            "runtime_seconds": time.perf_counter() - t0,
            "peak_memory_mb": peak_memory_mb(),
        }

    setattr(local, "coords", graph.coords)
    d = len(local.paulis)
    target = d - 1
    base = {
        "local_type_id": type_id,
        "root_vertex": root,
        "root_coord": jsonable_coord(graph.coords[root]),
        "multiplicity": next((typ.multiplicity for typ in enum.types if typ.type_id == type_id), 1),
        "family": normalize_family(params.family),
        "lattice": graph.lattice,
        "boundary": graph.boundary,
        "graph_L": graph.L,
        "R": params.R,
        "k": params.k,
        "mode": params.mode,
        "d_U": d,
        "support_count": local.support_count,
        "target_rank": target,
        "coordinate_hash": local.coordinate_hash,
        "prime": params.prime,
        "backend": "cpp_active_sparse_modular_elimination",
        "witness_weight_cap": witness_weight_cap,
        "witness_generation": {
            "rule": (
                "Python random.Random(seed).choice([-3,-2,-1,1,2,3]) "
                "on coordinates with Pauli weight <= witness_weight_cap; zero otherwise"
            ),
            "values": [-3, -2, -1, 1, 2, 3],
        },
    }
    if d < 2:
        base.update(
            {
                "status": "failed_math_obstruction",
                "message": "|U_c| < 2",
                "runtime_seconds": time.perf_counter() - t0,
                "peak_memory_mb": peak_memory_mb(),
            }
        )
        return base

    last_rank = None
    last_build = None
    for attempt in range(params.max_witnesses):
        witness_seed = params.seed + 1_000_003 * type_id + attempt
        h = active_witness_vector(local.paulis, witness_seed, witness_weight_cap)
        active_h_size = sum(1 for value in h if value)
        build = build_active_matrix(local, h, params.max_pair_checks)
        last_build = build
        if build.status != "ok":
            base.update(
                {
                    "status": "failed_resource_limit",
                    "message": build.message,
                    "witness_seed": witness_seed,
                    "witness_attempt": attempt,
                    "active_h_size": active_h_size,
                    "W_count": None,
                    "nnz": None,
                    "pair_checks": build.pair_checks,
                    "runtime_seconds": time.perf_counter() - t0,
                    "peak_memory_mb": peak_memory_mb(),
                }
            )
            return base
        if not check_integer_null(build.rows, h):
            base.update(
                {
                    "status": "implementation_error",
                    "message": "exact integer null-vector check failed",
                    "witness_seed": witness_seed,
                    "witness_attempt": attempt,
                    "active_h_size": active_h_size,
                    "W_count": len(build.row_keys),
                    "nnz": build.nnz,
                    "row_hash": build.row_hash,
                    "integer_null_check": False,
                    "runtime_seconds": time.perf_counter() - t0,
                    "peak_memory_mb": peak_memory_mb(),
                }
            )
            return base
        if len(build.row_keys) < target:
            base.update(
                {
                    "status": "failed_math_obstruction",
                    "message": "|W_c| < |U_c|-1, so rank target is impossible",
                    "witness_seed": witness_seed,
                    "witness_attempt": attempt,
                    "active_h_size": active_h_size,
                    "W_count": len(build.row_keys),
                    "nnz": build.nnz,
                    "row_hash": build.row_hash,
                    "integer_null_check": True,
                    "runtime_seconds": time.perf_counter() - t0,
                    "peak_memory_mb": peak_memory_mb(),
                }
            )
            return base

        rank = cpp_modular_rank(build.rows, d, params.prime, target)
        last_rank = rank
        if rank.resource_limited:
            base.update(
                {
                    "status": "failed_resource_limit",
                    "message": rank.message,
                    "witness_seed": witness_seed,
                    "witness_attempt": attempt,
                    "active_h_size": active_h_size,
                    "W_count": len(build.row_keys),
                    "nnz": build.nnz,
                    "pair_checks": build.pair_checks,
                    "row_hash": build.row_hash,
                    "integer_null_check": True,
                    "partial_rank_mod_p": rank.rank,
                    "build_runtime_seconds": build.runtime_seconds,
                    "rank_runtime_seconds": rank.runtime_seconds,
                    "runtime_seconds": time.perf_counter() - t0,
                    "peak_memory_mb": peak_memory_mb(),
                }
            )
            return base
        if rank.rank == target:
            base.update(
                {
                    "status": "exactly_certified",
                    "message": (
                        "exact integer null check and exact C++ odd-prime "
                        "finite-field rank check passed"
                    ),
                    "witness_seed": witness_seed,
                    "witness_attempt": attempt,
                    "witness_hash": hash_witness(h),
                    "active_h_size": active_h_size,
                    "W_count": len(build.row_keys),
                    "nnz": build.nnz,
                    "pair_checks": build.pair_checks,
                    "anticommuting_pairs": build.anticommuting_pairs,
                    "skipped_pairs_not_rooted": build.skipped_not_rooted,
                    "row_hash": build.row_hash,
                    "integer_null_check": True,
                    "rank_mod_p": rank.rank,
                    "rank_target_reached": rank.target_reached,
                    "rank_processed_rows": rank.processed_rows,
                    "rank_input_nnz": rank.input_nnz,
                    "build_runtime_seconds": build.runtime_seconds,
                    "rank_runtime_seconds": rank.runtime_seconds,
                    "runtime_seconds": time.perf_counter() - t0,
                    "peak_memory_mb": peak_memory_mb(),
                }
            )
            return base

    base.update(
        {
            "status": "inconclusive",
            "message": f"no active witness certified within {params.max_witnesses} attempts",
            "attempts": params.max_witnesses,
            "W_count": len(last_build.row_keys) if last_build else None,
            "nnz": last_build.nnz if last_build else None,
            "pair_checks": last_build.pair_checks if last_build else None,
            "row_hash": last_build.row_hash if last_build else None,
            "integer_null_check": True if last_build else None,
            "last_rank_mod_p": last_rank.rank if last_rank else None,
            "runtime_seconds": time.perf_counter() - t0,
            "peak_memory_mb": peak_memory_mb(),
        }
    )
    return base


def software_versions() -> Dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "certifier_backend": "cpp_active_sparse",
    }


def peak_memory_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024


if __name__ == "__main__":
    raise SystemExit(main())
