from __future__ import annotations

import json
import os
import platform
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .families import LocalPauliLimitExceeded, generate_local_paulis, normalize_family
from .graphs import generate_graph, normalize_boundary, normalize_lattice, safe_periodic_L
from .matrix import build_local_matrix, check_integer_null
from .rank import modular_rank
from .types import TypeEnumeration, enumerate_rooted_types


WITNESS_VALUES = (-3, -2, -1, 1, 2, 3)


@dataclass
class CaseParams:
    family: str
    lattice: str
    boundary: str
    R: int
    k: Optional[int] = None
    mode: str = "theorem"
    prime: int = 2147483647
    max_witnesses: int = 20
    max_pair_checks: int = 5_000_000
    max_rank_seconds: Optional[float] = None
    max_dimension: Optional[int] = None
    seed: int = 1729
    L: Optional[int] = None
    open_checks: int = 2
    backend: str = "python_sparse"


def certify_case(params: CaseParams) -> Dict[str, object]:
    family = normalize_family(params.family)
    lattice = normalize_lattice(params.lattice)
    boundary = normalize_boundary(params.boundary)
    t0 = time.perf_counter()
    enum = enumerate_rooted_types(lattice, boundary, params.R, params.L, params.open_checks)
    local_results = []
    for typ in enum.types:
        result = certify_local_type(params, enum, typ.type_id, typ.root)
        local_results.append(result)

    exact = enum.stabilized and local_results and all(r["status"] == "exactly_certified" for r in local_results)
    if exact:
        overall = "exactly_certified"
    elif any(r["status"] == "failed_math_obstruction" for r in local_results):
        overall = "failed_math_obstruction"
    elif any(r["status"] == "failed_resource_limit" for r in local_results):
        overall = "failed_resource_limit"
    else:
        overall = "inconclusive"
    if not enum.stabilized:
        overall = "inconclusive"

    anchoring = None
    if family == "exact_two_body_no_fields":
        anchoring = {
            "nonempty_support_respecting_anchoring": enum.graph.is_connected() and len(enum.graph.vertices) >= 2,
            "reason": "connected graph with at least two vertices" if enum.graph.is_connected() and len(enum.graph.vertices) >= 2 else "graph is disconnected or too small",
        }

    return {
        "schema_version": 1,
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
        "anchoring": anchoring,
        "overall_status": overall,
        "local_types": local_results,
        "runtime_seconds": time.perf_counter() - t0,
        "peak_memory_mb": peak_memory_mb(),
    }


def certify_local_type(params: CaseParams, enum: TypeEnumeration, type_id: int, root: int) -> Dict[str, object]:
    t0 = time.perf_counter()
    graph = enum.graph
    try:
        local = generate_local_paulis(
            graph, root, params.family, params.R, params.k, params.mode, max_paulis=params.max_dimension
        )
    except LocalPauliLimitExceeded as exc:
        return {
            "local_type_id": type_id,
            "root_vertex": root,
            "root_coord": jsonable_coord(graph.coords[root]),
            "multiplicity": next((t.multiplicity for t in enum.types if t.type_id == type_id), 1),
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
            "coordinate_hash": None,
            "prime": params.prime,
            "backend": "python_sparse_modular_elimination",
            "status": "failed_resource_limit",
            "message": str(exc),
            "W_count": None,
            "nnz": None,
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
        "multiplicity": next((t.multiplicity for t in enum.types if t.type_id == type_id), 1),
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
        "backend": "python_sparse_modular_elimination",
        "witness_generation": {
            "rule": "Python random.Random(seed).choice([-3,-2,-1,1,2,3]) for each coordinate in local coordinate order",
            "values": list(WITNESS_VALUES),
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
    if d * d > params.max_pair_checks:
        base.update(
            {
                "status": "failed_resource_limit",
                "message": f"ordered pair scan {d*d} exceeds limit {params.max_pair_checks}",
                "W_count": None,
                "nnz": None,
                "pair_checks": 0,
                "runtime_seconds": time.perf_counter() - t0,
                "peak_memory_mb": peak_memory_mb(),
            }
        )
        return base

    last_rank = None
    last_build = None
    for attempt in range(params.max_witnesses):
        witness_seed = params.seed + 1_000_003 * type_id + attempt
        h = witness_vector(d, witness_seed)
        build = build_local_matrix(local, h, params.max_pair_checks)
        last_build = build
        if build.status != "ok":
            base.update(
                {
                    "status": "failed_resource_limit",
                    "message": build.message,
                    "W_count": None,
                    "nnz": None,
                    "pair_checks": build.pair_checks,
                    "runtime_seconds": time.perf_counter() - t0,
                    "peak_memory_mb": peak_memory_mb(),
                }
            )
            return base
        null_ok = check_integer_null(build.rows, h)
        if not null_ok:
            base.update(
                {
                    "status": "implementation_error",
                    "message": "exact integer null-vector check failed",
                    "witness_seed": witness_seed,
                    "witness_attempt": attempt,
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
                    "W_count": len(build.row_keys),
                    "nnz": build.nnz,
                    "row_hash": build.row_hash,
                    "integer_null_check": True,
                    "runtime_seconds": time.perf_counter() - t0,
                    "peak_memory_mb": peak_memory_mb(),
                }
            )
            return base
        rank = modular_rank(build.rows, d, params.prime, target=target, max_seconds=params.max_rank_seconds)
        last_rank = rank
        if rank.resource_limited:
            base.update(
                {
                    "status": "failed_resource_limit",
                    "message": rank.message,
                    "witness_seed": witness_seed,
                    "witness_attempt": attempt,
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
                    "message": "exact integer null check and exact odd-prime finite-field rank check passed",
                    "witness_seed": witness_seed,
                    "witness_attempt": attempt,
                    "witness_hash": hash_witness(h),
                    "W_count": len(build.row_keys),
                    "nnz": build.nnz,
                    "pair_checks": build.pair_checks,
                    "anticommuting_pairs": build.anticommuting_pairs,
                    "skipped_pairs_not_rooted": build.skipped_not_rooted,
                    "row_hash": build.row_hash,
                    "integer_null_check": True,
                    "rank_mod_p": rank.rank,
                    "rank_target_reached": rank.target_reached,
                    "pivot_columns": rank.pivot_columns[: min(len(rank.pivot_columns), 2000)],
                    "pivot_columns_truncated": len(rank.pivot_columns) > 2000,
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
            "message": f"no witness certified within {params.max_witnesses} attempts",
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


def witness_vector(d: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.choice(WITNESS_VALUES) for _ in range(d)]


def hash_witness(h: Sequence[int]) -> str:
    import hashlib

    data = json.dumps(list(h), separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def jsonable_coord(coord: object) -> object:
    if isinstance(coord, tuple):
        return [jsonable_coord(x) for x in coord]
    return coord


def software_versions() -> Dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "certifier_backend": "pure_python",
    }


def peak_memory_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024


def write_certificate(cert: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2, sort_keys=True)
