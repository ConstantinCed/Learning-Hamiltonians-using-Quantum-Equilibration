from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .families import LocalPauliSet
from .pauli import PauliString, anticommutes, commutator_sign, pauli_jsonable


@dataclass
class MatrixBuildResult:
    status: str
    rows: List[Dict[int, int]]
    row_keys: List[PauliString]
    row_hash: str
    nnz: int
    pair_checks: int
    anticommuting_pairs: int
    skipped_not_rooted: int
    runtime_seconds: float
    message: str = ""


def build_local_matrix(local: LocalPauliSet, h: Sequence[int], max_pair_checks: Optional[int] = None) -> MatrixBuildResult:
    t0 = time.perf_counter()
    U = local.paulis
    d = len(U)
    site_to_indices: Dict[int, List[int]] = {}
    root_codes: List[int] = []
    root_bit_index = local.root_bit.bit_length() - 1 if local.root_bit else -1
    for j, p in enumerate(U):
        mask = p.support_mask
        while mask:
            bit = mask & -mask
            site_to_indices.setdefault(bit.bit_length() - 1, []).append(j)
            mask ^= bit
        if root_bit_index >= 0:
            root_codes.append((((p.x >> root_bit_index) & 1) << 1) | ((p.z >> root_bit_index) & 1))
        else:
            root_codes.append(0)

    row_map: Dict[PauliString, Dict[int, int]] = {}
    pair_checks = 0
    anti = 0
    skipped = 0
    seen_generation = [0] * d
    generation = 0
    for i, u in enumerate(U):
        generation += 1
        candidate_indices: List[int] = []
        mask = u.support_mask
        while mask:
            bit = mask & -mask
            for j in site_to_indices.get(bit.bit_length() - 1, []):
                if seen_generation[j] != generation:
                    seen_generation[j] = generation
                    candidate_indices.append(j)
            mask ^= bit
        for j in candidate_indices:
            if root_codes[i] == root_codes[j]:
                continue
            pair_checks += 1
            if max_pair_checks is not None and pair_checks > max_pair_checks:
                return MatrixBuildResult(
                    status="resource_limit",
                    rows=[],
                    row_keys=[],
                    row_hash="",
                    nnz=0,
                    pair_checks=pair_checks,
                    anticommuting_pairs=anti,
                    skipped_not_rooted=skipped,
                    runtime_seconds=time.perf_counter() - t0,
                    message=f"candidate pair scan exceeded limit {max_pair_checks}",
                )
            v = U[j]
            if not anticommutes(u, v):
                continue
            anti += 1
            w = u.xor(v)
            if not (w.support_mask & local.root_bit):
                skipped += 1
                continue
            row = row_map.setdefault(w, {})
            val = commutator_sign(u, v) * h[j]
            new_val = row.get(i, 0) + val
            if new_val:
                row[i] = new_val
            elif i in row:
                del row[i]

    ordered = sorted(row_map.items(), key=lambda item: item[0].sort_key())
    row_keys = [k for k, _ in ordered]
    rows = [dict(sorted(row.items())) for _, row in ordered]
    nnz = sum(len(row) for row in rows)
    row_hash = hash_row_list(row_keys, local.active_vertices, _coords_from_local(local))
    return MatrixBuildResult(
        status="ok",
        rows=rows,
        row_keys=row_keys,
        row_hash=row_hash,
        nnz=nnz,
        pair_checks=pair_checks,
        anticommuting_pairs=anti,
        skipped_not_rooted=skipped,
        runtime_seconds=time.perf_counter() - t0,
    )


def check_integer_null(rows: Sequence[Dict[int, int]], h: Sequence[int]) -> bool:
    for row in rows:
        if sum(value * h[col] for col, value in row.items()) != 0:
            return False
    return True


def hash_row_list(row_keys: Sequence[PauliString], active_vertices: Sequence[int], coords: Dict[int, object]) -> str:
    payload = [pauli_jsonable(p, active_vertices, coords) for p in row_keys]
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _coords_from_local(local: LocalPauliSet) -> Dict[int, object]:
    # LocalPauliSet stores only active vertices; the graph coordinate mapping is
    # attached dynamically by core before hashing.  This fallback makes tests
    # over synthetic local sets possible.
    coords = getattr(local, "coords", None)
    if coords is not None:
        return coords
    return {v: v for v in local.active_vertices}
