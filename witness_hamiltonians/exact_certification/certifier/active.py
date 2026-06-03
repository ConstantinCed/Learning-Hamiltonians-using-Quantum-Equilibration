from __future__ import annotations

import random
import time
from typing import Dict, List, Sequence

from .matrix import MatrixBuildResult, hash_row_list
from .pauli import PauliString, anticommutes, commutator_sign


WITNESS_VALUES = (-3, -2, -1, 1, 2, 3)


def active_witness_vector(paulis: Sequence[PauliString], seed: int, weight_cap: int) -> List[int]:
    """Deterministic integer witness supported on Pauli strings up to weight_cap."""

    rng = random.Random(seed)
    h = [0] * len(paulis)
    for idx, pauli in enumerate(paulis):
        if pauli.support_size <= weight_cap:
            h[idx] = rng.choice(WITNESS_VALUES)
    return h


def build_active_matrix(local, h: Sequence[int], max_pair_checks: int | None = None) -> MatrixBuildResult:
    """Build B_c(h) exactly when h is sparse.

    The full coordinate set U_c is retained as columns.  We only enumerate
    commutator pairs [P_i, P_j] with h_j != 0, which is exact because all other
    terms have zero coefficient in the witness.
    """

    t0 = time.perf_counter()
    U = local.paulis
    d = len(U)
    active_indices = [j for j, hj in enumerate(h) if hj != 0]
    site_to_active_indices: Dict[int, List[int]] = {}
    root_codes: List[int] = []
    root_bit_index = local.root_bit.bit_length() - 1 if local.root_bit else -1

    for j in active_indices:
        mask = U[j].support_mask
        while mask:
            bit = mask & -mask
            site_to_active_indices.setdefault(bit.bit_length() - 1, []).append(j)
            mask ^= bit

    for pauli in U:
        if root_bit_index >= 0:
            root_codes.append(
                (((pauli.x >> root_bit_index) & 1) << 1)
                | ((pauli.z >> root_bit_index) & 1)
            )
        else:
            root_codes.append(0)

    row_map: Dict[PauliString, Dict[int, int]] = {}
    pair_checks = 0
    anticommuting_pairs = 0
    skipped_not_rooted = 0
    seen_generation = [0] * d
    generation = 0

    for i, u in enumerate(U):
        generation += 1
        candidate_indices: List[int] = []
        mask = u.support_mask
        while mask:
            bit = mask & -mask
            for j in site_to_active_indices.get(bit.bit_length() - 1, []):
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
                    anticommuting_pairs=anticommuting_pairs,
                    skipped_not_rooted=skipped_not_rooted,
                    runtime_seconds=time.perf_counter() - t0,
                    message=f"candidate pair scan exceeded limit {max_pair_checks}",
                )

            v = U[j]
            if not anticommutes(u, v):
                continue
            anticommuting_pairs += 1
            w = u.xor(v)
            if not (w.support_mask & local.root_bit):
                skipped_not_rooted += 1
                continue
            row = row_map.setdefault(w, {})
            val = commutator_sign(u, v) * h[j]
            new_val = row.get(i, 0) + val
            if new_val:
                row[i] = new_val
            elif i in row:
                del row[i]

    ordered = sorted(row_map.items(), key=lambda item: item[0].sort_key())
    row_keys = [key for key, _row in ordered]
    rows = [dict(sorted(row.items())) for _key, row in ordered]
    nnz = sum(len(row) for row in rows)
    row_hash = hash_row_list(row_keys, local.active_vertices, getattr(local, "coords"))

    return MatrixBuildResult(
        status="ok",
        rows=rows,
        row_keys=row_keys,
        row_hash=row_hash,
        nnz=nnz,
        pair_checks=pair_checks,
        anticommuting_pairs=anticommuting_pairs,
        skipped_not_rooted=skipped_not_rooted,
        runtime_seconds=time.perf_counter() - t0,
    )
