from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class RankResult:
    rank: int
    target_reached: bool
    pivot_columns: List[int]
    runtime_seconds: float
    backend: str = "python_sparse_modular_elimination"
    resource_limited: bool = False
    message: str = ""


def modular_rank(
    rows: Sequence[Dict[int, int]],
    ncols: int,
    p: int,
    target: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> RankResult:
    if p == 2 or p <= 2:
        raise ValueError("rank prime must be an odd prime")
    t0 = time.perf_counter()
    pivots: Dict[int, Dict[int, int]] = {}
    pivot_columns: List[int] = []
    ordered_rows = sorted(rows, key=lambda r: (len(r), min(r) if r else -1))
    for source in ordered_rows:
        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            return RankResult(
                rank=len(pivot_columns),
                target_reached=False,
                pivot_columns=pivot_columns,
                runtime_seconds=time.perf_counter() - t0,
                resource_limited=True,
                message=f"rank time exceeded {max_seconds} seconds",
            )
        row = {c: (v % p) for c, v in source.items() if v % p}
        while row:
            if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
                return RankResult(
                    rank=len(pivot_columns),
                    target_reached=False,
                    pivot_columns=pivot_columns,
                    runtime_seconds=time.perf_counter() - t0,
                    resource_limited=True,
                    message=f"rank time exceeded {max_seconds} seconds",
                )
            pc = min(row)
            coeff = row[pc]
            pivot = pivots.get(pc)
            if pivot is None:
                inv = pow(coeff, p - 2, p)
                normalized = {c: (v * inv) % p for c, v in row.items() if (v * inv) % p}
                pivots[pc] = normalized
                pivot_columns.append(pc)
                if target is not None and len(pivot_columns) >= target:
                    return RankResult(
                        rank=len(pivot_columns),
                        target_reached=True,
                        pivot_columns=pivot_columns,
                        runtime_seconds=time.perf_counter() - t0,
                    )
                break
            factor = coeff
            for c, pv in pivot.items():
                nv = (row.get(c, 0) - factor * pv) % p
                if nv:
                    row[c] = nv
                elif c in row:
                    del row[c]
    return RankResult(
        rank=len(pivot_columns),
        target_reached=(target is not None and len(pivot_columns) >= target),
        pivot_columns=pivot_columns,
        runtime_seconds=time.perf_counter() - t0,
    )


def rational_rank_from_rows(rows: Sequence[Dict[int, int]], ncols: int) -> int:
    mat = [[Fraction(0) for _ in range(ncols)] for _ in rows]
    for r, row in enumerate(rows):
        for c, v in row.items():
            mat[r][c] = Fraction(v)
    return rational_rank_dense(mat)


def rational_rank_dense(mat: List[List[Fraction]]) -> int:
    if not mat:
        return 0
    m = len(mat)
    n = len(mat[0])
    rank = 0
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if mat[r][col]:
                pivot = r
                break
        if pivot is None:
            continue
        mat[row], mat[pivot] = mat[pivot], mat[row]
        pv = mat[row][col]
        mat[row] = [x / pv for x in mat[row]]
        for r in range(m):
            if r != row and mat[r][col]:
                factor = mat[r][col]
                mat[r] = [a - factor * b for a, b in zip(mat[r], mat[row])]
        rank += 1
        row += 1
        if row == m:
            break
    return rank


def gamma_rank_from_B(rows: Sequence[Dict[int, int]], row_support_sizes: Sequence[int], ncols: int) -> int:
    gamma = [[Fraction(0) for _ in range(ncols)] for _ in range(ncols)]
    for row, supp_size in zip(rows, row_support_sizes):
        if not row:
            continue
        weight = Fraction(1, supp_size)
        items = list(row.items())
        for c1, v1 in items:
            for c2, v2 in items:
                gamma[c1][c2] += weight * v1 * v2
    return rational_rank_dense(gamma)
