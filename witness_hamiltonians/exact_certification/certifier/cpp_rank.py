from __future__ import annotations

import os
import pathlib
import re
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class CppRankResult:
    rank: int
    target_reached: bool
    processed_rows: int
    input_nnz: int
    runtime_seconds: float
    backend: str = "cpp_sparse_modular_elimination"
    resource_limited: bool = False
    message: str = ""


def cpp_modular_rank(
    rows: Sequence[Dict[int, int]],
    ncols: int,
    p: int,
    target: int,
) -> CppRankResult:
    binary = ensure_rank_binary()
    with tempfile.NamedTemporaryFile(prefix="exact_rows_", suffix=".bin", delete=False) as tmp:
        path = tmp.name
    try:
        write_rows_binary(rows, ncols, p, path)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(binary), path, str(target)],
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode not in (0, 1):
            return CppRankResult(
                rank=0,
                target_reached=False,
                processed_rows=0,
                input_nnz=0,
                runtime_seconds=elapsed,
                resource_limited=True,
                message=(proc.stderr or proc.stdout).strip(),
            )
        parsed = parse_rank_output(proc.stdout)
        rank = parsed["rank"]
        return CppRankResult(
            rank=rank,
            target_reached=rank >= target,
            processed_rows=parsed["processed_rows"],
            input_nnz=parsed["input_nnz"],
            runtime_seconds=elapsed,
        )
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def write_rows_binary(rows: Sequence[Dict[int, int]], ncols: int, p: int, path: str) -> None:
    with open(path, "wb") as f:
        f.write(b"EXR1")
        f.write(struct.pack("<QQII", len(rows), ncols, p, len(rows)))
        for row in rows:
            f.write(struct.pack("<I", len(row)))
            for col, val in sorted(row.items()):
                f.write(struct.pack("<Ii", int(col), int(val)))


def ensure_rank_binary() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent
    source = here / "rank_modp.cpp"
    binary = here / "rank_modp"
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    compiler = os.environ.get("CXX", "clang++")
    subprocess.run(
        [compiler, "-O3", "-std=c++17", str(source), "-o", str(binary)],
        check=True,
    )
    return binary


def parse_rank_output(stdout: str) -> Dict[str, int]:
    match = re.search(
        r"rank\s+(\d+)\s+target\s+(\d+)\s+processed_rows\s+(\d+)/(\d+)\s+input_nnz\s+(\d+)",
        stdout,
    )
    if not match:
        raise ValueError(f"could not parse rank output: {stdout!r}")
    return {
        "rank": int(match.group(1)),
        "target": int(match.group(2)),
        "processed_rows": int(match.group(3)),
        "total_rows": int(match.group(4)),
        "input_nnz": int(match.group(5)),
    }
