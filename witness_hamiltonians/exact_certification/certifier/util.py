from __future__ import annotations


def popcount(x: int) -> int:
    return bin(x).count("1")
