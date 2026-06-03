from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


Coord = object


@dataclass
class Graph:
    lattice: str
    boundary: str
    L: int
    vertices: Tuple[int, ...]
    coords: Dict[int, Coord]
    adj: Dict[int, Tuple[int, ...]]

    def neighbors(self, v: int) -> Tuple[int, ...]:
        return self.adj[v]

    def distances(self, root: int, max_radius: Optional[int] = None) -> Dict[int, int]:
        dist = {root: 0}
        q: deque[int] = deque([root])
        while q:
            v = q.popleft()
            nd = dist[v] + 1
            if max_radius is not None and nd > max_radius:
                continue
            for nb in self.adj[v]:
                if nb not in dist:
                    dist[nb] = nd
                    q.append(nb)
        return dist

    def ball(self, root: int, radius: int) -> Set[int]:
        return set(self.distances(root, radius))

    def is_connected(self) -> bool:
        if not self.vertices:
            return True
        return len(self.distances(self.vertices[0])) == len(self.vertices)


def _make_graph(lattice: str, boundary: str, L: int, coords_list: Sequence[Coord], edge_coords: Iterable[Tuple[Coord, Coord]]) -> Graph:
    coord_to_id = {coord: i for i, coord in enumerate(coords_list)}
    adj_sets: Dict[int, Set[int]] = {i: set() for i in range(len(coords_list))}
    for a, b in edge_coords:
        if a == b:
            continue
        ia = coord_to_id.get(a)
        ib = coord_to_id.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        adj_sets[ia].add(ib)
        adj_sets[ib].add(ia)
    coords = {i: coord for coord, i in coord_to_id.items()}
    adj = {i: tuple(sorted(nbs)) for i, nbs in adj_sets.items()}
    return Graph(lattice=lattice, boundary=boundary, L=L, vertices=tuple(range(len(coords_list))), coords=coords, adj=adj)


def generate_graph(lattice: str, boundary: str, L: int) -> Graph:
    lattice = normalize_lattice(lattice)
    boundary = normalize_boundary(boundary)
    if lattice == "chain":
        return _chain(boundary, L)
    if lattice == "square":
        return _square(boundary, L)
    if lattice == "triangular":
        return _triangular(boundary, L)
    if lattice == "honeycomb":
        return _honeycomb(boundary, L)
    if lattice == "cubic":
        return _cubic(boundary, L)
    raise ValueError(f"unknown lattice {lattice}")


def normalize_lattice(lattice: str) -> str:
    aliases = {"1d": "chain", "path": "chain", "cycle": "chain", "3d_cubic": "cubic"}
    return aliases.get(lattice, lattice)


def normalize_boundary(boundary: str) -> str:
    aliases = {"torus": "periodic", "cycle": "periodic", "path": "open"}
    return aliases.get(boundary, boundary)


def safe_periodic_L(R: int) -> int:
    return max(4 * R + 3, 7)


def open_start_L(R: int) -> int:
    return max(4 * R + 3, 7)


def _chain(boundary: str, L: int) -> Graph:
    coords = list(range(L))
    edges = []
    for i in range(L - 1):
        edges.append((i, i + 1))
    if boundary == "periodic" and L > 2:
        edges.append((L - 1, 0))
    return _make_graph("chain", boundary, L, coords, edges)


def _square(boundary: str, L: int) -> Graph:
    coords = [(i, j) for i in range(L) for j in range(L)]
    edges = []
    for i, j in coords:
        for di, dj in [(1, 0), (0, 1)]:
            ni, nj = i + di, j + dj
            if boundary == "periodic":
                ni %= L
                nj %= L
            if 0 <= ni < L and 0 <= nj < L:
                edges.append(((i, j), (ni, nj)))
    return _make_graph("square", boundary, L, coords, edges)


def _triangular(boundary: str, L: int) -> Graph:
    coords = [(i, j) for i in range(L) for j in range(L)]
    edges = []
    for i, j in coords:
        for di, dj in [(1, 0), (0, 1), (1, -1)]:
            ni, nj = i + di, j + dj
            if boundary == "periodic":
                ni %= L
                nj %= L
            if 0 <= ni < L and 0 <= nj < L:
                edges.append(((i, j), (ni, nj)))
    return _make_graph("triangular", boundary, L, coords, edges)


def _honeycomb(boundary: str, L: int) -> Graph:
    coords = [(i, j, s) for i in range(L) for j in range(L) for s in (0, 1)]
    edges = []
    for i in range(L):
        for j in range(L):
            a = (i, j, 0)
            for bi, bj in [(i, j), (i - 1, j), (i, j - 1)]:
                ni, nj = bi, bj
                if boundary == "periodic":
                    ni %= L
                    nj %= L
                if 0 <= ni < L and 0 <= nj < L:
                    edges.append((a, (ni, nj, 1)))
    return _make_graph("honeycomb", boundary, L, coords, edges)


def _cubic(boundary: str, L: int) -> Graph:
    coords = [(i, j, k) for i in range(L) for j in range(L) for k in range(L)]
    edges = []
    for i, j, k in coords:
        for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            if boundary == "periodic":
                ni %= L
                nj %= L
                nk %= L
            if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L:
                edges.append(((i, j, k), (ni, nj, nk)))
    return _make_graph("cubic", boundary, L, coords, edges)
