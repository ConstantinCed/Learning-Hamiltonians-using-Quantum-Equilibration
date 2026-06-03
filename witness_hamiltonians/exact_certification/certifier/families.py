from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .graphs import Graph
from .pauli import PauliString, pauli_from_labeled_support, pauli_jsonable


class LocalPauliLimitExceeded(RuntimeError):
    def __init__(self, message: str, support_count: int, pauli_count: int):
        super().__init__(message)
        self.support_count = support_count
        self.pauli_count = pauli_count


@dataclass
class LocalPauliSet:
    family: str
    R: int
    k: Optional[int]
    mode: str
    root: int
    active_vertices: Tuple[int, ...]
    root_bit: int
    paulis: Tuple[PauliString, ...]
    support_count: int
    coordinate_hash: str


def normalize_family(family: str) -> str:
    aliases = {
        "two_body": "exact_two_body_no_fields",
        "exact_two_body": "exact_two_body_no_fields",
        "two_body_no_fields": "exact_two_body_no_fields",
        "xyz": "xyz_chain",
    }
    return aliases.get(family, family)


def generate_local_paulis(
    graph: Graph,
    root: int,
    family: str,
    R: int,
    k: Optional[int],
    mode: str = "theorem",
    max_paulis: Optional[int] = None,
) -> LocalPauliSet:
    family = normalize_family(family)
    if mode not in {"theorem", "centered"}:
        raise ValueError("mode must be theorem or centered")
    if family == "dense":
        supports = _dense_supports(graph, root, R, int(k or 0), mode)
        paulis = _all_labelings(graph, supports, max_paulis=max_paulis)
    elif family == "exact_two_body_no_fields":
        supports = _two_body_supports(graph, root, R, mode)
        paulis = _all_labelings(graph, supports, max_paulis=max_paulis)
    elif family == "xyz_chain":
        paulis = _xyz_paulis(graph, root, R, mode)
        if max_paulis is not None and len(paulis) > max_paulis:
            raise LocalPauliLimitExceeded(
                f"local coordinate count exceeded limit {max_paulis}",
                support_count=len({tuple(sorted(v for v, _ in term)) for term in paulis}),
                pauli_count=len(paulis),
            )
        supports = {tuple(_support_from_pauli_global_placeholder) for _support_from_pauli_global_placeholder in []}
    else:
        raise ValueError(f"unknown family {family}")

    if family != "xyz_chain":
        active_vertices = tuple(sorted({v for support in supports for v in support}))
        vertex_to_bit = {v: i for i, v in enumerate(active_vertices)}
        strings: List[PauliString] = []
        for labeled in paulis:
            strings.append(pauli_from_labeled_support(labeled, vertex_to_bit))
        pauli_tuple = tuple(sorted(set(strings), key=lambda p: p.sort_key()))
        support_count = len(supports)
    else:
        labeled_terms = paulis
        active_vertices = tuple(sorted({v for term in labeled_terms for v, _ in term}))
        vertex_to_bit = {v: i for i, v in enumerate(active_vertices)}
        strings = [pauli_from_labeled_support(term, vertex_to_bit) for term in labeled_terms]
        pauli_tuple = tuple(sorted(set(strings), key=lambda p: p.sort_key()))
        support_count = len({tuple(sorted(v for v, _ in term)) for term in labeled_terms})

    root_bit = 1 << active_vertices.index(root) if root in active_vertices else 0
    coordinate_hash = hash_pauli_list(pauli_tuple, active_vertices, graph.coords)
    return LocalPauliSet(
        family=family,
        R=R,
        k=k,
        mode=mode,
        root=root,
        active_vertices=active_vertices,
        root_bit=root_bit,
        paulis=pauli_tuple,
        support_count=support_count,
        coordinate_hash=coordinate_hash,
    )


def _dense_supports(graph: Graph, root: int, R: int, k: int, mode: str) -> Set[Tuple[int, ...]]:
    if k < 1:
        raise ValueError("dense family requires k >= 1")
    ball_root = graph.ball(root, R)
    centers = {root} if mode == "centered" else ball_root
    ball_cache: Dict[int, Set[int]] = {}

    def ball(v: int) -> Set[int]:
        if v not in ball_cache:
            ball_cache[v] = graph.ball(v, R)
        return ball_cache[v]

    supports: Set[Tuple[int, ...]] = set()
    for p in sorted(centers):
        candidates = sorted(ball(p))
        others = [q for q in candidates if q != p]
        for size in range(1, min(k, len(candidates)) + 1):
            for rest in itertools.combinations(others, size - 1):
                support = tuple(sorted((p,) + rest))
                if mode == "centered" and root not in support:
                    continue
                if _diameter_at_most_R(support, R, ball):
                    supports.add(support)
    return supports


def _two_body_supports(graph: Graph, root: int, R: int, mode: str) -> Set[Tuple[int, ...]]:
    ball_root = graph.ball(root, R)
    centers = {root} if mode == "centered" else ball_root
    supports: Set[Tuple[int, ...]] = set()
    for p in sorted(centers):
        for q, d in graph.distances(p, R).items():
            if q != p and 1 <= d <= R:
                support = tuple(sorted((p, q)))
                if mode == "centered" and root not in support:
                    continue
                supports.add(support)
    return supports


def _diameter_at_most_R(support: Sequence[int], R: int, ball_fn) -> bool:
    for i, a in enumerate(support):
        ba = ball_fn(a)
        for b in support[i + 1 :]:
            if b not in ba:
                return False
    return True


def _all_labelings(
    graph: Graph, supports: Iterable[Tuple[int, ...]], max_paulis: Optional[int] = None
) -> List[Tuple[Tuple[int, str], ...]]:
    out: List[Tuple[Tuple[int, str], ...]] = []
    sorted_supports = sorted(supports)
    for support in sorted_supports:
        for labels in itertools.product(("X", "Y", "Z"), repeat=len(support)):
            out.append(tuple(zip(support, labels)))
            if max_paulis is not None and len(out) > max_paulis:
                raise LocalPauliLimitExceeded(
                    f"local coordinate count exceeded limit {max_paulis}",
                    support_count=len(sorted_supports),
                    pauli_count=len(out),
                )
    return out


def _xyz_paulis(graph: Graph, root: int, R: int, mode: str) -> List[Tuple[Tuple[int, str], ...]]:
    if graph.lattice != "chain":
        raise ValueError("xyz_chain family is only implemented for the chain")
    ball_root = graph.ball(root, R)
    terms: List[Tuple[Tuple[int, str], ...]] = []
    single_sites = ball_root if mode == "theorem" else {root}
    for v in sorted(single_sites):
        for label in ("X", "Y", "Z"):
            terms.append(((v, label),))
    edge_supports = set()
    for v in sorted(ball_root):
        for nb in graph.adj[v]:
            support = tuple(sorted((v, nb)))
            if mode == "centered" and root not in support:
                continue
            edge_supports.add(support)
    for a, b in sorted(edge_supports):
        for label in ("X", "Y", "Z"):
            terms.append(((a, label), (b, label)))
    return terms


def hash_pauli_list(paulis: Sequence[PauliString], active_vertices: Sequence[int], coords: Dict[int, object]) -> str:
    payload = [pauli_jsonable(p, active_vertices, coords) for p in paulis]
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def centered_count(graph: Graph, root: int, family: str, R: int, k: Optional[int]) -> int:
    return len(generate_local_paulis(graph, root, family, R, k, mode="centered").paulis)
