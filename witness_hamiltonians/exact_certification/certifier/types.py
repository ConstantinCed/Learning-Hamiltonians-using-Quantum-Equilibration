from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .graphs import Graph, generate_graph, open_start_L, safe_periodic_L
from .util import popcount


@dataclass
class RootedBall:
    vertices: Tuple[int, ...]
    root: int
    adj_bits: Tuple[int, ...]
    distances: Tuple[int, ...]
    degrees: Tuple[int, ...]
    invariant: Tuple[object, ...]


@dataclass
class RootedType:
    type_id: int
    graph: Graph
    root: int
    ball: RootedBall
    multiplicity: int = 1


@dataclass
class TypeEnumeration:
    graph: Graph
    types: List[RootedType]
    radius: int
    stabilized: bool
    stabilization_Ls: List[int]
    previous_type_count: Optional[int] = None
    iso_checks: int = 0


def rooted_ball(graph: Graph, root: int, radius: int) -> RootedBall:
    dist_map = graph.distances(root, radius)
    others = sorted((v for v in dist_map if v != root), key=lambda v: (dist_map[v], repr(graph.coords[v]), v))
    vertices = (root,) + tuple(others)
    idx = {v: i for i, v in enumerate(vertices)}
    adj_bits: List[int] = []
    degrees: List[int] = []
    for v in vertices:
        bits = 0
        deg = 0
        for nb in graph.adj[v]:
            if nb in idx:
                bits |= 1 << idx[nb]
                deg += 1
        adj_bits.append(bits)
        degrees.append(deg)
    distances = tuple(dist_map[v] for v in vertices)
    invariant = _necessary_invariant(tuple(adj_bits), distances, tuple(degrees))
    return RootedBall(vertices=vertices, root=root, adj_bits=tuple(adj_bits), distances=distances, degrees=tuple(degrees), invariant=invariant)


def _necessary_invariant(adj_bits: Tuple[int, ...], distances: Tuple[int, ...], degrees: Tuple[int, ...]) -> Tuple[object, ...]:
    n = len(adj_bits)
    edge_count = sum(popcount(a) for a in adj_bits) // 2
    per_vertex = []
    for i in range(n):
        neigh_dist = []
        bits = adj_bits[i]
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            neigh_dist.append(distances[j])
            bits ^= bit
        per_vertex.append((i == 0, distances[i], degrees[i], tuple(sorted(neigh_dist))))
    return (n, edge_count, tuple(sorted(per_vertex)))


def deduplicate_rooted_types(graph: Graph, radius: int) -> Tuple[List[RootedType], int]:
    reps_by_inv: Dict[Tuple[object, ...], List[RootedType]] = {}
    types: List[RootedType] = []
    iso_checks = 0
    for root in sorted(graph.vertices, key=lambda v: (repr(graph.coords[v]), v)):
        ball = rooted_ball(graph, root, radius)
        bucket = reps_by_inv.setdefault(ball.invariant, [])
        matched = None
        for rep in bucket:
            iso_checks += 1
            if rooted_balls_isomorphic(ball, rep.ball):
                matched = rep
                break
        if matched is None:
            typ = RootedType(type_id=len(types), graph=graph, root=root, ball=ball, multiplicity=1)
            types.append(typ)
            bucket.append(typ)
        else:
            matched.multiplicity += 1
    return types, iso_checks


def enumerate_rooted_types(lattice: str, boundary: str, R: int, L: Optional[int] = None, open_checks: int = 2) -> TypeEnumeration:
    radius = 2 * R
    if boundary == "periodic":
        L0 = L or safe_periodic_L(R)
        graph = generate_graph(lattice, boundary, L0)
        if graph.lattice in {"chain", "square", "triangular", "cubic"}:
            ball = rooted_ball(graph, graph.vertices[0], radius)
            types = [RootedType(type_id=0, graph=graph, root=graph.vertices[0], ball=ball, multiplicity=len(graph.vertices))]
            iso_checks = 0
        elif graph.lattice == "honeycomb":
            roots = []
            seen_sublattices = set()
            for v in sorted(graph.vertices, key=lambda x: (repr(graph.coords[x]), x)):
                coord = graph.coords[v]
                sub = coord[2] if isinstance(coord, tuple) and len(coord) == 3 else None
                if sub not in seen_sublattices:
                    seen_sublattices.add(sub)
                    roots.append(v)
            sample_graph = Graph(
                lattice=graph.lattice,
                boundary=graph.boundary,
                L=graph.L,
                vertices=tuple(roots),
                coords=graph.coords,
                adj=graph.adj,
            )
            types = []
            iso_checks = 0
            for root in roots:
                ball = rooted_ball(graph, root, radius)
                matched = None
                for rep in types:
                    iso_checks += 1
                    if ball.invariant == rep.ball.invariant and rooted_balls_isomorphic(ball, rep.ball):
                        matched = rep
                        break
                if matched is None:
                    types.append(RootedType(type_id=len(types), graph=graph, root=root, ball=ball, multiplicity=len(graph.vertices) // len(roots)))
                else:
                    matched.multiplicity += len(graph.vertices) // len(roots)
        else:
            types, iso_checks = deduplicate_rooted_types(graph, radius)
        for i, typ in enumerate(types):
            typ.type_id = i
        return TypeEnumeration(graph=graph, types=types, radius=radius, stabilized=True, stabilization_Ls=[L0], iso_checks=iso_checks)

    L0 = L or open_start_L(R)
    last_graph = None
    last_types: Optional[List[RootedType]] = None
    last_iso = 0
    checked: List[int] = []
    prev_count: Optional[int] = None
    stabilized = False
    for step in range(max(1, open_checks)):
        cur_L = L0 + step
        graph = generate_graph(lattice, boundary, cur_L)
        candidate_roots = open_boundary_depth_roots(graph, radius)
        if candidate_roots is None:
            types, iso_checks = deduplicate_rooted_types(graph, radius)
        else:
            types, iso_checks = deduplicate_roots(graph, radius, candidate_roots)
        checked.append(cur_L)
        if last_types is not None:
            prev_count = len(last_types)
            if rooted_type_sets_isomorphic(last_types, types):
                stabilized = True
                last_graph = graph
                last_types = types
                last_iso += iso_checks
                break
        last_graph = graph
        last_types = types
        last_iso += iso_checks
    assert last_graph is not None and last_types is not None
    for i, typ in enumerate(last_types):
        typ.type_id = i
    return TypeEnumeration(
        graph=last_graph,
        types=last_types,
        radius=radius,
        stabilized=stabilized,
        stabilization_Ls=checked,
        previous_type_count=prev_count,
        iso_checks=last_iso,
    )


def deduplicate_roots(graph: Graph, radius: int, roots: Sequence[int]) -> Tuple[List[RootedType], int]:
    reps_by_inv: Dict[Tuple[object, ...], List[RootedType]] = {}
    types: List[RootedType] = []
    iso_checks = 0
    for root in sorted(set(roots), key=lambda v: (repr(graph.coords[v]), v)):
        ball = rooted_ball(graph, root, radius)
        bucket = reps_by_inv.setdefault(ball.invariant, [])
        matched = None
        for rep in bucket:
            iso_checks += 1
            if rooted_balls_isomorphic(ball, rep.ball):
                matched = rep
                break
        if matched is None:
            typ = RootedType(type_id=len(types), graph=graph, root=root, ball=ball, multiplicity=1)
            types.append(typ)
            bucket.append(typ)
        else:
            matched.multiplicity += 1
    return types, iso_checks


def open_boundary_depth_roots(graph: Graph, radius: int) -> Optional[List[int]]:
    if graph.boundary != "open" or graph.lattice not in {"chain", "square", "triangular", "cubic"}:
        return None
    coord_to_id = {coord: v for v, coord in graph.coords.items()}
    L = graph.L
    if graph.lattice == "chain":
        dims = 1
    elif graph.lattice in {"square", "triangular"}:
        dims = 2
    else:
        dims = 3

    per_dim_values = []
    for _ in range(dims):
        values = set()
        for depth in range(radius + 1):
            if depth < radius:
                values.add(depth)
                values.add(L - 1 - depth)
            else:
                values.add(radius)
                values.add(L - 1 - radius)
        per_dim_values.append(sorted(v for v in values if 0 <= v < L))

    roots = []
    for combo in itertools.product(*per_dim_values):
        coord = combo[0] if dims == 1 else tuple(combo)
        v = coord_to_id.get(coord)
        if v is not None:
            roots.append(v)
    return roots


def rooted_type_sets_isomorphic(a: Sequence[RootedType], b: Sequence[RootedType]) -> bool:
    if len(a) != len(b):
        return False
    used = [False] * len(b)
    for ta in a:
        found = False
        for j, tb in enumerate(b):
            if used[j]:
                continue
            if ta.ball.invariant == tb.ball.invariant and rooted_balls_isomorphic(ta.ball, tb.ball):
                used[j] = True
                found = True
                break
        if not found:
            return False
    return True


def rooted_balls_isomorphic(a: RootedBall, b: RootedBall) -> bool:
    if a.invariant != b.invariant:
        return False
    n = len(a.adj_bits)
    if n != len(b.adj_bits):
        return False
    colors_a = _refined_colors(a.adj_bits, a.distances, a.degrees)
    colors_b = _refined_colors(b.adj_bits, b.distances, b.degrees)
    if sorted(colors_a) != sorted(colors_b):
        return False
    if colors_a[0] != colors_b[0]:
        return False

    color_to_b_bits: Dict[int, int] = {}
    for j, col in enumerate(colors_b):
        color_to_b_bits[col] = color_to_b_bits.get(col, 0) | (1 << j)

    mapping = [-1] * n
    reverse = [-1] * n
    mapping[0] = 0
    reverse[0] = 0
    used_b = 1

    all_a_bits_by_color: Dict[int, int] = {}
    all_b_bits_by_color: Dict[int, int] = {}
    for i, col in enumerate(colors_a):
        all_a_bits_by_color[col] = all_a_bits_by_color.get(col, 0) | (1 << i)
    for i, col in enumerate(colors_b):
        all_b_bits_by_color[col] = all_b_bits_by_color.get(col, 0) | (1 << i)

    def choose_vertex() -> Optional[int]:
        best = None
        best_key = None
        mapped_a_bits = 0
        for i, m in enumerate(mapping):
            if m >= 0:
                mapped_a_bits |= 1 << i
        for i in range(n):
            if mapping[i] >= 0:
                continue
            candidates = color_to_b_bits[colors_a[i]] & ~used_b
            mapped_neighbors = popcount(a.adj_bits[i] & mapped_a_bits)
            key = (popcount(candidates), -mapped_neighbors, a.distances[i], i)
            if best_key is None or key < best_key:
                best_key = key
                best = i
        return best

    def consistent(i: int, j: int, used_b_now: int) -> bool:
        for ia, jb in enumerate(mapping):
            if jb < 0:
                continue
            edge_a = bool(a.adj_bits[i] & (1 << ia))
            edge_b = bool(b.adj_bits[j] & (1 << jb))
            if edge_a != edge_b:
                return False
        unmapped_a_mask = 0
        for ia, jb in enumerate(mapping):
            if jb < 0 and ia != i:
                unmapped_a_mask |= 1 << ia
        unused_b_mask = ((1 << n) - 1) & ~used_b_now & ~(1 << j)
        for col in all_a_bits_by_color:
            ca = popcount(a.adj_bits[i] & unmapped_a_mask & all_a_bits_by_color[col])
            cb = popcount(b.adj_bits[j] & unused_b_mask & all_b_bits_by_color[col])
            if ca != cb:
                return False
        return True

    def rec(depth: int, used_b_now: int) -> bool:
        if depth == n:
            return True
        i = choose_vertex()
        if i is None:
            return True
        candidates = color_to_b_bits[colors_a[i]] & ~used_b_now
        while candidates:
            bit = candidates & -candidates
            j = bit.bit_length() - 1
            candidates ^= bit
            if not consistent(i, j, used_b_now):
                continue
            mapping[i] = j
            reverse[j] = i
            if rec(depth + 1, used_b_now | bit):
                return True
            mapping[i] = -1
            reverse[j] = -1
        return False

    return rec(1, used_b)


def _refined_colors(adj_bits: Tuple[int, ...], distances: Tuple[int, ...], degrees: Tuple[int, ...]) -> Tuple[int, ...]:
    colors: Tuple[object, ...] = tuple((i == 0, distances[i], degrees[i]) for i in range(len(adj_bits)))
    while True:
        sigs = []
        for i, bits in enumerate(adj_bits):
            neigh = []
            cur = bits
            while cur:
                bit = cur & -cur
                j = bit.bit_length() - 1
                neigh.append(colors[j])
                cur ^= bit
            sigs.append((colors[i], tuple(sorted(neigh))))
        palette: Dict[object, int] = {sig: j for j, sig in enumerate(sorted(set(sigs)))}
        next_colors: List[int] = [palette[sig] for sig in sigs]
        next_tuple = tuple(next_colors)
        if next_tuple == colors:
            return next_tuple
        colors = next_tuple
