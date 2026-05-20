"""Graphs, local Pauli families, and dense rank search for the witness check."""

import itertools
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np
import networkx as nx


PAULIS = ("X", "Y", "Z")
PAULI_TO_XZ = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}
XZ_TO_PAULI = {
    (0, 0): "I",
    (1, 0): "X",
    (1, 1): "Y",
    (0, 1): "Z",
}

_single_mul = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"), ("Y", "I"): (1, "Y"), ("Z", "I"): (1, "Z"),
    ("X", "X"): (1, "I"), ("Y", "Y"): (1, "I"), ("Z", "Z"): (1, "I"),
    ("X", "Y"): (1j, "Z"), ("Y", "Z"): (1j, "X"), ("Z", "X"): (1j, "Y"),
    ("Y", "X"): (-1j, "Z"), ("Z", "Y"): (-1j, "X"), ("X", "Z"): (-1j, "Y"),
}


def json_safe(obj: Any) -> Any:
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    return obj


def pauli_mul(op1: Tuple[str, ...], op2: Tuple[str, ...]) -> Tuple[complex, Tuple[str, ...]]:
    phase = 1
    out = []
    for a, b in zip(op1, op2):
        ph, c = _single_mul[(a, b)]
        phase *= ph
        out.append(c)
    return phase, tuple(out)


def commute_parity(op1: Tuple[str, ...], op2: Tuple[str, ...]) -> int:
    cnt = 0
    for a, b in zip(op1, op2):
        if a != "I" and b != "I" and a != b:
            cnt += 1
    return cnt % 2


def pauli_tuple_to_xz(op: Tuple[str, ...]) -> Tuple[int, int]:
    x = 0
    z = 0
    for i, p in enumerate(op):
        xi, zi = PAULI_TO_XZ[p]
        if xi:
            x |= (1 << i)
        if zi:
            z |= (1 << i)
    return x, z


def symplectic_commute_parity(x1: int, z1: int, x2: int, z2: int) -> int:
    return ((bin(x1 & z2).count("1") + bin(z1 & x2).count("1")) & 1)


def dedup_ops(U: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    seen = set()
    out = []
    for op in U:
        if op not in seen:
            seen.add(op)
            out.append(op)
    return out


def dedup_term_dicts(V: List[Dict[int, str]]) -> List[Dict[int, str]]:
    seen = set()
    out = []
    for op in V:
        key = tuple(sorted(op.items()))
        if key not in seen:
            seen.add(key)
            out.append(op)
    return out


def path_graph(n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    nx.set_node_attributes(G, {i: i for i in G.nodes()}, "coord")
    return G


def cycle_graph(n: int) -> nx.Graph:
    G = path_graph(n)
    if n >= 3:
        G.add_edge(n - 1, 0)
    return G


def grid_graph(Lx: int, Ly: int, periodic: bool = False) -> nx.Graph:
    G = nx.Graph()
    for x in range(Lx):
        for y in range(Ly):
            G.add_node((x, y))
    for x in range(Lx):
        for y in range(Ly):
            if periodic:
                G.add_edge((x, y), ((x + 1) % Lx, y))
                G.add_edge((x, y), (x, (y + 1) % Ly))
            else:
                if x + 1 < Lx:
                    G.add_edge((x, y), (x + 1, y))
                if y + 1 < Ly:
                    G.add_edge((x, y), (x, y + 1))
    return nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="coord")


def cubic_graph(Lx: int, Ly: int, Lz: int, periodic: bool = False) -> nx.Graph:
    G = nx.Graph()
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                G.add_node((x, y, z))
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                u = (x, y, z)
                if periodic:
                    G.add_edge(u, ((x + 1) % Lx, y, z))
                    G.add_edge(u, (x, (y + 1) % Ly, z))
                    G.add_edge(u, (x, y, (z + 1) % Lz))
                else:
                    if x + 1 < Lx:
                        G.add_edge(u, (x + 1, y, z))
                    if y + 1 < Ly:
                        G.add_edge(u, (x, y + 1, z))
                    if z + 1 < Lz:
                        G.add_edge(u, (x, y, z + 1))
    return nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="coord")


def triangular_torus_graph(Lx: int, Ly: int) -> nx.Graph:
    G = nx.Graph()
    for x in range(Lx):
        for y in range(Ly):
            G.add_node((x, y))
    for x in range(Lx):
        for y in range(Ly):
            u = (x, y)
            nbrs = [
                ((x + 1) % Lx, y),
                (x, (y + 1) % Ly),
                ((x + 1) % Lx, (y - 1) % Ly),
            ]
            for v in nbrs:
                if u != v:
                    G.add_edge(u, v)
    return nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="coord")


def triangular_open_graph(Lx: int, Ly: int) -> nx.Graph:
    G = nx.Graph()
    for x in range(Lx):
        for y in range(Ly):
            G.add_node((x, y))
    for x in range(Lx):
        for y in range(Ly):
            u = (x, y)
            nbrs = [
                (x + 1, y),
                (x, y + 1),
                (x + 1, y - 1),
            ]
            for vx, vy in nbrs:
                if 0 <= vx < Lx and 0 <= vy < Ly:
                    G.add_edge(u, (vx, vy))
    return nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="coord")


def honeycomb_torus_graph(Lx: int, Ly: int) -> nx.Graph:
    G = nx.Graph()
    for x in range(Lx):
        for y in range(Ly):
            G.add_node((x, y, 0))
            G.add_node((x, y, 1))
    for x in range(Lx):
        for y in range(Ly):
            a = (x, y, 0)
            b = (x, y, 1)
            G.add_edge(a, b)
            G.add_edge(a, ((x - 1) % Lx, y, 1))
            G.add_edge(a, (x, (y - 1) % Ly, 1))
    return nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="coord")


def honeycomb_open_graph(Lx: int, Ly: int) -> nx.Graph:
    G = nx.Graph()
    for x in range(Lx):
        for y in range(Ly):
            G.add_node((x, y, 0))
            G.add_node((x, y, 1))
    for x in range(Lx):
        for y in range(Ly):
            a = (x, y, 0)
            b = (x, y, 1)
            G.add_edge(a, b)
            if x - 1 >= 0:
                G.add_edge(a, (x - 1, y, 1))
            if y - 1 >= 0:
                G.add_edge(a, (x, y - 1, 1))
    return nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="coord")


def ball_nodes(G: nx.Graph, root: int, R: int) -> set:
    return set(nx.single_source_shortest_path_length(G, root, cutoff=R).keys())


def diameter_leq_in_graph(G: nx.Graph, subset: Iterable[int], R: int) -> bool:
    S = list(subset)
    for i in range(len(S)):
        dist = nx.single_source_shortest_path_length(G, S[i], cutoff=R)
        for j in range(i + 1, len(S)):
            if S[j] not in dist:
                return False
    return True


def local_patch_nodes(G: nx.Graph, root: int, R_patch: int) -> List[int]:
    return sorted(ball_nodes(G, root, R_patch))


def induced_patch_graph(G: nx.Graph, patch_nodes: List[int]) -> nx.Graph:
    return G.subgraph(patch_nodes).copy()


def relabel_patch_nodes(patch_nodes: List[int]) -> Dict[int, int]:
    return {v: i for i, v in enumerate(patch_nodes)}


def local_dense_family_direct(
    G: nx.Graph,
    root: int,
    R_patch: int,
    k: int,
    R_geom: int,
) -> Tuple[List[Tuple[str, ...]], List[int], int]:
    patch_nodes = local_patch_nodes(G, root, R_patch)
    patch_map = relabel_patch_nodes(patch_nodes)
    root_patch = patch_map[root]
    Gp = induced_patch_graph(G, patch_nodes)

    nodes_local = sorted(Gp.nodes())
    U = []
    n_patch = len(patch_nodes)

    for s in range(1, k + 1):
        for S in itertools.combinations(nodes_local, s):
            if diameter_leq_in_graph(Gp, S, R_geom):
                local_inds = [patch_map[v] for v in S]
                for letters in itertools.product(PAULIS, repeat=s):
                    op = ["I"] * n_patch
                    for idx, letter in zip(local_inds, letters):
                        op[idx] = letter
                    U.append(tuple(op))

    return dedup_ops(U), patch_nodes, root_patch


def _support_diameter_leq(
    dist_by_node: Dict[int, Dict[int, int]],
    support: Tuple[int, ...],
    R: int,
) -> bool:
    for i, u in enumerate(support):
        dist = dist_by_node[u]
        for v in support[i + 1:]:
            if dist.get(v, R + 1) > R:
                return False
    return True


def formal_dense_family_uc(
    G: nx.Graph,
    root: int,
    R_local: int,
    k: int,
    R_geom: int,
) -> Tuple[List[Tuple[str, ...]], List[int], int]:
    """Build the proof's formal U_c for the dense local Pauli family.

    U_c consists of all global weight-<=k, diameter-<=R_geom terms whose
    support intersects B(root, R_local).  The localized patch contains the full
    support of every such term, not just the root ball.
    """
    core = ball_nodes(G, root, R_local)
    supports = set()
    dist_by_center = {
        s: nx.single_source_shortest_path_length(G, s, cutoff=R_geom)
        for s in core
    }

    for center in sorted(core):
        near = sorted(dist_by_center[center])
        for size in range(1, k + 1):
            for rest in itertools.combinations([v for v in near if v != center], size - 1):
                support = tuple(sorted((center,) + rest))
                supports.add(support)

    patch_nodes = sorted({v for support in supports for v in support})
    patch_map = relabel_patch_nodes(patch_nodes)
    root_patch = patch_map[root]
    dist_by_node = {
        v: nx.single_source_shortest_path_length(G, v, cutoff=R_geom)
        for v in patch_nodes
    }

    U = []
    n_patch = len(patch_nodes)
    for support in sorted(supports, key=lambda S: (len(S), S)):
        if not _support_diameter_leq(dist_by_node, support, R_geom):
            continue
        local_inds = [patch_map[v] for v in support]
        for letters in itertools.product(PAULIS, repeat=len(support)):
            op = ["I"] * n_patch
            for idx, letter in zip(local_inds, letters):
                op[idx] = letter
            U.append(tuple(op))

    return dedup_ops(U), patch_nodes, root_patch


def build_local_from_global_terms(
    G: nx.Graph,
    V_global: List[Dict[int, str]],
    root: int,
    R_patch: int,
) -> Tuple[List[Tuple[str, ...]], List[int], int]:
    B = ball_nodes(G, root, R_patch)
    patch_nodes = sorted(B)
    patch_map = relabel_patch_nodes(patch_nodes)
    root_patch = patch_map[root]

    U = []
    for op in V_global:
        if set(op.keys()) & B:
            loc = ["I"] * len(patch_nodes)
            for v, p in op.items():
                if v in patch_map:
                    loc[patch_map[v]] = p
            U.append(tuple(loc))

    return dedup_ops(U), patch_nodes, root_patch


def build_formal_uc_from_global_terms(
    G: nx.Graph,
    V_global: List[Dict[int, str]],
    root: int,
    R_local: int,
) -> Tuple[List[Tuple[str, ...]], List[int], int]:
    core = ball_nodes(G, root, R_local)
    selected = [op for op in V_global if set(op.keys()) & core]
    patch_nodes = sorted({v for op in selected for v in op})
    patch_map = relabel_patch_nodes(patch_nodes)
    root_patch = patch_map[root]

    U = []
    for op in selected:
        loc = ["I"] * len(patch_nodes)
        for v, p in op.items():
            loc[patch_map[v]] = p
        U.append(tuple(loc))

    return dedup_ops(U), patch_nodes, root_patch


def xyz_fields_family(G: nx.Graph) -> List[Dict[int, str]]:
    V = []
    for v in G.nodes():
        V.append({v: "X"})
        V.append({v: "Y"})
        V.append({v: "Z"})
    for u, v in G.edges():
        V.append({u: "X", v: "X"})
        V.append({u: "Y", v: "Y"})
        V.append({u: "Z", v: "Z"})
    return V


def full_nn_2body_all_fields_family(G: nx.Graph) -> List[Dict[int, str]]:
    # On path/cycle graphs this coincides with the dense k=2, R=1 family.
    # It is retained for reproducing legacy full_nn_2body_all_fields results.
    V = []
    for v in G.nodes():
        V += [{v: "X"}, {v: "Y"}, {v: "Z"}]
    for u, v in G.edges():
        for a in PAULIS:
            for b in PAULIS:
                V.append({u: a, v: b})
    return V


def full_2body_no_fields_family(G: nx.Graph, R_geom: int) -> List[Dict[int, str]]:
    V = []
    nodes = sorted(G.nodes())
    for i, u in enumerate(nodes):
        dist = nx.single_source_shortest_path_length(G, u, cutoff=R_geom)
        for v in nodes[i + 1 :]:
            if v not in dist:
                continue
            for a in PAULIS:
                for b in PAULIS:
                    V.append({u: a, v: b})
    return V


def build_Wc(U_ops: List[Tuple[str, ...]], root_patch: int) -> List[Tuple[str, ...]]:
    W = []
    seen = set()
    xz = [pauli_tuple_to_xz(op) for op in U_ops]

    for i, u in enumerate(U_ops):
        x1, z1 = xz[i]
        for j, v in enumerate(U_ops):
            if i == j:
                continue
            x2, z2 = xz[j]
            if symplectic_commute_parity(x1, z1, x2, z2) == 1:
                _, w = pauli_mul(u, v)
                if w[root_patch] != "I" and w not in seen:
                    seen.add(w)
                    W.append(w)

    return W


def commutator_matrix_for_h(
    U_ops: List[Tuple[str, ...]],
    W_ops: List[Tuple[str, ...]],
    h: np.ndarray,
) -> np.ndarray:
    w_index = {w: r for r, w in enumerate(W_ops)}
    m = len(U_ops)
    C = np.zeros((len(W_ops), m), dtype=np.complex128)

    for iu, u in enumerate(U_ops):
        for iv, v in enumerate(U_ops):
            if commute_parity(u, v) == 0:
                continue
            ph_uv, w = pauli_mul(u, v)
            ph_vu, _ = pauli_mul(v, u)
            coeff = h[iv] * (ph_uv - ph_vu)
            if coeff == 0:
                continue
            r = w_index.get(w)
            if r is not None:
                C[r, iu] += coeff

    return C


def dense_matrix_memory_gb(n_rows: int, n_cols: int, dtype_bytes: int = 16) -> float:
    return (n_rows * n_cols * dtype_bytes) / (1024 ** 3)


def witness_search_dense_rank(
    U_ops: List[Tuple[str, ...]],
    root_patch: int,
    trials: int = 400,
    seed: int = 0,
    coeff_bound: int = 3,
    memory_cap_gb: float = 8.0,
) -> Dict[str, Any]:
    W_ops = build_Wc(U_ops, root_patch)
    m = len(U_ops)
    target = m - 1
    est_gb = dense_matrix_memory_gb(len(W_ops), m)

    if est_gb > memory_cap_gb:
        return {
            "status": "skipped_memory_cap",
            "found_witness": False,
            "best_rank": None,
            "target_rank": int(target),
            "Uc_size": int(m),
            "Wc_size": int(len(W_ops)),
            "estimated_dense_gb": float(est_gb),
            "best_h_real": None,
        }

    rng = np.random.default_rng(seed)
    best_rank = -1
    best_h = None

    for _ in range(trials):
        h = rng.integers(-coeff_bound, coeff_bound + 1, size=m).astype(np.float64)
        if np.all(np.abs(h) < 1e-12):
            h[0] = 1.0

        C = commutator_matrix_for_h(U_ops, W_ops, h)
        rank = int(np.linalg.matrix_rank(C, tol=1e-9))

        if rank > best_rank:
            best_rank = rank
            best_h = h.copy()

        if rank == target:
            return {
                "status": "ok",
                "found_witness": True,
                "best_rank": int(rank),
                "target_rank": int(target),
                "Uc_size": int(m),
                "Wc_size": int(len(W_ops)),
                "estimated_dense_gb": float(est_gb),
                "best_h_real": [float(x) for x in best_h],
            }

    return {
        "status": "ok",
        "found_witness": False,
        "best_rank": int(best_rank),
        "target_rank": int(target),
        "Uc_size": int(m),
        "Wc_size": int(len(W_ops)),
        "estimated_dense_gb": float(est_gb),
        "best_h_real": None if best_h is None else [float(x) for x in best_h],
    }


@dataclass
class Job:
    tag: str
    family: str
    graph_kind: str
    graph_args: Tuple
    root: int
    R_patch: int
    trials: int
    seed: int
    k: Optional[int] = None
    R_geom: Optional[int] = None
    coeff_bound: int = 3
    root_label: str = "bulk"
    boundary: str = "periodic"
    local_mode: str = "rooted_ball"
    root_coord: Optional[Any] = None
    covered_root_count: Optional[int] = None
    covered_root_sample: Optional[List[Any]] = None
    coverage_note: Optional[str] = None
    witness_weight_cap: Optional[int] = None


def make_graph(kind: str, args: Tuple) -> nx.Graph:
    if kind == "cycle":
        return cycle_graph(*args)
    if kind == "path":
        return path_graph(*args)
    if kind == "grid_periodic":
        return grid_graph(*args, periodic=True)
    if kind == "grid_open":
        return grid_graph(*args, periodic=False)
    if kind == "cubic_periodic":
        return cubic_graph(*args, periodic=True)
    if kind == "cubic_open":
        return cubic_graph(*args, periodic=False)
    if kind == "triangular_torus":
        return triangular_torus_graph(*args)
    if kind == "triangular_open":
        return triangular_open_graph(*args)
    if kind == "honeycomb_torus":
        return honeycomb_torus_graph(*args)
    if kind == "honeycomb_open":
        return honeycomb_open_graph(*args)
    raise ValueError(kind)


def build_local_family_for_job(job: Job) -> Tuple[List[Tuple[str, ...]], List[int], int]:
    if job.family == "dense":
        G = make_graph(job.graph_kind, job.graph_args)
        if job.local_mode == "formal_uc":
            return formal_dense_family_uc(G, job.root, job.R_patch, job.k, job.R_geom)
        return local_dense_family_direct(G, job.root, job.R_patch, job.k, job.R_geom)

    if job.family == "xyz":
        G = make_graph(job.graph_kind, job.graph_args)
        V = xyz_fields_family(G)
        if job.local_mode == "formal_uc":
            return build_formal_uc_from_global_terms(G, V, job.root, job.R_patch)
        return build_local_from_global_terms(G, V, job.root, job.R_patch)

    if job.family == "full_nn_2body_all_fields":
        G = make_graph(job.graph_kind, job.graph_args)
        V = full_nn_2body_all_fields_family(G)
        if job.local_mode == "formal_uc":
            return build_formal_uc_from_global_terms(G, V, job.root, job.R_patch)
        return build_local_from_global_terms(G, V, job.root, job.R_patch)

    if job.family == "full_2body_no_fields":
        G = make_graph(job.graph_kind, job.graph_args)
        V = full_2body_no_fields_family(G, job.R_geom)
        if job.local_mode == "formal_uc":
            return build_formal_uc_from_global_terms(G, V, job.root, job.R_patch)
        return build_local_from_global_terms(G, V, job.root, job.R_patch)

    raise ValueError(job.family)


def run_job(job: Job, memory_cap_gb: float) -> Dict[str, Any]:
    t0 = time.time()
    U_ops, patch_nodes, root_patch = build_local_family_for_job(job)
    out = witness_search_dense_rank(
        U_ops=U_ops,
        root_patch=root_patch,
        trials=job.trials,
        seed=job.seed,
        coeff_bound=job.coeff_bound,
        memory_cap_gb=memory_cap_gb,
    )
    elapsed = time.time() - t0
    return {
        "tag": job.tag,
        "family": job.family,
        "graph_kind": job.graph_kind,
        "graph_args": job.graph_args,
        "k": job.k,
        "R_geom": job.R_geom,
        "R_patch": job.R_patch,
        "root": job.root,
        "root_label": job.root_label,
        "boundary": job.boundary,
        "local_mode": job.local_mode,
        "root_coord": job.root_coord,
        "covered_root_count": job.covered_root_count,
        "covered_root_sample": job.covered_root_sample,
        "coverage_note": job.coverage_note,
        "witness_weight_cap": job.witness_weight_cap,
        "patch_sites": len(patch_nodes),
        "elapsed_sec": float(elapsed),
        **out,
    }
