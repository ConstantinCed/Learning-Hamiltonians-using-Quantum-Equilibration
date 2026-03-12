
import argparse
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
    return (((x1 & z2).bit_count() + (z1 & x2).bit_count()) & 1)


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
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


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
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


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
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


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
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


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
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


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
    V = []
    for v in G.nodes():
        V += [{v: "X"}, {v: "Y"}, {v: "Z"}]
    for u, v in G.edges():
        for a in PAULIS:
            for b in PAULIS:
                V.append({u: a, v: b})
    return V


def kitaev_honeycomb_fields_fixed(Lx: int, Ly: int, periodic: bool = True) -> Tuple[nx.Graph, List[Dict[int, str]]]:
    if periodic:
        Graw = nx.Graph()
        for x in range(Lx):
            for y in range(Ly):
                Graw.add_node((x, y, 0))
                Graw.add_node((x, y, 1))

        colored_edges = []
        for x in range(Lx):
            for y in range(Ly):
                a = (x, y, 0)
                colored_edges += [
                    (a, (x, y, 1), "X"),
                    (a, ((x - 1) % Lx, y, 1), "Y"),
                    (a, (x, (y - 1) % Ly, 1), "Z"),
                ]
    else:
        Graw = nx.Graph()
        for x in range(Lx):
            for y in range(Ly):
                Graw.add_node((x, y, 0))
                Graw.add_node((x, y, 1))

        colored_edges = []
        for x in range(Lx):
            for y in range(Ly):
                a = (x, y, 0)
                colored_edges.append((a, (x, y, 1), "X"))
                if x - 1 >= 0:
                    colored_edges.append((a, (x - 1, y, 1), "Y"))
                if y - 1 >= 0:
                    colored_edges.append((a, (x, y - 1, 1), "Z"))

    G = nx.convert_node_labels_to_integers(Graw, ordering="sorted", label_attribute="old")
    old = nx.get_node_attributes(G, "old")
    inv = {old[i]: i for i in old}

    V = []
    for v in G.nodes():
        V += [{v: "X"}, {v: "Y"}, {v: "Z"}]

    seen = set()
    for a, b, t in colored_edges:
        ia, ib = inv[a], inv[b]
        key = tuple(sorted((ia, ib))) + (t,)
        if key in seen:
            continue
        seen.add(key)
        V.append({ia: t, ib: t})

    return G, V


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
    if kind == "triangular_torus":
        return triangular_torus_graph(*args)
    if kind == "honeycomb_torus":
        return honeycomb_torus_graph(*args)
    if kind == "honeycomb_open":
        return honeycomb_open_graph(*args)
    raise ValueError(kind)


def build_local_family_for_job(job: Job) -> Tuple[List[Tuple[str, ...]], List[int], int]:
    if job.family == "dense":
        G = make_graph(job.graph_kind, job.graph_args)
        return local_dense_family_direct(G, job.root, job.R_patch, job.k, job.R_geom)

    if job.family == "xyz":
        G = make_graph(job.graph_kind, job.graph_args)
        V = xyz_fields_family(G)
        return build_local_from_global_terms(G, V, job.root, job.R_patch)

    if job.family == "full_nn_2body_all_fields":
        G = make_graph(job.graph_kind, job.graph_args)
        V = full_nn_2body_all_fields_family(G)
        return build_local_from_global_terms(G, V, job.root, job.R_patch)

    if job.family == "kitaev_honey_2d":
        periodic = job.graph_kind == "honeycomb_torus"
        G, V = kitaev_honeycomb_fields_fixed(*job.graph_args, periodic=periodic)
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
        "patch_sites": len(patch_nodes),
        "elapsed_sec": float(elapsed),
        **out,
    }


def fast_validation_jobs() -> List[Job]:
    jobs = []

    for k, Rgeom, n in [
        (2, 1, 9),
        (2, 2, 11),
        (2, 3, 13),
        (3, 1, 9),
        (3, 2, 11),
        (2, 4, 17),
        (3, 3, 15),
    ]:
        jobs.append(Job(
            tag="d1_dense",
            family="dense",
            graph_kind="cycle",
            graph_args=(n,),
            root=0,
            R_patch=Rgeom,
            trials=300,
            seed=10 + 100 * Rgeom + 10 * k + n,
            k=k,
            R_geom=Rgeom,
        ))

    for n in [9, 11, 13]:
        jobs.append(Job("d1_xyz", "xyz", "cycle", (n,), 0, 1, 300, 1000 + n))
        jobs.append(Job("full_nn_2body_1d", "full_nn_2body_all_fields", "cycle", (n,), 0, 1, 300, 1100 + n))

    for Lx, Ly, k, Rgeom in [
        (4, 4, 2, 1),
        (4, 4, 3, 1),
        (5, 5, 2, 2),
        (5, 5, 3, 2),
    ]:
        jobs.append(Job(
            tag="d2_dense",
            family="dense",
            graph_kind="grid_periodic",
            graph_args=(Lx, Ly),
            root=0,
            R_patch=Rgeom,
            trials=300,
            seed=2000 + 100 * Rgeom + 10 * k + Lx + Ly,
            k=k,
            R_geom=Rgeom,
        ))

    for Lx, Ly, k, Rgeom in [
        (4, 4, 2, 1),
        (4, 4, 3, 1),
        (5, 5, 2, 2),
    ]:
        jobs.append(Job(
            tag="tri_dense",
            family="dense",
            graph_kind="triangular_torus",
            graph_args=(Lx, Ly),
            root=0,
            R_patch=Rgeom,
            trials=300,
            seed=3000 + 100 * Rgeom + 10 * k + Lx + Ly,
            k=k,
            R_geom=Rgeom,
        ))

    for Lx, Ly, k, Rgeom in [
        (3, 3, 2, 1),
        (3, 3, 3, 1),
        (4, 4, 2, 2),
    ]:
        jobs.append(Job(
            tag="honey_dense",
            family="dense",
            graph_kind="honeycomb_torus",
            graph_args=(Lx, Ly),
            root=0,
            R_patch=Rgeom,
            trials=300,
            seed=4000 + 100 * Rgeom + 10 * k + Lx + Ly,
            k=k,
            R_geom=Rgeom,
        ))

    for Lx, Ly, Lz, k, Rgeom in [
        (3, 3, 3, 2, 1),
        (3, 3, 3, 3, 1),
    ]:
        jobs.append(Job(
            tag="d3_dense",
            family="dense",
            graph_kind="cubic_periodic",
            graph_args=(Lx, Ly, Lz),
            root=0,
            R_patch=Rgeom,
            trials=300,
            seed=5000 + 100 * Rgeom + 10 * k + Lx + Ly + Lz,
            k=k,
            R_geom=Rgeom,
        ))

    for Lx, Ly in [(3, 3), (4, 4), (5, 5)]:
        jobs.append(Job("kitaev_honey_2d", "kitaev_honey_2d", "honeycomb_torus", (Lx, Ly), 0, 1, 300, 6000 + Lx + Ly))

    return jobs


def medium_push_jobs() -> List[Job]:
    jobs = []

    for k, Rgeom, n in [
        (2, 5, 21),
        (3, 4, 17),
        (3, 5, 21),
        (4, 2, 13),
        (4, 3, 17),
        (4, 4, 21),
    ]:
        jobs.append(Job(
            tag="push_d1_dense",
            family="dense",
            graph_kind="cycle",
            graph_args=(n,),
            root=0,
            R_patch=Rgeom,
            trials=500,
            seed=10000 + 100 * Rgeom + 10 * k + n,
            k=k,
            R_geom=Rgeom,
        ))

    for Lx, Ly, k, Rgeom in [
        (5, 5, 3, 2),
        (6, 6, 2, 2),
        (6, 6, 2, 3),
        (6, 6, 3, 2),
        (7, 7, 2, 3),
    ]:
        jobs.append(Job(
            tag="push_d2_dense",
            family="dense",
            graph_kind="grid_periodic",
            graph_args=(Lx, Ly),
            root=0,
            R_patch=Rgeom,
            trials=500,
            seed=11000 + 100 * Rgeom + 10 * k + Lx + Ly,
            k=k,
            R_geom=Rgeom,
        ))

    for Lx, Ly, k, Rgeom in [
        (5, 5, 3, 2),
        (6, 6, 2, 2),
        (6, 6, 2, 3),
        (6, 6, 3, 2),
    ]:
        jobs.append(Job(
            tag="push_tri_dense",
            family="dense",
            graph_kind="triangular_torus",
            graph_args=(Lx, Ly),
            root=0,
            R_patch=Rgeom,
            trials=500,
            seed=12000 + 100 * Rgeom + 10 * k + Lx + Ly,
            k=k,
            R_geom=Rgeom,
        ))

    for Lx, Ly, k, Rgeom in [
        (4, 4, 3, 2),
        (5, 5, 2, 3),
        (5, 5, 3, 2),
        (6, 6, 2, 3),
    ]:
        jobs.append(Job(
            tag="push_honey_dense",
            family="dense",
            graph_kind="honeycomb_torus",
            graph_args=(Lx, Ly),
            root=0,
            R_patch=Rgeom,
            trials=500,
            seed=13000 + 100 * Rgeom + 10 * k + Lx + Ly,
            k=k,
            R_geom=Rgeom,
        ))

    for Lx, Ly, Lz, k, Rgeom in [
        (3, 3, 3, 3, 2),
        (4, 4, 4, 2, 2),
        (4, 4, 4, 3, 1),
    ]:
        jobs.append(Job(
            tag="push_d3_dense",
            family="dense",
            graph_kind="cubic_periodic",
            graph_args=(Lx, Ly, Lz),
            root=0,
            R_patch=Rgeom,
            trials=500,
            seed=14000 + 100 * Rgeom + 10 * k + Lx + Ly + Lz,
            k=k,
            R_geom=Rgeom,
        ))

    for n in [15, 17, 21, 25]:
        jobs.append(Job("push_d1_xyz", "xyz", "cycle", (n,), 0, 1, 500, 15000 + n))
        jobs.append(Job("push_full_nn_2body_1d", "full_nn_2body_all_fields", "cycle", (n,), 0, 1, 500, 16000 + n))

    for Lx, Ly in [(5, 5), (6, 6)]:
        jobs.append(Job("push_kitaev_honey_2d", "kitaev_honey_2d", "honeycomb_torus", (Lx, Ly), 0, 1, 500, 17000 + Lx + Ly))

    return jobs


def obc_checks_jobs() -> List[Job]:
    jobs = []

    n = 31
    jobs.append(Job(
        tag="obc_d1_dense_endpoint",
        family="dense",
        graph_kind="path",
        graph_args=(n,),
        root=0,
        R_patch=3,
        trials=500,
        seed=21001,
        k=3,
        R_geom=3,
        root_label="endpoint",
    ))
    jobs.append(Job(
        tag="obc_d1_dense_bulk",
        family="dense",
        graph_kind="path",
        graph_args=(n,),
        root=n // 2,
        R_patch=3,
        trials=500,
        seed=21002,
        k=3,
        R_geom=3,
        root_label="bulk",
    ))
    jobs.append(Job(
        tag="obc_d1_xyz_endpoint",
        family="xyz",
        graph_kind="path",
        graph_args=(n,),
        root=0,
        R_patch=1,
        trials=500,
        seed=21003,
        root_label="endpoint",
    ))
    jobs.append(Job(
        tag="obc_d1_xyz_bulk",
        family="xyz",
        graph_kind="path",
        graph_args=(n,),
        root=n // 2,
        R_patch=1,
        trials=500,
        seed=21004,
        root_label="bulk",
    ))
    jobs.append(Job(
        tag="obc_d1_fullnn_endpoint",
        family="full_nn_2body_all_fields",
        graph_kind="path",
        graph_args=(n,),
        root=0,
        R_patch=1,
        trials=500,
        seed=21005,
        root_label="endpoint",
    ))
    jobs.append(Job(
        tag="obc_d1_fullnn_bulk",
        family="full_nn_2body_all_fields",
        graph_kind="path",
        graph_args=(n,),
        root=n // 2,
        R_patch=1,
        trials=500,
        seed=21006,
        root_label="bulk",
    ))

    Lx, Ly = 11, 11
    coords = sorted([(x, y) for x in range(Lx) for y in range(Ly)])
    inv_sq = {coords[i]: i for i in range(len(coords))}
    corner = inv_sq[(0, 0)]
    edge = inv_sq[(0, Ly // 2)]
    bulk = inv_sq[(Lx // 2, Ly // 2)]

    for root, label, seed in [(corner, "corner", 22001), (edge, "edge", 22002), (bulk, "bulk", 22003)]:
        jobs.append(Job(
            tag=f"obc_d2_dense_{label}",
            family="dense",
            graph_kind="grid_open",
            graph_args=(Lx, Ly),
            root=root,
            R_patch=3,
            trials=500,
            seed=seed,
            k=2,
            R_geom=3,
            root_label=label,
        ))

    jobs.append(Job(
        tag="obc_d2_dense_bulk_stronger",
        family="dense",
        graph_kind="grid_open",
        graph_args=(Lx, Ly),
        root=bulk,
        R_patch=2,
        trials=500,
        seed=22004,
        k=3,
        R_geom=2,
        root_label="bulk",
    ))

    Lx, Ly = 8, 8
    coords = sorted([(x, y, s) for x in range(Lx) for y in range(Ly) for s in (0, 1)])
    inv_h = {coords[i]: i for i in range(len(coords))}
    honey_edge = inv_h[(0, 0, 0)]
    honey_bulk = inv_h[(Lx // 2, Ly // 2, 0)]

    jobs.append(Job(
        tag="obc_honey_dense_edge",
        family="dense",
        graph_kind="honeycomb_open",
        graph_args=(Lx, Ly),
        root=honey_edge,
        R_patch=3,
        trials=500,
        seed=23001,
        k=2,
        R_geom=3,
        root_label="edge",
    ))
    jobs.append(Job(
        tag="obc_honey_dense_bulk",
        family="dense",
        graph_kind="honeycomb_open",
        graph_args=(Lx, Ly),
        root=honey_bulk,
        R_patch=3,
        trials=500,
        seed=23002,
        k=2,
        R_geom=3,
        root_label="bulk",
    ))
    jobs.append(Job(
        tag="obc_honey_dense_bulk_stronger",
        family="dense",
        graph_kind="honeycomb_open",
        graph_args=(Lx, Ly),
        root=honey_bulk,
        R_patch=2,
        trials=500,
        seed=23003,
        k=3,
        R_geom=2,
        root_label="bulk",
    ))

    jobs.append(Job(
        tag="obc_kitaev_honey_edge",
        family="kitaev_honey_2d",
        graph_kind="honeycomb_open",
        graph_args=(Lx, Ly),
        root=honey_edge,
        R_patch=2,
        trials=500,
        seed=23004,
        root_label="edge",
    ))
    jobs.append(Job(
        tag="obc_kitaev_honey_bulk",
        family="kitaev_honey_2d",
        graph_kind="honeycomb_open",
        graph_args=(Lx, Ly),
        root=honey_bulk,
        R_patch=2,
        trials=500,
        seed=23005,
        root_label="bulk",
    ))

    return jobs


def print_result(info: Dict[str, Any]) -> None:
    br = info["best_rank"]
    tr = info["target_rank"]
    brs = str(br) if br is not None else "None"
    kval = "-" if info["k"] is None else str(info["k"])
    rgval = "-" if info["R_geom"] is None else str(info["R_geom"])
    print(
        f"{info['tag']:24s} "
        f"status={info['status']:18s} "
        f"found={str(info['found_witness']):5s} "
        f"best_rank={brs:>6s} "
        f"target={tr:6d} "
        f"Uc={info['Uc_size']:6d} "
        f"Wc={info['Wc_size']:8d} "
        f"patch_sites={info['patch_sites']:4d} "
        f"root={info['root_label']:8s} "
        f"k={kval:>2s} "
        f"R={rgval:>2s} "
        f"Rpatch={info['R_patch']:>2d} "
        f"est_dense_gb={info['estimated_dense_gb']:.3f} "
        f"graph={info['graph_kind']} args={info['graph_args']} "
        f"family={info['family']} "
        f"elapsed={info['elapsed_sec']:.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="fast_validation", choices=["fast_validation", "medium_push", "obc_checks"])
    parser.add_argument("--memory-cap-gb", type=float, default=8.0)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if args.suite == "fast_validation":
        jobs = fast_validation_jobs()
    elif args.suite == "medium_push":
        jobs = medium_push_jobs()
    elif args.suite == "obc_checks":
        jobs = obc_checks_jobs()
    else:
        raise ValueError(args.suite)

    print(f"Running {len(jobs)} jobs in suite={args.suite} with memory_cap_gb={args.memory_cap_gb}")
    results = []

    for idx, job in enumerate(jobs, start=1):
        print(
            f"[{idx}/{len(jobs)}] starting {job.tag} family={job.family} "
            f"graph={job.graph_kind} args={job.graph_args} "
            f"root={job.root_label} k={job.k} R={job.R_geom} Rpatch={job.R_patch}"
        )
        info = run_job(job, memory_cap_gb=args.memory_cap_gb)
        print_result(info)
        results.append(info)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(json_safe(results), f, indent=2)

    print("\nSUMMARY")
    for info in results:
        print_result(info)


if __name__ == "__main__":
    main()
