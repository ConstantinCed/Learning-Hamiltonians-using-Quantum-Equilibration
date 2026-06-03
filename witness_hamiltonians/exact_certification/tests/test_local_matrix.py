import random

from certifier.families import centered_count, generate_local_paulis
from certifier.graphs import generate_graph
from certifier.matrix import build_local_matrix, check_integer_null
from certifier.rank import gamma_rank_from_B, modular_rank, rational_rank_from_rows


def test_local_matrix_has_exact_integer_null_vector():
    graph = generate_graph("chain", "open", 9)
    root = 4
    local = generate_local_paulis(graph, root, "dense", R=1, k=2, mode="theorem")
    setattr(local, "coords", graph.coords)
    rng = random.Random(21)
    for _ in range(5):
        h = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in local.paulis]
        build = build_local_matrix(local, h)
        assert build.status == "ok"
        assert check_integer_null(build.rows, h)


def test_tiny_rational_rank_matches_large_prime_rank():
    graph = generate_graph("chain", "open", 7)
    root = 3
    local = generate_local_paulis(graph, root, "dense", R=1, k=1, mode="theorem")
    setattr(local, "coords", graph.coords)
    h = [1, -1, 2, 3, -2, 1, -3, 2, 1]
    build = build_local_matrix(local, h)
    q_rank = rational_rank_from_rows(build.rows, len(local.paulis))
    fp_rank = modular_rank(build.rows, len(local.paulis), 2147483647).rank
    assert q_rank == fp_rank


def test_tiny_gram_rank_matches_B_rank_over_Q():
    graph = generate_graph("chain", "open", 7)
    root = 3
    local = generate_local_paulis(graph, root, "dense", R=1, k=1, mode="theorem")
    setattr(local, "coords", graph.coords)
    h = [1, -1, 2, 3, -2, 1, -3, 2, 1]
    build = build_local_matrix(local, h)
    b_rank = rational_rank_from_rows(build.rows, len(local.paulis))
    row_support_sizes = [w.support_size for w in build.row_keys]
    gamma_rank = gamma_rank_from_B(build.rows, row_support_sizes, len(local.paulis))
    assert gamma_rank == b_rank


def test_row_localization_and_root_condition():
    graph = generate_graph("square", "periodic", 7)
    root = 0
    local = generate_local_paulis(graph, root, "dense", R=1, k=2, mode="theorem")
    setattr(local, "coords", graph.coords)
    h = [1 if i % 2 else -1 for i in range(len(local.paulis))]
    build = build_local_matrix(local, h)
    assert all(w.support_mask & local.root_bit for w in build.row_keys)
    assert all(0 <= c < len(local.paulis) for row in build.rows for c in row)


def test_theorem_mode_is_larger_than_centered_mode():
    graph = generate_graph("chain", "open", 9)
    root = 4
    theorem = generate_local_paulis(graph, root, "dense", R=1, k=2, mode="theorem")
    centered = centered_count(graph, root, "dense", R=1, k=2)
    assert len(theorem.paulis) > centered
