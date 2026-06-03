from certifier.graphs import generate_graph
from certifier.types import deduplicate_rooted_types, enumerate_rooted_types, rooted_ball, rooted_balls_isomorphic


def test_periodic_chain_square_cubic_have_one_bulk_type_at_R1():
    assert len(enumerate_rooted_types("chain", "periodic", R=1).types) == 1
    assert len(enumerate_rooted_types("square", "periodic", R=1).types) == 1
    assert len(enumerate_rooted_types("cubic", "periodic", R=1).types) == 1


def test_open_chain_R2_produces_boundary_and_bulk_types():
    enum = enumerate_rooted_types("chain", "open", R=2)
    assert enum.stabilized
    assert len(enum.types) == 5


def test_exact_rooted_isomorphism_deduplicates_identical_types():
    graph = generate_graph("chain", "periodic", 9)
    a = rooted_ball(graph, 0, 4)
    b = rooted_ball(graph, 3, 4)
    assert rooted_balls_isomorphic(a, b)
    types, _ = deduplicate_rooted_types(graph, 4)
    assert len(types) == 1
