import random

from certifier.pauli import (
    LABEL_TO_BITS,
    PauliString,
    anticommutes,
    commutator_sign,
    one_site_product,
    product_phase_exp,
)


def test_one_qubit_multiplication_table():
    assert one_site_product("X", "Y") == (1, "Z")
    assert one_site_product("Y", "X") == (3, "Z")
    assert one_site_product("Y", "Z") == (1, "X")
    assert one_site_product("Z", "Y") == (3, "X")
    assert one_site_product("Z", "X") == (1, "Y")
    assert one_site_product("X", "Z") == (3, "Y")


def test_anticommutation_parity_against_bruteforce_matrices():
    rng = random.Random(10)
    for n in range(1, 5):
        for _ in range(30):
            u = random_pauli(n, rng)
            v = random_pauli(n, rng)
            AB = matmul(pauli_matrix(u, n), pauli_matrix(v, n))
            BA = matmul(pauli_matrix(v, n), pauli_matrix(u, n))
            brute_anticommutes = AB == neg_matrix(BA)
            assert anticommutes(u, v) == brute_anticommutes


def test_multi_qubit_product_signs_against_bruteforce_matrices():
    rng = random.Random(11)
    for n in range(1, 5):
        for _ in range(40):
            u = random_pauli(n, rng)
            v = random_pauli(n, rng)
            phase = product_phase_exp(u, v)
            lhs = matmul(pauli_matrix(u, n), pauli_matrix(v, n))
            rhs = scale_matrix(pauli_matrix(u.xor(v), n), phase_to_gauss(phase))
            assert lhs == rhs


def test_sigma_antisymmetry_for_anticommuting_pairs():
    rng = random.Random(12)
    for n in range(1, 5):
        for _ in range(50):
            u = random_pauli(n, rng)
            v = random_pauli(n, rng)
            if anticommutes(u, v):
                assert commutator_sign(v, u) == -commutator_sign(u, v)


def random_pauli(n, rng):
    x = 0
    z = 0
    labels = ["I", "X", "Y", "Z"]
    for i in range(n):
        xb, zb = LABEL_TO_BITS[rng.choice(labels)]
        if xb:
            x |= 1 << i
        if zb:
            z |= 1 << i
    return PauliString(x, z)


def pauli_matrix(p, n):
    mats = []
    for i in range(n):
        mats.append(one_site_matrix(p.label_at_bit(i)))
    out = mats[0]
    for m in mats[1:]:
        out = kron(out, m)
    return out


def one_site_matrix(label):
    z0 = (0, 0)
    one = (1, 0)
    minus_one = (-1, 0)
    i = (0, 1)
    minus_i = (0, -1)
    if label == "I":
        return [[one, z0], [z0, one]]
    if label == "X":
        return [[z0, one], [one, z0]]
    if label == "Z":
        return [[one, z0], [z0, minus_one]]
    if label == "Y":
        return [[z0, minus_i], [i, z0]]
    raise ValueError(label)


def gadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def gneg(a):
    return (-a[0], -a[1])


def gmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def phase_to_gauss(exp):
    return [(1, 0), (0, 1), (-1, 0), (0, -1)][exp % 4]


def matmul(A, B):
    m = len(A)
    n = len(B[0])
    k = len(B)
    out = [[(0, 0) for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = (0, 0)
            for t in range(k):
                s = gadd(s, gmul(A[i][t], B[t][j]))
            out[i][j] = s
    return out


def kron(A, B):
    out = []
    for row_a in A:
        for row_b in B:
            row = []
            for a in row_a:
                for b in row_b:
                    row.append(gmul(a, b))
            out.append(row)
    return out


def neg_matrix(A):
    return [[gneg(x) for x in row] for row in A]


def scale_matrix(A, scalar):
    return [[gmul(scalar, x) for x in row] for row in A]
