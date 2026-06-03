from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .util import popcount

LABEL_TO_BITS: Dict[str, Tuple[int, int]] = {
    "I": (0, 0),
    "X": (1, 0),
    "Z": (0, 1),
    "Y": (1, 1),
}

BITS_TO_LABEL: Dict[Tuple[int, int], str] = {v: k for k, v in LABEL_TO_BITS.items()}

# Phase exponent e means i**e.  This table uses Hermitian representatives
# I, X, Y, Z and is the source of truth for all commutator signs.
ONE_SITE_PRODUCT: Dict[Tuple[str, str], Tuple[int, str]] = {
    ("I", "I"): (0, "I"),
    ("I", "X"): (0, "X"),
    ("I", "Y"): (0, "Y"),
    ("I", "Z"): (0, "Z"),
    ("X", "I"): (0, "X"),
    ("Y", "I"): (0, "Y"),
    ("Z", "I"): (0, "Z"),
    ("X", "X"): (0, "I"),
    ("Y", "Y"): (0, "I"),
    ("Z", "Z"): (0, "I"),
    ("X", "Y"): (1, "Z"),
    ("Y", "X"): (3, "Z"),
    ("Y", "Z"): (1, "X"),
    ("Z", "Y"): (3, "X"),
    ("Z", "X"): (1, "Y"),
    ("X", "Z"): (3, "Y"),
}


@dataclass(frozen=True, order=True)
class PauliString:
    """A phase-free Pauli string represented by x/z bit masks."""

    x: int
    z: int

    @property
    def support_mask(self) -> int:
        return self.x | self.z

    @property
    def support_size(self) -> int:
        return popcount(self.support_mask)

    def label_at_bit(self, bit_index: int) -> str:
        xb = (self.x >> bit_index) & 1
        zb = (self.z >> bit_index) & 1
        return BITS_TO_LABEL[(xb, zb)]

    def xor(self, other: "PauliString") -> "PauliString":
        return PauliString(self.x ^ other.x, self.z ^ other.z)

    def sort_key(self) -> Tuple[int, int, int, int]:
        return (self.support_size, self.support_mask, self.x, self.z)


def one_site_product(a: str, b: str) -> Tuple[int, str]:
    return ONE_SITE_PRODUCT[(a, b)]


def anticommutes(u: PauliString, v: PauliString) -> bool:
    parity_mask = (u.x & v.z) ^ (u.z & v.x)
    return bool(popcount(parity_mask) & 1)


def product_phase_exp(u: PauliString, v: PauliString) -> int:
    """Return e in P_u P_v = i**e P_{u xor v}."""

    phase = 0
    mask = u.support_mask | v.support_mask
    while mask:
        bit = mask & -mask
        i = bit.bit_length() - 1
        phase_site, _ = one_site_product(u.label_at_bit(i), v.label_at_bit(i))
        phase = (phase + phase_site) & 3
        mask ^= bit
    return phase


def commutator_sign(u: PauliString, v: PauliString) -> int:
    """Return sigma_uv from [P_u,P_v] = 2i sigma_uv P_{u xor v}.

    The caller must pass an anticommuting pair.  For such a pair
    P_u P_v = i sigma_uv P_{u xor v}.
    """

    if not anticommutes(u, v):
        raise ValueError("commutator_sign requires an anticommuting pair")
    phase = product_phase_exp(u, v)
    if phase == 1:
        return 1
    if phase == 3:
        return -1
    raise AssertionError(f"anticommuting product had unexpected phase i**{phase}")


def pauli_from_labeled_support(
    labeled_support: Sequence[Tuple[int, str]], vertex_to_bit: Dict[int, int]
) -> PauliString:
    x = 0
    z = 0
    for vertex, label in labeled_support:
        bit = 1 << vertex_to_bit[vertex]
        xb, zb = LABEL_TO_BITS[label]
        if xb:
            x |= bit
        if zb:
            z |= bit
    return PauliString(x, z)


def pauli_to_labeled_support(
    p: PauliString, active_vertices: Sequence[int], coords: Dict[int, object]
) -> List[Tuple[object, str]]:
    out: List[Tuple[object, str]] = []
    mask = p.support_mask
    while mask:
        bit = mask & -mask
        i = bit.bit_length() - 1
        out.append((coords[active_vertices[i]], p.label_at_bit(i)))
        mask ^= bit
    return out


def pauli_jsonable(
    p: PauliString, active_vertices: Sequence[int], coords: Dict[int, object]
) -> List[List[object]]:
    return [[_jsonable_coord(c), label] for c, label in pauli_to_labeled_support(p, active_vertices, coords)]


def _jsonable_coord(coord: object) -> object:
    if isinstance(coord, tuple):
        return [_jsonable_coord(x) for x in coord]
    return coord


def support_vertices(p: PauliString, active_vertices: Sequence[int]) -> Tuple[int, ...]:
    vertices: List[int] = []
    mask = p.support_mask
    while mask:
        bit = mask & -mask
        i = bit.bit_length() - 1
        vertices.append(active_vertices[i])
        mask ^= bit
    return tuple(vertices)


def labels_for_support(p: PauliString) -> Tuple[Tuple[int, str], ...]:
    labels: List[Tuple[int, str]] = []
    mask = p.support_mask
    while mask:
        bit = mask & -mask
        i = bit.bit_length() - 1
        labels.append((i, p.label_at_bit(i)))
        mask ^= bit
    return tuple(labels)
