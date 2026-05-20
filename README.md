# Learning Hamiltonians using Quantum Equilibration

Hamiltonian witness computations and TFIM scaling experiments.

## Structure

- `witness_hamiltonians/` — witness-Hamiltonian non-degeneracy certifications
- `TFIM/` — Transverse Field Ising Model scaling
- `Hamiltonian reconstruction/` — reconstruction algorithms
- `Weak equilibration/` — equilibration studies

## Notes

Witness-Hamiltonian results certify non-degeneracy of local commutator matrices
for dense Pauli families on periodic and open-boundary lattices (chain, square,
triangular, honeycomb, cubic) and for the structured 1D XYZ chain. The full
nearest-neighbour all-fields chain is included in the dense `k=2, R=1` chain
case, rather than treated as a separate structured family. See
`witness_hamiltonians/README.md` for details.
