# Hamiltonian Witness Periodic Results

This repository contains Python code and JSON outputs for numerical certification of local nondegeneracy for periodic local Pauli Hamiltonian families.

## Files

- `witness_structured.py`: main script used for the periodic certification sweeps
- `structured_fast_50.json`: validation results
- `structured_push_50.json`: larger periodic push results

## Notes

The computations use real coefficient vectors so that the Hamiltonians are Hermitian.
The main periodic results cover dense geometric Pauli families on:
- 1D chains
- square lattices
- triangular lattices
- honeycomb lattices
- cubic lattices

and also structured families including:
- 1D XYZ
- full 1D nearest-neighbor two-body Pauli family
- Kitaev honeycomb with local fields
