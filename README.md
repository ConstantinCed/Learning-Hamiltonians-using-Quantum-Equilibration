# Learning Hamiltonians using Quantum Equilibration

This repository contains Python code and data for Hamiltonian witness computations and TFIM scaling experiments.

## Repository Structure

- `witness_hamiltonians/witness_structured.py`: main script used for periodic certification sweeps
- `witness_hamiltonians/structured_fast_50.json`: validation results
- `witness_hamiltonians/structured_push_50.json`: larger periodic push results
- `TFIM/AH_commutator_scaling/TFIM_commutator_AH_scaling.py`: TFIM commutator scaling script
- `TFIM/AU_scaling_vs_t/tfim__A_U_comm_exponent_vs_t.py`: TFIM AU commutator exponent vs time sweep
- `TFIM/AU_scaling_vs_t/alpha_vs_t.csv`: CSV output from AU scaling sweep

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
