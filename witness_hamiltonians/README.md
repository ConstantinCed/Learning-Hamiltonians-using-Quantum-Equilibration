# Witness Hamiltonians

Numerical witnesses certifying that the local commutator matrix `C(h)` attains
maximal rank `|U_c| - 1` for various Hamiltonian families. One witness certifies
the generic statement (rank condition is Zariski-open).

## Layout

```
witness_hamiltonians/
├── README.md
├── run_witness.py          driver: defines jobs and dispatches to backends
├── consolidate.py          print coverage summary
├── witness_structured.py    core library (graphs, families, dense rank)
├── additional_runs.py       sparse Gram-matrix backend
├── dense/                   generic local family U_c^{(k,R)}
│   ├── cycle.json
│   ├── grid_periodic.json
│   ├── triangular_torus.json
│   ├── honeycomb_torus.json
│   └── cubic_periodic.json
├── xyz/cycle.json
├── full_nn_2body_all_fields/cycle.json
└── kitaev_honey_2d/honeycomb_torus.json
```

## Reproducing

Results are checked in. To redo:

```
python3 run_witness.py --dry-run                 # list jobs
python3 run_witness.py                           # run missing cells
python3 run_witness.py --families dense          # restrict to family
python3 run_witness.py --lattices cycle cubic_periodic
python3 run_witness.py --force                   # re-run everything
python3 run_witness.py --memory-cap-gb 16        # raise memory cap
```

Backend dispatch: `|U_c| <= 2000` → dense, `> 2000` → sparse Gram.

## Coverage

```
python3 consolidate.py
```

Current dense-family coverage:

| lattice              | k = 2 R range | k = 3 R range | k = 4 R range |
|----------------------|---------------|---------------|---------------|
| 1D periodic chain    | 1..5          | 1..5          | 2..4          |
| 2D square (periodic) | 1..3          | 1..2          | 1, 2          |
| 2D triangular torus  | 1..3          | 1..2          | 1             |
| 2D honeycomb torus   | 1..3          | 1..2          | 1, 2          |
| 3D cubic (periodic)  | 1..2          | 1..2          | 1             |

Structured NN families (R_patch = 1):

| family                       | lattice           | sizes L      |
|------------------------------|-------------------|--------------|
| `xyz`                        | cycle             | 9, 11, ..., 25 |
| `full_nn_2body_all_fields`   | cycle             | 9, 11, ..., 25 |
| `kitaev_honey_2d`            | honeycomb torus   | 3, 4, 5, 6   |
