# Witness Hamiltonians: local non-degeneracy certifications

This folder contains the numerical witnesses that certify, for several
Hamiltonian families on standard lattices, that the local commutator
matrix `C(h)` attains its maximal possible rank `|U_c| - 1` for a
single integer coefficient vector `h`. Because the rank condition is
Zariski-open, one such witness certifies the generic statement.

Whenever the periodic size `L` satisfies `L >= 2 * R_patch + 2` the
local patch is isomorphic to the corresponding ball in the infinite
lattice, so the certification at one such `L` transfers verbatim to
every larger `L` and to the thermodynamic limit.

## Layout

```
witness_hamiltonians/
├── README.md                       this file
├── run_witness.py                  single driver: defines every job
├── consolidate.py                  print combined coverage table
├── witness_structured.py           core library (graphs, families, dense rank)
├── additional_runs.py              sparse Gram-matrix backend
├── dense/                          generic local family U_c^{(k,R)}
│   ├── cycle.json                  1D periodic chain
│   ├── grid_periodic.json          2D square torus
│   ├── triangular_torus.json       2D triangular torus
│   ├── honeycomb_torus.json        2D honeycomb torus
│   └── cubic_periodic.json         3D cubic torus
├── xyz/cycle.json                  XYZ chain + on-site X,Y,Z fields
├── full_nn_2body_all_fields/cycle.json   full NN 2-body + on-site fields
└── kitaev_honey_2d/honeycomb_torus.json  Kitaev honeycomb + on-site fields
```

Each JSON file is a list of result entries (one per `(L, k, R, R_patch)`
cell that was certified); the `found_witness` field flags successes.

## Reproducing the data

The pre-computed results are checked in; you do not need to re-run.
To redo any subset:

```
python3 run_witness.py --dry-run                 # list jobs
python3 run_witness.py                           # run only the missing cells
python3 run_witness.py --families dense          # restrict to a family
python3 run_witness.py --lattices cycle cubic_periodic
python3 run_witness.py --force                   # re-run everything
python3 run_witness.py --memory-cap-gb 16        # raise the dense backend cap
```

The driver auto-selects the dense or sparse backend based on `|U_c|`;
results are merged into the appropriate `<family>/<lattice>.json`,
deduplicating by `(graph_args, k, R_geom, R_patch, root_label)` and
preferring entries that found a witness.

## Coverage summary

Print the (k, R) coverage of each family at any time:

```
python3 consolidate.py
```

Current dense-family coverage:

| lattice              | k = 2 R range | k = 3 R range | k = 4 R range |
|----------------------|---------------|---------------|---------------|
| 1D periodic chain    | 1..5          | 1..5          | 2..4          |
| 2D square (periodic) | 1..3          | 1..2          | -             |
| 2D triangular torus  | 1..3          | 1..2          | -             |
| 2D honeycomb torus   | 1..3          | 1..2          | -             |
| 3D cubic (periodic)  | 1..2          | 1..2          | -             |

Structured nearest-neighbour families (all with `R_patch = 1`):

| family                       | lattice           | sizes L      |
|------------------------------|-------------------|--------------|
| `xyz`                        | cycle             | 9, 11, ..., 25 |
| `full_nn_2body_all_fields`   | cycle             | 9, 11, ..., 25 |
| `kitaev_honey_2d`            | honeycomb torus   | 3, 4, 5, 6   |
