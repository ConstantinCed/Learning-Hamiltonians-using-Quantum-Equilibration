# Witness Hamiltonians

This directory contains the exact theorem-level local nondegeneracy certificate
bundle. The final coverage is `81/81` planned cases exactly certified.

For each rooted local type `c`, the certifier constructs

```
U_c = {u in V : supp(P_u) intersects B(c,R)}
```

then builds the integer matrix `B_c(h_0)`, checks `B_c(h_0) h_0 = 0` exactly
over `Z`, and verifies `rank_{F_p} B_c(h_0) = |U_c|-1` over the odd prime
`p = 2147483647`.

Full rank over `F_p` exhibits a nonzero integer minor, hence proves the
corresponding rational/full-theorem rank condition. Since the rank condition is
Zariski-open, one exact witness proves the generic statement for that rooted
local type.

## Coverage

| family | lattices | boundaries | parameters | cases |
|---|---|---|---|---:|
| dense finite-range Pauli | chain, square, triangular, honeycomb, cubic | open, periodic | `R in {1,2}`, `k in {2,3,4}` | 60 |
| exact two-body/no-field | chain, square, triangular, honeycomb, cubic | open, periodic | `R in {1,2}` | 20 |
| `xyz_chain` | chain | periodic | `R=1`, `k=2` | 1 |

All certificates are in `exact_certification/certificates_final/`.

## Reproduction

Run from `witness_hamiltonians/exact_certification/`.

```
python3 run_tests.py
python3 verify_certificate.py certificates_final/001_dense_chain_open_R1_k2.json
python3 certify.py report --out-dir certificates_final certificates_final/*.json
```

To verify every certificate:

```
python3 - <<'PY'
import glob
import subprocess

for path in sorted(glob.glob("certificates_final/[0-9][0-9][0-9]_*.json")):
    subprocess.run(["python3", "verify_certificate.py", path], check=True)
PY
```

The original 71 exact certificates use a pure-Python exact sparse modular
backend. The 10 densest certificates use `certify_active.py`, which keeps the
full theorem-level `U_c` as columns but uses an integer witness supported only
on Pauli terms of weight at most 2. Matrix construction skips zero-coefficient
witness terms, and rank is computed exactly by `certifier/rank_modp.cpp` over
`F_2147483647`.
