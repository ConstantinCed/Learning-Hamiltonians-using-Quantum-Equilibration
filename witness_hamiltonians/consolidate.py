"""Pretty-print consolidated witness-Hamiltonian coverage."""

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BOUNDARIES = ("periodic", "open_boundary")


def iter_rows():
    for boundary in BOUNDARIES:
        boundary_dir = os.path.join(HERE, boundary)
        if not os.path.isdir(boundary_dir):
            continue
        for fam in sorted(os.listdir(boundary_dir)):
            fam_dir = os.path.join(boundary_dir, fam)
            if not os.path.isdir(fam_dir):
                continue
            for fn in sorted(os.listdir(fam_dir)):
                if not fn.endswith(".json"):
                    continue
                path = os.path.join(fam_dir, fn)
                with open(path) as f:
                    for row in json.load(f):
                        yield boundary, fam, fn, row


canon = {}
for boundary, fam_dir, fn, r in iter_rows():
    key = (
        boundary,
        r["family"],
        r["graph_kind"],
        tuple(r["graph_args"]),
        r.get("k"),
        r.get("R_geom"),
        r["R_patch"],
        r.get("local_mode", "rooted_ball"),
        r.get("root_label", "bulk"),
    )
    fw = r.get("found_witness", False)
    src = f"{boundary}/{fam_dir}/{fn}"
    if fw:
        canon[key] = ("OK", src)
    elif key not in canon:
        flag = "SKIP" if r.get("status") == "skipped_gram_cap" else "NO_WIT"
        canon[key] = (flag, src)

groups = defaultdict(list)
for (boundary, fam, gk, ga, k, Rg, Rp, mode, root_label), (flag, fn) in canon.items():
    groups[(boundary, gk, fam)].append((ga, k, Rg, Rp, mode, root_label, flag, fn))

print("CONSOLIDATED CERTIFICATION RESULTS")
print("=" * 88)
for (boundary, gk, fam), rows in sorted(groups.items()):
    rows.sort()
    print(f"\n[{boundary} / {gk}] family = {fam}   ({len(rows)} entries)")
    for ga, k, Rg, Rp, mode, root_label, flag, fn in rows:
        kstr = f"k={k}" if k is not None else "k=- "
        Rstr = f"R={Rg}" if Rg is not None else "R=- "
        root = f"root={root_label}"
        print(
            f"   L={ga}  {kstr}  {Rstr}  Rpatch={Rp}  "
            f"mode={mode:11s}  {root:28s} {flag:5s} [{fn}]"
        )

print("\n" + "=" * 88)
print("DENSE-FAMILY COVERAGE (k vs R, by boundary and lattice)")
print("=" * 88)
dense_groups = defaultdict(list)
for (boundary, fam, gk, _ga, k, Rg, _Rp, mode, _root_label), (flag, _fn) in canon.items():
    if fam == "dense":
        dense_groups[(boundary, gk, mode, k, Rg)].append(flag)

dense_cov = defaultdict(set)
dense_partial = defaultdict(set)
for (boundary, gk, mode, k, Rg), flags in dense_groups.items():
    if all(flag == "OK" for flag in flags):
        dense_cov[(boundary, gk, mode)].add((k, Rg))
    elif any(flag == "OK" for flag in flags):
        dense_partial[(boundary, gk, mode)].add((k, Rg))

label = {
    "cycle": "1D periodic chain",
    "path": "1D open chain",
    "grid_periodic": "2D square periodic",
    "grid_open": "2D square open",
    "triangular_torus": "2D triangular torus",
    "triangular_open": "2D triangular open",
    "honeycomb_torus": "2D honeycomb torus",
    "honeycomb_open": "2D honeycomb open",
    "cubic_periodic": "3D cubic periodic",
    "cubic_open": "3D cubic open",
}
order = [
    ("periodic", "cycle"),
    ("open_boundary", "path"),
    ("periodic", "grid_periodic"),
    ("open_boundary", "grid_open"),
    ("periodic", "triangular_torus"),
    ("open_boundary", "triangular_open"),
    ("periodic", "honeycomb_torus"),
    ("open_boundary", "honeycomb_open"),
    ("periodic", "cubic_periodic"),
    ("open_boundary", "cubic_open"),
]
for boundary, gk in order:
    modes = sorted({
        mode
        for b, g, mode in set(dense_cov) | set(dense_partial)
        if b == boundary and g == gk
    })
    for mode in modes:
        pairs = sorted(dense_cov.get((boundary, gk, mode), set()))
        partial_pairs = sorted(dense_partial.get((boundary, gk, mode), set()))
        if not pairs and not partial_pairs:
            continue
        print(f"\n{label[gk]} ({boundary}/{gk}, mode={mode}):")
        by_k = defaultdict(list)
        for k, R in pairs:
            by_k[k].append(R)
        for k in sorted(by_k):
            Rs = sorted(by_k[k])
            print(f"   k = {k}   R in {{{', '.join(map(str, Rs))}}}")
        if partial_pairs:
            by_k_partial = defaultdict(list)
            for k, R in partial_pairs:
                by_k_partial[k].append(R)
            for k in sorted(by_k_partial):
                Rs = sorted(by_k_partial[k])
                print(
                    f"   k = {k}   R partially certified/skipped in "
                    f"{{{', '.join(map(str, Rs))}}}"
                )

print("\n" + "=" * 88)
print("EXACT TWO-BODY / NO-FIELD COVERAGE (R, by boundary and lattice)")
print("=" * 88)
two_body_groups = defaultdict(list)
for (boundary, fam, gk, _ga, _k, Rg, _Rp, mode, _root_label), (flag, _fn) in canon.items():
    if fam == "full_2body_no_fields":
        two_body_groups[(boundary, gk, mode, Rg)].append(flag)

two_body_cov = defaultdict(set)
two_body_partial = defaultdict(set)
for (boundary, gk, mode, Rg), flags in two_body_groups.items():
    if all(flag == "OK" for flag in flags):
        two_body_cov[(boundary, gk, mode)].add(Rg)
    elif any(flag == "OK" for flag in flags):
        two_body_partial[(boundary, gk, mode)].add(Rg)

for boundary, gk in order:
    modes = sorted({
        mode
        for b, g, mode in set(two_body_cov) | set(two_body_partial)
        if b == boundary and g == gk
    })
    for mode in modes:
        Rs = sorted(two_body_cov.get((boundary, gk, mode), set()))
        partial_Rs = sorted(two_body_partial.get((boundary, gk, mode), set()))
        if not Rs and not partial_Rs:
            continue
        print(f"\n{label[gk]} ({boundary}/{gk}, mode={mode}):")
        if Rs:
            print(f"   R in {{{', '.join(map(str, Rs))}}}")
        if partial_Rs:
            print(
                f"   R partially certified/skipped in "
                f"{{{', '.join(map(str, partial_Rs))}}}"
            )
