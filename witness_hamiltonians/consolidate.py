"""Pretty-print consolidated witness-Hamiltonian coverage."""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

canon = {}
for fam in sorted(os.listdir(HERE)):
    fam_dir = os.path.join(HERE, fam)
    if not os.path.isdir(fam_dir) or fam.startswith(".") or fam.startswith("_"):
        continue
    for fn in sorted(os.listdir(fam_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(fam_dir, fn)) as f:
            for r in json.load(f):
                key = (
                    r["family"],
                    r["graph_kind"],
                    tuple(r["graph_args"]),
                    r.get("k"),
                    r.get("R_geom"),
                    r["R_patch"],
                )
                fw = r.get("found_witness", False)
                src = f"{fam}/{fn}"
                if fw:
                    canon[key] = ("OK", src)
                elif key not in canon:
                    canon[key] = (r.get("status", "?"), src)

groups = defaultdict(list)
for (fam, gk, ga, k, Rg, Rp), (flag, fn) in canon.items():
    groups[(gk, fam)].append((ga, k, Rg, Rp, flag, fn))

print("CONSOLIDATED CERTIFICATION RESULTS")
print("=" * 72)
for (gk, fam), rows in sorted(groups.items()):
    rows.sort()
    print(f"\n[{gk}] family = {fam}   ({len(rows)} entries)")
    for ga, k, Rg, Rp, flag, fn in rows:
        kstr = f"k={k}" if k is not None else "k=- "
        Rstr = f"R={Rg}" if Rg is not None else "R=- "
        print(f"   L={ga}  {kstr}  {Rstr}  Rpatch={Rp}   {flag:5s}   [{fn}]")

print("\n" + "=" * 72)
print("CLEAN DENSE-FAMILY COVERAGE (k vs R, by lattice)")
print("=" * 72)
dense_cov = defaultdict(set)
for (fam, gk, ga, k, Rg, Rp), (flag, fn) in canon.items():
    if fam == "dense" and flag == "OK":
        dense_cov[gk].add((k, Rg))

label = {
    "cycle": "1D periodic chain",
    "grid_periodic": "2D square (periodic)",
    "triangular_torus": "2D triangular (torus)",
    "honeycomb_torus": "2D honeycomb (torus)",
    "cubic_periodic": "3D cubic (periodic)",
}
order = ["cycle", "grid_periodic", "triangular_torus",
         "honeycomb_torus", "cubic_periodic"]
for gk in order:
    print(f"\n{label[gk]} ({gk}):")
    pairs = sorted(dense_cov.get(gk, set()))
    by_k = defaultdict(list)
    for k, R in pairs:
        by_k[k].append(R)
    for k in sorted(by_k):
        Rs = sorted(by_k[k])
        print(f"   k = {k}   R \u2208 {{{', '.join(map(str, Rs))}}}")
