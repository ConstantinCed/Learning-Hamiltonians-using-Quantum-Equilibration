from __future__ import annotations

import csv
import json
import os
from typing import Dict, Iterable, List, Sequence


def write_reports(certificate_paths: Sequence[str], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    certs = []
    for path in certificate_paths:
        with open(path, "r", encoding="utf-8") as f:
            cert = json.load(f)
        cert["_path"] = path
        certs.append(cert)

    summary_path = os.path.join(out_dir, "summary.csv")
    report_path = os.path.join(out_dir, "report.md")
    exact_tex_path = os.path.join(out_dir, "exact_certified_cases.tex")
    other_tex_path = os.path.join(out_dir, "probable_inconclusive_cases.tex")
    rows = [summary_row(c) for c in certs]
    write_summary_csv(rows, summary_path)
    write_markdown_report(certs, rows, report_path)
    write_latex_table([r for r in rows if r["overall_status"] == "exactly_certified"], exact_tex_path, "Exactly certified cases")
    write_latex_table([r for r in rows if r["overall_status"] != "exactly_certified"], other_tex_path, "Probable or inconclusive cases")
    return {
        "summary_csv": summary_path,
        "markdown_report": report_path,
        "exact_latex": exact_tex_path,
        "other_latex": other_tex_path,
    }


def summary_row(cert: Dict[str, object]) -> Dict[str, object]:
    p = cert["parameters"]
    locals_ = cert.get("local_types", [])
    exact_types = sum(1 for r in locals_ if r.get("status") == "exactly_certified")
    max_d = max((r.get("d_U") or 0 for r in locals_), default=0)
    max_w = max((r.get("W_count") or 0 for r in locals_), default=0)
    max_nnz = max((r.get("nnz") or 0 for r in locals_), default=0)
    return {
        "family": p.get("family"),
        "lattice": p.get("lattice"),
        "boundary": p.get("boundary"),
        "R": p.get("R"),
        "k": "" if p.get("k") is None else p.get("k"),
        "overall_status": cert.get("overall_status"),
        "type_count": cert.get("type_enumeration", {}).get("type_count"),
        "exact_type_count": exact_types,
        "max_d_U": max_d,
        "max_W_count": max_w,
        "max_nnz": max_nnz,
        "runtime_seconds": f"{cert.get('runtime_seconds', 0):.3f}",
        "peak_memory_mb": f"{cert.get('peak_memory_mb', 0):.1f}",
        "certificate": cert.get("_path"),
    }


def write_summary_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    fields = [
        "family",
        "lattice",
        "boundary",
        "R",
        "k",
        "overall_status",
        "type_count",
        "exact_type_count",
        "max_d_U",
        "max_W_count",
        "max_nnz",
        "runtime_seconds",
        "peak_memory_mb",
        "certificate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(certs: Sequence[Dict[str, object]], rows: Sequence[Dict[str, object]], path: str) -> None:
    exact = [r for r in rows if r["overall_status"] == "exactly_certified"]
    probable = [r for r in rows if r["overall_status"] == "probable"]
    inconclusive = [r for r in rows if r["overall_status"] == "inconclusive"]
    obstruction = [r for r in rows if r["overall_status"] == "failed_math_obstruction"]
    resource = [r for r in rows if r["overall_status"] == "failed_resource_limit"]
    densest = [
        r
        for r in rows
        if r["family"] == "dense" and r["lattice"] == "cubic" and str(r["R"]) == "2"
    ]
    lines = [
        "# Exact local nondegeneracy certification report",
        "",
        "## Method",
        "",
        "For each rooted local type $c$, we constructed the theorem-level local coordinate set",
        "$\\mathcal U_c = \\{u \\in \\mathcal V : \\operatorname{supp}(P_u) \\cap B(c,R) \\neq \\varnothing\\}$.",
        "For an integer witness $h_0$, we constructed $B_c(h_0)$ exactly, verified $B_c(h_0)h_0 = 0$ over $\\mathbb Z$, and verified $\\operatorname{rank}_{\\mathbb F_p}B_c(h_0) = |\\mathcal U_c| - 1$ over an odd prime field. This proves $S_c \\not\\equiv 0$ for that rooted local type.",
        "",
        "## Exactly certified cases",
        "",
        markdown_table(exact),
        "",
        "## Probable or inconclusive cases",
        "",
        markdown_table(probable + inconclusive),
        "",
        "## Failed due to mathematical rank obstruction",
        "",
        markdown_table(obstruction),
        "",
        "## Failed due to resource/time limits",
        "",
        markdown_table(resource),
        "",
        "## Densest cases attempted",
        "",
        markdown_table(densest),
        "",
        "## Certificate files",
        "",
    ]
    for cert in certs:
        lines.append(f"- `{cert['_path']}`")
    lines.extend(
        [
            "",
            "## Reproduction commands",
            "",
            "```bash",
            "python3 certify.py batch --plan plans/all_claimed_cases.yaml --out-dir certificates",
            "python3 verify_certificate.py certificates/<certificate>.json",
            "```",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def markdown_table(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return "_None._"
    fields = ["family", "lattice", "boundary", "R", "k", "overall_status", "type_count", "exact_type_count", "max_d_U", "max_W_count", "max_nnz"]
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(f, "")) for f in fields) + "|")
    return "\n".join(lines)


def write_latex_table(rows: Sequence[Dict[str, object]], path: str, caption: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}\n\\centering\n")
        f.write("\\begin{tabular}{llllrrrr}\n")
        f.write("Family & Lattice & Boundary & $R$ & $k$ & Types & Max $|\\mathcal U_c|$ & Max nnz \\\\\n")
        f.write("\\hline\n")
        for r in rows:
            f.write(
                f"{tex(r['family'])} & {tex(r['lattice'])} & {tex(r['boundary'])} & {r['R']} & {r['k']} & {r['type_count']} & {r['max_d_U']} & {r['max_nnz']} \\\\\n"
            )
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{tex(caption)}}}\n")
        f.write("\\end{table}\n")


def tex(value: object) -> str:
    return str(value).replace("_", "\\_")
