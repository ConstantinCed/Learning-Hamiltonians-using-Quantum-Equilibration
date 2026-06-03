#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List

from certifier.core import CaseParams, certify_case, write_certificate
from certifier.plan import load_plan
from certifier.reporting import write_reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Exact local nondegeneracy certifier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cert = sub.add_parser("certify", help="certify one case")
    add_case_args(p_cert)
    p_cert.add_argument("--out", required=True)

    p_batch = sub.add_parser("batch", help="run a plan")
    p_batch.add_argument("--plan", required=True)
    p_batch.add_argument("--out-dir", required=True)
    p_batch.add_argument("--stage", default=None, help="optional stage filter")
    p_batch.add_argument("--max-cases", type=int, default=None)
    p_batch.add_argument("--max-pair-checks", type=int, default=None, help="override plan pair-scan limit")
    p_batch.add_argument("--max-witnesses", type=int, default=None, help="override plan witness limit")
    p_batch.add_argument("--max-rank-seconds", type=float, default=None, help="override plan exact-rank time limit per witness")
    p_batch.add_argument("--max-dimension", type=int, default=None, help="override plan local coordinate dimension limit")

    p_report = sub.add_parser("report", help="regenerate reports from certificate JSON files")
    p_report.add_argument("--out-dir", required=True)
    p_report.add_argument("certificates", nargs="+")

    args = parser.parse_args(argv)
    if args.cmd == "certify":
        params = params_from_args(args)
        cert = certify_case(params)
        write_certificate(cert, args.out)
        print(f"{args.out}: {cert['overall_status']}")
        return 0 if cert["overall_status"] == "exactly_certified" else 2
    if args.cmd == "batch":
        cases = load_plan(args.plan)
        if args.stage:
            cases = [c for c in cases if str(c.get("stage")) == args.stage]
        if args.max_cases is not None:
            cases = cases[: args.max_cases]
        os.makedirs(args.out_dir, exist_ok=True)
        paths: List[str] = []
        for idx, case in enumerate(cases, 1):
            if args.max_pair_checks is not None:
                case["max_pair_checks"] = args.max_pair_checks
            if args.max_witnesses is not None:
                case["max_witnesses"] = args.max_witnesses
            if args.max_rank_seconds is not None:
                case["max_rank_seconds"] = args.max_rank_seconds
            if args.max_dimension is not None:
                case["max_dimension"] = args.max_dimension
            params = params_from_mapping(case)
            name = case_filename(params, idx)
            path = os.path.join(args.out_dir, name)
            print(f"[{idx}/{len(cases)}] {name}")
            cert = certify_case(params)
            write_certificate(cert, path)
            print(f"  {cert['overall_status']} in {cert['runtime_seconds']:.2f}s")
            paths.append(path)
        report_paths = write_reports(paths, args.out_dir)
        print("reports:")
        for k, v in report_paths.items():
            print(f"  {k}: {v}")
        return 0
    if args.cmd == "report":
        write_reports(args.certificates, args.out_dir)
        return 0
    return 1


def add_case_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--family", required=True)
    parser.add_argument("--lattice", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--mode", default="theorem")
    parser.add_argument("--backend", default="python_sparse")
    parser.add_argument("--prime", type=int, default=2147483647)
    parser.add_argument("--max-witnesses", type=int, default=20)
    parser.add_argument("--max-pair-checks", type=int, default=5_000_000)
    parser.add_argument("--max-rank-seconds", type=float, default=None)
    parser.add_argument("--max-dimension", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--open-checks", type=int, default=2)


def params_from_args(args) -> CaseParams:
    return CaseParams(
        family=args.family,
        lattice=args.lattice,
        boundary=args.boundary,
        R=args.R,
        k=args.k,
        mode=args.mode,
        prime=args.prime,
        max_witnesses=args.max_witnesses,
        max_pair_checks=args.max_pair_checks,
        max_rank_seconds=args.max_rank_seconds,
        max_dimension=args.max_dimension,
        seed=args.seed,
        L=args.L,
        open_checks=args.open_checks,
        backend=args.backend,
    )


def params_from_mapping(case: Dict[str, object]) -> CaseParams:
    return CaseParams(
        family=str(case["family"]),
        lattice=str(case["lattice"]),
        boundary=str(case["boundary"]),
        R=int(case["R"]),
        k=None if case.get("k") in (None, "") else int(case["k"]),
        mode=str(case.get("mode", "theorem")),
        prime=int(case.get("prime", 2147483647)),
        max_witnesses=int(case.get("max_witnesses", 20)),
        max_pair_checks=int(case.get("max_pair_checks", 5_000_000)),
        max_rank_seconds=None if case.get("max_rank_seconds") in (None, "") else float(case["max_rank_seconds"]),
        max_dimension=None if case.get("max_dimension") in (None, "") else int(case["max_dimension"]),
        seed=int(case.get("seed", 1729)),
        L=None if case.get("L") in (None, "") else int(case["L"]),
        open_checks=int(case.get("open_checks", 2)),
        backend=str(case.get("backend", "python_sparse")),
    )


def case_filename(params: CaseParams, idx: int) -> str:
    k = "none" if params.k is None else str(params.k)
    raw = f"{idx:03d}_{params.family}_{params.lattice}_{params.boundary}_R{params.R}_k{k}.json"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


if __name__ == "__main__":
    raise SystemExit(main())
