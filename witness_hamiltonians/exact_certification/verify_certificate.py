#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from certifier.core import CaseParams, witness_vector, hash_witness
from certifier.active import active_witness_vector, build_active_matrix
from certifier.cpp_rank import cpp_modular_rank
from certifier.families import generate_local_paulis
from certifier.graphs import generate_graph
from certifier.matrix import build_local_matrix, check_integer_null


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify exact local certificate JSON")
    parser.add_argument("certificate")
    args = parser.parse_args(argv)
    with open(args.certificate, "r", encoding="utf-8") as f:
        cert = json.load(f)
    ok = verify_certificate(cert)
    print(f"{args.certificate}: {'verified' if ok else 'verification failed'}")
    return 0 if ok else 1


def verify_certificate(cert) -> bool:
    params_raw = cert["parameters"]
    params = CaseParams(
        family=params_raw["family"],
        lattice=params_raw["lattice"],
        boundary=params_raw["boundary"],
        R=int(params_raw["R"]),
        k=params_raw.get("k"),
        mode=params_raw.get("mode", "theorem"),
        prime=int(params_raw["prime"]),
        max_witnesses=int(params_raw.get("max_witnesses", 20)),
        max_pair_checks=int(params_raw.get("max_pair_checks", 5_000_000)),
        max_rank_seconds=None if params_raw.get("max_rank_seconds") in (None, "") else float(params_raw["max_rank_seconds"]),
        max_dimension=None if params_raw.get("max_dimension") in (None, "") else int(params_raw["max_dimension"]),
        seed=int(params_raw.get("seed", 1729)),
    )
    graph_L = int(cert["type_enumeration"]["graph_L"])
    graph = generate_graph(params.lattice, params.boundary, graph_L)
    all_ok = True
    for local_cert in cert.get("local_types", []):
        if local_cert.get("status") != "exactly_certified":
            continue
        root = int(local_cert["root_vertex"])
        local = generate_local_paulis(graph, root, params.family, params.R, params.k, params.mode)
        setattr(local, "coords", graph.coords)
        if len(local.paulis) != int(local_cert["d_U"]):
            print(f"type {local_cert['local_type_id']}: d_U mismatch")
            all_ok = False
            continue
        if local.coordinate_hash != local_cert.get("coordinate_hash"):
            print(f"type {local_cert['local_type_id']}: coordinate hash mismatch")
            all_ok = False
            continue
        witness_weight_cap = local_cert.get(
            "witness_weight_cap",
            cert.get("parameters", {}).get("witness_weight_cap"),
        )
        if witness_weight_cap in (None, ""):
            h = witness_vector(len(local.paulis), int(local_cert["witness_seed"]))
            build = build_local_matrix(local, h, params.max_pair_checks)
        else:
            h = active_witness_vector(
                local.paulis,
                int(local_cert["witness_seed"]),
                int(witness_weight_cap),
            )
            build = build_active_matrix(local, h, params.max_pair_checks)
        if hash_witness(h) != local_cert.get("witness_hash"):
            print(f"type {local_cert['local_type_id']}: witness hash mismatch")
            all_ok = False
            continue
        if build.status != "ok":
            print(f"type {local_cert['local_type_id']}: rebuild failed: {build.message}")
            all_ok = False
            continue
        checks = [
            (len(build.row_keys) == int(local_cert["W_count"]), "W_count mismatch"),
            (build.nnz == int(local_cert["nnz"]), "nnz mismatch"),
            (build.row_hash == local_cert.get("row_hash"), "row hash mismatch"),
            (check_integer_null(build.rows, h), "integer null check failed"),
        ]
        for condition, message in checks:
            if not condition:
                print(f"type {local_cert['local_type_id']}: {message}")
                all_ok = False
        rank = cpp_modular_rank(
            build.rows,
            len(local.paulis),
            params.prime,
            target=int(local_cert["target_rank"]),
        )
        if rank.rank != int(local_cert["target_rank"]):
            print(f"type {local_cert['local_type_id']}: rank mismatch {rank.rank}")
            all_ok = False
    if cert.get("overall_status") == "exactly_certified":
        exact_count = sum(1 for r in cert.get("local_types", []) if r.get("status") == "exactly_certified")
        if exact_count != cert.get("type_enumeration", {}).get("type_count"):
            print("overall exact certificate does not certify every rooted local type")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    raise SystemExit(main())
