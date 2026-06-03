#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import sys
import traceback


def main() -> int:
    root = pathlib.Path(__file__).parent
    tests = sorted((root / "tests").glob("test_*.py"))
    failures = 0
    total = 0
    for path in tests:
        name = "local_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            total += 1
            func = getattr(module, attr)
            try:
                func()
                print(f"PASS {path.name}::{attr}")
            except Exception:
                failures += 1
                print(f"FAIL {path.name}::{attr}")
                traceback.print_exc()
    print(f"{total - failures}/{total} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
