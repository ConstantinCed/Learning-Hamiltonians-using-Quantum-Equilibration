from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Tuple


PLURAL_KEYS = {
    "families": "family",
    "lattices": "lattice",
    "boundaries": "boundary",
    "Rs": "R",
    "ks": "k",
}


def load_plan(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    cases = parse_simple_yaml_cases(text)
    expanded: List[Dict[str, object]] = []
    for case in cases:
        expanded.extend(expand_case(case))
    return expanded


def parse_simple_yaml_cases(text: str) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    cur = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped == "cases:":
            continue
        if stripped.startswith("- "):
            if cur is not None:
                cases.append(cur)
            cur = {}
            rest = stripped[2:].strip()
            if rest:
                key, value = _split_kv(rest)
                cur[key] = _parse_value(value)
            continue
        if cur is None:
            continue
        key, value = _split_kv(stripped)
        cur[key] = _parse_value(value)
    if cur is not None:
        cases.append(cur)
    return cases


def _split_kv(line: str) -> Tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"expected key: value line, got {line!r}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_value(value: str) -> object:
    if value == "":
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(part.strip()) for part in inner.split(",")]
    if value in {"null", "None", "none"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        return value.strip("'\"")


def expand_case(case: Dict[str, object]) -> List[Dict[str, object]]:
    normalized: Dict[str, object] = {}
    for key, value in case.items():
        normalized[PLURAL_KEYS.get(key, key)] = value
    varying = []
    fixed = {}
    for key, value in normalized.items():
        if isinstance(value, list):
            varying.append((key, value))
        else:
            fixed[key] = value
    if not varying:
        return [fixed]
    keys = [k for k, _ in varying]
    values = [v for _, v in varying]
    out = []
    for combo in itertools.product(*values):
        item = dict(fixed)
        item.update(dict(zip(keys, combo)))
        out.append(item)
    return out
