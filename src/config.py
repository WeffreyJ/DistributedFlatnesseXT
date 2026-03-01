"""Configuration loading utilities."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


def _parse_scalar(text: str) -> Any:
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(tok.strip()) for tok in inner.split(",")]
    try:
        if "." in text or "e" in low:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _simple_yaml_load(raw: str) -> dict[str, Any]:
    """Minimal YAML loader supporting nested maps and flow-style lists."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, val = line.strip().partition(":")
        val = val.strip()
        if " #" in val:
            val = val.split(" #", 1)[0].strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if val == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(val)
    return root


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    """Load YAML config into a nested SimpleNamespace."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = f.read()
    if yaml is not None:
        data = yaml.safe_load(raw)
    else:
        data = _simple_yaml_load(raw)
    return _to_ns(data)
