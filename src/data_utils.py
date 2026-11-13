"""Shared data-root resolution and JSONL I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_DATA_CANDIDATES = [
    REPO_ROOT / "data",
    Path("/content/data/train+public_test"),
    Path("/content/data"),
    Path(r"C:\Users\nduy1\Desktop\antigravity_shit\lifemate\data"),
]


def _has_split(base: Path, split: str | None) -> bool:
    if not base.is_dir():
        return False
    if split:
        return (base / split / f"{split}.jsonl").is_file()
    return (base / "train" / "train.jsonl").is_file() or (base / "public_test" / "public_test.jsonl").is_file()


def resolve_data_root(root: str | Path | None = None, split: str | None = None) -> Path:
    """Resolve the competition dataset root directory across local Windows and Colab."""
    if root:
        cand = Path(root)
        # Allow passing either `<root>` or `<root>/<split>` transparently.
        if split and (cand / f"{split}.jsonl").is_file() and not (cand / split).is_dir():
            return cand.parent
        if _has_split(cand, split):
            return cand
        for sub in cand.iterdir() if cand.is_dir() else []:
            if _has_split(sub, split):
                return sub
        raise FileNotFoundError(f"Could not find LifeMate split={split!r} under {cand}")

    for cand in _DEFAULT_DATA_CANDIDATES:
        if _has_split(cand, split):
            return cand
        if cand.is_dir():
            for sub in cand.iterdir():
                if sub.is_dir() and _has_split(sub, split):
                    return sub

    raise FileNotFoundError(
        "Could not locate LifeMate dataset root. Pass `--root <path_to_data>`."
    )


def read_jsonl(path: Path) -> list[dict]:
    """Read non-empty lines from a UTF-8 JSONL file."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write dicts as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
