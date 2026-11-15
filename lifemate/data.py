"""Input handling; only the organizer runtime opens environment state."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def data_root(path, split):
    base = Path(path).resolve()
    if (base / f"{split}.jsonl").is_file() and base.name == split:
        base = base.parent
    if not (base / split / f"{split}.jsonl").is_file():
        raise FileNotFoundError(f"Expected {base / split / (split + '.jsonl')}")
    return base


def load_runtime(root, split):
    """Import the package supplied WITH this split, never an old cached module."""
    folder = Path(root) / split
    if not (folder / "zaic_tools" / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing organizer runtime: {folder / 'zaic_tools'}")
    for key in list(sys.modules):
        if key == "zaic_tools" or key.startswith("zaic_tools."):
            del sys.modules[key]
    sys.path.insert(0, str(folder))
    try:
        module = importlib.import_module("zaic_tools")
    finally:
        sys.path.remove(str(folder))
    if Path(module.__file__).resolve().parent != (folder / "zaic_tools").resolve():
        raise RuntimeError("Loaded zaic_tools from the wrong split")
    return module


def select_rows(root, split, ids_path=None, limit=None):
    rows = read_jsonl(Path(root) / split / f"{split}.jsonl")
    all_ids = [r["id"] for r in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate dataset IDs")
    if ids_path:
        payload = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        ids = payload["ids"] if isinstance(payload, dict) else payload
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or not set(ids) <= set(all_ids):
            raise ValueError("IDs must be unique and present in the selected split")
        selected = set(ids)
        rows = [r for r in rows if r["id"] in selected]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("Empty dataset selection")
    return rows


def audio_path(root, split, relative):
    base = (Path(root) / split).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise FileNotFoundError(f"Missing audio or path outside split: {relative}")
    return path


def public_input(row):
    """Explicit allowlist: labels, final_state and arbitrary extra fields never reach the model."""
    context = row.get("context", {})
    return {
        "id": row["id"], "audio": row.get("audio"),
        "context": {k: context[k] for k in
                    ("current_time", "timezone", "location", "user_profile", "installed_apps") if k in context},
        "history": [{k: turn[k] for k in ("role", "audio", "text", "tool_calls", "tool_results") if k in turn}
                    for turn in row.get("history", [])],
    }
