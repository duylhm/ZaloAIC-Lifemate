"""Pure-stdlib Kaggle source/data discovery, also exercised by local tests."""
from __future__ import annotations
import hashlib
import shutil
import zipfile
from pathlib import Path


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe ZIP entry: {info.filename}")
        z.extractall(destination)


def source_roots(base):
    base = Path(base)
    candidates = {p.parent for p in base.rglob("source_revision.txt") if (p.parent / "lifemate" / "runner.py").is_file()}
    return sorted(candidates)


def stage_source(source, work):
    source, work = Path(source), Path(work)
    if source.is_file():
        h = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        destination = work / f"source-archive-{h}"
        safe_extract(source, destination)
        candidates = source_roots(destination)
    elif source.is_dir():
        candidates = source_roots(source)
    else:
        raise FileNotFoundError(source)
    if len(candidates) != 1:
        raise ValueError(f"Need exactly one v2 source root; found {candidates}")
    original = candidates[0]
    h = hashlib.sha256()
    files = [p for p in original.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for p in sorted(files):
        h.update(p.relative_to(original).as_posix().encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    repo = work / f"lifemate-v2-{h.hexdigest()[:12]}"
    if original.resolve() != repo.resolve():
        shutil.copytree(original, repo, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    return repo


def find_data(source, work):
    source = Path(source)
    if source.is_file():
        target = Path(work) / "lifemate-data"
        safe_extract(source, target)
        source = target
    roots = sorted({p.parent.parent for p in source.rglob("train.jsonl")
                    if p.parent.name == "train" and (p.parent / "zaic_tools" / "__init__.py").is_file()})
    if len(roots) != 1:
        raise ValueError(f"Need exactly one dataset root containing train/train.jsonl; found {roots}")
    return roots[0]
