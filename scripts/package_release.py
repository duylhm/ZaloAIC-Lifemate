"""Package source only. Never redistribute organizer data, old outputs or weights."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)
FOLDERS = ("lifemate", "scripts", "tests", "notebooks", "docker", "training_code", "docs")
FILES = ("README.md", "requirements.txt", "requirements-train.txt", "source_revision.txt",
         "run_inference.py", "evaluate.py", "runtime_config.json", ".dockerignore", ".gitignore")

source = DIST / "LifeMate_v2_Source.zip"
paths = [ROOT / name for name in FILES]
paths += [p for folder in FOLDERS for p in (ROOT / folder).rglob("*")
          if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(paths):
        z.write(path, path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(source) as z:
    assert z.testzip() is None
    assert not any("state.json" in n or n.endswith(".wav") or "backup" in n for n in z.namelist())
kit = DIST / "LifeMate_v2_Kit.zip"
with zipfile.ZipFile(kit, "w", zipfile.ZIP_DEFLATED) as z:
    for path in (source, ROOT / "notebooks/LifeMate_v2_Kaggle.ipynb", ROOT / "README.md",
                 ROOT / "docs/VALIDATION.md", ROOT / "docs/EXPERIMENT_PLAN.md"):
        z.write(path, path.name)
with zipfile.ZipFile(kit) as z:
    assert z.testzip() is None
print(json.dumps({p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in (source, kit)}, indent=2))
