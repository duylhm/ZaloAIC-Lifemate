import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lifemate.data import file_hash
from lifemate.health import require_smoke


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    for name in ("model", "revision", "asr", "asr-model", "asr-revision"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seconds-per-sample", type=float, required=True)
    a = p.parse_args()
    output = Path(a.output)
    summary_path = output.with_suffix(".summary.json")
    manifest_path = output.with_suffix(".run") / "manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        raise ValueError("No completed smoke for this RUN_NAME. Run cell 4 first.")
    manifest = json.loads(manifest_path.read_text())
    code = {f.name: file_hash(f) for f in (Path(__file__).resolve().parents[1] / "lifemate").glob("*.py")}
    if manifest.get("code") != code:
        raise ValueError("Source changed since smoke. Use a new RUN_NAME and run cell 4.")
    require_smoke(json.loads(summary_path.read_text()), manifest,
                  {k: v for k, v in vars(a).items() if k != "output"})
    print("Smoke passed: five audio samples without timeout/context errors. Pilot is enabled.")


if __name__ == "__main__":
    main()
