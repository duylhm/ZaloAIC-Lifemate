"""Download pinned models for offline deployment; run on Kaggle, not in Docker inference."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pins", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    pins = json.loads(Path(a.pins).read_text(encoding="utf-8"))
    for key in ("llm", "asr"):
        entry = pins[key]
        opts = {"allow_patterns": ["*.int8.onnx", "tokens.txt", "config.json", "README.md"]} if key == "asr" and "gipformer" in entry["repo_id"].lower() else {}
        snapshot_download(entry["repo_id"], revision=entry["revision"], local_dir=str(Path(a.output) / key), **opts)
    (Path(a.output) / "pins.json").write_text(json.dumps(pins, indent=2), encoding="utf-8")
    print("Offline models ready:", a.output)

if __name__ == "__main__":
    main()
