"""No-argument offline Docker entrypoint."""
import json
import os
from pathlib import Path
from .runner import parser, run


def main():
    root = Path(os.environ.get("LIFEMATE_DATA", "/data"))
    selected = os.environ.get("LIFEMATE_SPLIT")
    if not selected:
        choices = [s for s in ("private_test", "public_test", "train") if (root / s / f"{s}.jsonl").is_file()]
        if len(choices) != 1:
            raise ValueError(f"Expected one split in /data or explicit LIFEMATE_SPLIT. Found {choices}")
        selected = choices[0]
    config = json.loads(Path("/code/runtime_config.json").read_text())
    if config.get("asr") not in ("gipformer", "phowhisper"):
        raise ValueError("Offline deployment requires a live ASR backend")
    args = parser().parse_args(["--root", str(root), "--split", selected,
        "--model", config["model"], "--asr", config["asr"], "--asr-model", config["asr_model"],
        "--offline", "--output", os.environ.get("LIFEMATE_OUTPUT", "/result/submission.jsonl")])
    run(args)

if __name__ == "__main__":
    main()
