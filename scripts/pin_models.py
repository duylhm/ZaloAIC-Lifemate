"""Pick immutable model revisions available before the competition cutoff."""
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

CUTOFF = datetime(2026, 10, 26, tzinfo=timezone(timedelta(hours=7)))


def pin(repo_id, revision=None):
    from huggingface_hub import HfApi
    api = HfApi()
    commits = api.list_repo_commits(repo_id, revision=revision or "main")
    eligible = [c for c in commits if c.created_at < CUTOFF]
    if not eligible:
        raise ValueError(f"No revision predating {CUTOFF.isoformat()} for {repo_id}")
    commit = eligible[0]
    if revision and commits[0].commit_id != commit.commit_id:
        raise ValueError("Requested revision is newer than the permitted cutoff")
    return {"repo_id": repo_id, "revision": commit.commit_id, "date": commit.created_at.isoformat()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--revision")
    p.add_argument("--asr-model", default="g-group-ai-lab/gipformer1.5-68M-rnnt")
    p.add_argument("--asr-revision")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    data = {"llm": pin(a.model, a.revision), "asr": pin(a.asr_model, a.asr_revision)}
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
