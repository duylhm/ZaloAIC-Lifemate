"""Official local evaluator for Zalo AI Challenge 2026 — LifeMate.

Implements the exact scoring rules from the competition portal:
1. Out-of-scope samples (|G| == 0):
   Score = 1.0 if submitted tool_calls is also empty (|S| == 0), else 0.0.
2. Non-empty gold samples (|G| > 0):
   - State match: submitted state_hash must equal gold state_hash, else Score = 0.0.
   - Tool-call match: a submitted call matches a gold call when:
     * tool name is identical
     * every non-fuzzy argument is identical
     * every fuzzy argument (marked `x-match: fuzzy` in catalog) has token-F1 > 0.6
       after lowercasing and Unicode NFC.
     Each call can match at most one call (order is not considered).
   - If hash_sys != hash_gold or M == 0: Score = 0.0
   - Else: P = M / (M + U), R = M / |G|, Score = 2*P*R / (P + R)
3. Final score = (100 / N) * sum(Score_i)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def load_catalog_fuzzy_map(catalog_path: Path) -> dict[str, set[str]]:
    """Return {tool_name: set_of_fuzzy_param_names} from <split>_tools_catalog.json."""
    if not catalog_path.is_file():
        return {}
    cat = json.loads(catalog_path.read_text(encoding="utf-8"))
    fuzzy_map: dict[str, set[str]] = {}
    for app in cat.get("apps", []):
        for tool in app.get("tools", []):
            props = tool.get("parameters", {}).get("properties", {})
            f_keys = {k for k, v in props.items() if isinstance(v, dict) and v.get("x-match") == "fuzzy"}
            if f_keys:
                fuzzy_map[tool["name"]] = f_keys
    return fuzzy_map


def tokenize_nfc(text: str) -> list[str]:
    """Lowercase, Unicode NFC normalize, and extract word/number tokens."""
    norm = unicodedata.normalize("NFC", str(text)).lower()
    return re.findall(r"[^\W_]+", norm)


def token_f1(pred_str: str, gold_str: str) -> float:
    """Compute token-level F1 between two strings after lowercasing and Unicode NFC."""
    pred_toks = tokenize_nfc(pred_str)
    gold_toks = tokenize_nfc(gold_str)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    pred_cnt = Counter(pred_toks)
    gold_cnt = Counter(gold_toks)
    common = sum((pred_cnt & gold_cnt).values())
    if common == 0:
        return 0.0
    p = common / len(pred_toks)
    r = common / len(gold_toks)
    return 2 * p * r / (p + r)


def normalize_non_fuzzy(tool_name: str, key: str, val):
    """Normalize non-fuzzy values (e.g. sys.ask_missing_info params list is a set of param names)."""
    if tool_name == "sys.ask_missing_info" and key == "params" and isinstance(val, list):
        return sorted(str(x) for x in val)
    return val


def calls_match(
    sub_call: dict,
    gold_call: dict,
    fuzzy_map: dict[str, set[str]],
) -> bool:
    """Check whether a single submitted tool call matches a gold tool call."""
    t_sub = sub_call.get("name")
    t_gold = gold_call.get("name")
    if not t_sub or t_sub != t_gold:
        return False

    args_sub = sub_call.get("arguments") or {}
    args_gold = gold_call.get("arguments") or {}
    if not isinstance(args_sub, dict) or not isinstance(args_gold, dict):
        return False

    # Filter out None/empty values just like zaic_tools.validate.EMPTY
    empty_vals = (None, "", [], {})
    clean_sub = {k: v for k, v in args_sub.items() if v not in empty_vals}
    clean_gold = {k: v for k, v in args_gold.items() if v not in empty_vals}

    if set(clean_sub.keys()) != set(clean_gold.keys()):
        return False

    fuzzy_keys = fuzzy_map.get(t_sub, set())
    for k, g_val in clean_gold.items():
        s_val = clean_sub[k]
        if k in fuzzy_keys and isinstance(s_val, str) and isinstance(g_val, str):
            if token_f1(s_val, g_val) <= 0.6:
                return False
        else:
            if normalize_non_fuzzy(t_sub, k, s_val) != normalize_non_fuzzy(t_gold, k, g_val):
                return False

    return True


def max_bipartite_matches(
    sub_calls: list[dict],
    gold_calls: list[dict],
    fuzzy_map: dict[str, set[str]],
) -> int:
    """Compute maximum one-to-one matching M between submitted calls and gold calls."""
    n_sub = len(sub_calls)
    n_gold = len(gold_calls)
    adj: list[list[int]] = [[] for _ in range(n_sub)]
    for i, sc in enumerate(sub_calls):
        for j, gc in enumerate(gold_calls):
            if calls_match(sc, gc, fuzzy_map):
                adj[i].append(j)

    match_gold = [-1] * n_gold

    def dfs(u: int, visited: list[bool]) -> bool:
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_gold[v] == -1 or dfs(match_gold[v], visited):
                match_gold[v] = u
                return True
        return False

    matched = 0
    for i in range(n_sub):
        visited = [False] * n_gold
        if dfs(i, visited):
            matched += 1
    return matched


def score_sample(
    sub_calls: list[dict],
    sub_hash: str,
    gold_calls: list[dict],
    gold_hash: str,
    fuzzy_map: dict[str, set[str]],
) -> dict:
    """Compute official score and diagnostic metrics for one sample."""
    g_len = len(gold_calls)
    s_len = len(sub_calls)

    if g_len == 0:
        ok = s_len == 0
        return {
            "score": 1.0 if ok else 0.0,
            "f1_unconditional": 1.0 if ok else 0.0,
            "state_match": sub_hash == gold_hash,
            "M": 0,
            "G": 0,
            "S": s_len,
            "category": "out_of_scope",
        }

    state_ok = sub_hash == gold_hash
    m = max_bipartite_matches(sub_calls, gold_calls, fuzzy_map)
    u = s_len - m
    if m == 0:
        f1 = 0.0
    else:
        p = m / (m + u) if (m + u) > 0 else 0.0
        r = m / g_len
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    score = f1 if (state_ok and m > 0) else 0.0

    if any(c.get("name") == "sys.ask_missing_info" for c in gold_calls):
        cat = "ask_missing_info"
    elif g_len == 1:
        cat = "single_call"
    else:
        cat = "multi_call"

    return {
        "score": score,
        "f1_unconditional": f1,
        "state_match": state_ok,
        "M": m,
        "G": g_len,
        "S": s_len,
        "category": cat,
    }


def evaluate_predictions(
    data_root: Path,
    split: str,
    submission_path: Path,
    limit: int | None = None,
) -> dict:
    """Evaluate a submission.jsonl file against gold labels of `split` (e.g. 'train')."""
    sys.path.insert(0, str(data_root / split))
    from zaic_tools import ToolEnv  # noqa: E402

    fuzzy_map = load_catalog_fuzzy_map(data_root / split / f"{split}_tools_catalog.json")

    subs: dict[str, dict] = {}
    with open(submission_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                subs[row["id"]] = row

    results = []
    by_cat: dict[str, list[float]] = {}

    with open(data_root / split / f"{split}.jsonl", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if limit is not None and idx >= limit:
                break
            item = json.loads(line)
            qid = item["id"]
            gold_calls = item["label"]["tool_calls"]

            # Compute gold state_hash via ToolEnv replay
            gold_env = ToolEnv(split, qid, root=data_root)
            for gc in gold_calls:
                gold_env.call(gc["name"], gc.get("arguments") or {})
            gold_hash = gold_env.get_state_hash()

            sub = subs.get(qid, {"tool_calls": [], "state_hash": ""})
            sub_calls = sub.get("tool_calls") or []

            # Replay submitted calls as the organizer does to verify state_hash
            sub_env = ToolEnv(split, qid, root=data_root)
            for sc in sub_calls:
                sub_env.call(sc.get("name", ""), sc.get("arguments") or {})
            replayed_sub_hash = sub_env.get_state_hash()

            res = score_sample(sub_calls, replayed_sub_hash, gold_calls, gold_hash, fuzzy_map)
            res["id"] = qid
            res["submitted_hash_matches_replay"] = (sub.get("state_hash") == replayed_sub_hash)
            results.append(res)
            by_cat.setdefault(res["category"], []).append(res["score"])

    n = len(results)
    final_score = 100.0 * sum(r["score"] for r in results) / n if n else 0.0
    tiebreak_f1 = 100.0 * sum(r["f1_unconditional"] for r in results) / n if n else 0.0
    state_acc = 100.0 * sum(1 for r in results if r["state_match"]) / n if n else 0.0

    summary = {
        "samples": n,
        "final_score": round(final_score, 4),
        "tiebreak_tool_f1": round(tiebreak_f1, 4),
        "state_match_rate": round(state_acc, 4),
        "by_category": {
            k: {
                "count": len(v),
                "score": round(100.0 * sum(v) / len(v), 2),
            }
            for k, v in sorted(by_cat.items())
        },
    }
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate LifeMate submission.jsonl on labeled split")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--submission", type=Path, help="Path to submission.jsonl (omit to run gold self-check)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.submission is None:
        # Self-test: generate gold submission in memory and verify 100.0 score
        sys.path.insert(0, str(args.root / args.split))
        from zaic_tools import ToolEnv

        tmp_sub = args.root / "gold_check_sub.jsonl"
        with open(args.root / args.split / f"{args.split}.jsonl", encoding="utf-8") as fin, open(tmp_sub, "w", encoding="utf-8") as fout:
            for idx, line in enumerate(fin):
                if args.limit and idx >= args.limit:
                    break
                item = json.loads(line)
                env = ToolEnv(args.split, item["id"], root=args.root)
                for tc in item["label"]["tool_calls"]:
                    env.call(tc["name"], tc.get("arguments") or {})
                fout.write(json.dumps({
                    "id": item["id"],
                    "tool_calls": env.get_tool_calls(),
                    "state_hash": env.get_state_hash(),
                }, ensure_ascii=False) + "\n")
        summary = evaluate_predictions(args.root, args.split, tmp_sub, limit=args.limit)
        tmp_sub.unlink(missing_ok=True)
        print("Gold Self-Check Summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        summary = evaluate_predictions(args.root, args.split, args.submission, limit=args.limit)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
