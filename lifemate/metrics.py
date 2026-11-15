"""Local approximation of the published scoring rule, with replayed hashes."""
from __future__ import annotations

import unicodedata
from collections import Counter

from .prompt import catalog_tools


def token_f1(a, b):
    a = Counter(unicodedata.normalize("NFC", a).lower().split())
    b = Counter(unicodedata.normalize("NFC", b).lower().split())
    n = sum((a & b).values())
    return 2 * n / (sum(a.values()) + sum(b.values())) if a or b else 1.0


def calls_match(a, b, tools):
    if a.get("name") != b.get("name"):
        return False
    # The organizer treats None/empty optional values as absent.
    left = {k: v for k, v in a.get("arguments", {}).items() if v not in (None, "", [], {})}
    right = {k: v for k, v in b.get("arguments", {}).items() if v not in (None, "", [], {})}
    if left.keys() != right.keys():
        return False
    props = tools.get(a["name"], {}).get("parameters", {}).get("properties", {})
    for key, value in left.items():
        other = right[key]
        if props.get(key, {}).get("x-match") == "fuzzy" and isinstance(value, str) and isinstance(other, str):
            if token_f1(value, other) <= 0.6:
                return False
        elif a["name"] == "sys.ask_missing_info" and key == "params":
            if sorted(value) != sorted(other):
                return False
        elif type(value) is not type(other) or value != other:
            return False
    return True


def matched_count(submitted, gold, tools):
    edges = [[j for j, b in enumerate(gold) if calls_match(a, b, tools)] for a in submitted]
    matches = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matches or augment(matches[j], seen):
                matches[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(submitted)))


def evaluate(rows, predictions, runtime, root, split, catalog, deadline_ms=None, failed_ids=None):
    expected = {r["id"] for r in rows}
    by_id = {p["id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != expected:
        raise ValueError("Predictions must cover selected IDs exactly once")
    tools = catalog_tools(catalog)
    details = []
    for row in rows:
        pred = by_id[row["id"]]
        # Separate evaluator replay, never exposed to the inference agent.
        gold = row["label"]["tool_calls"]
        env = runtime.ToolEnv(split, row["id"], root=root)
        for call in gold:
            env.call(call["name"], call["arguments"])
        same = pred["state_hash"] == env.get_state_hash()
        count = matched_count(pred["tool_calls"], gold, tools)
        denom = len(gold) + len(pred["tool_calls"])
        f1 = 2 * count / denom if denom else 1.0
        score = (f1 if same else 0.0) if gold else float(not pred["tool_calls"])
        timeout = deadline_ms is not None and pred.get("time", 0) > deadline_ms
        failed = row["id"] in (failed_ids or set())
        details.append({"id": row["id"], "score": 0.0 if timeout or failed else score, "state_match": same,
                        "tool_f1": f1, "timeout": timeout,
                        "failed": failed,
                        "category": "oos" if not gold else "ask" if any(c["name"] == "sys.ask_missing_info" for c in gold)
                        else "multi" if len(gold) > 1 else "single"})
    summary = {key: 100 * sum(r[key] for r in details) / len(details) for key in ("score", "state_match", "tool_f1")}
    summary.update(n=len(details), timeouts=sum(r["timeout"] for r in details), failed=sum(r["failed"] for r in details))
    return summary, details


def verify(rows, predictions, runtime, root, split):
    ids = [p["id"] for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) != {r["id"] for r in rows}:
        raise ValueError("Submission has missing, unknown or duplicate IDs")
    for pred in predictions:
        if not isinstance(pred.get("tool_calls"), list):
            raise ValueError("tool_calls must be a list")
        env = runtime.ToolEnv(split, pred["id"], root=root)
        for call in pred["tool_calls"]:
            if not env.validate(call["name"], call["arguments"])["ok"]:
                raise ValueError(f"Schema invalid: {pred['id']}: {call}")
            env.call(call["name"], call["arguments"])
        if env.get_state_hash() != pred.get("state_hash"):
            raise ValueError(f"Replay hash mismatch: {pred['id']}")
