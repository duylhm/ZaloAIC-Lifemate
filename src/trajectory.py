"""Causal DAG ReAct trajectory grouping, augmentation, ChatML formatting, and loss masking."""

from __future__ import annotations

import copy
import json
import random
import re
from typing import Any

from .prompt_builder import build_messages

_CONDITIONAL_CUES = (
    "nếu",
    "trường hợp",
    "miễn là",
    "còn không",
    "không thì",
    "chưa tới",
    "vượt",
    "chạm mốc",
    "ví đủ",
    "tài khoản đủ",
    "trên ",
    "dưới ",
)

_CONDITION_CHECK_TOOLS = {
    "weather.get_forecast",
    "gold_price.get_price",
    "exchange_rate.convert",
    "exchange_rate.get_rate",
    "stocks.get_quote",
    "ewallet.get_balance",
    "banking.get_balance",
}

_PRECONDITION_READ_TOOLS = _CONDITION_CHECK_TOOLS | {"bill_payment.list_bills"}

_TRIVIAL_SCALARS = {
    "ok",
    "true",
    "false",
    "none",
    "null",
    "vnd",
    "vn",
    "on",
    "off",
    "high",
    "low",
    "open",
    "closed",
    "pending",
    "paid",
}

# Safe Vietnamese ASR phonetic confusions applied ONLY to trigger/action cues,
# never to substrings matching gold argument literals.
_PHONETIC_NOISE_PAIRS: list[tuple[str, str]] = [
    ("báo thức", "bảo thích"),
    ("nhắc tôi", "nhất tôi"),
    ("nhắc mình", "nhất mình"),
    ("thời tiết", "thờ tiết"),
    ("chuyển khoản", "chuyến khoản"),
    ("tài khoản", "tài khoan"),
    ("đặt lịch", "đặc lịch"),
    ("xóa lịch", "xoa lịch"),
    ("hủy lịch", "thủy lịch"),
    ("bật đèn", "bậc đèn"),
    ("mở rèm", "mỡ rèm"),
    ("gọi điện", "gợi điện"),
    ("cập nhật", "cập nhập"),
    ("kiểm tra", "kiễm tra"),
    ("trạng thái", "trạng thai"),
    ("tin nhắn", "tín nhắn"),
    ("hóa đơn", "hoa đơn"),
]


def _extract_scalar_tokens(obj: Any, min_str_len: int = 2) -> tuple[set[str], set[str]]:
    """Collect exact normalized scalars and substring-matchable IDs/names from nested JSON."""
    exact: set[str] = set()
    substrs: set[str] = set()

    def walk(val: Any) -> None:
        if isinstance(val, dict):
            for v in val.values():
                walk(v)
        elif isinstance(val, list):
            for v in val:
                walk(v)
        elif isinstance(val, bool) or val is None:
            return
        elif isinstance(val, (int, float)):
            if abs(val) >= 100:
                s = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
                exact.add(s)
        elif isinstance(val, str):
            s = val.strip()
            low = s.lower()
            if len(s) < min_str_len or low in _TRIVIAL_SCALARS:
                return
            exact.add(low)
            if len(s) >= 4:
                substrs.add(low)

    walk(obj)
    return exact, substrs


def _results_overlap_arguments(tool_result: Any, arguments: dict) -> bool:
    """Return True if a prior tool_result supplies values or IDs consumed by arguments."""
    res_exact, res_sub = _extract_scalar_tokens(tool_result)
    if not res_exact and not res_sub:
        return False
    arg_exact, arg_sub = _extract_scalar_tokens(arguments)
    if res_exact & arg_exact:
        return True
    for r_tok in res_sub:
        for a_tok in arg_sub:
            if r_tok in a_tok:
                return True
    return False


def is_conditional_request(text: str) -> bool:
    """Detect conditional triggers where condition probes must resolve before actions."""
    low = (text or "").lower()
    return any(cue in low for cue in _CONDITIONAL_CUES)


def group_trajectory_into_steps(
    sample: dict,
    catalog_tools: dict[str, dict],
) -> list[dict[str, list[dict]]]:
    """Partition flat gold tool_calls into causally valid ReAct execution steps.

    Independent reads batch into one step to minimize turns, while respecting:
    1. Data dependencies (a later call consumes a literal ID/value from a prior result).
    2. Conditional barriers (weather/price/balance checks run before gated actions).
    3. Read-before-write & same-app write-before-read ordering.
    4. Immediate termination at `sys.ask_missing_info` (meta).
    """
    label = sample.get("label") or {}
    calls: list[dict] = label.get("tool_calls") or []
    results: list[dict] = label.get("tool_results") or []
    if not calls:
        return []

    conditional = is_conditional_request(sample.get("audio_text", ""))
    n = len(calls)
    step_of = [0] * n

    meta_info: list[tuple[str, str, str]] = []
    for c in calls:
        name = c["name"]
        app = name.split(".")[0]
        kind = (catalog_tools.get(name) or {}).get("kind", "write")
        if name == "sys.ask_missing_info":
            kind = "meta"
        meta_info.append((name, app, kind))

    for i in range(n):
        name_i, app_i, kind_i = meta_info[i]
        args_i = calls[i].get("arguments") or {}
        earliest = 0

        for j in range(i):
            name_j, app_j, kind_j = meta_info[j]
            res_j = results[j] if j < len(results) else {}

            if _results_overlap_arguments(res_j, args_i):
                earliest = max(earliest, step_of[j] + 1)
                continue

            if conditional and name_j in _CONDITION_CHECK_TOOLS and name_i not in _PRECONDITION_READ_TOOLS:
                earliest = max(earliest, step_of[j] + 1)
                continue

            if kind_j == "read" and kind_i in ("write", "meta"):
                earliest = max(earliest, step_of[j] + 1)
                continue

            if kind_j == "write" and kind_i == "read" and app_j == app_i:
                earliest = max(earliest, step_of[j] + 1)
                continue

            if kind_j == "write" and kind_i == "write" and app_j == app_i:
                earliest = max(earliest, step_of[j])

        if kind_i == "meta":
            earliest = max(earliest, max(step_of[:i], default=0))

        step_of[i] = earliest

    max_step = max(step_of)
    steps: list[dict[str, list[dict]]] = []
    for s_idx in range(max_step + 1):
        idxs = [i for i, st in enumerate(step_of) if st == s_idx]
        if not idxs:
            continue
        steps.append({
            "tool_calls": [copy.deepcopy(calls[i]) for i in idxs],
            "tool_results": [copy.deepcopy(results[i]) if i < len(results) else {"ok": True} for i in idxs],
        })
    return steps


def sample_dropout_apps(
    sample: dict,
    rng: random.Random,
    max_drop_ratio: float = 0.7,
) -> set[str]:
    """Select a subset of non-gold installed apps to drop during SFT augmentation."""
    installed = set((sample.get("context") or {}).get("installed_apps") or [])
    gold_calls = (sample.get("label") or {}).get("tool_calls") or []
    gold_apps = {c["name"].split(".")[0] for c in gold_calls} - {"sys"}
    # Also keep any apps already referenced in history turns so history stays coherent.
    for turn in sample.get("history") or []:
        for tc in turn.get("tool_calls") or []:
            gold_apps.add(tc["name"].split(".")[0])

    candidates = sorted(installed - gold_apps)
    if not candidates or max_drop_ratio <= 0:
        return set()

    drop_rate = rng.uniform(0.0, max_drop_ratio)
    k = int(round(len(candidates) * drop_rate))
    if k <= 0:
        return set()
    return set(rng.sample(candidates, k))


def inject_safe_phonetic_noise(sample: dict, rng: random.Random, prob: float = 0.2) -> dict:
    """Inject realistic Vietnamese ASR verb confusion without corrupting gold arguments."""
    if rng.random() >= prob:
        return sample

    text = sample.get("audio_text", "")
    if not text:
        return sample

    # Collect argument string literals so we never mutate an entity/fuzzy parameter.
    gold_literals: set[str] = set()
    for call in (sample.get("label") or {}).get("tool_calls") or []:
        exact, _ = _extract_scalar_tokens(call.get("arguments") or {}, min_str_len=2)
        gold_literals.update(exact)

    matches: list[tuple[str, str]] = []
    low_text = text.lower()
    for clean, noisy in _PHONETIC_NOISE_PAIRS:
        if clean in low_text and not any(clean in lit or lit in clean for lit in gold_literals):
            matches.append((clean, noisy))

    if not matches:
        return sample

    clean, noisy = rng.choice(matches)
    pattern = re.compile(re.escape(clean), re.IGNORECASE)
    corrupted = pattern.sub(noisy, text, count=1)
    if corrupted == text:
        return sample

    out = copy.deepcopy(sample)
    out["audio_text"] = corrupted
    return out


def sample_to_chatml(
    sample: dict,
    catalog_tools: dict[str, dict],
    drop_apps: set[str] | None = None,
    shuffle_rng: random.Random | None = None,
    asr_alt: str | None = None,
) -> list[dict[str, str]]:
    """Convert a training sample into a complete multi-turn ChatML SFT conversation."""
    steps = group_trajectory_into_steps(sample, catalog_tools)

    # Terminate right after `sys.ask_missing_info` without tool_results or extra `[]`.
    ended_by_ask = bool(
        steps
        and any(tc.get("name") == "sys.ask_missing_info" for tc in steps[-1]["tool_calls"])
    )

    if ended_by_ask:
        messages = build_messages(
            sample,
            catalog_tools,
            step_trace=steps[:-1],
            drop_apps=drop_apps,
            shuffle_rng=shuffle_rng,
            asr_alt=asr_alt,
        )
        messages.append({
            "role": "assistant",
            "content": json.dumps({"tool_calls": steps[-1]["tool_calls"]}, ensure_ascii=False),
        })
        return messages

    messages = build_messages(
        sample,
        catalog_tools,
        step_trace=steps,
        drop_apps=drop_apps,
        shuffle_rng=shuffle_rng,
        asr_alt=asr_alt,
    )
    messages.append({
        "role": "assistant",
        "content": '{"tool_calls": []}',
    })
    return messages


def _apply_chat_template_compat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> list[int]:
    """Call tokenizer.apply_chat_template while disabling Qwen3 thinking when supported."""
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)

    # transformers >= 5.x or BatchEncoding may return a dict/Encoding with `input_ids`.
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def tokenize_chatml_with_masking(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int = 8192,
) -> dict[str, list[int]]:
    """Tokenize multi-turn ChatML conversation with labels=-100 on all non-assistant tokens.

    For ChatML tokenizers (`<|im_start|>` / `<|im_end|>`), masks the role header
    `<|im_start|>assistant\n` (plus any empty Qwen3 `<think>\n\n</think>\n\n` block)
    and unmasks strictly the assistant content + its closing `<|im_end|>`.
    """
    full_ids = _apply_chat_template_compat(tokenizer, messages, add_generation_prompt=False)
    labels = [-100] * len(full_ids)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    unk_id = getattr(tokenizer, "unk_token_id", None)

    if (
        isinstance(im_start_id, int)
        and isinstance(im_end_id, int)
        and im_start_id != unk_id
        and im_end_id != unk_id
        and im_start_id in full_ids
    ):
        asst_prefix_candidates = [
            tokenizer.encode("assistant\n", add_special_tokens=False),
            tokenizer.encode("assistant", add_special_tokens=False),
        ]
        think_end_tok = tokenizer.convert_tokens_to_ids("</think>")
        newline_id = tokenizer.encode("\n", add_special_tokens=False)

        i = 0
        n = len(full_ids)
        while i < n:
            if full_ids[i] == im_start_id:
                matched_header_len = 0
                for cand in asst_prefix_candidates:
                    if cand and full_ids[i + 1 : i + 1 + len(cand)] == cand:
                        matched_header_len = 1 + len(cand)
                        break

                if matched_header_len > 0:
                    start = i + matched_header_len
                    # Skip optional empty `<think>...</think>\n\n` injected by Qwen3 templates.
                    if isinstance(think_end_tok, int) and think_end_tok != unk_id:
                        search_limit = min(n, start + 12)
                        for p in range(start, search_limit):
                            if full_ids[p] == think_end_tok:
                                start = p + 1
                                while start < n and full_ids[start] in newline_id:
                                    start += 1
                                break

                    end = start
                    while end < n and full_ids[end] != im_end_id:
                        labels[end] = full_ids[end]
                        end += 1
                    if end < n and full_ids[end] == im_end_id:
                        labels[end] = full_ids[end]
                    i = end + 1
                    continue
            i += 1
    else:
        # Generic fallback via incremental apply_chat_template boundaries.
        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prompt_ids = _apply_chat_template_compat(
                tokenizer, messages[:idx], add_generation_prompt=True
            )
            with_resp_ids = _apply_chat_template_compat(
                tokenizer, messages[: idx + 1], add_generation_prompt=False
            )
            start = len(prompt_ids)
            end = min(len(with_resp_ids), len(full_ids))
            for k in range(start, end):
                labels[k] = full_ids[k]

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
