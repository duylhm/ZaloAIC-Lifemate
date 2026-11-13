"""Schema-aware sanitization, type coercion, and output parsing guardrails."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(?:[.,]\d{3})+$")
_TZ_SUFFIX_RE = re.compile(r"(Z|[+-]\d{2}:\d{2})$")


def _strip_noise_wrappers(text: str) -> str:
    """Remove optional Qwen3 thinking blocks and markdown code fences."""
    cleaned = _THINK_RE.sub("", text or "")
    # Handle truncated <think> without closing tag by keeping text after </think> if present,
    # or dropping only the unclosed <think>... wrapper if JSON follows.
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    cleaned = cleaned.strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    return cleaned


def _extract_balanced_json_candidates(text: str) -> list[Any]:
    """Scan text for balanced `{...}` or `[...]` JSON segments in order."""
    candidates: list[Any] = []
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch in "{[":
            try:
                obj, end = decoder.raw_decode(text, idx)
                candidates.append(obj)
                idx = end
                continue
            except json.JSONDecodeError:
                pass
        idx += 1
    return candidates


def parse_tool_calls(text: str) -> list[dict]:
    """Extract normalized `[{"name": ..., "arguments": {...}}]` list from LLM output."""
    cleaned = _strip_noise_wrappers(text)
    if not cleaned:
        return []

    parsed_objs: list[Any] = []
    try:
        parsed_objs.append(json.loads(cleaned))
    except json.JSONDecodeError:
        parsed_objs = _extract_balanced_json_candidates(cleaned)

    raw_calls: list[Any] = []
    for obj in parsed_objs:
        if isinstance(obj, dict):
            if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
                raw_calls = obj["tool_calls"]
                break
            if "name" in obj:
                raw_calls = [obj]
                break
        elif isinstance(obj, list):
            if all(isinstance(x, dict) and "name" in x for x in obj):
                raw_calls = obj
                break

    out: list[dict] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        args = item.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append({"name": name, "arguments": args})
    return out


def _infer_param_format(spec: dict) -> str:
    fmt = spec.get("format", "")
    if fmt:
        return fmt
    desc = spec.get("description", "")
    if "YYYY-MM-DDTHH:MM" in desc or "ISO-8601" in desc:
        return "date-time"
    if "YYYY-MM-DD" in desc:
        return "date"
    if "HH:MM" in desc:
        return "time"
    return ""


def _sample_tz_suffix(sample: dict) -> str:
    cur = str((sample.get("context") or {}).get("current_time") or "")
    m = re.search(r"([+-]\d{2}:\d{2}|Z)$", cur)
    if not m:
        return "+07:00"
    return "+00:00" if m.group(1) == "Z" else m.group(1)


def _coerce_integer(val: Any) -> Any:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        # Extract leading numeric token if followed by units like "2 tiếng", "500.000 VND"
        token = re.match(r"^(-?[\d.,]+)", s)
        if token:
            num_str = token.group(1)
            if _THOUSANDS_RE.match(num_str):
                return int(num_str.replace(".", "").replace(",", ""))
            try:
                f = float(num_str.replace(",", "."))
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
    return val


def _coerce_number(val: Any) -> Any:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        s = val.strip()
        token = re.match(r"^(-?[\d.,]+)", s)
        if token:
            num_str = token.group(1)
            if _THOUSANDS_RE.match(num_str):
                return int(num_str.replace(".", "").replace(",", ""))
            # Single comma as decimal point (e.g. "1,5")
            if num_str.count(",") == 1 and "." not in num_str:
                num_str = num_str.replace(",", ".")
            try:
                f = float(num_str)
                return int(f) if f.is_integer() else f
            except ValueError:
                pass
    return val


def _coerce_temporal(val: str, fmt: str, tz_suffix: str) -> str:
    s = unicodedata.normalize("NFC", val).strip()
    if fmt == "date":
        if re.match(r"^\d{4}-\d{2}-\d{2}T", s):
            return s[:10]
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", s)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{y:04d}-{mo:02d}-{d:02d}"
        return s

    if fmt == "time":
        m = re.match(r"^(\d{1,2})[:hH](\d{2})(?::\d{2})?$", s)
        if m:
            return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
        return s

    if fmt == "date-time":
        s = s.replace(" ", "T")
        m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})(?::(\d{2}))?(Z|[+-]\d{2}:\d{2})?$", s)
        if m:
            base = m.group(1)
            sec = m.group(2) or "00"
            tz = m.group(3) or tz_suffix
            if tz == "Z":
                tz = "+00:00"
            return f"{base}:{sec}{tz}"
        if not _TZ_SUFFIX_RE.search(s) and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", s):
            return f"{s}{tz_suffix}"
        return s

    return s


def _coerce_scalar_value(val: Any, schema: dict, tz_suffix: str) -> Any:
    ptype = schema.get("type")
    if isinstance(val, str):
        val = unicodedata.normalize("NFC", val).strip()

    if ptype == "integer":
        return _coerce_integer(val)
    if ptype == "number":
        return _coerce_number(val)
    if ptype == "boolean":
        if isinstance(val, str):
            low = val.lower()
            if low in {"true", "1", "yes", "có"}:
                return True
            if low in {"false", "0", "no", "không"}:
                return False
        return val
    if ptype == "string" and val is not None:
        s = str(val)
        fmt = _infer_param_format(schema)
        if fmt:
            s = _coerce_temporal(s, fmt, tz_suffix)
        # Match enum case-insensitively if exact enum missing.
        enums = schema.get("enum")
        if isinstance(enums, list) and s not in enums:
            low_s = s.lower()
            for e in enums:
                if str(e).lower() == low_s:
                    return e
        return s
    if ptype == "array" and isinstance(val, list):
        item_schema = schema.get("items") or {}
        if item_schema.get("type") == "object":
            item_props = item_schema.get("properties") or {}
            cleaned_items = []
            for el in val:
                if isinstance(el, dict):
                    cleaned_el = {}
                    for k, v in el.items():
                        if k in item_props and v not in (None, "", [], {}):
                            cleaned_el[k] = _coerce_scalar_value(v, item_props[k], tz_suffix)
                    if cleaned_el:
                        cleaned_items.append(cleaned_el)
            return cleaned_items
        return [_coerce_scalar_value(el, item_schema, tz_suffix) for el in val if el not in (None, "", [], {})]
    return val


def sanitize_and_repair_call(
    name: str,
    arguments: dict | None,
    sample: dict,
    catalog_tools: dict[str, dict],
) -> tuple[str, dict]:
    """Normalize tool name and clean/coerce arguments against runtime tool catalog."""
    raw_name = (name or "").strip()
    args = dict(arguments or {})
    ctx = sample.get("context") or {}
    installed = set(ctx.get("installed_apps") or []) | {"sys"}

    # Recover unqualified tool name if unique among installed apps.
    if "." not in raw_name:
        matches = [
            f"{app}.{raw_name}"
            for app in sorted(installed)
            if f"{app}.{raw_name}" in catalog_tools
        ]
        if len(matches) == 1:
            raw_name = matches[0]

    tool_spec = catalog_tools.get(raw_name)
    if not tool_spec:
        cleaned = {k: v for k, v in args.items() if v not in (None, "", [], {}, "null", "None")}
        return raw_name, cleaned

    props = (tool_spec.get("parameters") or {}).get("properties") or {}
    tz_suffix = _sample_tz_suffix(sample)

    cleaned_args: dict[str, Any] = {}
    for k, v in args.items():
        if v in (None, "", [], {}, "null", "None"):
            continue
        if props and k not in props:
            continue
        p_schema = props.get(k, {})
        coerced = _coerce_scalar_value(v, p_schema, tz_suffix)
        if coerced not in (None, "", [], {}):
            cleaned_args[k] = coerced

    # Gold-grounded default conventions.
    if raw_name == "weather.get_forecast" and "location" in cleaned_args:
        user_city = unicodedata.normalize("NFC", str((ctx.get("user_profile") or {}).get("city") or "")).strip().lower()
        loc_val = unicodedata.normalize("NFC", str(cleaned_args["location"])).strip().lower()
        if loc_val in {user_city, "ở đây", "hiện tại", "địa phương", ""}:
            cleaned_args.pop("location", None)

    if raw_name == "alarm.create" and cleaned_args.get("repeat") == "once":
        cleaned_args.pop("repeat", None)

    if raw_name == "sys.ask_missing_info" and isinstance(cleaned_args.get("params"), str):
        cleaned_args["params"] = [cleaned_args["params"]]

    return raw_name, cleaned_args
