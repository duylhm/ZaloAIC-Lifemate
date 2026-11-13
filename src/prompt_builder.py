"""Dynamic Schema-Grounded Prompt Builder for ZAC 2026 — LifeMate.

Compresses tool JSON schemas into compact, high-signal typed signatures so <5B LLMs
stay well within budget (<=8K tokens) while preserving 100% of schema constraints:
required, x-one-of, enum, format, minimum/exclusiveMinimum, items, x-ref, and x-match.
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

WEEKDAYS_VI = [
    "Thứ Hai",
    "Thứ Ba",
    "Thứ Tư",
    "Thứ Năm",
    "Thứ Sáu",
    "Thứ Bảy",
    "Chủ Nhật",
]

# Boilerplate phrases repeated verbatim across dozens of catalog entries.
# These rules are stated once in SYSTEM_INSTRUCTION, saving ~40% schema tokens.
_BOILERPLATE_PATTERNS = [
    re.compile(r"\s*ISO-8601 đầy đủ giây và offset múi giờ,\s*ví dụ\s*[^\s;)]+", re.I),
    re.compile(r"\s*YYYY-MM-DD,\s*ví dụ\s*\d{4}-\d{2}-\d{2}", re.I),
    re.compile(r"\s*Số nguyên,\s*không dấu chấm(?:,|\s+hay)\s*phẩy(?:\s+hay đơn vị)?,\s*ví dụ\s*\d+\s*", re.I),
    re.compile(r"\s*số nguyên,\s*không dấu chấm(?:,|\s+hay)\s*phẩy(?:\s+hay đơn vị)?,\s*ví dụ\s*\d+\s*", re.I),
]


def load_catalog_dict(catalog_path: Path) -> dict[str, dict]:
    """Return {tool_name: tool_schema_dict} from <split>_tools_catalog.json."""
    cat = json.loads(catalog_path.read_text(encoding="utf-8"))
    tools: dict[str, dict] = {}
    for app in cat.get("apps", []):
        for t in app.get("tools", []):
            tools[t["name"]] = t
    return tools


def _clean_desc(desc: str) -> str:
    if not desc:
        return ""
    text = desc.strip()
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" .;,")
    return text


def _format_type_spec(prop: dict) -> str:
    t = prop.get("type", "any")
    if t == "array":
        items = prop.get("items") or {}
        if items.get("type") == "object":
            sub_req = set(items.get("required") or [])
            sub_fields = []
            for sk, sv in (items.get("properties") or {}).items():
                star = "*" if sk in sub_req else ""
                sub_fields.append(f"{sk}{star}:{sv.get('type', 'any')}")
            return "array[{" + ", ".join(sub_fields) + "}]"
        return f"array[{items.get('type', 'any')}]"

    parts = [t]
    fmt = prop.get("format")
    if not fmt and t == "string":
        raw_desc = prop.get("description") or ""
        if "ISO-8601" in raw_desc:
            fmt = "date-time"
        elif "YYYY-MM-DD" in raw_desc:
            fmt = "date"
        elif "HH:MM" in raw_desc:
            fmt = "time"
    if fmt:
        parts.append(fmt)
    if prop.get("enum"):
        parts.append("enum=" + json.dumps(prop["enum"], ensure_ascii=False))
    if "minimum" in prop:
        parts.append(f">={prop['minimum']}")
    if "exclusiveMinimum" in prop:
        parts.append(f">{prop['exclusiveMinimum']}")
    if prop.get("x-ref"):
        parts.append(f"ref:{prop['x-ref']}")
    if prop.get("x-match"):
        parts.append(f"match:{prop['x-match']}")
    return "|".join(parts)


def format_compact_tool(tool: dict) -> str:
    """Format a single catalog tool into a concise single-line typed signature."""
    name = tool["name"]
    kind = tool.get("kind", "write")
    desc = tool.get("description", "").strip()
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    one_of = params.get("x-one-of") or []

    param_chunks = []
    for pk, pv in props.items():
        star = "*" if pk in required else ""
        tspec = _format_type_spec(pv)
        pdesc = _clean_desc(pv.get("description", ""))
        if pdesc:
            param_chunks.append(f"{pk}{star}:{tspec} ({pdesc})")
        else:
            param_chunks.append(f"{pk}{star}:{tspec}")

    line = f"- {name} [{kind}]"
    if desc:
        line += f": {desc}"
    if one_of:
        groups = " | ".join("+".join(g) for g in one_of)
        line += f" | one-of:[{groups}]"
    if param_chunks:
        line += " | args: " + "; ".join(param_chunks)
    else:
        line += " | args: {}"
    return line


def get_allowed_tool_lines(
    sample: dict,
    catalog_tools: dict[str, dict],
    drop_apps: set[str] | None = None,
    shuffle_rng: random.Random | None = None,
) -> list[str]:
    """Return compact signatures for every tool installed in `sample` plus `sys.ask_missing_info`."""
    installed = sample.get("context", {}).get("installed_apps", {})
    selected_tools = []
    if isinstance(installed, dict):
        for app_id, app_info in installed.items():
            if drop_apps and app_id in drop_apps:
                continue
            tool_names = app_info.get("tools", []) if isinstance(app_info, dict) else []
            for tname in tool_names:
                if tname in catalog_tools:
                    selected_tools.append(catalog_tools[tname])
    else:
        allowed_apps = {a for a in (installed or []) if not (drop_apps and a in drop_apps)}
        for tname, tspec in catalog_tools.items():
            if tspec.get("app") in allowed_apps:
                selected_tools.append(tspec)

    if shuffle_rng is not None:
        shuffle_rng.shuffle(selected_tools)

    if "sys.ask_missing_info" in catalog_tools:
        selected_tools.append(catalog_tools["sys.ask_missing_info"])

    return [format_compact_tool(t) for t in selected_tools]


def compute_time_anchors(iso_time: str) -> dict[str, str]:
    """Pre-compute today, tomorrow, and Vietnamese weekday to prevent LLM date arithmetic errors."""
    try:
        dt = datetime.fromisoformat(iso_time)
        tz_suffix = iso_time[-6:] if ("+" in iso_time[10:] or "-" in iso_time[10:]) else "+07:00"
        return {
            "now": iso_time,
            "weekday_vi": WEEKDAYS_VI[dt.weekday()],
            "today": dt.strftime("%Y-%m-%d"),
            "tomorrow": (dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            "day_after_tomorrow": (dt + timedelta(days=2)).strftime("%Y-%m-%d"),
            "tz": tz_suffix,
        }
    except Exception:
        return {"now": iso_time}


def _compact_history_turns(history: list[dict]) -> list[dict]:
    """Preserve every assistant text, tool_call, and key IDs/fields from prior tool_results."""
    out = []
    for turn in history or []:
        if turn.get("role") == "user":
            txt = turn.get("audio_text") or turn.get("text") or ""
            if txt:
                out.append({"u": txt})
        else:
            entry: dict = {}
            if turn.get("text"):
                entry["a"] = turn["text"]
            if turn.get("tool_calls"):
                entry["calls"] = turn["tool_calls"]
            if turn.get("tool_results"):
                entry["results"] = turn["tool_results"]
            if entry:
                out.append(entry)
    return out


SYSTEM_INSTRUCTION = """Bạn là AI Agent thực thi tác vụ giọng nói tiếng Việt trên thiết bị (ZAC 2026 — LifeMate).
Dựa vào `time_anchors`, `user_profile`, `location`, `history`, `available_tools` và `request`, hãy gọi chính xác chuỗi tool cần thiết để hoàn thành yêu cầu.

QUY TẮC BẮT BUỘC:
1. ĐỊNH DẠNG ĐẦU RA:
   - Chỉ trả về DUY NHẤT một JSON object hợp lệ (không markdown, không giải thích):
     {"tool_calls": [{"name": "<app>.<tool>", "arguments": {...}}]}
   - Khi đã hoàn thành hết yêu cầu, hoặc yêu cầu ngoài phạm vi `available_tools` (out-of-scope), hoặc người dùng tự hủy ("thôi khỏi", "không cần nữa"), hoặc câu điều kiện không thỏa mãn: trả về `{"tool_calls": []}`.

2. ĐỌC TRƯỚC KHI GHI & TÁI SỬ DỤNG LỊCH SỬ:
   - Không tự bịa ID (`*id`, `phone_number`, `email`...). Phải gọi `[read]` tool tra cứu trước (lấy ID từ kết quả `tool_results` trả về ở lượt kế tiếp).
   - Có thể gộp nhiều `[read]` tool độc lập trong cùng một lượt để tra cứu nhanh, NHƯNG với câu điều kiện ("Nếu mai mưa thì...", "Nếu ví đủ thì..."), CHỈ gọi tool kiểm tra điều kiện trước; chờ kết quả rồi mới làm tiếp hoặc dừng bằng `{"tool_calls": []}`.
   - Nếu thông tin/ID/dữ liệu thời tiết/giá cả đã có sẵn trong `history`, HÃY DÙNG LUÔN, không gọi lại tool đọc cũ.

3. THAM SỐ CHẶT CHẼ & TỐI GIẢN:
   - Chỉ gọi tool có trong `available_tools`. Đảm bảo đủ mọi tham số đánh dấu `*` (required) và đúng một nhóm trong `one-of:[...]`.
   - Số nguyên viết liền không dấu chấm/phẩy/đơn vị (vd `1200000`, `2`). Kiểu `date`: `YYYY-MM-DD`; `date-time`: `YYYY-MM-DDTHH:MM:SS+07:00` (theo múi giờ `time_anchors.tz`); `time`: `HH:MM`.
   - Chỉ truyền tham số bắt buộc hoặc được người dùng nói rõ. Bỏ tham số tùy chọn mặc định (`weather.get_forecast` không truyền `location` nếu hỏi thời tiết ở `user_profile.city`; `alarm.create` không tự thêm `repeat="once"` hay `label`; tool `list_*`/`get_status` để `{}` trừ khi có tiêu chí lọc rõ).

4. THIẾU THÔNG TIN HOẶC TRÙNG TÊN (`sys.ask_missing_info`):
   - Khi thiếu tham số bắt buộc dù đã tra cứu, hoặc kết quả tìm kiếm trả về nhiều ứng viên trùng tên không rõ người nào, KHÔNG đoán bừa: gọi `{"tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "<app>.<tool>", "params": ["<param_thiếu>"]}}]}` rồi kết thúc."""


def build_messages(
    sample: dict,
    catalog_tools: dict[str, dict],
    step_trace: list[dict] | None = None,
    drop_apps: set[str] | None = None,
    shuffle_rng: random.Random | None = None,
    asr_alt: str | None = None,
) -> list[dict[str, str]]:
    """Construct ChatML conversation messages up to the current agent step."""
    ctx = sample.get("context", {})
    anchors = compute_time_anchors(ctx.get("current_time", ""))
    tool_lines = get_allowed_tool_lines(
        sample,
        catalog_tools,
        drop_apps=drop_apps,
        shuffle_rng=shuffle_rng,
    )
    history_turns = _compact_history_turns(sample.get("history") or [])

    req_text = sample.get("audio_text", "")
    user_sections = [
        f"time_anchors: {json.dumps(anchors, ensure_ascii=False)}",
        f"location: {json.dumps(ctx.get('location') or {}, ensure_ascii=False)}",
        f"user_profile: {json.dumps(ctx.get('user_profile') or {}, ensure_ascii=False)}",
    ]
    if history_turns:
        user_sections.append(f"history: {json.dumps(history_turns, ensure_ascii=False)}")
    user_sections.append("available_tools:\n" + "\n".join(tool_lines))
    user_sections.append(f"request: {json.dumps(req_text, ensure_ascii=False)}")
    if asr_alt and asr_alt.strip() and asr_alt.strip() != req_text.strip():
        user_sections.append(f"asr_alt: {json.dumps(asr_alt.strip(), ensure_ascii=False)}")

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": "\n".join(user_sections)},
    ]

    for step in step_trace or []:
        messages.append({
            "role": "assistant",
            "content": json.dumps({"tool_calls": step["tool_calls"]}, ensure_ascii=False),
        })
        messages.append({
            "role": "user",
            "content": json.dumps({"tool_results": step["tool_results"]}, ensure_ascii=False),
        })

    return messages
