"""Prompt Builder for Qwen2.5-3B-Instruct in Zalo AI Challenge — LifeMate.
Strictly local, offline, no external API.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

def compute_time_anchors(iso_time: str) -> dict[str, str]:
    try:
        dt = datetime.fromisoformat(iso_time)
        tz_suffix = iso_time[-6:] if ("+" in iso_time[10:] or "-" in iso_time[10:]) else "+07:00"
        return {
            "current_time": iso_time,
            "weekday_vi": WEEKDAYS_VI[dt.weekday()],
            "today_date": dt.strftime("%Y-%m-%d"),
            "tomorrow_date": (dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            "day_after_tomorrow_date": (dt + timedelta(days=2)).strftime("%Y-%m-%d"),
            "tz_offset": tz_suffix,
        }
    except Exception:
        return {"current_time": iso_time}

def filter_catalog(catalog_dict: dict, installed_apps: set[str]) -> list[dict]:
    """Return only tools belonging to apps installed in this session, plus 'sys' tools."""
    filtered_tools = []
    for app in catalog_dict.get("apps", []):
        app_name = app.get("name", "")
        if app_name in installed_apps or app_name == "sys":
            for tool in app.get("tools", []):
                filtered_tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                })
    return filtered_tools

def build_system_prompt(filtered_tools: list[dict], current_state: dict) -> str:
    tools_str = json.dumps(filtered_tools, ensure_ascii=False, indent=1)
    
    # State summary: only include relevant non-empty keys to save context tokens
    compact_state = {}
    for k, v in current_state.items():
        if v:
            compact_state[k] = v
    state_str = json.dumps(compact_state, ensure_ascii=False)

    return f"""Bạn là trợ lý AI điều khiển giọng nói LifeMate trong Zalo AI Challenge 2026.
Nhiệm vụ của bạn là phân tích câu nói của người dùng và sinh danh sách các lệnh gọi công cụ (tool_calls) phù hợp nhất.

DANH SÁCH CÔNG CỤ ĐƯỢC PHÉP DÙNG:
{tools_str}

DỮ LIỆU HIỆN TẠI TRONG THIẾT BỊ (DATABASE STATE):
{state_str}

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ 100%):
1. OUT-OF-SCOPE HOẶC HỦY BỎ:
   - Nếu câu nói hỏi về tính năng không có trong danh sách công cụ (như khách sạn, vé xem phim, visa, spa, xe máy...) -> Trả về mảng rỗng: []
   - Nếu người dùng hủy lệnh ("à thôi khỏi", "thôi không cần nữa", "cứ để nguyên đó") -> Trả về: []
2. CÂU HỎI ĐIỀU KIỆN THỜI TIẾT ("Nếu mai mưa thì X, không mưa thì thôi"):
   - Lượt đầu tiên CHỈ gọi duy nhất: `weather.get_forecast` với ngày tương ứng. KHÔNG tự tiện sửa lịch hay báo thức nếu chưa biết kết quả mưa.
3. KHÔNG ĐƯỢC TỰ BỊA ID:
   - Thao tác lịch (`calendar`): lấy `event_id` từ DATABASE STATE.
   - Thao tác hóa đơn (`bill_payment`): lấy `bill_id` từ DATABASE STATE. Tham số nguồn tiền là `source` (ví dụ "ewallet" hoặc "banking").
   - Thao tác nhắc nhở (`reminders.create`): tham số là `content` và `time`.
   - Thao tác báo thức (`alarm.create`): tham số `time` (ISO-8601 kèm múi giờ).
   - Thao tác liên hệ (`contacts.search`): nếu tìm thấy nhiều hơn 1 người trùng tên hoặc thiếu thông tin bắt buộc, gọi `sys.ask_missing_info`.
4. ĐỊNH DẠNG ĐẦU RA:
   - CHỈ trả về duy nhất một khối JSON array hợp lệ, KHÔNG giải thích gì thêm:
[
  {{"name": "tên_tool", "arguments": {{"tham_số": "giá_trị"}}}}
]"""

def build_user_prompt(sample: dict, audio_text: str) -> str:
    ctx = sample.get("context", {})
    curr_time = ctx.get("current_time", "")
    anchors = compute_time_anchors(curr_time)
    user_prof = ctx.get("user_profile") or {}
    history = sample.get("history", [])

    history_str = ""
    if history:
        turns = []
        for h in history:
            role = h.get("role", "")
            t = h.get("audio_text") or h.get("text") or ""
            calls = h.get("tool_calls", [])
            call_names = [c["name"] for c in calls] if calls else []
            turns.append(f"- {role}: {t} (tools: {call_names})")
        history_str = "\nLịch sử hội thoại gần đây:\n" + "\n".join(turns)

    return f"""Thời gian hiện tại: {curr_time} ({anchors.get('weekday_vi', '')}, hôm nay là {anchors.get('today_date', '')}, ngày mai là {anchors.get('tomorrow_date', '')})
Thông tin người dùng: {json.dumps(user_prof, ensure_ascii=False)}{history_str}

Người dùng vừa nói: "{audio_text}"

Hãy sinh JSON array các tool calls phù hợp:"""
