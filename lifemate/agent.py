"""One execution session per sample; one repair attempt per sample."""
from __future__ import annotations

import json
import re
import time

from .prompt import allowed_tools, build_messages, compact


def parse_response(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"tool_calls"} or not isinstance(value["tool_calls"], list):
        raise ValueError('Expected {"tool_calls": [...]}')
    calls = value["tool_calls"]
    if len(calls) > 1:
        raise ValueError("Return at most ONE call; wait for its result before the next call")
    if calls:
        call = calls[0]
        if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict)):
            raise ValueError("Each call needs a string name and an arguments object")
    return calls


def scalar_strings(value):
    if isinstance(value, dict):
        return set().union(*(scalar_strings(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(scalar_strings(v) for v in value)) if value else set()
    return {value} if isinstance(value, str) else set()


def unseen_references(args, schema, known, request_text):
    """Check declared foreign keys, including object/array fields; never inspect hidden state."""
    errors = []
    if schema.get("x-ref"):
        values = args if isinstance(args, list) else [args]
        for value in values:
            if isinstance(value, str) and value not in known and not re.search(
                    r"(?<!\w)" + re.escape(value) + r"(?!\w)", request_text):
                errors.append(value)
    if isinstance(args, dict):
        for key, value in args.items():
            errors.extend(unseen_references(value, schema.get("properties", {}).get(key, {}), known, request_text))
    elif isinstance(args, list):
        for value in args:
            errors.extend(unseen_references(value, schema.get("items", {}), known, request_text))
    return errors


class Agent:
    def __init__(self, backend, catalog, max_steps=16):
        self.backend, self.catalog, self.max_steps = backend, catalog, max_steps

    def run(self, sample, env, deadline):
        messages = build_messages(sample, self.catalog)
        allowed = allowed_tools(sample, self.catalog)
        known = scalar_strings(sample.get("history", [])) | scalar_strings(sample["context"])
        request_text = sample["text"] + "\n" + "\n".join(t.get("text", "") for t in sample.get("history", []))
        executed = set()
        trace, repairs = [], 0
        stop = "max_steps"
        for step in range(self.max_steps):
            if time.monotonic() >= deadline:
                stop = "timeout"
                break
            raw = self.backend.generate(messages, deadline)
            messages.append({"role": "assistant", "content": raw})
            entry = {"step": step, "response": raw}
            trace.append(entry)
            if time.monotonic() >= deadline:
                stop = "timeout"
                break
            try:
                calls = parse_response(raw)
                if not calls:
                    stop = "done"
                    break
                call = calls[0]
                name, args = call["name"], call["arguments"]
                if name not in allowed:
                    raise ValueError("Tool is not installed: " + name)
                extra = set(args) - set(allowed[name].get("parameters", {}).get("properties", {}))
                if extra:
                    raise ValueError("Unknown arguments: " + ", ".join(sorted(extra)))
                validation = env.validate(name, args)
                if not validation["ok"]:
                    raise ValueError(str(validation.get("error")))
                unseen = unseen_references(args, allowed[name].get("parameters", {}), known, request_text)
                if unseen:
                    raise ValueError("Look up these IDs before using them: " + ", ".join(unseen))
                # A changed state can justify the same call again (e.g. redeem TWO vouchers).
                # Repeated reads/no-op writes in the same state are a loop.
                signature = (json.dumps(call, ensure_ascii=False, sort_keys=True), env.get_state_hash())
                if signature in executed:
                    raise ValueError("This exact call already ran. Use its result; do not repeat it")
            except (ValueError, TypeError) as error:
                entry["rejected"] = str(error)
                if repairs >= 1:
                    stop = "repair_exhausted"
                    break
                repairs += 1
                messages.append({"role": "user", "content": compact({"validation_error": str(error),
                    "executed": False, "instruction": "Correct the next action once; prior successful calls remain executed."})})
                continue
            result = env.call(name, args)
            executed.add(signature)
            known.update(scalar_strings(result))
            entry.update(call=call, result=result)
            messages.append({"role": "user", "content": compact({"tool": name, "tool_result": result})})
            if name == "sys.ask_missing_info":
                stop = "asked"
                break
        return {"id": sample["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}, {
            "stop": stop, "repairs": repairs, "steps": trace,
        }
