"""Offline Multi-Turn ReAct Agent with pre-call schema validation & self-correction."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Protocol

from .guardrails import parse_tool_calls, sanitize_and_repair_call
from .prompt_builder import build_messages


class ChatBackend(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class NoopBackend:
    """Zero-call fallback backend for CPU sanity checks and baseline verification."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        return '{"tool_calls": []}'


class HeuristicBackend:
    """Lightweight deterministic CPU backend for offline integration testing (non-neural)."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        # Stop on second turn once read results return.
        if len(messages) > 2:
            return '{"tool_calls": []}'

        user_msg = messages[-1]["content"]
        req_m = re.search(r"request:\s*(\".*\")", user_msg)
        req = json.loads(req_m.group(1)).lower() if req_m else ""
        avail = user_msg.split("available_tools:\n", 1)[-1].split("\nrequest:", 1)[0]

        calls: list[dict] = []
        if "thời tiết" in req or "mưa" in req:
            if "weather.get_forecast " in avail:
                anchors_m = re.search(r"time_anchors:\s*(\{.*?\})", user_msg)
                anchors = json.loads(anchors_m.group(1)) if anchors_m else {}
                date_val = anchors.get("tomorrow") if "mai" in req else anchors.get("today")
                if date_val:
                    calls.append({"name": "weather.get_forecast", "arguments": {"date": date_val}})
        elif "giá vàng" in req and "gold_price.get_price " in avail:
            calls.append({"name": "gold_price.get_price", "arguments": {}})
        elif "xem lịch" in req and "calendar.list_events " in avail:
            calls.append({"name": "calendar.list_events", "arguments": {}})

        return json.dumps({"tool_calls": calls}, ensure_ascii=False)


class HFTransformersBackend:
    """Local HuggingFace Transformers inference backend (supports 4-bit QLoRA & Qwen3 /no_think)."""

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.max_new_tokens = max_new_tokens
        tokenizer_src = adapter_path if adapter_path and local_files_only else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "torch_dtype": dtype,
        }
        if load_in_4bit and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if adapter_path:
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                local_files_only=local_files_only,
            )
        else:
            self.model = base_model
        self.model.eval()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

        raw_eos = self.tokenizer.eos_token_id
        eos_candidates = list(raw_eos) if isinstance(raw_eos, (list, tuple)) else [raw_eos]
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != self.tokenizer.unk_token_id:
            eos_candidates.append(im_end)
        self.eos_ids = list(dict.fromkeys(x for x in eos_candidates if isinstance(x, int)))

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = self._render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_ids,
            )
        gen_ids = out_ids[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


class OpenAICompatibleBackend:
    """HTTP client backend for local vLLM / OpenAI-compatible servers."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", max_tokens: int = 512) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


class LifeMateAgent:
    """Multi-turn schema-grounded ReAct agent executing calls against a fresh ToolEnv."""

    def __init__(
        self,
        backend: ChatBackend,
        catalog_tools: dict[str, dict],
        max_turns: int = 8,
        max_total_calls: int = 12,
    ) -> None:
        self.backend = backend
        self.catalog_tools = catalog_tools
        self.max_turns = max_turns
        self.max_total_calls = max_total_calls

    def run_sample(
        self,
        sample: dict,
        env: Any,
        asr_alt: str | None = None,
    ) -> tuple[list[dict], str]:
        """Run ReAct loop until termination, returning `(env.get_tool_calls(), env.get_state_hash())`."""
        installed = set((sample.get("context") or {}).get("installed_apps") or []) | {"sys"}
        step_trace: list[dict[str, list[dict]]] = []
        executed_sigs: set[tuple[str, str]] = set()
        executed_count = 0
        consecutive_empty_turns = 0

        for _ in range(self.max_turns):
            messages = build_messages(
                sample,
                self.catalog_tools,
                step_trace=step_trace,
                asr_alt=asr_alt,
            )
            raw_reply = self.backend.generate(messages)
            parsed_calls = parse_tool_calls(raw_reply)
            if not parsed_calls:
                break

            turn_calls: list[dict] = []
            turn_results: list[dict] = []
            valid_in_turn = 0
            asked_missing = False

            for raw_call in parsed_calls:
                if executed_count >= self.max_total_calls:
                    break

                name, args = sanitize_and_repair_call(
                    raw_call.get("name", ""),
                    raw_call.get("arguments"),
                    sample,
                    self.catalog_tools,
                )
                app = name.split(".")[0] if "." in name else ""
                turn_calls.append({"name": name, "arguments": args})

                if app not in installed or name not in self.catalog_tools:
                    turn_results.append({
                        "ok": False,
                        "schema_validation_error": f"Tool {name!r} is not in available_tools for this sample.",
                    })
                    continue

                sig = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if sig in executed_sigs:
                    turn_results.append({
                        "ok": False,
                        "schema_validation_error": f"Duplicate call skipped ({name}); already executed earlier.",
                    })
                    continue

                # Critical competition rule: validate BEFORE calling env.call(),
                # because rejected env.call() still gets recorded in env.get_tool_calls().
                val = env.validate(name, args)
                if not val.get("ok"):
                    turn_results.append({
                        "ok": False,
                        "schema_validation_error": val.get("error", "Invalid tool arguments"),
                    })
                    continue

                res = env.call(name, args)
                executed_sigs.add(sig)
                executed_count += 1
                valid_in_turn += 1
                turn_results.append(res)

                if name == "sys.ask_missing_info":
                    asked_missing = True
                    break

            if asked_missing or executed_count >= self.max_total_calls:
                break

            if valid_in_turn == 0:
                consecutive_empty_turns += 1
                if consecutive_empty_turns >= 2:
                    break
            else:
                consecutive_empty_turns = 0

            step_trace.append({
                "tool_calls": turn_calls,
                "tool_results": turn_results,
            })

        return env.get_tool_calls(), env.get_state_hash()
