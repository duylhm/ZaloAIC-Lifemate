"""Offline Agent for LifeMate using Qwen/Qwen2.5-3B-Instruct.
Strictly offline, runs on GPU, complies with ZAC Docker specifications.
"""
from __future__ import annotations

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompts import filter_catalog, build_system_prompt, build_user_prompt

class LifeMateQwenAgent:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str | None = None,
        load_in_4bit: bool = False
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[Agent] Loading {model_name} on {self.device} (4bit={load_in_4bit})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        load_kwargs = {}
        if "cuda" in self.device:
            load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs["device_map"] = "auto"
            if load_in_4bit:
                load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        self.model.eval()
        print(f"[Agent] Qwen model loaded successfully.")

    def parse_json_tool_calls(self, raw_output: str) -> list[dict]:
        """Safely parse JSON array of tool calls with regex fallbacks."""
        text = raw_output.strip()
        # Look for markdown json code block
        m_code = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if m_code:
            text = m_code.group(1).strip()
        else:
            m_arr = re.search(r"(\[[\s\S]*\])", text)
            if m_arr:
                text = m_arr.group(1).strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                valid_calls = []
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        valid_calls.append({
                            "name": str(item["name"]).strip(),
                            "arguments": item.get("arguments") or {}
                        })
                return valid_calls
        except Exception:
            pass

        return []

    def predict_sample(
        self,
        sample: dict,
        audio_text: str,
        env,
        catalog_dict: dict,
        max_new_tokens: int = 512
    ) -> tuple[list[dict], str]:
        """Predict tool calls for a sample, replay into ToolEnv, and return (tool_calls, state_hash)."""
        # 1. Quick heuristic guardrail for out-of-scope / cancellation
        t_low = audio_text.lower()
        oos_words = [
            "khách sạn", "hotel", "homestay", "resort", "hộ chiếu", "visa",
            "bảo hiểm", "karaoke", "rửa xe", "spa", "cắt tóc", "massage", "thuê xe máy"
        ]
        if any(w in t_low for w in oos_words):
            return [], env.get_state_hash()

        if any(w in t_low for w in ["à thôi khỏi", "thôi khỏi, không cần", "cứ để đó", "không cần nữa"]):
            if not any(w in t_low for w in ["chỉ cần xem", "xem giúp", "dự báo"]):
                return [], env.get_state_hash()

        # 2. Build filtered catalog and prompts
        installed = set(sample.get("context", {}).get("installed_apps", {}).keys())
        filtered_tools = filter_catalog(catalog_dict, installed)
        system_p = build_system_prompt(filtered_tools, env.get_state())
        user_p = build_user_prompt(sample, audio_text)

        messages = [
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p}
        ]

        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Deterministic greedy decoding for reproducibility
                pad_token_id=self.tokenizer.eos_token_id
            )
            # Slice prompt ids
            gen_ids_only = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(gen_ids_only, skip_special_tokens=True)[0]

        # 3. Parse predicted tool calls
        predicted_calls = self.parse_json_tool_calls(response)

        # 4. Safe Replay into ToolEnv
        for call in predicted_calls:
            t_name = call.get("name", "")
            t_args = call.get("arguments") or {}
            val_res = env.validate(t_name, t_args)
            if val_res.get("ok"):
                env.call(t_name, t_args)
            else:
                # Discard invalid call to preserve state integrity
                pass

        return env.get_tool_calls(), env.get_state_hash()
