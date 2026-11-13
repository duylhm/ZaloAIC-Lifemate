"""Offline unit & integration test suite for the LifeMate ReAct + SFT pipeline."""

from __future__ import annotations

import ast
import json
import random
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent import HeuristicBackend, LifeMateAgent, NoopBackend  # noqa: E402
from src.asr import hydrate_sample_transcripts, load_asr_cache  # noqa: E402
from src.data_utils import read_jsonl, resolve_data_root  # noqa: E402
from src.guardrails import parse_tool_calls, sanitize_and_repair_call  # noqa: E402
from src.prompt_builder import build_messages, load_catalog_dict  # noqa: E402
from src.trajectory import (  # noqa: E402
    group_trajectory_into_steps,
    inject_safe_phonetic_noise,
    sample_dropout_apps,
    sample_to_chatml,
    tokenize_chatml_with_masking,
)


def _try_data_root() -> Path | None:
    try:
        return resolve_data_root(None, split="train")
    except FileNotFoundError:
        return None


class _MockChatMLTokenizer:
    """Minimal byte-level ChatML tokenizer simulating Qwen2.5 & Qwen3 templates."""

    def __init__(self, with_qwen3_think: bool = False) -> None:
        self.with_qwen3_think = with_qwen3_think
        self.unk_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self._special = {
            "<|im_start|>": 1000,
            "<|im_end|>": 1001,
            "<think>": 1002,
            "</think>": 1003,
        }

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._special.get(tok, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [b + 10 for b in text.encode("utf-8")]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        for idx, msg in enumerate(messages):
            ids.append(self._special["<|im_start|>"])
            ids.extend(self.encode(f"{msg['role']}\n"))
            if msg["role"] == "assistant" and self.with_qwen3_think and not enable_thinking:
                ids.extend([self._special["<think>"], *self.encode("\n\n"), self._special["</think>"], *self.encode("\n\n")])
            ids.extend(self.encode(msg["content"]))
            ids.append(self._special["<|im_end|>"])
            ids.extend(self.encode("\n"))
        if add_generation_prompt:
            ids.append(self._special["<|im_start|>"])
            ids.extend(self.encode("assistant\n"))
            if self.with_qwen3_think and not enable_thinking:
                ids.extend([self._special["<think>"], *self.encode("\n\n"), self._special["</think>"], *self.encode("\n\n")])
        return ids


class PipelineTests(unittest.TestCase):
    def test_no_state_json_leak_and_python310_fstring_safety(self) -> None:
        py_files = [
            *REPO_ROOT.glob("*.py"),
            *(REPO_ROOT / "src").glob("*.py"),
            *(REPO_ROOT / "scripts").glob("*.py"),
        ]
        self.assertFalse((REPO_ROOT / "src" / "prompts.py").exists(), "Legacy state-leaking prompts.py must be deleted")

        for pf in py_files:
            src = pf.read_text(encoding="utf-8")
            ast.parse(src, filename=str(pf))
            if pf.name not in {"evaluate.py"} and pf.parent == REPO_ROOT:
                self.assertNotIn("state.json", src, f"Unauthorized state.json reference in {pf}")
            if pf.parent.name == "src":
                self.assertNotIn("state.json", src, f"Unauthorized state.json reference in {pf}")

            # Ensure Python 3.10/3.11 compatible f-strings (no backslashes inside {...}).
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.JoinedStr):
                    for val in node.values:
                        if isinstance(val, ast.FormattedValue):
                            seg = ast.get_source_segment(src, val) or ""
                            self.assertNotIn("\\", seg, f"Backslash in f-string expression in {pf}: {seg}")

    def test_guardrails_parsing_and_coercion(self) -> None:
        raw = "<think>reasoning here</think>\n```json\n{\"tool_calls\": [{\"name\": \"weather.get_forecast\", \"arguments\": {\"location\": \"TP.HCM\", \"date\": \"29/11/2026\"}}]}\n```"
        calls = parse_tool_calls(raw)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "weather.get_forecast")

        mock_catalog = {
            "weather.get_forecast": {
                "name": "weather.get_forecast",
                "kind": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                    },
                },
            },
            "ewallet.transfer": {
                "name": "ewallet.transfer",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "contact_id": {"type": "string"},
                        "amount": {"type": "integer"},
                    },
                },
            },
            "alarm.create": {
                "name": "alarm.create",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string", "description": "HH:MM"},
                        "repeat": {"type": "string"},
                    },
                },
            },
        }
        sample = {"context": {"current_time": "2026-11-28T08:00:00+07:00", "user_profile": {"city": "TP.HCM"}, "installed_apps": ["weather", "ewallet", "alarm"]}}

        # Drops default city location & converts DD/MM/YYYY -> YYYY-MM-DD.
        name, args = sanitize_and_repair_call("weather.get_forecast", calls[0]["arguments"], sample, mock_catalog)
        self.assertEqual(name, "weather.get_forecast")
        self.assertEqual(args, {"date": "2026-11-29"})

        # Coerces Vietnamese thousands & drops hallucinated parameter keys.
        _, t_args = sanitize_and_repair_call(
            "ewallet.transfer",
            {"contact_id": "c_01", "amount": "1.200.000 VND", "fake_field": "x"},
            sample,
            mock_catalog,
        )
        self.assertEqual(t_args, {"contact_id": "c_01", "amount": 1200000})

        # Normalizes alarm time and strips default repeat="once".
        _, a_args = sanitize_and_repair_call(
            "alarm.create",
            {"time": "7h05", "repeat": "once"},
            sample,
            mock_catalog,
        )
        self.assertEqual(a_args, {"time": "07:05"})

    def test_tokenize_chatml_masking_qwen25_and_qwen3(self) -> None:
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "REQ"},
            {"role": "assistant", "content": '{"tool_calls": [{"name": "a.b", "arguments": {}}]}'},
            {"role": "user", "content": '{"tool_results": [{"ok": true}]}'},
            {"role": "assistant", "content": '{"tool_calls": []}'},
        ]
        for q3 in (False, True):
            tok = _MockChatMLTokenizer(with_qwen3_think=q3)
            out = tokenize_chatml_with_masking(tok, msgs, max_length=2048)
            ids = out["input_ids"]
            labels = out["labels"]
            self.assertEqual(len(ids), len(labels))
            unmasked_bytes = bytes([x - 10 for x in labels if x not in (-100, 1001)]).decode("utf-8")
            self.assertEqual(unmasked_bytes, msgs[2]["content"] + msgs[4]["content"])
            # Closing <|im_end|> (1001) on BOTH assistant turns must be supervised.
            self.assertEqual(sum(1 for x in labels if x == 1001), 2)

    def test_prompt_builder_drop_apps_and_asr_alt(self) -> None:
        sample = {
            "id": "s_mock",
            "audio_text": "Kiểm tra thời tiết hôm nay",
            "context": {
                "current_time": "2026-11-28T08:00:00+07:00",
                "location": {"city": "TP.HCM"},
                "user_profile": {"city": "TP.HCM"},
                "installed_apps": ["weather", "stocks"],
            },
            "history": [],
        }
        cat = {
            "weather.get_forecast": {"name": "weather.get_forecast", "app": "weather", "kind": "read", "parameters": {"properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}}, "required": ["date"]}},
            "stocks.get_quote": {"name": "stocks.get_quote", "app": "stocks", "kind": "read", "parameters": {"properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
            "sys.ask_missing_info": {"name": "sys.ask_missing_info", "app": "sys", "kind": "meta", "parameters": {"properties": {"tool_name": {"type": "string"}, "params": {"type": "array"}}, "required": ["tool_name", "params"]}},
        }
        msgs = build_messages(sample, cat, drop_apps={"stocks"}, asr_alt="Kiểm tra thợ tiết hôm nay")
        user_txt = msgs[1]["content"]
        self.assertIn("weather.get_forecast", user_txt)
        self.assertNotIn("stocks.get_quote", user_txt)
        self.assertIn("asr_alt:", user_txt)
        self.assertIn("date*:string|date", user_txt)
        # Also verify real dict format for installed_apps
        sample_dict = dict(sample)
        sample_dict["context"] = dict(sample["context"], installed_apps={
            "weather": {"tools": ["weather.get_forecast"]},
            "stocks": {"tools": ["stocks.get_quote"]},
        })
        msgs2 = build_messages(sample_dict, cat, drop_apps={"stocks"})
        self.assertIn("weather.get_forecast", msgs2[1]["content"])
        self.assertNotIn("stocks.get_quote", msgs2[1]["content"])
        self.assertEqual(NoopBackend().generate(msgs), '{"tool_calls": []}')

    def test_notebook_valid_json(self) -> None:
        nb_path = REPO_ROOT / "notebooks" / "LifeMate_Colab_Inference.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertGreaterEqual(len(nb.get("cells", [])), 5)


class DatasetDependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = _try_data_root()
        if cls.data_root is not None:
            sys.path.insert(0, str(cls.data_root / "train"))
            import zaic_tools

            cls.zaic_tools = zaic_tools
            cls.train_cat = load_catalog_dict(cls.data_root / "train" / "train_tools_catalog.json")
            cls.train_rows = read_jsonl(cls.data_root / "train" / "train.jsonl")
            cls.pub_cat = load_catalog_dict(cls.data_root / "public_test" / "public_test_tools_catalog.json")
            cls.pub_rows = read_jsonl(cls.data_root / "public_test" / "public_test.jsonl")

    def test_prompt_dropout_and_chatml_all_1545_train(self) -> None:
        if self.data_root is None:
            self.skipTest("Local dataset not present")

        rng = random.Random(42)
        for sample in self.train_rows:
            drop = sample_dropout_apps(sample, rng, max_drop_ratio=0.7)
            msgs = sample_to_chatml(sample, self.train_cat, drop_apps=drop, shuffle_rng=rng)
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[1]["role"], "user")
            self.assertEqual(msgs[-1]["role"], "assistant")

            steps = group_trajectory_into_steps(sample, self.train_cat)
            flat_gold = [tc for st in steps for tc in st["tool_calls"]]
            flat_asst = []
            for m in msgs:
                if m["role"] == "assistant":
                    flat_asst.extend(json.loads(m["content"])["tool_calls"])
            self.assertEqual(flat_asst, flat_gold)

            # Assert ask_missing_info ends immediately with no trailing empty [] turn.
            if any(c["name"] == "sys.ask_missing_info" for c in flat_gold):
                self.assertIn("sys.ask_missing_info", msgs[-1]["content"])

            noisy = inject_safe_phonetic_noise(sample, rng, prob=0.5)
            self.assertIn("audio_text", noisy)

    def test_agent_validate_self_correction_and_no_pollution(self) -> None:
        if self.data_root is None:
            self.skipTest("Local dataset not present")

        sample = next(r for r in self.train_rows if "weather" in (r["context"].get("installed_apps") or []))
        env = self.zaic_tools.ToolEnv("train", sample["id"], root=self.data_root)

        class _SelfCorrectingMockBackend:
            def __init__(self) -> None:
                self.turn = 0
                self.seen_error = False

            def generate(self, messages: list[dict[str, str]]) -> str:
                self.turn += 1
                if self.turn == 1:
                    # Emit one missing-required call (rejected by env.validate) + one disallowed app tool
                    return json.dumps({
                        "tool_calls": [
                            {"name": "weather.get_forecast", "arguments": {}},
                            {"name": "nonexistent_app.foo", "arguments": {}},
                        ]
                    })
                if self.turn == 2:
                    self.seen_error = "schema_validation_error" in messages[-1]["content"]
                    return json.dumps({
                        "tool_calls": [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-21"}}]
                    })
                return '{"tool_calls": []}'

        backend = _SelfCorrectingMockBackend()
        agent = LifeMateAgent(backend=backend, catalog_tools=self.train_cat)
        calls, state_h = agent.run_sample(sample, env)
        self.assertTrue(backend.seen_error)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], {"name": "weather.get_forecast", "arguments": {"date": "2026-11-21"}})
        self.assertTrue(isinstance(state_h, str) and len(state_h) > 0)

    def test_splits_integrity_no_leakage(self) -> None:
        if self.data_root is None:
            self.skipTest("Local dataset not present")

        sp = json.loads((REPO_ROOT / "splits" / "val_all.json").read_text(encoding="utf-8"))
        train_ids = set(sp["train_ids"])
        val_unseen = set(sp["val_unseen_ids"])
        val_indomain = set(sp["val_indomain_ids"])
        holdout = set(sp["holdout_apps"])

        self.assertFalse(train_ids & val_unseen)
        self.assertFalse(train_ids & val_indomain)
        self.assertFalse(val_unseen & val_indomain)
        self.assertEqual(len(train_ids) + len(val_unseen) + len(val_indomain), len(self.train_rows))

        for r in self.train_rows:
            if r["id"] in train_ids:
                gold_apps = {c["name"].split(".")[0] for c in r["label"]["tool_calls"]}
                self.assertFalse(gold_apps & holdout)

    def test_public_test_asr_hydration_and_heuristic_schema_valid(self) -> None:
        if self.data_root is None:
            self.skipTest("Local dataset not present")

        pho_cache = load_asr_cache(REPO_ROOT / "asr_public_test_pho.json")
        alt_cache = load_asr_cache(REPO_ROOT / "asr_public_test.json")
        agent = LifeMateAgent(backend=HeuristicBackend(), catalog_tools=self.pub_cat)

        for r in self.pub_rows[:15]:
            hyd, asr_alt = hydrate_sample_transcripts(
                r,
                split_dir=self.data_root / "public_test",
                split_name="public_test",
                primary_cache=pho_cache,
                alt_cache=alt_cache,
            )
            self.assertTrue(hyd["audio_text"])
            for t in hyd.get("history") or []:
                if t.get("role") == "user":
                    self.assertTrue(t.get("audio_text"), f"Missing history ASR on {r['id']}")

            env = self.zaic_tools.ToolEnv("public_test", r["id"], root=self.data_root)
            calls, _ = agent.run_sample(hyd, env, asr_alt=asr_alt)
            for c in calls:
                v = env.validate(c["name"], c.get("arguments") or {})
                self.assertTrue(v.get("ok"), f"Invalid call generated: {c} -> {v}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
