import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

from lifemate.agent import Agent, parse_response
from lifemate.asr import hydrate
from lifemate.data import public_input
from lifemate.metrics import calls_match, matched_count, verify
from lifemate.prompt import build_messages
from lifemate.splits import groups, make_splits


CATALOG = {"apps": [{"id": "app", "tools": [
    {"name": "app.list", "kind": "read", "parameters": {"properties": {}}},
    {"name": "app.set", "kind": "write", "parameters": {"required": ["item_id"],
        "properties": {"item_id": {"type": "string", "x-ref": "app.items"}}}},
    {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"properties": {"tool_name": {}, "params": {}}}},
]}]}
SAMPLE = {"id": "x", "text": "đổi cái đó", "context": {"installed_apps": {"app": {"tools": ["app.list", "app.set"]}}}, "history": []}


class FakeEnv:
    def __init__(self):
        self.calls = []
    def validate(self, name, args):
        return {"ok": name != "app.set" or bool(args.get("item_id")), "error": "missing item_id"}
    def call(self, name, args):
        self.calls.append({"name": name, "arguments": args})
        return {"ok": True, "items": [{"id": "i1"}]}
    def get_tool_calls(self):
        return self.calls
    def get_state_hash(self):
        return "hash"


class Scripted:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []
    def generate(self, messages, deadline):
        self.messages = copy.deepcopy(messages)
        return next(self.outputs)


def call(name, args=None):
    return json.dumps({"tool_calls": [{"name": name, "arguments": args or {}}]})


class CoreTests(unittest.TestCase):
    def test_garbage_is_not_silent_done(self):
        for bad in ('hello', '{}', '{"tool_calls":null}', '{"tool_calls":[{},{}]}'):
            with self.assertRaises(ValueError):
                parse_response(bad)
        self.assertEqual(parse_response('```json\n{"tool_calls":[]}\n```'), [])

    def test_dependency_observation_before_write(self):
        b = Scripted([call("app.list"), call("app.set", {"item_id": "i1"}), '{"tool_calls":[]}'])
        env = FakeEnv()
        result, trace = Agent(b, CATALOG).run(SAMPLE, env, time.monotonic() + 10)
        self.assertEqual([c["name"] for c in result["tool_calls"]], ["app.list", "app.set"])
        self.assertIn("tool_result", b.messages[-1]["content"])
        self.assertEqual(trace["stop"], "done")

    def test_invalid_call_repairs_without_execution(self):
        b = Scripted([call("app.set"), call("app.list"), '{"tool_calls":[]}'])
        env = FakeEnv()
        _, trace = Agent(b, CATALOG).run(SAMPLE, env, time.monotonic() + 10)
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(trace["repairs"], 1)

    def test_unseen_id_and_second_invalid_stop(self):
        b = Scripted([call("app.set", {"item_id": "invented"}), call("app.set", {"item_id": "another"})])
        env = FakeEnv()
        _, trace = Agent(b, CATALOG).run(SAMPLE, env, time.monotonic() + 10)
        self.assertEqual(env.calls, [])
        self.assertEqual(trace["stop"], "repair_exhausted")

    def test_history_not_replayed(self):
        sample = copy.deepcopy(SAMPLE)
        sample["history"] = [{"role": "assistant", "tool_calls": [{"name": "app.list", "arguments": {}}],
                               "tool_results": [{"items": [{"id": "i1"}]}]}]
        b = Scripted([call("app.set", {"item_id": "i1"}), '{"tool_calls":[]}'])
        env = FakeEnv()
        Agent(b, CATALOG).run(sample, env, time.monotonic() + 10)
        self.assertEqual([c["name"] for c in env.calls], ["app.set"])

    def test_duplicate_call_does_not_execute_twice(self):
        b = Scripted([call("app.list"), call("app.list"), '{"tool_calls":[]}'])
        env = FakeEnv()
        Agent(b, CATALOG).run(SAMPLE, env, time.monotonic() + 10)
        self.assertEqual(len(env.calls), 1)

    def test_repeated_write_allowed_after_state_change(self):
        class ChangingEnv(FakeEnv):
            def get_state_hash(self):
                return str(len(self.calls))
        sample = copy.deepcopy(SAMPLE)
        sample["text"] = "đổi hai lần cho i1"
        b = Scripted([call("app.set", {"item_id": "i1"}), call("app.set", {"item_id": "i1"}), '{"tool_calls":[]}'])
        env = ChangingEnv()
        Agent(b, CATALOG).run(sample, env, time.monotonic() + 10)
        self.assertEqual(len(env.calls), 2)

    def test_deadline_prevents_call(self):
        env = FakeEnv()
        _, trace = Agent(Scripted([]), CATALOG).run(SAMPLE, env, time.monotonic() - 1)
        self.assertEqual(trace["stop"], "timeout")

    def test_prompt_has_no_label_or_future_state(self):
        row = {**SAMPLE, "label": {"secret": "GOLD_SENTINEL"}, "final_state": {"hidden": "STATE_SENTINEL"}}
        sample = public_input(row)
        sample["text"] = "request"
        prompt = json.dumps(build_messages(sample, CATALOG))
        self.assertNotIn("GOLD_SENTINEL", prompt)
        self.assertNotIn("STATE_SENTINEL", prompt)

    def test_live_asr_ignores_gold_current_and_history(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "train"
            base.mkdir()
            (base / "current.wav").touch()
            (base / "history.wav").touch()
            row = {**SAMPLE, "audio": "current.wav", "audio_text": "SECRET",
                   "history": [{"role": "user", "audio": "history.wav", "audio_text": "HISTORY_SECRET"}]}
            class ASR:
                def transcribe(self, path):
                    return path.stem
            sample = hydrate(row, d, "train", "gipformer", ASR(), time.monotonic() + 10)
            self.assertEqual(sample["text"], "current")
            self.assertEqual(sample["history"][0]["text"], "history")
            self.assertNotIn("SECRET", json.dumps(sample))

    def test_audio_groups_transitive(self):
        rows = [{"id": "1", "audio": "a", "history": [{"audio": "b"}]},
                {"id": "2", "audio": "b", "history": [{"audio": "c"}]}, {"id": "3", "audio": "c"}]
        self.assertEqual(len(groups(rows)), 1)

    def test_split_disjoint_and_reproducible(self):
        rows = [{"id": str(i), "audio": str(i), "label": {"tool_calls": []}} for i in range(40)]
        rows[1]["audio"] = rows[0]["audio"]
        parts = make_splits(rows, pilot_size=3, indomain_size=12)
        self.assertEqual(parts, make_splits(rows, pilot_size=3, indomain_size=12))
        sets = [set(r["audio"] for r in split) for split in parts.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_matching_one_to_one(self):
        c = {"name": "app.list", "arguments": {}}
        self.assertEqual(matched_count([c, c], [c], {}), 1)
        self.assertFalse(calls_match({"name": "x", "arguments": {"x": True}}, {"name": "x", "arguments": {"x": 1}}, {}))


if __name__ == "__main__":
    unittest.main()
