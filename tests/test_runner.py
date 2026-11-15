import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from lifemate.data import atomic_json
from lifemate.models import NoopBackend
from lifemate.runner import parser, run

RUNTIME = '''
class ToolEnv:
    def __init__(self, split, question_id, root=None):
        self.calls = []
    def validate(self, name, args): return {"ok": True}
    def call(self, name, args):
        self.calls.append({"name":name,"arguments":args})
        return {"ok":True}
    def get_tool_calls(self): return self.calls
    def get_state_hash(self): return "fixturehash"
'''


class RunnerTests(unittest.TestCase):
    def fixture(self, d):
        base = Path(d)
        train = base / 'train'
        (train / 'zaic_tools').mkdir(parents=True)
        (train / 'zaic_tools/__init__.py').write_text(RUNTIME)
        row = {"id": "train_1", "audio_text": "ngoài phạm vi", "context": {}, "history": [], "label": {"tool_calls": []}}
        (train / 'train.jsonl').write_text(json.dumps(row), encoding='utf-8')
        atomic_json(train / 'train_tools_catalog.json', {"apps": []})
        return parser().parse_args(['--root', str(base), '--asr', 'gold', '--backend', 'noop',
                                   '--output', str(base / 'result.jsonl'), '--evaluate'])

    def test_resume_and_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            args = self.fixture(d)
            first = run(args)
            saved = (Path(d) / 'result.jsonl').read_bytes()
            class NeverGenerate(NoopBackend):
                def generate(self, messages, deadline):
                    raise AssertionError('Completed checkpoint must not generate again')
            second = run(args, backend=NeverGenerate())
            self.assertEqual(first, second)
            self.assertEqual(saved, (Path(d) / 'result.jsonl').read_bytes())
            args.seed += 1
            with self.assertRaisesRegex(ValueError, 'configuration/data/code changed'):
                run(args)

    def test_gold_disallowed_on_test(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(d)
            args.split = 'public_test'
            with self.assertRaisesRegex(ValueError, 'Gold transcripts'):
                run(args)

    def test_no_export_on_missing_audio(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            args = self.fixture(d)
            args.asr = 'gipformer'
            file = Path(d) / 'train/train.jsonl'
            row = json.loads(file.read_text())
            row['audio'] = 'missing.wav'
            file.write_text(json.dumps(row))
            class DummyASR:
                path = 'noop'
                def transcribe(self, path):
                    raise AssertionError('must check audio exists first')
            with self.assertRaises(FileNotFoundError):
                run(args, asr=DummyASR())
            self.assertFalse(Path(args.output).exists())
