import unittest
from types import SimpleNamespace
from lifemate.health import abort_slow_train, require_smoke
from lifemate.models import compute_dtype


class HealthTests(unittest.TestCase):
    def test_timeout_smoke_blocks_pilot(self):
        with self.assertRaisesRegex(ValueError, "Smoke failed"):
            require_smoke({"n": 5, "stops": {"timeout": 4}}, {"config": {}}, {})

    def test_changed_model_blocks_stale_smoke(self):
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            require_smoke({"n": 5}, {"config": {"model": "old"}}, {"model": "new"})

    def test_successful_smoke_allows_pilot(self):
        require_smoke({"n": 5, "stops": {"done": 5}}, {"config": {"model": "new"}}, {"model": "new"})

    def test_abort_train_but_never_partial_public(self):
        traces = [{"stop": "timeout"}] * 3 + [{"stop": "done"}] * 2
        with self.assertRaisesRegex(RuntimeError, "Stopped early"):
            abort_slow_train("train", traces)
        abort_slow_train("public_test", traces)
        abort_slow_train("train", traces[:4])

    def test_pre_ampere_does_not_use_emulated_bfloat16(self):
        for major, expected in ((6, "fp16"), (7, "fp16"), (8, "bf16")):
            torch = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda _: (major, 0),
                                                        is_bf16_supported=lambda: True),
                                    float16="fp16", bfloat16="bf16")
            self.assertEqual(compute_dtype(torch), expected)
