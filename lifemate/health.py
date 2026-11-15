"""Stop expensive benchmarks when the small end-to-end probe already failed."""


def require_smoke(summary, manifest, expected_config):
    config = manifest.get("config", {})
    mismatch = [key for key, value in expected_config.items() if config.get(key) != value]
    if mismatch:
        raise ValueError("Smoke configuration differs: " + ", ".join(mismatch) + ". Run cell 4 with the current settings.")
    if summary.get("n", 0) < 5:
        raise ValueError("Run the five-sample audio smoke in cell 4 before continuing.")
    stops = summary.get("stops", {})
    if summary.get("timeouts", 0) or stops.get("timeout", 0) or stops.get("input_or_context_error", 0):
        raise ValueError("Smoke failed: timeout/context errors. Stop and fix runtime before pilot/public; do not increase the 30-second limit.")


def abort_slow_train(split, traces):
    # Apply on resume too: saved failed records must not consume another full pilot.
    if split == "train" and len(traces) >= 5 and sum(t.get("stop") == "timeout" for t in traces[:5]) >= 3:
        raise RuntimeError("Stopped early: at least 3 of the first 5 TRAIN samples timed out. "
                           "Check GPU/kernels or try Qwen/Qwen3-4B with a new run name. "
                           "Partial checkpoints are saved; no complete submission was exported.")
