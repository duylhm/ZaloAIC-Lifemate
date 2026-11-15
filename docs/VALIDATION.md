# Validation performed locally — 2026-11-15

## v2.1 runtime fix

The supplied Kaggle log confirms Qwen3.5 weights load, but shows fallback
causal-convolution/DeltaNet kernels and 10/11 displayed rows timing out.
New tests cover native dtype selection on pre-Ampere/Ampere, failed/stale smoke
rejection, and early TRAIN timeout termination without truncating public output.
Qwen3-4B is now the next-probe default. Its end-to-end speed and scores still
need to be measured on Kaggle; local CPU tests cannot establish those results.

## What was actually run

- Unit/integration tests: JSON parsing, repair without execution, reference
  checks, history reuse, same-state loop prevention, legitimate repeated writes,
  deadline handling, label exclusion, live-ASR routing for current/history
  audio, grouped splits, ZIP/folder bootstrap, ZIP path escape rejection,
  resumable checkpoints, config mismatch refusal and missing-audio failure.
- CPU oracle replay: **1,545 TRAIN rows**, using the organizer's real ToolEnv.
  **100% final-state match**, **99.9871 local score**. The oracle supplies gold
  calls; this is NOT an LLM, ASR or leaderboard result.
- One mismatch remains: `train_1696` requests `chat_6142` in its gold calls,
  although that ID is absent from the current request/history and the preceding
  `chat_ott.list_chats` result. The reference guard rejects that call. Replaying
  the gold call does not change state. This is retained as a label/reference
  discrepancy; no inference exception is keyed to that sample ID.
- TRAIN JSONL matches the original local competition ZIP by SHA256:
  `24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8`.
- Group splitting checksummed **5,036 actual audio files**, including history.
  Result: **1,340 training / 30 pilot / 175 confirm** rows. Sizes differ from
  the previous notebook's splits; the notebook prints actual counts.
- CPU noop pilot exercised the full runner, JSONL export, schema/replay
  verification and evaluation. Its 13.33 score reflects four OOS cases among
  30 samples, NOT a model baseline.
- All generated notebook code cells were compiled. Both source ZIP and
  extracted-folder staging were tested. `Qwen3_5ForCausalLM` imports successfully
  with the locally installed Transformers 5.19.0.

## Still unverified

- Qwen3.5-4B weight loading, 4-bit generation and memory use on Kaggle GPU.
- Gipformer/PhoWhisper decoding and measured ASR accuracy. Local dependency
  download was blocked by network/DNS restrictions, so live-ASR smoke must run
  in notebook cell 4. Model import is not an inference test.
- Gold-versus-ASR pilot and confirmation scores; no new public submission exists.
- 30-second performance on scoring hardware. Generation/ASR deadlines are
  cooperative; overruns are recorded and score zero in local timed evaluation.
- Docker build and offline GPU run. The local Docker daemon is unavailable.

The previous implementation and uncommitted files are backed up. Creating a
new Git branch failed because this session cannot write `.git`; no commit or
push has been performed.
