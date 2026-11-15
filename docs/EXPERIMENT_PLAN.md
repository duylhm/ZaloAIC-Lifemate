# LifeMate rebuild decisions — 2026-11-15

## Runtime update after Kaggle feedback

The user's Qwen3.5 run loaded successfully but used fallback PyTorch DeltaNet
kernels. Ten of the first eleven displayed samples timed out (roughly 30–51 s).
Switch the next probe to Qwen3-4B with the same agent and ASR. No score or speed
improvement is yet claimed. Preserve 30 s and require a passing five-audio probe.
Qwen3.5 remains a candidate after compatible kernels and hardware are established.
FP16 is now selected for pre-Ampere CUDA hardware even if PyTorch reports emulated
BF16 support. The screenshot alone does not establish the user's GPU model.

## Original baseline design

- Qwen/Qwen3.5-4B, text-only HF class, greedy decoding, thinking disabled, NF4.
- Gipformer 1.5 68M ONNX INT8 on CPU, greedy search; transcribe latest and prior user audio.
- Installed tool schemas + app descriptions + complete public context/history.
- One call per step. One schema/format/reference repair per sample. One ToolEnv.
- Stop on finish, ask, time limit, step limit or exhausted repair.
- Same-state duplicate guard; repeated state-changing operations remain possible.
- No saved public transcripts, external inference APIs, hidden state reads or
  label-conditioned inference. TRAIN gold is an explicit diagnostics mode.

## Measurement order

1. Audio-to-JSONL smoke on 5 pilot IDs, including replay and timing.
2. 30 pilot rows (may exceed slightly to preserve connected audio groups):
   gold versus Gipformer with identical model and logic.
3. If gold is materially better, compare PhoWhisper on the same pilot. Its CPU
   fallback avoids VRAM contention but can exceed 30 seconds. Do not assume faster
   inference or better end-to-end scores from an ASR paper's WER.
4. If gold is low, inspect failures by selected tool, arguments, early finish,
   ask, schema repair and context/time limit. Change ONE prompt/model factor.
5. Confirm the selected change on the disjoint confirmation groups. Use >=5
   points and improvement across multiple examples as an investment gate for
   further work, not as a statistical guarantee or leaderboard forecast.
6. Freeze configuration, run all public IDs, verify replay, then test Docker
   offline. Reserve time based on measured mean/p95 and observed timeout rate.

## Splits and training gate

Group all current and historical audio by SHA256, including transitive sharing.
Hold out target apps bike_rental, exchange_rate, medication, package_tracking.
Add a category-balanced in-domain holdout, then divide validation groups into
pilot and confirm. Training rows remain available separately. This intentionally
overrepresents rare OOS/ask categories; report category counts and do not claim
the pilot is an unbiased leaderboard estimate. No k-fold today.

Do not reuse an old adapter trained on these validation IDs. No QLoRA is enabled
yet. If justified, use assistant-token-only loss on replay-checked TRAIN
trajectories. Any ASR augmentation must come from those same training audio IDs;
check changed numbers, names, negation and dates before retaining the original
label. Group before augmentation. No validation or test augmentation for training.
Ensemble/model soup are deferred because no independently validated checkpoints
exist and inference/time budgets are tight.

## Sources used for design

- BFCL (ICML 2025): https://proceedings.mlr.press/v267/patil25a.html
- ToolACE (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/663865ea167425c6c562cb0b6bcf76c7-Abstract-Conference.html
- Qwen text-only integration: https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Qwen checkpoint: https://huggingface.co/Qwen/Qwen3.5-4B
- Gipformer checkpoint: https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt
- Gipformer source: https://github.com/ggroup-ai-lab/gipformer
- Reference LifeMate experiments: https://github.com/Datyth/ZaloAI-Challenge-lifemate
- PhoWhisper source: https://github.com/VinAIResearch/PhoWhisper
- Organizer: https://challenge.zalo.ai/portal/lifemate

The implementation in `lifemate/` is newly written. Reference-repo scores are
not measurements of this implementation. Model revisions must predate the
competition cutoff; notebook cell 3 selects eligible immutable revisions.
