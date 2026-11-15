# Training status

This submission candidate is a zero-shot baseline. No LifeMate labels have been
used to update model weights. Inference uses public Qwen and ASR checkpoints,
with immutable revisions recorded in `models/pins.json` during export.

TRAIN labels are used only for local evaluation, split construction and oracle
plumbing tests. Public/private test inputs are never labelled for training.

One short QLoRA experiment is conditional on measured agent errors with gold
transcripts, a material gain on the independent confirmation set, and remaining
runtime budget. It has NOT been run or selected. No training result is claimed.
Cross-validation, synthetic app augmentation, ensemble and model soup are not
part of this baseline.
