# Offline deployment candidate

Status: entrypoints and packaging prepared; GPU execution and Docker build must
be measured on the deployment host. This local machine has no CUDA GPU and the
Docker daemon is not running. Do not treat this as a tested final image.

1. Run Kaggle notebook cell 8 after selecting a configuration. It writes a
   self-contained `lifemate-v2-deploy/` build context with `models/llm`,
   `models/asr`, `models/pins.json` and `runtime_config.json`.
2. Download that folder. Build with network enabled:

   ```bash
   docker build -f docker/Dockerfile -t lifemate-v2 .
   ```

3. Test with networking disabled and one split under the input root:

   ```bash
   docker run --rm --gpus all --network none \
     -v /absolute/data:/data:ro -v /absolute/result:/result \
     lifemate-v2
   ```

   If the root contains multiple splits, set `-e LIFEMATE_SPLIT=public_test`.
   Runtime imports `/data/<split>/zaic_tools`; no organizer package or dataset
   is baked into the image. `/code/predict.sh` takes no arguments and writes
   `/result/submission.jsonl`. Time is measured in milliseconds before each
   sample's environment initialization and audio reads. Model loading is excluded.

4. Check exact ID coverage and replay hashes using `evaluate.py` with the same
   split. Review `*.summary.json` for maximum time and timeout count. Confirm
   output equivalence to Kaggle; this has not yet been established.

5. Jupyter: override entrypoint with `/code/start_jupyter.sh`, map port 8888,
   open `predict_notebook.ipynb`. It writes `/result/jupyter_submission.jsonl`.

The 30-second generation limit is cooperative: a CUDA prefill/kernel or one ASR
call can overrun it. Overruns are recorded as timeouts and count as zero in local
timed evaluation. Measure actual A30 performance before finalizing; the code
does not claim an enforced operating-system wall-clock cutoff.

Use the organizer's current Docker instructions as final authority for image
upload and base-image compatibility. This folder does not upload anything.
