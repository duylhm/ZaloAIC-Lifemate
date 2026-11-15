FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt jupyterlab
COPY lifemate /code/lifemate
COPY run_inference.py evaluate.py /code/
COPY docker/predict.sh docker/start_jupyter.sh /code/
COPY docker/predict_notebook.ipynb /code/predict_notebook.ipynb
COPY training_code /code/training_code
COPY models /code/models
COPY runtime_config.json /code/runtime_config.json
RUN chmod +x /code/predict.sh /code/start_jupyter.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ENTRYPOINT ["/code/predict.sh"]
