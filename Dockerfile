# TensorFlow 2.13 requires CUDA 11.8, cuDNN 8 and Python <= 3.11
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        gcc \
        g++ \
        libffi-dev \
        musl-dev \
        git \
        python3.10 python3-pip && \
    apt-get clean

ENV PYTHONIOENCODING=utf-8

WORKDIR /app

RUN adduser --uid 1000 --disabled-password --gecos "app" app && \
    chown -R app:app /app
USER 1000

ENV PATH="/home/app/.local/bin:${PATH}"

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

COPY --chown=app:app . .

# Include the default vocoder in the image so that it is not downloaded at startup
RUN python3 -c "from speechbrain.inference.vocoders import HIFIGAN; from tts_worker.config import tts_config; HIFIGAN.from_hparams(source=tts_config.SPEECHBRAIN_HIFIGAN_MODEL)"

# oneDNN mishandles the empty Conv1D input when building single-speaker models on CPU
ENV TF_ENABLE_ONEDNN_OPTS=0

ENTRYPOINT ["python3", "main.py"]
