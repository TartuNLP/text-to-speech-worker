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

RUN adduser --disabled-password --gecos "app" app && \
    chown -R app:app /app
USER app

ENV PATH="/home/app/.local/bin:${PATH}"

COPY --chown=app:app requirements.txt .
RUN pip install -r requirements.txt && \
    rm requirements.txt

COPY --chown=app:app . .

ENTRYPOINT ["python3", "main.py"]
