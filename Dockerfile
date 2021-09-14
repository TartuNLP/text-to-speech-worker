FROM continuumio/miniconda3 as build
RUN apt-get update && \
    apt-get install -y build-essential && \
    conda install -c conda-forge conda-pack mamba

COPY environments/environment.yml .
RUN mamba env create -f environment.yml -n venv && \
    rm environment.yml && \
    conda-pack -n venv -o /tmp/env.tar && \
    mkdir /venv &&  \
    cd /venv && \
    tar xf /tmp/env.tar && \
    rm /tmp/env.tar && \
    /venv/bin/conda-unpack && \
    conda clean -afy && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyc' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    conda env remove -n venv

FROM debian:buster

ENV PYTHONIOENCODING=utf-8
WORKDIR /app
VOLUME /app/models

RUN adduser --disabled-password --gecos "app" app && \
    chown -R app:app /app
USER app

COPY --from=build --chown=app:app /venv /venv
ENV PATH="/venv/bin:${PATH}"
COPY --chown=app:app . .

RUN python -c "import nltk; nltk.download(\"punkt\")";

RUN echo "python tts_worker.py --worker \$WORKER_NAME" > entrypoint.sh

ENTRYPOINT ["bash", "entrypoint.sh"]
