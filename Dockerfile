FROM tensorflow/tensorflow:2.6.0

COPY environments/requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt
RUN python -c "import nltk; nltk.download(\"punkt\"); nltk.download(\"cmudict\")";

WORKDIR /app
VOLUME /app/models
COPY . .

ENV WORKER_NAME=""
RUN echo "python tts_worker.py --worker \$WORKER_NAME" > entrypoint.sh

ENTRYPOINT ["bash", "entrypoint.sh"]
