# Estonian Text-to-Speech

This repository contains Estonian multi-speaker neural text-to-speech synthesis workers that process requests from
RabbitMQ.

The project is developed by the [NLP research group](https://tartunlp.ai) at the [University of Tartu](https://ut.ee).
Speech synthesis can also be tested in our [demo](https://www.neurokone.ee/).

## Models

[The releases section](https://github.com/TartuNLP/text-to-speech-worker/releases) contains the model files or their
download instructions. If a release does not specify the model information, the model from the previous release can
be used. We advise always using the latest available version to ensure best model quality and code compatibility.

The model configuration files included in `config/config.yaml` correspond to the following `models/` directory
structure:

```
models
├── hifigan
│   ├── ljspeech
│   │   ├── config.json
│   │   └── model.pt
│   ├── vctk
│   │   ├── config.json
│   │   └── model.pt
└── tts
    └── multispeaker
        ├── config.yaml
        └── model_weights.hdf5
```

## Setup

The TTS worker can be deployed using the docker image published alongside the repository. Each image version correlates
to a specific release. The required model file(s) are excluded from the image to reduce the image size and should be
downloaded from the releases section and their directory should be attached to the volume `/app/models`.

Logging configuration is loaded from `/app/config/logging.prod.ini` and service configuration from the
`/app/config/config.yaml` file. The included config is commented to illustrate how new model configurations could be
added.

The following environment variables should be configured when running the container:

- `MQ_USERNAME` - RabbitMQ username
- `MQ_PASSWORD` - RabbitMQ user password
- `MQ_HOST` - RabbitMQ host
- `MQ_PORT` (optional) - RabbitMQ port (`5672` by default)
- `MQ_EXCHANGE` (optional) - RabbitMQ exchange name (`text-to-speech` by default)
- `MQ_HEARTBEAT` (optional) - heartbeat interval (`60` seconds by default)
- `MQ_CONNECTION_NAME` (optional) - friendly connection name (`TTS worker` by default)
- `MKL_NUM_THREADS` (optional) - number of threads used for intra-op parallelism by PyTorch (used for the vocoder model)
  . `16` by default. If set to a blank value, it defaults to the number of CPU cores which may cause computational
  overhead when deployed on larger nodes. Alternatively, the `docker run` flag `--cpuset-cpus` can be used to control
  this. For more details, refer to the [performance and hardware requirements](#performance-and-hardware-requirements)
  section below.

By default, the container entrypoint is `main.py` without additional arguments, but arguments should be defined with the
`COMMAND` option. The only required flag is `--model-name` to select which model is loaded by the worker. The full list
of supported flags can be seen by running `python main.py -h`:

```commandline
usage: main.py [-h] [--model-config MODEL_CONFIG] [--model-name MODEL_NAME] [--log-config LOG_CONFIG]

A text-to-speech worker that processes incoming TTS requests via RabbitMQ.

optional arguments:
  -h, --help            show this help message and exit
  --model-config MODEL_CONFIG
                        The model config YAML file to load.
  --model-name MODEL_NAME
                        The model to load. Refers to the model name in the config file.
  --log-config LOG_CONFIG
                        Path to log config file.
```

The setup can be tested with the following sample `docker-compose.yml` configuration:

```yaml
version: '3'
services:
  rabbitmq:
    image: 'rabbitmq'
    environment:
      - RABBITMQ_DEFAULT_USER=${RABBITMQ_USER}
      - RABBITMQ_DEFAULT_PASS=${RABBITMQ_PASS}
  tts_api:
    image: ghcr.io/tartunlp/text-to-speech-api:latest
    environment:
      - MQ_HOST=rabbitmq
      - MQ_PORT=5672
      - MQ_USERNAME=${RABBITMQ_USER}
      - MQ_PASSWORD=${RABBITMQ_PASS}
    ports:
      - '8000:8000'
    depends_on:
      - rabbitmq
  tts_worker:
    image: ghcr.io/tartunlp/text-to-speech-worker:latest
    environment:
      - MQ_HOST=rabbitmq
      - MQ_PORT=5672
      - MQ_USERNAME=${RABBITMQ_USER}
      - MQ_PASSWORD=${RABBITMQ_PASS}
    command: [ "--model-name", "multispeaker" ]
    volumes:
      - ./models:/app/models
    depends_on:
      - rabbitmq
```

### Manual setup

The following steps have been tested on Ubuntu and is both CPU and GPU compatible (CUDA required).

- Clone this repository with submodules
- Install prerequisites:
    - GNU Compiler Collection (`sudo apt install build-essential`)
    - For a **CPU** installation we recommend using the included `requirements.txt` file in a clean environment (tested with
      Python 3.9)
      ```commandline
      pip install -r requirements.txt
      ```

    - For a **GPU** installation, use the `environment.yml` file instead.
        - Make sure you have the following prerequisites installed:
            - CUDA (see https://developer.nvidia.com/cuda-downloads)
            - Conda (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

        - Then create and activate a Conda environment with all dependencies:
          ```commandline
          conda env create -f environment.yml -n tts
          conda activate tts
          ```

- Download the models from the [releases section](https://github.com/TartuNLP/text-to-speech-worker/releases) and
  place inside the `models/` directory.

- Check the configuration files and change any defaults as needed. Make sure that the `model_path` parameter in
  `config/config.yaml` points to the model you just downloaded.

- Specify RabbitMQ connection parameters with environment variables or in a `config/.env` file as illustrated in the
  `config/sample.env`.

Run the worker with where `MODEL_NAME` matches the model name in your config file:

```commandline
python main.py --model-name $MODEL_NAME [--log-config config/logging.ini --config config/config.yaml]
```
