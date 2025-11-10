import logging.config
from argparse import ArgumentParser

from tts_worker.config import read_model_config
from tts_worker.synthesizer import Synthesizer
from tts_worker.mq_consumer import MQConsumer


def parse_args():
    parser = ArgumentParser(
        description="A text-to-speech worker that processes incoming TTS requests via RabbitMQ."
    )
    parser.add_argument('--model-config', type=str, default='config/config.yaml',
                        help="The model config YAML file to load.")
    parser.add_argument('--model-name', type=str,
                        help="The model to load. Refers to the model name in the config file.")
    parser.add_argument('--log-config', type=str, default='config/logging.prod.ini',
                        help="Path to log config file.")
    parser.add_argument('--max-input-length', type=int, default=0,
                        help="Optional max input length configuration - "
                             "the maximum number of characters that the model will ever try to synthesize in one go. "
                             "If not set, some limit will be calculated automatically during start-up and "
                             "any failed sentences are automatically retried in smaller chunks. The limit is specific "
                             "to hardware.")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.config.fileConfig(args.log_config)
    model_config = read_model_config(args.model_config, args.model_name)

    tts = Synthesizer(model_config, args.max_input_length)
    consumer = MQConsumer(tts)
    consumer.start()


if __name__ == "__main__":
    main()
