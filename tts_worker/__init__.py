import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformerTTS')))

from .config import tf_config, mq_config, read_model_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_config.CPP_MIN_LOG_LEVEL)
import tensorflow as tf

from .utils import clean, split_sentence
from .synthesizer import Synthesizer
from .mq_consumer import MQConsumer

logger = logging.getLogger(__name__)

# Tensorflow tries to allocate all memory on a GPU unless explicitly told otherwise.
# Does not affect allocation by Pytorch vocoders.
# TODO TF VRAM limit does not illustrate actual VRAM usage
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        if tf_config.VRAM_LIMIT:  # A memory limit for speech models
            tf.config.experimental.set_virtual_device_configuration(gpu, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(tf_config.VRAM_LIMIT))])
        else:  # Allocating on-the-go
            logger.warning("No VRAM usage limit for Tensorflow set.")
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    logger.error(e)
