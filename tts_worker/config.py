import os

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from yaml.loader import SafeLoader


class TTSConfig(BaseSettings):
    """
    Imports TTS worker configuration from environment variables
    - TF_VRAM_LIMIT: int - Limit for TensorFlow GPU memory usage in MB.
    - TTS_MAX_INPUT_LENGTH: int - Maximum number of characters that the model will try to synthesize in one go. 
    If not set (value is 0), the model's maximum input length is used and any failed sentences are automatically retried in smaller chunks.
    The limit is specific to hardware and VRAM availability/limits.
    - SPEECHBRAIN_HIFIGAN_MODEL: str - Huggingface model name for Speechbrain HiFi-GAN vocoder.
    """
    CPP_MIN_LOG_LEVEL: int = 2
    TF_VRAM_LIMIT: int = 8000
    TTS_MAX_INPUT_LENGTH: int = 0
    SPEECHBRAIN_HIFIGAN_MODEL: str = "speechbrain/tts-hifigan-libritts-22050Hz"

    def __init__(self, **values):
        super().__init__(**values)
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', str(self.CPP_MIN_LOG_LEVEL))


class MQConfig(BaseSettings):
    """
    Imports MQ configuration from environment variables
    """
    username: str = 'guest'
    password: str = 'guest'
    host: str = 'localhost'
    port: int = 5672
    exchange: str = 'text-to-speech'
    heartbeat: int = 60
    connection_name: str = 'TTS worker'
    x_priority: int = 0
    x_expires: int = 60000

    class Config:
        env_prefix = 'mq_'


class ModelConfig(BaseModel):
    model_name: str
    model_path: str
    frontend: str
    speakers: dict[str, int] # name, id


def read_model_config(file_path: str, model_name: str) -> ModelConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)
        model_config = ModelConfig(
            model_name=model_name,
            **config['tts_models'][model_name]
        )

    return model_config


tts_config = TTSConfig()
mq_config = MQConfig()
