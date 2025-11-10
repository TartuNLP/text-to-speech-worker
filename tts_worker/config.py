import yaml
from yaml.loader import SafeLoader
from typing import Dict

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class TFConfig(BaseSettings):
    CPP_MIN_LOG_LEVEL: int = 2
    VRAM_LIMIT: int = 1600

    class Config:
        env_file = 'config/.env'
        env_prefix = 'TF_'


class MQConfig(BaseSettings):
    """
    Imports MQ configuration from environment variables
    """
    host: str = 'localhost'
    port: int = 5672
    username: str = 'guest'
    password: str = 'guest'
    exchange: str = 'text-to-speech'
    heartbeat: int = 60
    connection_name: str = 'TTS worker'
    x_priority: int = 0
    x_expires: int = 60000

    class Config:
        env_file = 'config/.env'
        env_prefix = 'mq_'

class ModelConfig(BaseModel):
    model_name: str
    model_path: str
    frontend: str
    speakers: Dict[str, int] # name, id


def read_model_config(file_path: str, model_name: str) -> ModelConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)
        model_config = ModelConfig(
            model_name=model_name,
            **config['tts_models'][model_name]
        )

    return model_config


tf_config = TFConfig()
mq_config = MQConfig()
