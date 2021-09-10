import os
import sys
import io
import logging
import re
from typing import Dict, Any, Optional

import numpy as np
from scipy.io import wavfile
from nltk import sent_tokenize
from marshmallow import Schema, fields, ValidationError
from nauron import Worker, Response

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import settings

sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/TransformerTTS')
from TransformerTTS.utils.config_manager import Config
from vocoding.predictors import HiFiGANPredictor
from tts_preprocess_et.convert import convert_sentence

logger = logging.getLogger('tts')

# Tensorflow tries to allocate all memory on a GPU unless explicitly told otherwise.
# Does not affect allocation by Pytorch vocoders.
# TODO TF VRAM limit does not illustrate actual VRAM usage
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        if settings.TF_VRAM_LIMIT:  # Memory limit for speech models
            tf.config.experimental.set_virtual_device_configuration(gpu, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(settings.TF_VRAM_LIMIT))])
        else:  # Allocating on-the-go
            logger.warning("No VRAM usage limit for Tensorflow set.")
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    logger.error(e)


class TTSWorker(Worker):
    def __init__(self, config_path: str, checkpoint_path: str, vocoder_path: str):
        class TTSSchema(Schema):
            text = fields.Str(required=True)
            speaker = fields.Str()
            speed = fields.Float(missing=1, validate=lambda s: 0.5 <= s <= 2)
            application = fields.Str(allow_none=True, missing=None)

        self.silence = np.zeros(10000, dtype=np.int16)
        self.schema = TTSSchema

        self.config = Config(config_path=config_path)
        self.model = self.config.load_model(checkpoint_path=checkpoint_path)
        self.vocoder = HiFiGANPredictor.from_folder(vocoder_path)

        logger.info("Transformer-TTS initialized.")

    def process_request(self, body: Dict[str, Any], _: Optional[str] = None) -> Response:
        try:
            body = self.schema().load(body)
            logger.info(f"Request received: {{"
                        f"speaker: {body['speaker']}, "
                        f"speed: {body['speed']}}}")
            return Response(content=self._synthesize(body['text'], body['speed']), mimetype='audio/wav')
        except ValidationError as error:
            return Response(content=error.messages, http_status_code=400)
        except tf.errors.ResourceExhaustedError:
            return Response(content="Input contains sentences that are too long.", http_status_code=413)

    def _synthesize(self, text: str, speed: float = 1) -> bytes:
        """Convert text to speech waveform.
        Args:
          text (str) : Input text to be synthesized
          speed (float)
        """

        def clean(sent):
            sent = re.sub(r'[`´’\']', r'', sent)
            sent = re.sub(r'[()]', r', ', sent)
            try:
                sent = convert_sentence(sent)
            except Exception as ex:
                logger.error(str(ex), sent)
            sent = re.sub(r'[()[\]:;−­–…—]', r', ', sent)
            sent = re.sub(r'[«»“„”]', r'"', sent)
            sent = re.sub(r'[*\'\\/-]', r' ', sent)
            sent = re.sub(r'[`´’\']', r'', sent)
            sent = re.sub(r' +([.,!?])', r'\g<1>', sent)
            sent = re.sub(r', ?([.,?!])', r'\g<1>', sent)
            sent = re.sub(r'\.+', r'.', sent)

            sent = re.sub(r' +', r' ', sent)
            sent = re.sub(r'^ | $', r'', sent)
            sent = re.sub(r'^, ?', r'', sent)
            sent = sent.lower()
            sent = re.sub(re.compile(r'\s+'), ' ', sent)
            return sent

        waveforms = []

        # The quotation marks need to be unified, otherwise sentence tokenization won't work
        text = re.sub(r'[«»“„]', r'"', text)

        for i, sentence in enumerate(sent_tokenize(text, 'estonian')):
            logger.info(f"Original sentence {i}: {sentence}")
            sentence = clean(sentence)
            logger.info(f"Cleaned sentence {i}: {sentence}")
            out = self.model.predict(sentence, speed_regulator=speed)
            waveform = self.vocoder([out['mel'].numpy().T])
            if i != 0:
                waveforms.append(self.silence)
            waveforms.append(waveform[0])

        waveform = np.concatenate(waveforms)

        out = io.BytesIO()
        wavfile.write(out, 22050, waveform.astype(np.int16))

        return out.read()


if __name__ == '__main__':
    worker = TTSWorker(**settings.WORKER_PARAMETERS)

    worker.start(connection_parameters=settings.MQ_PARAMETERS,
                 service_name=settings.SERVICE_NAME,
                 routing_key=settings.ROUTES[0],
                 alt_routes=settings.ROUTES[1:])
