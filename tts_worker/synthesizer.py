import io
import logging
import re

import numpy as np
from scipy.io import wavfile
from nltk import sent_tokenize

import tensorflow as tf

from tts_worker.config import ModelConfig
from tts_worker.schemas import Request, Response, ResponseContent
from tts_worker.utils import clean, split_sentence

from tts_worker.vocoding.predictors import HiFiGANPredictor

from TransformerTTS.model.models import ForwardTransformer

logger = logging.getLogger(__name__)


class Synthesizer:
    def __init__(self, model_config: ModelConfig):
        self.model_name = model_config.model_name

        self.model = ForwardTransformer.load_model(model_config.model_path)
        self.vocoders = {}

        for speaker in model_config.speakers.values():
            if speaker.vocoder not in self.vocoders:
                self.vocoders[speaker.vocoder] = HiFiGANPredictor.from_folder(model_config.vocoders[speaker.vocoder])

        self.speakers = model_config.speakers

        self.frontend = model_config.frontend

        self.sampling_rate = self.model.config['sampling_rate']
        self.hop_length = self.model.config['hop_length']
        self.win_length = self.model.config['win_length']

        self.silence = np.zeros(self.sampling_rate // 2 - (self.sampling_rate // 2) % self.hop_length,
                                dtype=np.int16)  # ~0.5 sec
        self.silence_len = self.silence.shape[0] // self.hop_length

        self.gst_len = self.model.text_pipeline.tokenizer.zfill

        self.max_input_length = self.model.config['encoder_max_position_encoding'] - self.gst_len
        self.last_input_len = 0

        logger.debug(f"sampling rate: {self.sampling_rate}, "
                     f"hop length: {self.hop_length}, "
                     f"win length: {self.win_length}, "
                     f"gst length: {self.gst_len}, "
                     f"max input length: {self.max_input_length}")

        logger.info("Transformer-TTS initialized.")

    def process_request(self, request: Request) -> Response:
        logger.info(f"Request received: {{"
                    f"speaker: {request.speaker}, "
                    f"speed: {request.speed}}}")
        return self._synthesize(request.text, request.speaker, request.speed)

    def _synthesize(self, text: str, speaker: str, speed: float = 1) -> Response:
        """Convert text to speech waveform.
        Args:
          text (str) : Input text to be synthesized
          speed (float)
        """

        waveforms = []
        vocoder = self.vocoders[self.speakers[speaker].vocoder]

        # The quotation marks need to be unified, otherwise sentence tokenization won't work
        sentences = sent_tokenize(re.sub(r'[«»“„]', r'"', text), 'estonian')  # TODO: front-end specific?

        durations = []
        normalized_text = ""

        for i, sentence in enumerate(sentences):
            waveforms.append(self.silence)
            durations.append(self.silence_len)
            normalized_text += " "

            logger.debug(f"Original sentence {i} ({len(sentence)} chars): {sentence}")
            normalized_sentence = clean(sentence, self.model.config['alphabet'], frontend=self.frontend)
            logger.debug(f"Cleaned sentence {i} ({len(normalized_sentence)} chars): {normalized_sentence}")

            while True:
                try:
                    sent_durations = []
                    if len(normalized_sentence) > self.max_input_length:
                        inputs = split_sentence(normalized_sentence, max_len=self.max_input_length)
                        logger.debug(f'Sentence split into {len(inputs)} parts: '
                                     f'{[x[:10] + " ... " + x[-10:] for x in inputs]}')
                    else:
                        inputs = [normalized_sentence]

                    for input_sentence in inputs:
                        self.last_input_len = len(input_sentence)

                        tts_out = self.model.predict(input_sentence,
                                                     speed_regulator=speed,
                                                     speaker_id=self.speakers[speaker].speaker_id)
                        mel_spec = tts_out['mel'].numpy().T
                        sent_durations += np.rint(
                            tts_out['duration'].numpy().squeeze()
                        ).astype(int)[self.gst_len:].tolist()

                        logger.debug(f"Predicted mel-spectrogram dimensions: {mel_spec.shape}")

                        if mel_spec.size:  # don't send empty mel-spectrograms to vocoder
                            waveform = vocoder([mel_spec])[0]
                            waveforms.append(waveform)
                    normalized_text += ''.join(inputs)
                    durations += sent_durations
                    break
                except tf.errors.ResourceExhaustedError:
                    logger.warning(
                        f"Synthesis failed with max input length {self.max_input_length}, "
                        f"reducing max length to {int(self.last_input_len * 0.9)} and tying again...")
                    self.max_input_length = int(self.last_input_len * 0.9)

        waveforms.append(self.silence)
        durations.append(self.silence_len)
        normalized_text += " "

        waveform = np.concatenate(waveforms)

        out = io.BytesIO()
        wavfile.write(out, self.sampling_rate, waveform.astype(np.int16))

        result = Response(
            content=ResponseContent(audio=out.read(),
                                    text=text,
                                    normalized_text=normalized_text,
                                    duration_frames=durations,
                                    sampling_rate=self.sampling_rate,
                                    win_length=self.win_length,
                                    hop_length=self.hop_length))

        return result
