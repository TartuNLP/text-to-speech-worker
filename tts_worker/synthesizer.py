import io
import logging
import re

import numpy as np
import tensorflow as tf
import torch
from nltk import sent_tokenize
from scipy.io import wavfile
from speechbrain.inference.vocoders import HIFIGAN
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from TransformerTTS.model.models import ForwardTransformer
from tts_worker.config import ModelConfig, tts_config
from tts_worker.schemas import Request, Response, ResponseContent
from tts_worker.utils import clean, split_sentence

logger = logging.getLogger(__name__)


class Synthesizer:
    def __init__(self, model_config: ModelConfig):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocoder = HIFIGAN.from_hparams(
            source=tts_config.SPEECHBRAIN_HIFIGAN_MODEL,
            run_opts={"device": self.device},
        )
        logger.info(
            f"Loaded HiFi-GAN vocoder {tts_config.SPEECHBRAIN_HIFIGAN_MODEL} on device: {self.device}."
        )

        # Tensorflow tries to allocate all memory on a GPU unless explicitly told otherwise.
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    if tts_config.TF_VRAM_LIMIT > 0:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [
                                tf.config.experimental.VirtualDeviceConfiguration(
                                    memory_limit=tts_config.TF_VRAM_LIMIT
                                )
                            ],
                        )
                        logger.info(
                            f"Set TF VRAM limit to {tts_config.TF_VRAM_LIMIT} MB."
                        )
                        logger.warning(
                            "TensorFlow VRAM limit does not affect PyTorch VRAM usage!"
                        )
                    else:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info("Set TensorFlow memory usage set to dynamic.")
            except RuntimeError as e:
                logger.error(e)

        self.model_name = model_config.model_name

        self.model = ForwardTransformer.load_model(model_config.model_path)
        logger.info(
            f"Loaded Transformer-TTS model '{self.model_name}' from {model_config.model_path}."
        )

        self.speakers = model_config.speakers
        self.frontend = model_config.frontend

        self.sampling_rate = self.model.config["sampling_rate"]
        self.hop_length = self.model.config["hop_length"]
        self.win_length = self.model.config["win_length"]

        self.silence = np.zeros(
            self.sampling_rate // 2 - (self.sampling_rate // 2) % self.hop_length,
            dtype=np.float32,
        )  # ~0.5 sec
        self.silence_len = self.silence.shape[0] // self.hop_length

        self.gst_len = self.model.text_pipeline.tokenizer.zfill

        model_input_limit = (
            self.model.config["encoder_max_position_encoding"] - self.gst_len
        )

        if (
            tts_config.TTS_MAX_INPUT_LENGTH
            and tts_config.TTS_MAX_INPUT_LENGTH <= model_input_limit
        ):
            logger.info(
                f"Max input length set according to user configuration: {tts_config.TTS_MAX_INPUT_LENGTH}"
            )
            self.max_input_length = tts_config.TTS_MAX_INPUT_LENGTH
        else:
            if tts_config.TTS_MAX_INPUT_LENGTH > model_input_limit:
                logger.warning(
                    "The specified input length limit exceeds the model's maximum input size. Using model's limit instead."
                )
            if not tts_config.TTS_MAX_INPUT_LENGTH:
                logger.info(
                    "No maximum input length specified, using model's maximum input size."
                )
            self.max_input_length = model_input_limit

        logger.info("Transformer-TTS initialized.")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Frontend: {self.frontend}")
        logger.info(f"Speakers: {list(self.speakers.keys())}")
        logger.info(f"Sampling rate: {self.sampling_rate}")
        logger.info(f"Hop length: {self.hop_length}")
        logger.info(f"Win length: {self.win_length}")
        logger.info(f"GST length: {self.gst_len}")
        logger.info(f"Max input length: {self.max_input_length}")

    def process_request(self, request: Request) -> Response:
        logger.info(
            f"Request received: {{speaker: {request.speaker}, speed: {request.speed}}}"
        )
        response = self._synthesize(request.text, request.speaker, request.speed)
        return response

    def _synthesize(self, text: str, speaker: str, speed: float = 1) -> Response:
        """Convert text to speech waveform.
        Args:
          text (str) : Input text to be synthesized
          speed (float)
        """
        max_input_length = self.max_input_length
        waveforms = []

        # The quotation marks need to be unified, otherwise sentence tokenization won't work
        sentences = sent_tokenize(
            re.sub(r"[«»“„]", r'"', text), "estonian"
        )  # TODO: front-end specific

        durations = []
        normalized_text = ""

        for i, sentence in enumerate(sentences):
            waveforms.append(self.silence)
            durations.append(self.silence_len)
            normalized_text += " "

            logger.debug(f"Original sentence {i} ({len(sentence)} chars): {sentence}")
            normalized_sentence = clean(
                sentence, self.model.config["alphabet"], frontend=self.frontend
            )
            logger.debug(
                f"Cleaned sentence {i} ({len(normalized_sentence)} chars): {normalized_sentence}"
            )

            while True:
                last_input_len = max_input_length
                try:
                    sent_durations = []
                    sent_waveforms = []
                    if len(normalized_sentence) > max_input_length:
                        input_chunks = split_sentence(
                            normalized_sentence, max_len=max_input_length
                        )
                        logger.debug(
                            f"Sentence split into {len(input_chunks)} parts: "
                            f"{[x[:10] + ' ... ' + x[-10:] for x in input_chunks]}"
                        )
                    else:
                        input_chunks = [normalized_sentence]

                    for input_chunk in input_chunks:
                        last_input_len = len(input_chunk)

                        tts_out = self.model.predict(
                            input_chunk,
                            speed_regulator=speed,
                            speaker_id=self.speakers[speaker],
                        )
                        mel_spec = tts_out["mel"].numpy().T
                        sent_durations += (
                            np.rint(tts_out["duration"].numpy().squeeze())
                            .astype(int)[self.gst_len :]
                            .tolist()
                        )

                        logger.debug(
                            f"Predicted mel-spectrogram dimensions: {mel_spec.shape}"
                        )

                        if (
                            mel_spec.size
                        ):  # don't send empty mel-spectrograms to vocoder
                            tensor = torch.tensor(mel_spec).unsqueeze(0)
                            waveform = self.vocoder.decode_batch(
                                tensor.to(self.device)
                            ).squeeze()
                            sent_waveforms.append(waveform.cpu().numpy())
                    normalized_text += "".join(input_chunks)
                    durations += sent_durations
                    waveforms += sent_waveforms
                    break
                except (InvalidArgumentError, tf.errors.ResourceExhaustedError):
                    max_input_length = int(
                        min((last_input_len, max_input_length)) * 0.5
                    )
                    if max_input_length < 10:
                        logger.error(
                            "Synthesis failed, input could not be reduced further."
                        )
                        return Response(
                            status_code=500,
                            status="Synthesis failed, input could not be reduced further.",
                        )

                    logger.warning(
                        f"Synthesis failed, "
                        f"trying again with max input length {max_input_length} (original: {self.max_input_length})."
                    )

        waveforms.append(self.silence)
        durations.append(self.silence_len)
        normalized_text += " "

        waveform = np.concatenate(waveforms)

        out = io.BytesIO()
        wavfile.write(out, self.sampling_rate, waveform.astype(np.float32))

        result = Response(
            content=ResponseContent(
                audio=out.read(),
                text=text,
                normalized_text=normalized_text,
                duration_frames=durations,
                sampling_rate=self.sampling_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
            )
        )

        return result
