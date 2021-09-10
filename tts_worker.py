import os
import io
import logging
import re
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from nltk import sent_tokenize
from marshmallow import Schema, fields, validate, ValidationError
from nauron import Worker, Response

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import train
import audio
from train import build_model
from hparams import hparams, hparams_debug_string
from deepvoice3_pytorch import frontend

import settings

logger = logging.getLogger('tts')


class TTSWorker(Worker):
    def __init__(self, preset: str, checkpoint: str, allowed_speakers: List[int], trim_threshold: int = 5):
        class TTSSchema(Schema):
            text = fields.Str(required=True)
            speaker_id = fields.Int(missing=allowed_speakers[0],
                                    validate=validate.OneOf(allowed_speakers))

        self.schema = TTSSchema

        with open(preset) as f:
            hparams.parse_json(f.read())
        logger.debug(hparams_debug_string())

        self.trim_threshold = trim_threshold

        self._frontend = None
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = None
        self.silence = np.zeros(10000)

        # Presets
        with open(preset) as f:
            hparams.parse_json(f.read())

        self._frontend = getattr(frontend, hparams.frontend)
        train._frontend = self._frontend

        self.model = build_model()

        # Load checkpoints separately
        if self.use_cuda:
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["state_dict"])

        self.model.seq2seq.decoder.max_decoder_steps = hparams.max_positions - 2

        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.make_generation_fast_()

        logger.info("Deepvoice3 initialized.")

    def process_request(self, body: Dict[str, Any], _: Optional[str] = None) -> Response:
        try:
            body = self.schema().load(body)
            logger.info(f"Request received: {{"
                        f"speaker: {body['speaker_id']}, "
                        f"text: \"{body['text']}\""
                        f"}}")
        except ValidationError as error:
            return Response(content=error.messages, http_status_code=400)

        try:
            return Response(content=self._synthesize(body['text'], body['speaker_id']), mimetype='audio/wav')
        except ValueError:
            return Response(content="Input contains sentences that are too long.", http_status_code=413)

    def _synthesize(self, text: str, speaker_id: int) -> bytes:
        """Convert text to speech waveform given a deepvoice3 model.
        Args:
          text (str) : Input text to be synthesized
          speaker_id (int)
        """
        waveforms = []

        # The quotation marks need to be unified, otherwise sentence tokenization won't work
        text = re.sub(r'[«»“„]', r'"', text)

        for i, sentence in enumerate(sent_tokenize(text, 'estonian')):
            sequence = np.array(self._frontend.text_to_sequence(sentence))
            sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(self.device)
            text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(self.device)
            speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(self.device)

            if text_positions.size()[1] >= hparams.max_positions:
                raise ValueError("Input contains sentences that are too long.")

            # Greedy decoding
            with torch.no_grad():
                mel_outputs, linear_outputs, alignments, done = self.model(
                    sequence, text_positions=text_positions, speaker_ids=speaker_ids)

            linear_output = linear_outputs[0].cpu().data.numpy()
            alignment = alignments[0].cpu().data.numpy()

            # Predicted audio signal
            waveform = audio.inv_spectrogram(linear_output.T)

            # Cutting predicted signal to remove stuttering from the end of synthesized audio
            last_row = np.transpose(alignment)[-1]
            repetitions = np.where(last_row > 0)[0]
            if repetitions.size > self.trim_threshold:
                end = repetitions[self.trim_threshold]
                end = int(end * len(waveform) / last_row.size)
                waveform = waveform[:end]
            if i != 0:
                waveforms.append(self.silence)
            waveforms.append(waveform)

        waveform = np.concatenate(waveforms)

        out = io.BytesIO()
        audio.save_wav(waveform, out)
        return out.read()


if __name__ == '__main__':
    worker = TTSWorker(**settings.WORKER_PARAMETERS)

    worker.start(connection_parameters=settings.MQ_PARAMETERS,
                 service_name=settings.SERVICE_NAME,
                 routing_key=settings.ROUTES[0],
                 alt_routes=settings.ROUTES[1:])
