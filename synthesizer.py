# coding: utf-8
import io
import re
import numpy as np
import torch
from hparams import hparams
from deepvoice3_pytorch import frontend
from train import build_model
import audio
import train
from nltk import sent_tokenize


class Synthesizer:
    def __init__(self, preset, checkpoint_path, fast=True):
        self._frontend = None
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = None
        self.preset = preset
        self.silence = np.zeros(10000)

        # Presets
        with open(self.preset) as f:
            hparams.parse_json(f.read())

        self._frontend = getattr(frontend, hparams.frontend)
        train._frontend = self._frontend

        self.model = build_model()

        # Load checkpoints separately
        if self.use_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["state_dict"])

        # TODO handling longer inputs
        self.model.seq2seq.decoder.max_decoder_steps = hparams.max_positions-2

        self.model = self.model.to(self.device)
        self.model.eval()
        if fast:
            self.model.make_generation_fast_()

    def synthesize(self, text, speaker_id=0, threshold=5):
        """Convert text to speech waveform given a deepvoice3 model.

        Args:
          text (str) : Input text to be synthesized
          speaker_id (int)
          threshold (int) : Threshold for trimming stuttering at the end. Smaller threshold means more agressive
          trimming.
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
            if repetitions.size > threshold:
                end = repetitions[threshold]
                end = int(end * len(waveform) / last_row.size)
                waveform = waveform[:end]
            if i != 0:
                waveforms.append(self.silence)
            waveforms.append(waveform)

        waveform = np.concatenate(waveforms)

        out = io.BytesIO()
        audio.save_wav(waveform, out)
        return out
