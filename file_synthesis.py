# coding: utf-8
import os
from argparse import ArgumentParser, FileType
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
import audio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = ArgumentParser()
parser.add_argument('input', type=FileType('r'),
                    help="Input text file to synthesize.")
parser.add_argument('output', type=FileType('w'),
                    help="Output .wav file path.")
parser.add_argument('--checkpoint', type=FileType('r'), default='models/checkpoint.pth',
                    help="The checkpoint (model file) to load.")
parser.add_argument('--preset', type=FileType('r'), default='deepvoice3_pytorch/presets/eesti_konekorpus.json',
                    help="Model preset file.")
parser.add_argument('--speaker-id', type=int, default=0,
                    help="The ID of the speaker to use for synthesis.")
args = parser.parse_known_args()[0]

if __name__ == '__main__':
    args = parser.parse_known_args()[0]

    with open(args.preset.name, 'r', encoding='utf-8') as f:
        hparams.parse_json(f.read())

    synthesizer = Synthesizer(args.preset.name, args.checkpoint.name)

    with open(args.input.name, 'r', encoding='utf-8') as f:
        text = f.read()

    waveform = synthesizer.synthesize(text, args.speaker_id)
    audio.save_wav(waveform, args.output.name)
