# coding: utf-8
import os
import sys
import re
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import yaml
from yaml.loader import SafeLoader
from nltk import sent_tokenize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/TransformerTTS')
from TransformerTTS.utils.config_manager import Config
from vocoding.predictors import HiFiGANPredictor
from tts_preprocess_et.convert import convert_sentence


class Synthesizer:
    def __init__(self, config_path: str, checkpoint_path: str, vocoder_path: str):
        self.silence = np.zeros(10000, dtype=np.int16)
        self.config = Config(config_path=config_path)
        self.model = self.config.load_model(checkpoint_path=checkpoint_path)
        self.vocoder = HiFiGANPredictor.from_folder(vocoder_path)

        print("Transformer-TTS initialized.")

    def synthesize(self, text: str, speed: float = 1):
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
                print(f'ERROR: {str(ex)}, sentence: {sent}')
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
        sentences = sent_tokenize(text, 'estonian')

        for i, sentence in enumerate(tqdm(sentences, unit="sentence")):
            sentence = clean(sentence)
            out = self.model.predict(sentence, speed_regulator=speed)
            waveform = self.vocoder([out['mel'].numpy().T])
            if i != 0:
                waveforms.append(self.silence)
            waveforms.append(waveform[0])

        waveform = np.concatenate(waveforms)

        return waveform


if __name__ == '__main__':
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser()
    parser.add_argument('input', type=FileType('r'),
                        help="Input text file to synthesize.")
    parser.add_argument('output', type=FileType('w'),
                        help="Output .wav file path."),
    parser.add_argument('--speaker', type=str, required=True,
                        help="The name of the speaker to use for synthesis.")
    parser.add_argument('--speed', type=int, default=1,
                        help="Output speed multiplier.")
    parser.add_argument('--config', type=FileType('r'), default='config.yaml',
                        help="The config file to load.")
    args = parser.parse_known_args()[0]

    with open(args.config.name, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)['speakers'][args.speaker]

    synthesizer = Synthesizer(**config)

    with open(args.input.name, 'r', encoding='utf-8') as f:
        text = f.read()

    waveform = synthesizer.synthesize(text, speed=args.speed)
    wavfile.write(args.output.name, 22050, waveform.astype(np.int16))
