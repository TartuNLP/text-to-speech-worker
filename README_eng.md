# Estonian Text-to-Speech

Scripts for Estonian multispeaker speech synthesis from text file input.

Speech synthesis was developed in collaboration with the [Estonian Language Institute](http://portaal.eki.ee/).

Estonian text-to-speech can also be used via our [web demo](https://www.neurokone.ee). The components 
to run the same models via API have can be found [here](https://github.com/TartuNLP/text-to-speech-api)
and [here](https://github.com/TartuNLP/text-to-speech-worker).

## Requirements and installation

The following installation instructions have been tested on Ubuntu. The code is both CPU and GPU compatible
(CUDA required).

- Make sure you have the following prerequisites installed:
    - Conda (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
    - GNU Compiler Collection (run `sudo apt install build-essential`)

- Clone with submodules

```
git clone --recurse-submodules https://koodivaramu.eesti.ee/tartunlp/text-to-speech
```

- Create and activate a Conda environment with all dependencies.

```
cd text-to-speech
conda env create -f environments/environment.yml
conda activate transformer-tts
python -c 'import nltk; nltk.download("punkt")'
```

- Download our [TransformerTTS models](https://github.com/TartuNLP/text-to-speech-worker/releases/tag/v2.0.0) and 
  place them inside the `models/` directory.

## Usage

A file can be syntesized with the following command. Currently, only plain text files (utf-8) are supported and the
audio is saved in `.wav` format.

```
python synthesizer.py --speaker albert test.txt test.wav
```

More info about script usage can be found with the `--help` flag:

```
synthesizer.py [-h] [--speaker SPEAKER] [--speed SPEED] [--config CONFIG] input output

positional arguments:
  input                     Input text file to synthesize.
  output                    Output .wav file path.

optional arguments:
  -h, --help                show this help message and exit
  --speaker SPEAKER         The name of the speaker to use for synthesis.
  --speed SPEED             Output speed multiplier.
  --config CONFIG           The config file to load.
```