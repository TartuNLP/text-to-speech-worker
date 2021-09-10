# Estonian Text-to-Speech

Scripts for Estonian multispeaker speech synthesis from text file input. This repository contains the following
submodules:

- [Deep Voice 3 adaptation for Estonian](https://github.com/TartuNLP/deepvoice3_pytorch)
- [Estonian text-to-speech preprocessing scripts](https://github.com/TartuNLP/tts_preprocess_et)

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
conda env create -f environment.yml
conda activate deepvoice
pip install --no-deps -e "deepvoice3_pytorch/[bin]"
python -c 'import nltk; nltk.download("punkt"); nltk.download("cmudict")'
```

- Download our [Deep Voice 3 model](https://github.com/TartuNLP/deepvoice3_pytorch/releases/kratt-v1.2) and place it
  inside the `models/` directory. The model we reference to in this version supports six different speakers.

## Usage

A file can be syntesized with the following command. Currently, only plain text files (utf-8) are supported and the
audio is saved in `.wav` format.

```
python file_synthesis.py test.txt test.wav
```

More info about script usage can be found with the `--help` flag:

```
file_synthesis.py [-h] [--checkpoint CHECKPOINT] [--preset PRESET] [--speaker-id SPEAKER_ID] input output

positional arguments:
  input                     Input text file to synthesize.
  output                    Output .wav file path.

optional arguments:
  -h, --help                show this help message and exit
  --checkpoint CHECKPOINT   The checkpoint (model file) to load.
  --preset PRESET           Model preset file.
  --speaker-id SPEAKER_ID   The ID of the speaker to use for synthesis.
```