# Estonian Text-to-Speech API

A simple Flask API for Estonian multispeaker speech synthesis. This repository contains the following submodules:
- [Deep Voice 3 adaptation for Estonian](https://github.com/TartuNLP/deepvoice3_pytorch)
- [Estonian text-to-speech preprocessing scripts](https://github.com/TartuNLP/tts_preprocess_et)

Speech synthesis was developed in collaboration with the [Estonian Language Institute](http://portaal.eki.ee/).

A newer version of Estonian text-to-speech can be used via our [web demo](https://www.neurokone.ee). This repository will be updated to run the latest version in Fall 2021.
 
## API usage
To use the API, use the following POST request format.

POST `/api/v1.0/synthesize`

BODY (JSON):
```
{
    "text": "Tere."
    "speaker_id": 0
}
```
Upon such request, the server will return a binary stream of the synthesized audio in .wav format. The `speaker_id
` parameter is optional and by default, the first speaker is selected.

The [model](https://github.com/TartuNLP/deepvoice3_pytorch/releases/download/kratt-v1.2) we reference to in this version
 supports six different speakers.

## Requirements and installation

The following installation instructions have been tested on Ubuntu 18.04. The code is both CPU and GPU compatible.

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
- Download our [Deep Voice 3 model](https://github.com/TartuNLP/deepvoice3_pytorch/releases/download/kratt-v1.2/autosegment.pth)

- Create a configuration file and change any defaults as needed. Make sure that the `checkpoint` parameter points to
 the model file you just downloaded.
```
cp config.sample.json config.json
```

Configure a web server to run `tts_server.py` or test the API with:
```
export FLASK_APP=tts_server.py
flask run
```
