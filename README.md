# Eestikeelne kõnesüntees

Skriptid eestikeelse mitmehäälse kõnesünteesi kasutamiseks teksifaili põhjal.
 
Kõnesüntees on loodud koostöös [Eesti Keele Instituudiga](http://portaal.eki.ee/)

Kõnesünteesi on võimalik kasutada ka meie [veebidemos](https://www.neurokone.ee). Samade mudelite rakendusliidese 
komponendid on kättesaadavad [siit](https://github.com/TartuNLP/text-to-speech-api)
ja [siit](https://github.com/TartuNLP/text-to-speech-worker).

## Nõuded ja seadistamine

Siinseid instruktsioone on testitud Ubuntuga. Kood on nii CPU- kui GPU-sõbralik (vajab CUDA-t GPU kasutamiseks).

- Veendu, et järgmised komponendid on installitud:
    - Conda (loe lähemalt: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
    - GNU Compiler Collection (jooksuta: `sudo apt install build-essential`)

- Klooni see repositoorium koos alammoodulitega:
```
git clone --recurse-submodules https://koodivaramu.eesti.ee/tartunlp/text-to-speech
```
- Loo ja aktiveeri Conda keskond:
```
cd text-to-speech
conda env create -f environments/environment.yml
conda activate transformer-tts
python -c 'import nltk; nltk.download("punkt")'
```
- Lae alla meie [TransformerTTS mudelid](https://github.com/TartuNLP/text-to-speech-worker/releases/tag/v2.0.0) ja aseta need `models/` kausta.

## Kasutamine

Tekstifaili saab sünteesida järgmise käsuga. Hetkel oskab skript lugeda vaid toorteksti kujul faile ja salvestab 
väljundi `.wav` formaadis.

```
python synthesizer.py --speaker albert test.txt test.wav
```

Lisainfot skripti kasutamise kohta saab `--help` parameetriga:

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