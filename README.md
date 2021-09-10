# Eestikeelne kõnesüntees kõnesünteesi API

Skriptid eestikeelse mitmehäälse kõnesünteesi kasutamiseks teksifaili põhjal. Kood sisaldab alammooduleid, mis viitavad järgmistele kõnesünteesi komponentidele:
 - [eesti keelele kohandatud Deep Voice 3](https://github.com/TartuNLP/deepvoice3_pytorch)
 - [eestikeelse kõnesünteesi eeltöötlus](https://github.com/TartuNLP/tts_preprocess_et)
 
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
conda env create -f environment.yml
conda activate deepvoice
pip install --no-deps -e "deepvoice3_pytorch/[bin]"
python -c 'import nltk; nltk.download("punkt"); nltk.download("cmudict")'
```
- Lae alla meie [Deep Voice 3 mudel](https://github.com/TartuNLP/deepvoice3_pytorch/releases/kratt-v1.2) ja aseta 
  see `models/` kausta. Siin viidatud mudel toetab kuue kõneleja häält.

## Kasutamine

Tekstifaili saab sünteesida järgmise käsuga. Hetkel oskab skript lugeda vaid toorteksti kujul faile ja salvestab 
väljundi `.wav` formaadis.

```
python file_synthesis.py test.txt test.wav
```

Lisainfot skripti kasutamise kohta saab `--help` parameetriga:

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