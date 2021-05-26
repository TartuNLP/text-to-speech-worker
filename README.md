# Eestikeelse kõnesünteesi API

Siit repositooriumist leiab lihtsa API, mis võimaldab käivitada eestikeelse mitmehäälse kõnesünteesi
 serverit. Kood sisaldab alammooduleid, mis viitavad järgmistele kõnesünteesi komponentidele:
 - [eesti keelele kohandatud Deep Voice 3](https://github.com/TartuNLP/deepvoice3_pytorch)
 - [eestikeelse kõnesünteesi eeltöötlus](https://github.com/TartuNLP/tts_preprocess_et)
 
Kõnesüntees on loodud koostöös [Eesti Keele Instituudiga](http://portaal.eki.ee/) ja seda on võimalik
 kasutada ka meie [veebidemos](https://www.neurokone.ee).
 
## Kasutamine
API kasutamiseks tuleb veebiserverile saata järgmises formaadis POST päring, kus parameeter `text` viitab sünteesitavale tekstile ja `speaker_id` soovitud häälele.

POST `/api/v1.0/synthesize`

BODY (JSON):
```
{
    "text": "Tere."
    "speaker_id": 0
}
```
Server tagastab binaarsel kujul .wav formaadis helifaili. Parameeter `speaker_id` ei ole kohtustuslik ning vaikimisi kasutatakse esimest häält.

Käesolevas versioonis viidatud [mudel](https://github.com/TartuNLP/deepvoice3_pytorch/releases/tag/kratt-v1.2) toetab
 kuut erinevat häält.

## Nõuded ja seadistamine

Siinseid instruktsioone on testitud Ubuntu 18.04-ga. Kood on nii CPU- kui GPU-sõbralik.

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
- Lae alla meie [Deep Voice 3 mudel](https://github.com/TartuNLP/deepvoice3_pytorch/releases/download/kratt-v1.2/autosegment.pth)

- Loo konfiguratsiooni fail. Kontrolli, et parameeter `checkpoint` viitaks eelmises punktis alla laetud
 mudeli failile.
```
cp config.sample.json config.json
```

Seadista veebiserveri, mis jooksutaks `tts_server.py` faili või testi API kasutust nii:
```
export FLASK_APP=tts_server.py
flask run
```