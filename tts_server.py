# coding: utf-8
import os
import json
from flask_cors import CORS
from flask import Flask, send_file, jsonify
from flask_restful import Api, Resource, reqparse, abort
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


with open('config.json') as config_file:
    config = json.load(config_file)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
api = Api(app)
CORS(app)

with open(config['preset']) as f:
    hparams.parse_json(f.read())
print(hparams_debug_string())

synthesizer = Synthesizer(config['preset'], config['checkpoint'])
print("Deepvoice3 initialized.")


@app.route('/')
def index():
    return "Eestikeelse kõnesünteesi API"


def synthesize(args):
    text = args.get('text')
    speaker_id = args.get('speaker_id')

    if text == '' or speaker_id not in config['allowed_speakers']:
        speaker_id = config['allowed_speakers'][0]

    try:
        return synthesizer.synthesize(text, speaker_id, config['trim_threshold'])
    except ValueError:
        abort(413)


class AudioAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('text', type=str, required=True, help='No text provided', location='json')
        self.reqparse.add_argument('speaker_id', type=int, default=config['allowed_speakers'][0],
                                   help='No speaker id provided', location='json')
        super(AudioAPI, self).__init__()

    def post(self):
        data = synthesize(self.reqparse.parse_args())
        return send_file(data, mimetype='audio/wav')


class AudioAPIJSON(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('text', type=str, required=True, help='No text provided', location='json')
        self.reqparse.add_argument('speaker_id', type=int, default=config['allowed_speakers'][0],
                                   help='No speaker id provided', location='json')
        super(AudioAPIJSON, self).__init__()

    def post(self):
        data = synthesize(self.reqparse.parse_args())

        byte_str = data.getvalue()
        new_data = byte_str.decode('ISO-8859-1')
        return jsonify({'audio': new_data})


api.add_resource(AudioAPI, '/api/v1.0/synthesize', endpoint='audio')
api.add_resource(AudioAPIJSON, '/api/v1.0/synthesize/json', endpoint='audio-json')


if __name__ == '__main__':
    app.run(config['host'], config['port'], debug=True)
