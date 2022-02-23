from flask import Flask, request, jsonify
import app.torch_utils as utils
import logging
import io

app = Flask(__name__)


ALLOWED_EXTENSIONS = {'csv', 'txt'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 


stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
app.logger.addHandler(stream_handler)


@app.route('/predictfile', methods=['POST'], endpoint='predictfile')
def predictfile():
    if request.method == 'POST':
        fileobj = request.files.get('file', None)
        if fileobj is None or fileobj.filename == '':
            return jsonify({'error': 'no file'})
        if not allowed_file(fileobj.filename): 
            return jsonify({'error': 'file format not supported'})
        try: 
            file = io.TextIOWrapper(fileobj, encoding="utf-8")
            print('Parsing open textstream...', end='')
            lines = [("", utils.preprocess_NN(line)) for line in file.read().split('\n')] # (blank labels, [tokens])          
            print('COMPLETE.')
        except:
            return jsonify({'error': 'error in file parsing'})
        try:
            print('Sending for prediction...')
            prediction = utils.get_prediction(lines, batch_size=1)
            print('...PREDICTION COMPLETE.')
            return jsonify({'response': prediction}, {'metadata': utils.modelstats_loaded})
        except:
            return jsonify({'error': 'error in prediction'})
    return jsonify({'default result': 1})


@app.route('/predictjson', methods=['POST'], endpoint='predictjson')
def predictjson():
    if request.method == 'POST':
        if request.is_json == False:
            return jsonify({'error': 'no json received'})
        try: 
            print('Parsing json...', end='')
            examples = request.json['data']
            if examples is None or not examples:
                return jsonify({'error': 'empty json received'})
            lines = [("", utils.preprocess_NN(line)) for line in examples] # (blank labels, [tokens])          
            print('COMPLETE.')
        except:
            return jsonify({'error': 'error in file parsing'})
        try:
            print('Sending for prediction...')
            prediction = utils.get_prediction(lines, batch_size=1)
            print('...PREDICTION COMPLETE.')
            return jsonify({'response': prediction}, {'metadata': utils.modelstats_loaded})
        except:
            return jsonify({'error': 'error in prediction'})
    return jsonify({'default result': 1})