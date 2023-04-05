import numpy as np
from flask import Flask, request, render_template
from model import load
from utils import preprocessor
from utils.label import decode_label

app = Flask(__name__)

model = load.ini()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print(request)
        print(request.files)
        file = request.files['file']
        spec = preprocessor.preprocess(file)
        spec = spec[np.newaxis, ...]
        print(spec.shape)

        prediction = model.predict(spec)
        print(prediction[0])

        label = decode_label(prediction[0])
        print(label)

        return render_template('upload_success.html')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run()
