from flask import Flask, jsonify, request, render_template

import predictor
import util

app = Flask(__name__)

X = None
images = None

@app.route("/")
def pilot():
    return render_template('pilot.html')


@app.route("/predict", methods=['POST'])
def predict():
    rawImage = request.files['file']
    X = predictor.readAndNormalizeImg(rawImage)

    preds = predictor.predict(X)
    trueLabel = util.LABEL_DICT[preds[0]]

    data = {
        "prediction": trueLabel
    }
    response = jsonify(data)

    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
