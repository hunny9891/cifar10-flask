from flask import Flask
from flask import jsonify
import predictor
import util

app = Flask(__name__)

X = None
images = None

@app.route("/predict", methods=['GET'])
def predict():
    preds = predictor.predict(X)
    responseList = []
    for pred in preds:
        responseList.append(util.LABEL_DICT[pred])
    data = {
        "predictions": responseList
    }
    response = jsonify(data)

    return response

if __name__== "__main__":
    images, X = predictor.readAndNormalizeImg()
    app.run(port=5000)