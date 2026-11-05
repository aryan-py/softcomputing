from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import os

from neural_network import NeuralNetwork

app = Flask(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

nn = NeuralNetwork.load_model(os.path.join(STATIC_DIR, "model.json"))

with open(os.path.join(STATIC_DIR, "scaler.json"), "r") as f:
    scaler_params = json.load(f)

with open(os.path.join(STATIC_DIR, "stats.json"), "r") as f:
    stats = json.load(f)

CLASSES = ["Setosa", "Versicolor", "Virginica"]


@app.route("/")
def index():
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        float(data["sepal_length"]),
        float(data["sepal_width"]),
        float(data["petal_length"]),
        float(data["petal_width"]),
    ]

    mean = np.array(scaler_params["mean"])
    scale = np.array(scaler_params["scale"])
    X = (np.array(features) - mean) / scale
    X = X.reshape(1, -1)

    probs = nn.predict(X)[0]
    predicted_class = int(np.argmax(probs))

    return jsonify({
        "class": CLASSES[predicted_class],
        "class_index": predicted_class,
        "probabilities": {
            CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(3)
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
