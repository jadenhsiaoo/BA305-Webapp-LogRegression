# app.py
from flask import Flask, render_template, request, jsonify
from target import load_model_bundle, predict_single
from flask import send_from_directory
import os

app = Flask(__name__)

# Load model bundle (model + feature_means + feature list) at startup
model_bundle = load_model_bundle()


@app.route("/", methods=["GET"])
def index():
    # just renders templates/index.html
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}

    # data is a dict of {feature_name: string_or_number}
    prob_default, filled = predict_single(
        data,
        bundle=model_bundle,
        return_filled=True
    )

    return jsonify({
        "prob_default": prob_default,
        "filled_inputs": filled,
    })

# Route to serve evaluation graphs
@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory("results", filename)


if __name__ == "__main__":
    app.run(debug=True, port=3000)
