
from pathlib import Path

import joblib
import numpy as np

from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "clothing_demand_model.pkl"

app = Flask(__name__)

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python train_model.py first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "frontend.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    features = [
        "price",
        "discount_percent",
        "previous_sales",
        "advertising_spend",
        "season_demand_index"
    ]

    try:
        values = [float(data[name]) for name in features]
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Please enter valid values in every field."
        }), 400

    if not all(np.isfinite(values)):
        return jsonify({
            "error": "All values must be valid numbers."
        }), 400

    price, discount, previous_sales, advertising, season = values

    if price <= 0:
        return jsonify({"error": "Price must be greater than zero."}), 400

    if not 0 <= discount <= 100:
        return jsonify({
            "error": "Discount must be between 0 and 100."
        }), 400

    if previous_sales < 0 or advertising < 0:
        return jsonify({
            "error": "Sales and advertising spend cannot be negative."
        }), 400

    if not 0.5 <= season <= 1.5:
        return jsonify({
            "error": "Season index must be between 0.5 and 1.5."
        }), 400

    # Predict units sold
    prediction = model.predict([values])[0]

    # Demand cannot be negative
    prediction = max(0, prediction)

    return jsonify({
        "predicted_units_sold": int(round(prediction)),
        "message": "This is an estimated demand, not a guarantee."
    })


if __name__ == "__main__":
    app.run(debug=True)
