
from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)

# Load trained ML model
MODEL_PATH = Path("student_result_model.pkl")
model = joblib.load(MODEL_PATH)

# Input features
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Result Prediction API is running",
        "status": "success"
    }), 200


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    }), 200


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Please provide JSON input"
            }), 400

        # Check missing features
        missing_features = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_fields": missing_features
            }), 400

        # Prepare input data
        input_data = [[
            float(data[feature])
            for feature in FEATURES
        ]]

        # Predict student result
        prediction = model.predict(input_data)[0]

        prediction_code = int(prediction)

        result = "PASS" if prediction_code == 1 else "FAIL"

        return jsonify({
            "prediction": result,
            "prediction_code": prediction_code
        }), 200

    except ValueError:
        return jsonify({
            "error": "Feature values must be numeric"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# Run Flask application
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
