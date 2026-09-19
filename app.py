from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load your trained ML model
model = joblib.load("age_weight_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Your model has ONE input feature
        value = float(data["value"])

        prediction = model.predict([[value]])

        return jsonify({
            "success": True,
            "prediction": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)