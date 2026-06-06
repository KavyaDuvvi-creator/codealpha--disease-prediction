from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)
model = joblib.load("model.pkl")

# In-memory history store
prediction_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([{
        "Age":                     int(data["Age"]),
        "Gender":                  data["Gender"],
        "Cholesterol":             int(data["Cholesterol"]),
        "Blood Pressure":          int(data["Blood Pressure"]),
        "Heart Rate":              int(data["Heart Rate"]),
        "Smoking":                 data["Smoking"],
        "Alcohol Intake":          data["Alcohol Intake"],
        "Exercise Hours":          int(data["Exercise Hours"]),
        "Family History":          data["Family History"],
        "Diabetes":                data["Diabetes"],
        "Obesity":                 data["Obesity"],
        "Stress Level":            int(data["Stress Level"]),
        "Blood Sugar":             int(data["Blood Sugar"]),
        "Exercise Induced Angina": data["Exercise Induced Angina"],
        "Chest Pain Type":         data["Chest Pain Type"],
    }])

    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    prob_pct    = round(float(probability) * 100, 2)

    record = {
        "time":       datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "age":        int(data["Age"]),
        "gender":     data["Gender"],
        "chest_pain": data["Chest Pain Type"],
        "prediction": int(prediction),
        "probability": prob_pct,
    }
    prediction_history.insert(0, record)   # newest first

    return jsonify({
        "prediction":  int(prediction),
        "probability": prob_pct,
        "history":     prediction_history[:10]   # return latest 10
    })

@app.route("/history")
def history():
    return jsonify(prediction_history[:10])

@app.route("/clear", methods=["POST"])
def clear():
    prediction_history.clear()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)
