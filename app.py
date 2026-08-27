from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
le_location = joblib.load("le_location.pkl")
le_time = joblib.load("le_time.pkl")
le_risk = joblib.load("le_risk.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    location = request.form["location"]
    time = request.form["time"]
    crime_rate = int(request.form["crime_rate"])

    location = le_location.transform([location])[0]
    time = le_time.transform([time])[0]

    prediction = model.predict([[location,time,crime_rate]])
    result = le_risk.inverse_transform(prediction)[0]

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
