from flask import Flask, render_template, request
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/adaptivity")
def adaptivity():
    return render_template("adaptivity.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        funding_total_usd = float(request.form.get("funding_total_usd"))
        funding_rounds = float(request.form.get("funding_rounds"))
        age_first_funding_year = float(request.form.get("age_first_funding_year"))
        age_last_funding_year = float(request.form.get("age_last_funding_year"))
        age_first_milestone_year = float(request.form.get("age_first_milestone_year"))
        age_last_milestone_year = float(request.form.get("age_last_milestone_year"))
        relationships = float(request.form.get("relationships"))

        # Prepare input for model
        input_data = np.array([[ 
            funding_total_usd,
            funding_rounds,
            age_first_funding_year,
            age_last_funding_year,
            age_first_milestone_year,
            age_last_milestone_year,
            relationships
        ]])

        # Get prediction probability
        prob = model.predict_proba(input_data)[0][1]

        # Decision logic
        if prob >= 0.5:
            result = f"Startup will be Acquired ✅ (Success Probability: {prob:.2f})"
        else:
            result = f"Startup may be Closed ❌ (Success Probability: {prob:.2f})"

        return render_template("result.html", prediction_text=result)

    except:
        return render_template("result.html",
                               prediction_text="Please enter valid numeric values in all fields.")

if __name__ == "__main__":
    app.run(debug=True)
