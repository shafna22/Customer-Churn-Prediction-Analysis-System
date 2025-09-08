from flask import Flask, render_template, request
import numpy as np
import os
import json
import joblib  # use joblib since you saved models with it

app = Flask(__name__)

# --- Paths ---
MODEL_PATH = os.path.join("models", "churn_model.pkl")   # unified model name
SCALER_PATH = os.path.join("models", "scaler.pkl")
METADATA_PATH = os.path.join("models", "metadata.json")

# --- Load model, scaler, metadata ---
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
FEATURES = metadata["features"]  # exact feature order from training

# Define numeric columns (scaled during training)
NUMERIC_COLS = ["tenure", "MonthlyCharges", "SeniorCitizen"]


# --- Convert form to feature row ---
def build_feature_row(form):
    # Initialize all features to 0
    row_dict = {feat: 0 for feat in FEATURES}

    for field in form:
        val = form.get(field)

        # ✅ Handle Yes/No checkboxes
        if str(val).lower() == "yes":
            val = 1
        elif str(val).lower() == "no":
            val = 0

        # ✅ Handle categorical dummy (like gender_Male, InternetService_Fiber optic, etc.)
        dummy_col = f"{field}_{val}"
        if dummy_col in row_dict:
            row_dict[dummy_col] = 1
            continue  # stop here, don’t try to cast to float

        # ✅ Handle numeric columns
        if field in NUMERIC_COLS:
            try:
                row_dict[field] = float(val)
            except:
                row_dict[field] = 0.0

    # Ensure order matches training
    row = np.array([[row_dict[feat] for feat in FEATURES]], dtype=float)

    # ✅ Scale only numeric cols
    numeric_indices = [FEATURES.index(feat) for feat in NUMERIC_COLS if feat in FEATURES]
    if numeric_indices:
        row[:, numeric_indices] = scaler.transform(row[:, numeric_indices])

    return row




# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = build_feature_row(request.form)
        prob = model.predict_proba(row)[0, 1]
        pred = "Likely to Churn" if prob > 0.5 else "Not Likely to Churn"
        return render_template("result.html", prediction=pred, probability=round(prob * 100, 2))
    except Exception as e:
        return f"Error in prediction: {e}"


@app.route("/dashboard")
def dashboard():
    # Pick only the plots you want to display
    selected_plots = [
        "churn_distribution_donut.png",
        "churn_by_contract.png",
        "churn_by_payment.png",
        "churn_by_internet.png",
        "churn_by_gender.png",
        "churn_by_senior.png",
        "dist_tenure.png",
        "dist_MonthlyCharges.png",
        "dist_TotalCharges.png"
        
    ]
    return render_template("dashboard.html", plots=selected_plots)


if __name__ == "__main__":
    app.run(debug=True)
