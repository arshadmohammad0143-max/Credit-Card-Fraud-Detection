from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==========================================
# FILE PATHS
# ==========================================

MODEL_PATH = "model/fraud_model.pkl"
SCALER_PATH = "model/scaler.pkl"
DATA_PATH = "data/creditcard.csv"


# ==========================================
# LOAD MODEL, SCALER AND DATASET
# ==========================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

data = pd.read_csv(DATA_PATH)


# ==========================================
# IMPORTANT:
# USE EXACTLY THE 30 FEATURES USED FOR TRAINING
# ==========================================

FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


# Check that dataset has all required columns
missing_columns = [
    column for column in FEATURE_COLUMNS
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


print("========================================")
print("APPLICATION STARTED")
print("========================================")
print("Dataset shape:", data.shape)
print("Number of features:", len(FEATURE_COLUMNS))
print("Features:", FEATURE_COLUMNS)
print("========================================")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# FRAUD PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # --------------------------------------
        # Get transaction number from HTML
        # --------------------------------------

        transaction_number = int(
            request.form["transaction_number"]
        )

        # --------------------------------------
        # Check transaction number
        # --------------------------------------

        if transaction_number < 0 or transaction_number >= len(data):

            return render_template(
                "index.html",
                error="Invalid transaction number!"
            )

        # --------------------------------------
        # Get selected transaction
        # --------------------------------------

        transaction = data.iloc[transaction_number]

        # --------------------------------------
        # Get actual value
        # --------------------------------------

        actual_value = int(transaction["Class"])

        # --------------------------------------
        # IMPORTANT:
        # Select EXACTLY 30 model features
        # --------------------------------------

        features = transaction[FEATURE_COLUMNS]

        # Convert to DataFrame with one row
        features = pd.DataFrame(
            [features.values],
            columns=FEATURE_COLUMNS
        )

        print("----------------------------------------")
        print("Transaction number:", transaction_number)
        print("Features shape BEFORE scaling:", features.shape)
        print("Number of features:", len(features.columns))
        print("----------------------------------------")

        # --------------------------------------
        # Scale the 30 features
        # --------------------------------------

        features_scaled = scaler.transform(features)

        print(
            "Features shape AFTER scaling:",
            features_scaled.shape
        )

        # --------------------------------------
        # Prediction
        # --------------------------------------

        prediction = int(
            model.predict(features_scaled)[0]
        )

        # --------------------------------------
        # Probability
        # --------------------------------------

        probability = float(
            model.predict_proba(features_scaled)[0][1] * 100
        )

        # --------------------------------------
        # Transaction amount
        # --------------------------------------

        amount = float(transaction["Amount"])

        # --------------------------------------
        # Prediction result
        # --------------------------------------

        if prediction == 1:
            result = "FRAUD TRANSACTION"
        else:
            result = "NORMAL TRANSACTION"

        # --------------------------------------
        # Actual dataset result
        # --------------------------------------

        if actual_value == 1:
            actual_result = "FRAUD"
        else:
            actual_result = "NORMAL"

        # --------------------------------------
        # Print result in terminal
        # --------------------------------------

        print("========================================")
        print("TRANSACTION RESULT")
        print("========================================")
        print("Transaction:", transaction_number)
        print("Amount:", amount)
        print("Prediction:", prediction)
        print("Result:", result)
        print("Fraud Probability:", round(probability, 2), "%")
        print("Actual:", actual_result)
        print("========================================")

        # --------------------------------------
        # Send result to HTML
        # --------------------------------------

        return render_template(
            "index.html",
            show_result=True,
            prediction=prediction,
            result=result,
            probability=round(probability, 2),
            transaction_number=transaction_number,
            amount=round(amount, 2),
            actual=actual_result
        )

    except Exception as e:

        print("========================================")
        print("ERROR:")
        print(e)
        print("========================================")

        return render_template(
            "index.html",
            error=str(e)
        )

# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )