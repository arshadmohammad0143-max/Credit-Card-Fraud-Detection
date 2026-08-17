import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. LOAD DATA
# =========================================================

DATA_PATH = "data/creditcard.csv"

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)
print("Columns:")
print(data.columns.tolist())


# =========================================================
# 2. SEPARATE FEATURES AND TARGET
# =========================================================

X = data.drop(columns=["Class"])
y = data["Class"]

print("\nNumber of input features:", X.shape[1])
print("Target column: Class")


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# 4. SCALE ALL 30 FEATURES
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)


# =========================================================
# 5. TRAIN LOGISTIC REGRESSION
# =========================================================

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# =========================================================
# 6. PREDICTION
# =========================================================

y_pred = model.predict(X_test_scaled)


# =========================================================
# 7. MODEL PERFORMANCE
# =========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# =========================================================
# 8. SAVE MODEL AND SCALER
# =========================================================

joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n========================================")
print("FILES SAVED SUCCESSFULLY")
print("========================================")

print("Model  : model/fraud_model.pkl")
print("Scaler : model/scaler.pkl")

print("\nModel expects features:", model.n_features_in_)
print("Scaler expects features:", scaler.n_features_in_)