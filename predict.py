import pandas as pd
import joblib


# ============================================================
# LOAD MODEL, SCALER AND DATASET
# ============================================================

model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

data = pd.read_csv("data/creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]


# ============================================================
# TITLE
# ============================================================

print("=" * 60)
print("        CREDIT CARD FRAUD DETECTION SYSTEM")
print("=" * 60)

print("\n1. Test Normal Transaction")
print("2. Test Fraud Transaction")
print("3. Enter Transaction Number")
print("4. Exit")


# ============================================================
# USER CHOICE
# ============================================================

choice = input("\nEnter your choice: ")


# ============================================================
# SELECT TRANSACTION
# ============================================================

if choice == "1":

    # Find first normal transaction
    row_number = y[y == 0].index[0]

elif choice == "2":

    # Find first fraud transaction
    row_number = y[y == 1].index[0]

elif choice == "3":

    row_number = int(
        input(
            f"Enter transaction number (0-{len(data)-1}): "
        )
    )

elif choice == "4":

    print("\nProgram closed.")
    exit()

else:

    print("\nInvalid choice.")
    exit()


# ============================================================
# GET TRANSACTION
# ============================================================

transaction = X.iloc[[row_number]].copy()


# ============================================================
# SCALE AMOUNT
# ============================================================

transaction["Amount"] = scaler.transform(
    transaction[["Amount"]]
)


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(transaction)[0]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("                  RESULT")
print("=" * 60)

print("\nTransaction number:", row_number)
print("Transaction amount:", data.iloc[row_number]["Amount"])


if prediction == 1:

    print("\n🚨 FRAUD TRANSACTION DETECTED!")

else:

    print("\n✅ NORMAL TRANSACTION")


# ============================================================
# ACTUAL RESULT
# ============================================================

actual = y.iloc[row_number]

print("\nActual dataset result:")

if actual == 1:
    print("Actual: FRAUD")
else:
    print("Actual: NORMAL")


print("\n" + "=" * 60)