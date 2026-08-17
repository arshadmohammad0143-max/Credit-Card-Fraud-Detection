import pandas as pd

data = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully!")
print()
print("Number of rows and columns:")
print(data.shape)

print()
print("Column names:")
print(data.columns.tolist())

print()
print("First 5 rows:")
print(data.head())

print()
print("Fraud / Normal count:")
print(data["Class"].value_counts())