import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully!")

# Basic information
print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum().sum())

print("\nTransaction Class:")
print(data["Class"].value_counts())

# Create graph
data["Class"].value_counts().plot(kind="bar")

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.xticks([0, 1], ["Normal", "Fraud"])

plt.show()