import pandas as pd

# Load the transactions data
df = pd.read_csv("transactions.csv")

# Display the first 5 rows
print("First 5 transactions:")
print(df.head())

# Check the shape of the dataset
print("\nTotal rows, columns:", df.shape)
