import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("transactions.csv")

# Show original data types
print("Original data types:\n", df.dtypes)

# Convert categorical columns 'Type' and 'Location' to numerical using LabelEncoder
le_type = LabelEncoder()
df['Type_enc'] = le_type.fit_transform(df['Type'])

le_location = LabelEncoder()
df['Location_enc'] = le_location.fit_transform(df['Location'])

# Display the encoded columns
print("\nSample data with encoded columns:")
print(df[['Type', 'Type_enc', 'Location', 'Location_enc']].head())

# Save processed data to a new CSV (optional)
df.to_csv("transactions_processed.csv", index=False)
