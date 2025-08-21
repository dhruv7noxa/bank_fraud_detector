import pandas as pd
from sklearn.ensemble import IsolationForest

# Load processed data
df = pd.read_csv("transactions_processed.csv")

# Select features for training
features = ['Amount', 'Type_enc', 'Location_enc']

X = df[features]

# Train Isolation Forest model
model = IsolationForest(contamination=0.3, random_state=42)  # 30% expected fraud rate
model.fit(X)

# Predict anomalies: -1 = anomaly (fraud), 1 = normal
df['Prediction'] = model.predict(X)
df['Prediction_label'] = df['Prediction'].map({1: 'Normal', -1: 'Fraud'})

# Show results versus actual
print(df[['TransactionID', 'Amount', 'Type', 'Location', 'IsFraud', 'Prediction_label']])

# Save model predictions if needed
df.to_csv("transactions_with_predictions.csv", index=False)
