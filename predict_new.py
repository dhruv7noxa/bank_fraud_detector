import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")  # Suppress deprecation and user warnings

# Load processed data to fit encoders and model
df = pd.read_csv("transactions_processed.csv")
features = ['Amount', 'Type_enc', 'Location_enc']
X = df[features]

# Fit encoders on uppercase, stripped strings for both training and prediction
le_type = LabelEncoder()
le_type.fit(df['Type'].str.strip().str.upper())

le_location = LabelEncoder()
le_location.fit(df['Location'].str.strip().str.upper())

# Train Isolation Forest model
model = IsolationForest(contamination=0.3, random_state=42)
model.fit(X)

def predict_transaction(amount, txn_type, location):
    txn_type_clean = txn_type.strip().upper()
    location_clean = location.strip().upper()

    if txn_type_clean not in le_type.classes_:
        print(f"Unknown transaction type: {txn_type}")
        print("Valid types are:", [t.title() for t in le_type.classes_])
        return
    if location_clean not in le_location.classes_:
        print(f"Unknown location: {location}")
        print("Valid locations are:", [l.title() for l in le_location.classes_])
        return

    txn_type_enc = le_type.transform([txn_type_clean])[0]
    location_enc = le_location.transform([location_clean])

    # Use DataFrame so sklearn gets feature names
    features_vec = pd.DataFrame([[float(amount), int(txn_type_enc), int(location_enc)]],
                               columns=features)

    pred = model.predict(features_vec)
    return "Fraud" if pred == -1 else "Normal"

amount = float(input("Enter transaction amount: "))
txn_type = input("Enter transaction type (Online/ATM/POS): ")
location = input("Enter location (Delhi, Mumbai, Chennai, Jaipur, Unknown): ")

result = predict_transaction(amount, txn_type, location)
if result:
    print(f"Prediction: {result}")
