import pandas as pd
import matplotlib.pyplot as plt

# Load data with predictions
df = pd.read_csv("transactions_with_predictions.csv")

# Count occurrences of each prediction
counts = df['Prediction_label'].value_counts()

# Plot bar chart
counts.plot(kind='bar', color=['orange', 'green'])
plt.title('Fraud vs Normal Transaction Counts')
plt.xlabel('Transaction Type')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=0)
plt.show()
