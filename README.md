# bank_fraud_detector
Bank Transaction Fraud Detector uses machine learning to identify suspicious banking transactions automatically. Built with Python, it preprocesses data, trains an Isolation Forest model, and predicts fraud in real time. Easy to use and extensible for larger datasets and enhanced detection accuracy.

# Bank Transaction Fraud Detector
## Overview
This project implements a machine learning-based fraud detection system to automatically identify suspicious banking transactions. Using Python and the Isolation Forest anomaly detection algorithm, it analyzes transaction data such as amount, transaction type, and location, and flags potential fraud in real time.

## Features
Loads and preprocesses transaction data from CSV files.
Encodes categorical features (transaction type and location) for use in machine learning.
Trains an Isolation Forest model to detect anomalies (fraudulent transactions).
Supports interactive user input to predict fraud status of new transactions.
Provides case-insensitive and whitespace-tolerant input handling for ease of use.
Designed for extensibility with larger datasets and model improvements.

# Technologies Used

Python 3

pandas (data manipulation)

scikit-learn (machine learning: Isolation Forest, LabelEncoder)

VS Code (development environment)

# Installation
Before running the scripts, install the necessary Python libraries. Use the command below in your terminal or command prompt:

## bash
pip install pandas scikit-learn
Make sure you have Python 3 installed and pip package manager available.

### How It Works
#### Data Loading: 
Read banking transaction records from a CSV file.

#### Preprocessing: 
Convert categorical data into numerical format.

#### Model Training: 
Fit Isolation Forest on an encoded feature set.

#### Prediction: 
Detect anomalies in new transaction inputs, labeling them as “Fraud” or “Normal”.

#### User Interface: 
Command-line script allows users to input transaction details dynamically and immediately receive fraud predictions.

### Usage
Clone the repository.

Place your transaction CSV file in the project directory.

Run preprocessing and training scripts:

## bash
python preprocess.py
python train_model.py
Predict new transactions interactively with:

## bash
python predict_new.py
Enter the transaction amount, type (Online, ATM, POS), and location as prompted.

# Project Structure
### transactions.csv — Sample data with labeled transactions

### preprocess.py — Data preprocessing (encoding categorical features)

### train_model.py — Train the Isolation Forest model on processed data

### predict_new.py — Interactive script for predicting new transaction fraud status

### transactions_processed.csv — Encoded dataset generated during preprocessing

# Future Improvements
Integrate more transaction features—timestamp, device ID, etc.—for better accuracy.

Add web or GUI interface for real-time fraud alerting.

Deploy using REST API for integration with banking systems.

Experiment with other anomaly detection and classification algorithms.



# Author
Dhruv Meena - dhruv123meena@gmail.com
