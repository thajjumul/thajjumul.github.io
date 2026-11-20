"""
ONLINE PAYMENT FRAUD DETECTION
A simple classroom Machine Learning project.

Run:
    python app.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_FILE = "transaction_data.csv"


def create_sample_data():
    """Create a demonstration dataset with 2,000 synthetic transactions."""
    rows = []

    # A fixed seed makes the same sample data every time.
    data = pd.util.testing if False else None
    import random
    random.seed(42)

    for _ in range(2000):
        amount = round(random.uniform(100, 10000), 2)
        time = random.randint(0, 23)
        location_change = random.randint(0, 1)
        recent_txns = random.randint(0, 12)
        device_change = random.randint(0, 1)
        failed_attempts = random.randint(0, 4)

        # Simple rule used only to create teaching/demo labels.
        risk_score = 0
        if amount > 5000:
            risk_score += 1
        if time <= 5:
            risk_score += 1
        if location_change == 1:
            risk_score += 1
        if recent_txns >= 8:
            risk_score += 1
        if device_change == 1:
            risk_score += 1
        if failed_attempts >= 2:
            risk_score += 1

        # Label 1 means fraud; label 0 means genuine.
        fraud = 1 if risk_score >= 3 else 0

        rows.append([
            amount, time, location_change, recent_txns,
            device_change, failed_attempts, fraud
        ])

    columns = [
        "amount", "time", "location_change", "recent_txns",
        "device_change", "failed_attempts", "fraud"
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(DATA_FILE, index=False)
    return df


def train_model(df):
    """Separate input features and target, then train Random Forest."""
    X = df.drop("fraud", axis=1)
    y = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n========== MODEL PERFORMANCE ==========")
    print(f"Accuracy : {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0) * 100:.2f}%")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0) * 100:.2f}%")
    print(f"F1 Score : {f1_score(y_test, predictions, zero_division=0) * 100:.2f}%")
    print("=======================================\n")

    return model, X.columns


def get_integer(prompt, minimum, maximum):
    """Read and validate an integer from the user."""
    while True:
        try:
            value = int(input(prompt))
            if minimum <= value <= maximum:
                return value
            print(f"Please enter a number from {minimum} to {maximum}.")
        except ValueError:
            print("Please enter a valid whole number.")


def predict_transaction(model, feature_columns):
    print("\n========== ENTER TRANSACTION DETAILS ==========")

    while True:
        try:
            amount = float(input("Transaction amount (₹): "))
            if amount > 0:
                break
            print("Amount must be greater than zero.")
        except ValueError:
            print("Please enter a valid amount.")

    time = get_integer("Transaction hour (0-23): ", 0, 23)
    location_change = get_integer("Location changed? (0=No, 1=Yes): ", 0, 1)
    recent_txns = get_integer("Recent transactions (0-12): ", 0, 12)
    device_change = get_integer("Device changed? (0=No, 1=Yes): ", 0, 1)
    failed_attempts = get_integer("Failed attempts (0-4): ", 0, 4)

    new_transaction = pd.DataFrame(
        [[amount, time, location_change, recent_txns,
          device_change, failed_attempts]],
        columns=feature_columns
    )

    prediction = model.predict(new_transaction)[0]
    fraud_probability = model.predict_proba(new_transaction)[0][1] * 100

    print("\n=============== PREDICTION ===============")
    if prediction == 1:
        print("RESULT: FRAUDULENT TRANSACTION")
    else:
        print("RESULT: GENUINE TRANSACTION")
    print(f"Fraud probability: {fraud_probability:.2f}%")
    print("==========================================")
    print("Note: This is an educational demo, not a real banking decision.")


def main():
    print("==========================================")
    print("     ONLINE PAYMENT FRAUD DETECTION")
    print("==========================================")
    print("Preparing sample transaction data...")

    df = create_sample_data()
    print(f"Dataset created: {len(df)} transactions")
    print(f"Dataset saved as: {DATA_FILE}")

    model, feature_columns = train_model(df)
    predict_transaction(model, feature_columns)


if __name__ == "__main__":
    main()
