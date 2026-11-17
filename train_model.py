import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

PROJECT_DIR = r"C:\Users\venug\OneDrive\Desktop\ml projects\riskmangment"
MODEL_PATH = os.path.join(PROJECT_DIR, "risk_model.joblib")


def find_csv():
    """Find the CrimesOnWomen CSV in the project folder, whatever its exact name."""
    for name in os.listdir(PROJECT_DIR):
        if name.lower().endswith(".csv") and "crimes" in name.lower():
            return os.path.join(PROJECT_DIR, name)
    raise FileNotFoundError(f"No CrimesOnWomen CSV file found in: {PROJECT_DIR}")

CRIME_COLS = ["Rape", "K&A", "DD", "AoW", "AoM", "DV", "WT"]
FEATURES = ["Year"] + CRIME_COLS

df = pd.read_csv(find_csv())
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

df["Total_Crimes"] = df[CRIME_COLS].sum(axis=1)
df["Risk_Level"] = pd.qcut(df["Total_Crimes"], q=3, labels=["Low", "Medium", "High"])

le = LabelEncoder()
y = le.fit_transform(df["Risk_Level"])

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(df[FEATURES], y)

bundle = {
    "model": model,
    "model_name": "Logistic Regression",
    "label_encoder": le,
    "features": FEATURES,
}
joblib.dump(bundle, MODEL_PATH)
print("Model saved to:", MODEL_PATH)
