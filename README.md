# 🛡️ Women Safety Risk Prediction System

A Machine Learning web application built with **Python**, **Scikit-learn**, and **Streamlit** that classifies location safety into **SAFE**, **CAUTION**, or **DANGER** based on historical crimes-against-women data across Indian States and Union Territories.

---

## 🚀 Features
- **Location-Based Risk Lookup:** Select any State/UT and target year to analyze historical crime records and predict the safety verdict (`SAFE`, `CAUTION`, or `DANGER`) with model confidence scores.
- **Custom Crime Metrics Simulator:** Test the trained model directly by adjusting crime indicators (`Rape`, `K&A`, `DD`, `AoW`, `AoM`, `DV`, `WT`) and viewing class probabilities.
- **Actionable Safety Recommendations:** Displays tailored safety precautions and a detailed breakdown of reported crime categories.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
- **Frontend / Web App:** Streamlit

---

## 📂 Project Structure
```text
women-safety-risk-management/
├── app.py                     # Streamlit web application
├── train_model.py             # Model training and serialization script
├── risk_model.joblib          # Trained ML model bundle
├── CrimesOnWomenData[1].csv   # Historical crime dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation