<div align="center">

# 🛡️ Women Safety Risk Management & Prediction System

**An end-to-end Machine Learning web application that evaluates location-based safety risk levels (`SAFE`, `CAUTION`, `DANGER`) using historical crime analytics.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Model-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/venugopalats58-ctrl/women-safety-risk-management)

</div>

---

## 📌 Overview

The **Women Safety Risk Management & Prediction System** leverages supervised machine learning to classify regions into three actionable safety tiers—**SAFE**, **CAUTION**, and **DANGER**—based on multi-year reported crime indicators across Indian States and Union Territories.

Designed for quick risk assessment and interactive exploration, the application pairs historical regional crime lookups with a real-time custom crime simulator and tailored personal safety recommendations.

---

## ✨ Key Features

- **📍 State & Year Risk Analysis:** Select any Indian State/UT and target year to automatically retrieve historical records, run inference, and view the predicted risk level alongside model confidence.
- **🧮 Custom Crime Metrics Simulator:** Interactively input custom values across 7 crime categories and year to test model behavior and visualize class probability distributions in real time.
- **📊 Transparent Crime Breakdown:** Displays itemized case counts for every feature used by the model so predictions remain interpretable.
- **🚨 Actionable Safety Guidance:** Generates context-aware safety precautions tailored to the predicted risk tier (`SAFE`, `CAUTION`, or `DANGER`).

---

## 🧠 Model & Dataset Features

The classifier (`risk_model.joblib`) evaluates **8 numerical features**:

| Feature Code | Full Description | Type |
| :--- | :--- | :--- |
| **`Year`** | Reporting Year | Temporal |
| **`Rape`** | Reported Rape Cases | Crime Count |
| **`K&A`** | Kidnapping & Abduction | Crime Count |
| **`DD`** | Dowry Deaths | Crime Count |
| **`AoW`** | Assault on Women (Intent to Outrage Modesty) | Crime Count |
| **`AoM`** | Insult to the Modesty of Women | Crime Count |
| **`DV`** | Cruelty by Husband or Relatives (Domestic Violence) | Crime Count |
| **`WT`** | Immoral Traffic / Women Trafficking | Crime Count |

### Risk Classification Tiers
- 🟢 **SAFE (`Low` Risk):** Lower historical crime volume relative to national benchmarks.
- 🟠 **CAUTION (`Medium` Risk):** Moderate crime incidence requiring heightened situational awareness.
- 🔴 **DANGER (`High` Risk):** Elevated crime incidence requiring strict precautionary measures.

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, Joblib
- **Web Interface:** Streamlit

---

## 📂 Repository Structure

```text
women-safety-risk-management/
├── app.py                     # Streamlit interactive web application
├── train_model.py             # Model training, evaluation & serialization pipeline
├── risk_model.joblib          # Serialized model bundle (model, encoder, feature list)
├── CrimesOnWomenData[1].csv   # Historical crimes-against-women dataset
├── requirements.txt           # Project dependencies for local & cloud deployment
└── README.md                  # Project documentation
