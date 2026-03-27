# 🛣️ US Accident Severity Prediction

> End-to-end data pipeline and machine learning project analyzing 7.7 million US traffic accidents — from exploratory analysis on a local subset to a distributed big data pipeline on Google Cloud Platform, with a live interactive web app for real-time severity prediction.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ushbxrq5fblrr5abnu22e3.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-006AFF?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Polars](https://img.shields.io/badge/Polars-DataFrame-CD792C?style=for-the-badge&logo=polars&logoColor=white)](https://pola.rs/)
[![Apache Spark](https://img.shields.io/badge/Apache-Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![GCP](https://img.shields.io/badge/Google_Cloud-Dataproc-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/dataproc)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 📌 Project Overview

This project answers a real-world question: **can we predict how severe a traffic accident will be before responders arrive?**

Using the [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) dataset (7.7M records, 49 attributes), the project follows a two-phase approach:

1. **Local Analysis & Model Development** — Deep EDA with Polars, feature engineering, and ML model training in a Colab notebook on a stratified 500K-record sample.
2. **GCP Scale-Out** — The validated pipeline was lifted to Google Cloud Dataproc to process the full 7.7M record dataset using Apache Spark and Hive.

The final XGBoost model is deployed as a **live Streamlit web app** for interactive severity prediction.

---

## 📊 Phase 1: Exploratory Data Analysis

All EDA was performed using **Polars** for its high-performance, memory-efficient DataFrame operations on the full raw dataset.

### Dataset at a Glance

| Attribute | Value |
|---|---|
| Total Records | 7,728,394 |
| Features | 46 |
| Coverage | 49 US states, 2016–2023 |
| Target | Severity (1 = minor → 4 = major road closure) |

### Key Findings

**Severe class imbalance** — Severity 2 dominates at ~80% of all records, making naive accuracy a misleading metric. This guided the choice of class-weighted models.

| Severity Level | Description | Share |
|---|---|---|
| 1 — Minor | Low traffic impact | ~0.9% |
| 2 — Moderate | Most common scenario | ~79.7% |
| 3 — Significant | Road partially blocked | ~17.0% |
| 4 — Severe | Full road closure | ~2.6% |

**Geographic concentration** — Top 105 cities (0.77% of all cities) account for the vast majority of incidents. Miami, Houston, Dallas, and Charlotte lead in accident volume, reflecting both population density and reporting coverage.

**Temporal patterns** — Accidents spike during morning and evening rush hours and show distinct weekday vs. weekend distributions. Night-time accidents correlate with higher severity.

**Weather & road features** — Direct weather-to-severity correlations are weak; severity is better captured through **engineered interaction features** (e.g., night + low visibility) and **target-encoded risk scores** per location and weather condition.

**High-sparsity features excluded** — `End_Lat`, `End_Lng`, `Wind_Chill`, and `Precipitation` had >50% missing values and were dropped. `Year` and `Pressure` showed extreme multicollinearity (VIF > 4000) and were removed.

---

## ⚙️ Phase 2: Feature Engineering

A core focus of this project was crafting **29 high-signal features** from raw attributes:

### Target-Encoded Risk Scores
Rather than one-hot encoding high-cardinality fields, each category was replaced with its **mean historical severity** — capturing geographic and weather risk compactly.

| Feature | Description |
|---|---|
| `City_Sev_Avg` | Average severity for this city |
| `State_Sev_Avg` | Average severity for this state |
| `Weather_Condition_Sev_Avg` | Risk score for this weather type |
| `Wind_Direction_Sev_Avg` | Risk score for wind direction |

### Interaction & Derived Features

| Feature | Logic |
|---|---|
| `Night_Low_Vis` | Is night AND visibility < 2 miles |
| `High_Speed_Potential` | No traffic signal AND distance > 1 mile |
| `Impact_Intensity` | Distance / (duration + 1) — proxy for spread |
| `Log_Distance` | Log-transformed distance impacted |
| `Road_Feature_Count` | Count of nearby infrastructure features |
| `Is_Night` | Hour < 6 or > 20 |
| `Is_Weekend` | Extracted from timestamp |

### Temporal & Road Features
`Hour`, `Month`, and 11 binary road flags (`Junction`, `Traffic_Signal`, `Crossing`, `Stop`, `Railway`, etc.) round out the feature set.

---

## 🤖 Phase 3: Model Training & Selection

Models were trained on a **stratified 500K-record sample** (80/20 train-test split, class-balanced) in a Colab notebook.

The target was binarized: **Is_Severe = 1** if Severity ≥ 3, else 0.

### Random Forest vs. XGBoost

Both models used class-balancing strategies to handle the 4:1 imbalance.

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | 80% | 80% |
| Macro F1 | 0.73 | 0.73 |
| Recall (Severe class) | 0.80 | **0.81** |
| Precision (Severe class) | 0.48 | 0.48 |

**XGBoost was selected** as the production model for three reasons:
- Marginally higher recall on the severe class — more important than precision in safety-critical prediction
- Lower `max_depth` (6 vs. 12) means faster inference with equivalent accuracy
- Native `scale_pos_weight` parameter handles class imbalance more gracefully than Random Forest's `balanced_subsample`

The trained XGBoost model, scaler, and feature artifacts were serialized as `.pkl` files for deployment.

---

## ☁️ Phase 4: Scaling to Google Cloud Platform

The validated pipeline was scaled to the **full 7.7M record dataset** on GCP Dataproc, running Spark 3.3 on a managed cluster.

### Pipeline Stages

```
Raw CSV (GCS)
    │
    ▼
[1] PySpark Data Cleaning      → Deduplication, null filtering, type casting, outlier removal
    │
    ▼
[2] Hive EDA                   → External table over cleaned CSV, aggregation queries
    │
    ▼
[3] PySpark EDA                → Parallel analysis, coalesce to single output files
    │
    ▼
[4] Spark MLlib — Random Forest → Full-dataset model training (100 trees, depth 10, 80/20 split)
    │
    ▼
Predictions + Confusion Matrix saved to GCS
```

The distributed model on GCP confirmed the findings from the local notebook — validating that the feature engineering and model architecture generalize well beyond the 500K sample.

---

## 🌐 Live Web App

**[🚀 Try it live →](https://ushbxrq5fblrr5abnu22e3.streamlit.app/)**

Built with Streamlit, the app lets you simulate any accident scenario and instantly get a severity prediction from the XGBoost model.

### Inputs
- **Location** — State and city (dropdown, populated from training data)
- **Weather** — Condition, temperature, humidity, pressure, wind speed & direction
- **Time** — Hour of day, month, weekend toggle
- **Road** — Distance impacted, nearby road features (junction, signal, crossing, etc.)

### Output
- Binary classification: **High Severity** (Level 3–4) vs. **Low Severity** (Level 1–2)
- Probability score with a visual confidence bar

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Manipulation | Polars, pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Folium |
| Machine Learning | scikit-learn, XGBoost |
| Model Serialization | joblib |
| Web App | Streamlit |
| Big Data Processing | Apache Spark / PySpark |
| SQL Analytics | Apache Hive (HiveQL) |
| Cloud Platform | Google Cloud Dataproc |
| Cloud Storage | Google Cloud Storage (GCS) |
| Notebook Environment | Google Colab |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/US-Accidents-Severity-Prediction.git
cd US-Accidents-Severity-Prediction

# Install dependencies
pip install -r requirements.txt

# Launch the web app
cd "Accident Prediction App"
streamlit run app.py
```

All `.pkl` model artifacts are included in the `Accident Prediction App/` directory — no retraining required.

---

## 📁 Repository Structure

```
US-Accidents-Severity-Prediction/
├── Accident Prediction App/
│   ├── app.py                          # Streamlit app
│   ├── accident_severity_model.pkl     # Trained XGBoost model
│   ├── data_scaler.pkl                 # Fitted StandardScaler
│   ├── feature_columns.pkl             # Ordered feature names
│   ├── target_mappings.pkl             # Target-encoded averages
│   └── state_city_map.pkl              # State → city lookup
├── Scripts/
│   ├── Notebook/
│   │   └── Prediction_of_Accident_Severity.ipynb   # Full EDA + ML notebook
│   └── onGCP/
│       ├── DataCleaning/clean.py       # PySpark cleaning job
│       ├── EDA/
│       │   ├── eda_spark.py            # PySpark EDA
│       │   └── edaHive.txt             # HiveQL queries
│       ├── ML/severity.py              # Spark MLlib pipeline
│       └── SummaryStats/SummaryHiveQl.txt
├── Output/
│   ├── Cleaned_data/
│   ├── EDA_output/
│   │   ├── Hive/
│   │   └── Spark/
│   └── ML_output/
│       ├── ML_predictions_random_forest_*.csv
│       └── confusion_matrix_rf.png
├── requirements.txt
└── README.md
```

---

## 👤 Author

**Pradeepta Dey**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
