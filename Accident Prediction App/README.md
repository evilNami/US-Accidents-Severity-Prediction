# 🛡️ US Accident Severity Predictor

A machine learning web app that predicts the severity of US traffic accidents based on location, weather, time, and road conditions.

**[🚀 Live Demo →](https://ushbxrq5fblrr5abnu22e3.streamlit.app/)**

---

## What it does

Enter details about a traffic accident scenario — state, city, weather conditions, time of day, and nearby road features — and the model predicts whether the accident is likely to be:

- **High Severity** (Level 3–4): Significant road closure expected
- **Low Severity** (Level 1–2): Minor incident with low traffic impact

## Model

- **Dataset**: [US Accidents (March 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) — 7.7M records
- **Algorithm**: XGBoost Classifier (80% test accuracy)
- **Target**: Binary classification — Severe (≥ Level 3) vs. Not Severe
- **Features**: 29 features including weather, location target encodings, temporal factors, and road infrastructure flags

## Tech Stack

- Python, Streamlit
- XGBoost, scikit-learn
- pandas, numpy

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
