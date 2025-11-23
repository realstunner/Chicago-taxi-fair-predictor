# Chicago-taxi-fair-predictor
🚕 Chicago Taxi-Fare Predictor A friendly Machine Learning project using LightGBM  This project predicts taxi fares in Chicago using trip details like distance, duration, and time of day. It is designed as a beginner-friendly ML project, with clean code, simple feature engineering, and a prediction function you can run anytime.
A simple and friendly machine-learning project that predicts taxi fares in Chicago using a LightGBM regression model.

This project includes:
- A trained model (`taxi-model(1).joblib`)
- A prediction script (`app.py`) using Streamlit for local deployment
- Feature engineering for realistic fare estimation

---

## 📌 Project Overview
This model predicts the **total taxi fare** based on trip characteristics.  
The dataset is from Chicago Taxi Trips (public dataset), cleaned and processed.

The model uses **LightGBM**, chosen because:
- It is fast even on low-RAM systems (like 8GB)
- Works well with tabular datasets
- Produces accurate predictions after feature engineering

---

## 🧠 Features Used in Prediction

The model uses the following features:

### **Input Features**
| Feature | Description |
|--------|-------------|
| `TRIP_MILES` | Distance traveled in miles |
| `TRIP_SECONDS` | Trip duration in seconds |
| `TRIP_START_HOUR` | Hour of day the trip begins (0–23) |

### **Engineered Features**
| Feature | Description |
|--------|-------------|
| `RUSH_HOUR` | 1 → (7–9 AM or 4–6 PM), else 0 |
| `AVERAGE_SPEED_MPH` | Computed as `miles / (seconds / 3600)` |

### Why Feature Engineering?
It improves prediction accuracy by helping the model understand:
- Traffic effects  
- Driving speed  
- Morning/evening congestion

<img width="845" height="875" alt="image" src="https://github.com/user-attachments/assets/8daa8043-a870-46ad-bbfc-bfbe907a95c8" />
