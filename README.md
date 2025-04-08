# 🏎️ F1 Telemetry ML Analysis

This project explores the use of machine learning to analyze and predict performance from Formula 1 telemetry data. It combines traditional data science techniques with modern ML models to uncover insights, predict cornering behavior, and visualize driver performance over a race stint or qualifying lap.

## 📊 Features

- 🛠️ **Lap-by-lap data analysis** using throttle, brake, gear, RPM, speed, and DRS signals
- 📈 **Driver comparison tools** for sector deltas, overlays, and braking points
- 🤖 **Machine learning models** for:
  - Corner entry prediction
  - Pit stop window classification
  - DRS usage and effectiveness analysis
- 📉 Visualizations using `matplotlib`, `Plotly`, and `seaborn`
- ⚙️ Lightweight, modular data pipeline using `pandas` + `scikit-learn`

## 🧠 Models Used

- `XGBoost` for classification and regression of telemetry segments
- `LSTM` and `GRU` (planned) for time-series modeling of lap traces
- Custom logic for detecting driver behavior patterns

## 📁 File Structure (WIP)

