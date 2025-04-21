# 🏎️ F1 Fuel Optimization 

This project simulates how different fuel properties influence lap times in Formula 1, focusing specifically on performance at the Monaco Grand Prix. Using a dataset built from realistic parameters and race engineering logic, the goal is to train a machine learning model that predicts lap time and helps identify the optimal fuel blend.

---

## 📘 Overview

I generated a synthetic dataset of 100 different F1 fuel blends, based on known values from FIA regulations and fuel suppliers like BP, Shell, Petronas, Aramco, and ExxonMobil. Instead of modeling by team or supplier, I simplified the problem to focus purely on the chemical and physical properties of fuel. Each entry in the dataset represents a unique fuel configuration and its estimated impact on lap time at Monaco — a tight, low-speed, high-downforce circuit where every gram counts.

---

## 🧪 Dataset Parameters

Each row in the dataset includes:

- `RON`: Research Octane Number — impacts knock resistance
- `Energy Density (MJ/kg)`: Energy released per kg of fuel
- `Fuel Density (kg/m³)`: Mass of fuel per volume, used to calculate tank size
- `Combustion Temp (°C)`: Average temperature of combustion
- `Fuel Mass (kg)`: Total fuel needed for a full race (~4400 MJ target)
- `Fuel Burn per Lap (kg)`: Fuel consumption rate over 78 laps
- `Avg Fuel Mass (kg)`: Average fuel carried throughout the race (assumes linear burn)
- `Weight Penalty (s)`: Time lost due to average fuel mass (approx. 0.03s per kg/lap)
- `Fuel Volume (L)`: Total fuel volume required (based on mass and density)
- `Avg Lap Time (s)`: Estimated lap time, based on all above variables

---

## Lap Time Calculation Formula

To model the lap time, I used:

```python
lap_time = (
    72.0 (mid-level F1 car lap time)
    - (RON - 98) * 0.3
    - (Energy_Density - 43.5) * 0.5
    + (Combustion_Temp - 875) * 0.03
    + (Avg_Fuel_Mass * 0.03)
    + random_noise
)
```

---

## Project Goal

- The final goal is to build and train a machine learning model that can:
- Predict lap time from fuel properties.
- Identify optimal fuel configurations that minimize time.
- Factor in fuel mass and volume to stay within realistic packaging constraints.

---

## Roadmap:
- Generate synthetic dataset with realistic physics
- Build a regression model to predict lap times
- Optimize fuel blend to minimize time
- Expand to other circuits with different fuel load sensitivities
- Build an interactive dashboard for testing fuel strategies

---

---

Model Summary

| Model                 | Handles Non-Linearity             | Regularization            | Feature Selection       | Notes                          |
|----------------------|-----------------------------------|---------------------------|--------------------------|--------------------------------|
| Linear Regression     | ❌                                | ❌                        | ❌                       | Simple, fast                   |
| Ridge                 | ❌                                | ✅ L2                     | ❌                       | Penalizes big weights          |
| Lasso                 | ❌                                | ✅ L1                     | ✅                       | Shrinks unimportant features   |
| Random Forest         | ✅                                | ❌                        | ❌                       | Very accurate, black box       |
| Gradient Boosting     | ✅✅                              | ✅                        | ❌                       | Often top performer            |
| Polynomial (Linear)   | ✅ (via feature engineering)      | ✅ (with Ridge/Lasso)     | ✅                       | Can overfit                    |
| SVR                   | ✅                                | ✅                        | ❌                       | Works well on small data       |

---

## 🔍 Results & Insights

After training multiple regression models, the **Polynomial Regression with Ridge Regularization** gave the best results, achieving:

- **Training R²:** 0.91  
- **Validation R²:** 0.84  

Using this model, we simulated thousands of hypothetical fuel blends. By optimizing within FIA limits (≤110 kg fuel mass and ≤100 L volume), we were able to identify the ideal blend for **Monaco 2024**, targeting Charles Leclerc's average winning lap time of **1:50.199**.

This model predicted an optimal configuration that balanced energy density, octane rating, and fuel volume to squeeze out the best performance — all while staying legal under FIA constraints.

---

## 🚀 Future Improvements

To make the project even more realistic:

- Use **real telemetry or sim racing data** (via FastF1, iRacing telemetry exports)
- Simulate **burn rate per lap**, adjusting for mass drop-off
- Replace lap time formula with a **track dynamics model**
- Add **aero and tire models** to capture full car-fuel-lap interaction
- Extend to **race strategy modeling** (stints, undercut risk, etc.)

---

## 🧠 Credits & References

- FIA Technical Regulations  
- Shell Motorsport Fuel Research  
- BP Castrol Fuel Science Briefs  
- FastF1 Python Telemetry Toolkit  
- Motorsport Engineering literature from Oxford Brookes
