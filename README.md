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

##  🤝 Contributions
Have better data, ideas, or simulations? PRs and forks welcome! You can also reach out if you'd like to collaborate on building out the ML model or integrating it with track-specific simulations.

---
