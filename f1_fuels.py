## Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Lets import the data and make sure it's reading properly in pandas

data = pd.read_csv("Simulated_F1_Fuel_Dataset.csv")
print(data.head())

## Graph data for a visualization 
sns.set_theme(style="whitegrid")
sns.pairplot(data[["RON", "Energy Density (MJ/kg)", "Fuel Mass (kg)", "Avg Lap Time (s)"]])
# plt.show()

# This dictionary keeps track of the training and validation scores which we will later turn into a dataframe for comparison!
model_scores = []

# Start with linear regression, import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Seperate x and y
X = data.drop(columns=["Avg Lap Time (s)"])
y = data["Avg Lap Time (s)"]

# Create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

# Perform Linear Regression
lr = LinearRegression().fit(X_train, y_train)

# Linear regression coefficients and the model's training + validation scores
feature_names = X.columns

# print("\nLinear Model Coefficients")
# for name, coef in zip(X.columns, lr.coef_):
#     print(f"{name}: {coef:.2f}")
# print(f"\nIntercept: {lr.intercept_:.2f}")
# print("Training score: {:.2f}".format(lr.score(X_train, y_train)))
# print("Validation score: {:.2f}".format(lr.score(X_val, y_val)))

model_scores.append([
    "Linear Regression",
    round(lr.score(X_train, y_train), 2),
    round(lr.score(X_val, y_val), 2)
])

# Perform Ridge Regression
from sklearn.linear_model import Ridge

# Ridge regression coefficients and the model's training + validation scores
ridge = Ridge().fit(X_train, y_train)

# print("\nRidge Regression Model Coefficients")
# for name, coef in zip(X.columns, ridge.coef_):
#     print(f"{name}: {coef:.2f}")
# print("Training score: {:.2f}".format(ridge.score(X_train, y_train)))
# print("Validation score: {:.2f}".format(ridge.score(X_val, y_val)))

model_scores.append([
    "Ridge Regression",
    round(ridge.score(X_train, y_train), 2),
    round(ridge.score(X_val, y_val), 2)
])

# Perform Lasso Regression
from sklearn.linear_model import Lasso

# Lasso regression coefficients and the model's training + validation scores
lasso = Lasso(alpha=0.001).fit(X_train, y_train)

# print("\nLasso Regression Model Coefficients")
# for name, coef in zip(X.columns, lasso.coef_):
#     print(f"{name}: {coef:.2f}")
# print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
# print("Validation set score: {:.2f}".format(lasso.score(X_val, y_val)))
# print("Number of features used:", np.sum(lasso.coef_ != 0))

model_scores.append([
    "Lasso Regression",
    round(lasso.score(X_train, y_train), 2),
    round(lasso.score(X_val, y_val), 2)
])
# Perform Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(    n_estimators=100,
    max_depth=3,                
    min_samples_split=5,        
    min_samples_leaf=3,        
    max_features='sqrt',        
    random_state=42).fit(X_train, y_train)

# print("\nRandom Forest Regression")
# print("Training score: {:.2f}".format(rf.score(X_train, y_train)))
# print("Validation score: {:.2f}".format(rf.score(X_val, y_val)))

model_scores.append([
    "Random Forest Regression",
    round(rf.score(X_train, y_train), 2),
    round(rf.score(X_val, y_val), 2)
])

# Perform Gradient Boosting Regression
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=300,         
    learning_rate=0.02,         
    max_depth=2,               
    min_samples_leaf=4,         
    random_state=42).fit(X_train, y_train)

# print("\nGradient Boosting Regression")
# print("Training score: {:.2f}".format(gbr.score(X_train, y_train)))
# print("Validation score: {:.2f}".format(gbr.score(X_val, y_val)))

model_scores.append([
    "Gradient Boosting Regression",
    round(gbr.score(X_train, y_train), 2),
    round(gbr.score(X_val, y_val), 2)
])

# Perform Polynomial Regression (degree 2)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    Ridge(alpha=1.0)
).fit(X_train, y_train)

# print("\nPolynomial Regression (Degree 2)")
# print("Training score: {:.2f}".format(poly_model.score(X_train, y_train)))
# print("Validation score: {:.2f}".format(poly_model.score(X_val, y_val)))

model_scores.append([
    "Polynomial Regression (Deg 2)",
    round(poly_model.score(X_train, y_train), 2),
    round(poly_model.score(X_val, y_val), 2)
])

# Perform Polynomial Regression (degree 3)
poly_model_3 = make_pipeline(
    PolynomialFeatures(degree=3),
    Ridge(alpha=1.0)
).fit(X_train, y_train)

# print("\nPolynomial Regression (Degree 3)")
# print("Training score: {:.2f}".format(poly_model_3.score(X_train, y_train)))
# print("Validation score: {:.2f}".format(poly_model_3.score(X_val, y_val)))

model_scores.append([
    "Polynomial Regression (Deg 3)",
    round(poly_model_3.score(X_train, y_train), 2),
    round(poly_model_3.score(X_val, y_val), 2)
])

# Perform Polynomial Regression (degree 2) with Ridge Regression

poly_ridge_model = make_pipeline(
    PolynomialFeatures(degree=2),
    Ridge(alpha=1.0)
).fit(X_train, y_train)

# print("\nPolynomial Regression w/ Ridge (Degree 2)")
# print("Training score: {:.2f}".format(poly_ridge_model.score(X_train, y_train)))
# print("Validation score: {:.2f}".format(poly_ridge_model.score(X_val, y_val)))

model_scores.append([
    "Polynomial Regression (Deg 2 w/ Ridge)",
    round(poly_ridge_model.score(X_train, y_train), 2),
    round(poly_ridge_model.score(X_val, y_val), 2)
])

# Perform Support Vector Regression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(C=10, epsilon=0.1, kernel='rbf'))
]).fit(X_train, y_train)

# print("\nSupport Vector Regression")
# print("Training score: {:.2f}".format(svr.score(X_train, y_train)))
# print("Validation score: {:.2f}".format(svr.score(X_val, y_val)))

model_scores.append([
    "Support Vector Regression",
    round(svr.score(X_train, y_train), 2),
    round(svr.score(X_val, y_val), 2)
])

# Converting the results to a dataframe
results_df = pd.DataFrame(
    model_scores, 
    columns=["Model", "Train R²", "Validation R²"]
).sort_values(by="Validation R²", ascending=False)

# RESULTS!
print("\n📊 Model Performance Summary:")
print(results_df)

# Predict using your best model
y_pred = poly_ridge_model.predict(X_val)

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.8)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Lap Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Actual vs Predicted Lap Time – Polynomial + Ridge")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

print("\n🛠️ Lets target a specific lap time using our newly created model...")

# Now lets work backwards. I want to achieve a win. Lets use Charles's data from 2024 Monaco
# His finishing time wasL 2 hrs, 23 mins, 15.554 s - an average lap time of 1:50.199 (over 78 laps)

from scipy.optimize import minimize

# Target average lap time (in seconds)
target_lap_time = 110.199
energy_required = 3600  # MJ

# Define the optimization objective function
def objective(params):
    ron, ed, fd, temp = params
    fuel_mass = energy_required / ed
    avg_mass = fuel_mass / 2
    weight_penalty = avg_mass * 0.03
    predicted_lap_time = (
        90
        - ron * 0.05
        + fuel_mass * 0.01
        + weight_penalty * 5
        - ed * 0.5
    )
    return abs(predicted_lap_time - target_lap_time)

# Constraints
def volume_constraint(params):
    _, ed, fd, _ = params
    fuel_mass = energy_required / ed
    volume = (fuel_mass / fd) * 1000
    return 100 - volume

def mass_constraint(params):
    _, ed, _, _ = params
    fuel_mass = energy_required / ed
    return 110 - fuel_mass

constraints = [
    {'type': 'ineq', 'fun': volume_constraint},
    {'type': 'ineq', 'fun': mass_constraint}
]

# Bounds: [RON, Energy Density, Fuel Density, Temp]
bounds = [
    (97.5, 100.5),
    (43.5, 44.2),
    (740, 820),
    (870, 890)
]

# Run optimization
result = minimize(
    objective,
    x0=[99, 44.0, 790, 880],
    bounds=bounds,
    constraints=constraints,
    method='SLSQP'
)

# Extract optimized values
ron, ed, fd, temp = result.x
fuel_mass = energy_required / ed
volume = (fuel_mass / fd) * 1000
avg_mass = fuel_mass / 2
weight_penalty = avg_mass * 0.03
predicted_lap_time = (
    90
    - ron * 0.05
    + fuel_mass * 0.01
    + weight_penalty * 5
    - ed * 0.5
)

# Show results
final_blend = pd.DataFrame([{
    "RON": ron,
    "Energy Density (MJ/kg)": ed,
    "Fuel Density (kg/m³)": fd,
    "Combustion Temp (°C)": temp,
    "Fuel Mass (kg)": fuel_mass,
    "Fuel Volume (L)": volume,
    "Avg Fuel Mass (kg)": avg_mass,
    "Weight Penalty (s)": weight_penalty,
    "Predicted Lap Time (s)": predicted_lap_time
}])

print("\n🏁 Optimized Fuel Blend to Match Charles Leclerc's Average Lap Time (Monaco 2024):")
print(final_blend.T.round(3))