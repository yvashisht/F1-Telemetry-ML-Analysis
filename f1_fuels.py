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

print("\n🛠️ Starting fuel blend optimization (vectorized)...")

# Step 1: Generate a large number of random blends
n_samples = 5000  # Can be increased for more coverage
blend_data = pd.DataFrame({
    "RON": np.random.uniform(97.5, 100.5, n_samples),
    "Energy Density (MJ/kg)": np.random.uniform(43.5, 44.2, n_samples),
    "Fuel Density (kg/m³)": np.random.uniform(750, 810, n_samples),  # ⬅️ raised
    "Combustion Temp (°C)": np.random.uniform(870, 890, n_samples),
})

# Step 2: Calculate derived columns
blend_data["Fuel Mass (kg)"] = 4400 / blend_data["Energy Density (MJ/kg)"]
blend_data["Fuel Burn per Lap (kg)"] = blend_data["Fuel Mass (kg)"] / 78
blend_data["Avg Fuel Mass (kg)"] = blend_data["Fuel Mass (kg)"] / 2
blend_data["Weight Penalty (s)"] = blend_data["Avg Fuel Mass (kg)"] * 0.03
blend_data["Fuel Volume (L)"] = (blend_data["Fuel Mass (kg)"] / blend_data["Fuel Density (kg/m³)"]) * 1000

# Step 3: Filter by FIA constraints
legal_blends = blend_data[
    (blend_data["Fuel Mass (kg)"] <= 110) &
    (blend_data["Fuel Volume (L)"] <= 100)
].copy()

print(f"\n✅ {len(legal_blends)} FIA-legal fuel blends generated out of {n_samples} total.")

# Step 4: Predict lap time
if legal_blends.empty:
    print("❌ No valid blends found. Try loosening constraints or increasing sample size.")
else:
    try:
        legal_blends = legal_blends[X.columns]  # Match model input
        legal_blends["Predicted Lap Time (s)"] = poly_ridge_model.predict(legal_blends)
        print("✅ Prediction completed.")
    except Exception as e:
        print("❌ Prediction failed:", e)
        raise

    # Step 5: Show optimal configuration
    best_blend = legal_blends.sort_values("Predicted Lap Time (s)").head(1)
    print("\n🏎️ Optimal (Legal) Fuel Blend for Fastest Lap:")
    print(best_blend.T)

    # Step 6: Visualize
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=legal_blends,
        x="Fuel Volume (L)",
        y="Predicted Lap Time (s)",
        color="green",
        alpha=0.6
    )
    plt.axvline(100, color='black', linestyle='--', label='FIA Volume Limit (100L)')
    plt.xlabel("Fuel Volume (L)")
    plt.ylabel("Predicted Lap Time (s)")
    plt.title("Lap Time vs Fuel Volume – Only Legal Blends")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    print("\n✅ Script reached the end successfully!")
