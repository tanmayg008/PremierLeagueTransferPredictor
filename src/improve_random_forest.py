# ==========================================
# Lesson 12
# improve_random_forest.py
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# Load ML dataset
# ------------------------------------------

df = pd.read_csv("data/processed/premier_league_ml_dataset.csv")

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)
print(f"Dataset Shape : {df.shape}")

# ------------------------------------------
# Create Age Feature
# ------------------------------------------

df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])

# Calculate player's age as of 1 May 2026
season_date = pd.Timestamp("2026-05-01")

df["age"] = (
    (season_date - df["date_of_birth"]).dt.days / 365.25
).round(1)

print("\nAge feature created successfully.")

# ------------------------------------------
# Remove players without transfer value
# ------------------------------------------

df = df.dropna(subset=["market_value_in_eur"])

print(f"Players after removing missing target values: {len(df)}")

# ------------------------------------------
# Features for ML Model
# ------------------------------------------

feature_columns = [
    # Playing time
    "minutes",
    "appearances",

    # Performance
    "rating",
    "goals",
    "assists",
    "goal_contributions",
    "xG",
    "xA",

    # Shooting
    "totalShots",
    "shotsOnTarget",

    # Passing
    "pass_accuracy",
    "accuratePasses",
    "keyPasses",

    # Dribbling
    "successfulDribbles",

    # Defensive actions
    "tackles",
    "interceptions",
    "ballRecovery",

    # Engineered Features
    "goals_per90",
    "assists_per90",
    "goal_contributions_per90",
    "xG_per90",
    "xA_per90",
    "passes_per90",
    "key_passes_per90",
    "tackles_per90",
    "interceptions_per90",
    "recoveries_per90",
    "shot_accuracy",
    "big_chance_conversion",

    # New Feature
    "age",

    # Categorical Feature
    "position_tm"
]
# ------------------------------------------
# Debugging: Check available and missing columns
# ------------------------------------------

print("\n" + "=" * 60)
print("Available Columns in Dataset")
print("=" * 60)
print(df.columns.tolist())

missing_columns = [
    col for col in feature_columns
    if col not in df.columns
]

print("\n" + "=" * 60)
print("Missing Columns")
print("=" * 60)
print(missing_columns)

# Stop execution if feature names don't match dataset
if missing_columns:
    raise ValueError(
        "Feature list does not match dataset columns. "
        "Update feature_columns using the printed dataset columns above."
    )

# ------------------------------------------
# Prepare Features (X)
# ------------------------------------------

X = df[feature_columns].copy()

# Convert categorical Position column into numbers
X = pd.get_dummies(
    X,
    columns=["position_tm"],
    drop_first=True
)

# ------------------------------------------
# Target Variable (Log Transform)
# ------------------------------------------

y = np.log1p(df["market_value_in_eur"])

# ------------------------------------------
# Train/Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("Train/Test Split")
print("=" * 60)
print(f"Training Players : {len(X_train)}")
print(f"Testing Players  : {len(X_test)}")

# ------------------------------------------
# Train Random Forest Model
# ------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully.")

# ------------------------------------------
# Predictions
# ------------------------------------------

log_predictions = model.predict(X_test)

# Convert predictions back to Euros
predictions = np.expm1(log_predictions)
actual_values = np.expm1(y_test)

# ------------------------------------------
# Model Evaluation
# ------------------------------------------

mae = mean_absolute_error(actual_values, predictions)

rmse = np.sqrt(
    mean_squared_error(actual_values, predictions)
)

r2 = r2_score(actual_values, predictions)

print("\n" + "=" * 60)
print("Improved Random Forest Results")
print("=" * 60)
print(f"MAE : €{mae:,.0f}")
print(f"RMSE: €{rmse:,.0f}")
print(f"R²  : {r2:.3f}")

# ------------------------------------------
# Save Predictions
# ------------------------------------------
results = pd.DataFrame({
    "player_name": df.loc[y_test.index, "player_name_stats"],
    "actual_value": actual_values.round(0),
    "predicted_value": predictions.round(0)
})

results.to_csv(
    "data/processed/model_predictions_v2.csv",
    index=False
)

print("\nPredictions saved to:")
print("data/processed/model_predictions_v2.csv")
print("=" * 60)