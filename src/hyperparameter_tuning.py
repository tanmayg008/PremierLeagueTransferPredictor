# ==========================================
# Lesson 12 Part 2
# hyperparameter_tuning.py
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("data/processed/premier_league_ml_dataset.csv")

# ------------------------------------------
# Create Age Feature
# ------------------------------------------

df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])

season_date = pd.Timestamp("2026-05-01")

df["age"] = (
    (season_date - df["date_of_birth"]).dt.days / 365.25
).round(1)

# ------------------------------------------
# Remove Missing Target
# ------------------------------------------

df = df.dropna(subset=["market_value_in_eur"])

# ------------------------------------------
# Feature List
# ------------------------------------------

feature_columns = [
    "minutes",
    "appearances",
    "rating",
    "goals",
    "assists",
    "goal_contributions",
    "xG",
    "xA",
    "totalShots",
    "shotsOnTarget",
    "pass_accuracy",
    "accuratePasses",
    "keyPasses",
    "successfulDribbles",
    "tackles",
    "interceptions",
    "ballRecovery",
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
    "age",
    "position_tm"
]

X = df[feature_columns].copy()

X = pd.get_dummies(
    X,
    columns=["position_tm"],
    drop_first=True
)

# Log transformed target
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

print("=" * 60)
print("Train/Test Split Completed")
print("=" * 60)
print(f"Training Players : {len(X_train)}")
print(f"Testing Players  : {len(X_test)}")

# ------------------------------------------
# Baseline Cross Validation
# ------------------------------------------

baseline_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

cv_scores = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

print("\n" + "=" * 60)
print("5-Fold Cross Validation (Baseline)")
print("=" * 60)

print("Fold Scores:", np.round(cv_scores, 3))
print(f"Average CV R²: {cv_scores.mean():.3f}")

# ------------------------------------------
# Hyperparameter Grid
# ------------------------------------------

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2]
}

print("\nStarting Grid Search...")
print("This may take 2–5 minutes.\n")

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# ------------------------------------------
# Best Model
# ------------------------------------------

best_model = grid_search.best_estimator_

print("\n" + "=" * 60)
print("Best Hyperparameters")
print("=" * 60)

print(grid_search.best_params_)
print(f"Best CV R²: {grid_search.best_score_:.3f}")

# ------------------------------------------
# Test Set Evaluation
# ------------------------------------------

log_predictions = best_model.predict(X_test)

predictions = np.expm1(log_predictions)
actual_values = np.expm1(y_test)

mae = mean_absolute_error(actual_values, predictions)

rmse = np.sqrt(
    mean_squared_error(actual_values, predictions)
)

r2 = r2_score(actual_values, predictions)

print("\n" + "=" * 60)
print("Tuned Random Forest Results")
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
    "data/processed/model_predictions_tuned.csv",
    index=False
)

# ------------------------------------------
# Save Feature Importance
# ------------------------------------------

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

importance.to_csv(
    "data/processed/feature_importance_tuned.csv",
    index=False
)

print("\nFiles Saved Successfully!")
print("- model_predictions_tuned.csv")
print("- feature_importance_tuned.csv")