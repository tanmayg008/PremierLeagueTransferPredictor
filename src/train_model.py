# ==========================================================
# PREMIER LEAGUE TRANSFER VALUE PREDICTOR
# LESSON 11 : TRAIN RANDOM FOREST MODEL
# ==========================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# STEP 1 : LOAD ML READY DATASET
# ==========================================================

DATA_PATH = "data/processed/premier_league_ml_ready_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("ML READY DATASET LOADED")
print("=" * 50)

print(df.shape)
print(df.head())

# ==========================================================
# STEP 2 : CREATE FEATURES (X) AND TARGET (y)
# ==========================================================

X = df.drop(columns=["market_value_in_eur"])

y = df["market_value_in_eur"]

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)

# ==========================================================
# STEP 3 : TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Features :", X_train.shape)
print("Testing Features  :", X_test.shape)

print("Training Labels   :", y_train.shape)
print("Testing Labels    :", y_test.shape)

# ==========================================================
# STEP 4 : CREATE RANDOM FOREST MODEL
# ==========================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

print("\nRandom Forest Model Created!")

# ==========================================================
# STEP 5 : TRAIN MODEL
# ==========================================================

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================================================
# STEP 6 : MAKE PREDICTIONS
# ==========================================================

predictions = model.predict(X_test)

print("\nFirst Five Predictions:")
print(predictions[:5])

# ==========================================================
# STEP 7 : EVALUATE MODEL
# ==========================================================

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"MAE  : €{mae:,.0f}")
print(f"RMSE : €{rmse:,.0f}")
print(f"R²   : {r2:.3f}")

# ==========================================================
# STEP 8 : COMPARE ACTUAL VS PREDICTED VALUES
# ==========================================================

comparison_df = pd.DataFrame({
    "Actual Value": y_test,
    "Predicted Value": predictions
})

comparison_df["Difference"] = (
    comparison_df["Actual Value"] -
    comparison_df["Predicted Value"]
)

print("\nSample Predictions")
print(comparison_df.head(10))

# ==========================================================
# STEP 9 : FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features")
print(importance_df.head(15))

# ==========================================================
# STEP 10 : SAVE RESULTS
# ==========================================================

comparison_df.to_csv(
    "data/processed/model_predictions.csv",
    index=False
)

importance_df.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

print("\nPrediction results saved!")
print("Feature importance saved!")