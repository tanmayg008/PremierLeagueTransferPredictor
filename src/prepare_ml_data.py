# ==========================================================
# PREMIER LEAGUE TRANSFER VALUE PREDICTOR
# STEP 10 : PREPARE DATA FOR MACHINE LEARNING
# ==========================================================

import pandas as pd

# ==========================================================
# STEP 1 : LOAD MERGED DATASET
# ==========================================================

DATA_PATH = "data/processed/premier_league_ml_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("ML DATASET LOADED")
print("=" * 50)

print(df.shape)
print(df.head())

# ==========================================================
# STEP 2 : REMOVE PLAYERS WITHOUT MARKET VALUE
# ==========================================================

ml_df = df.dropna(subset=["market_value_in_eur"]).copy()

print("\nDataset after removing missing target values:")
print(ml_df.shape)

print("Players removed:", len(df) - len(ml_df))



# ==========================================================
# STEP 3 : SELECT FEATURES FOR MACHINE LEARNING
# ==========================================================

feature_columns = [
    "minutes",
    "appearances",
    "matchesStarted",
    "rating",

    "goals",
    "assists",
    "goal_contributions",
    "xG",
    "xA",

    "goals_per90",
    "assists_per90",
    "goal_contributions_per90",
    "xG_per90",
    "xA_per90",

    "passes_per90",
    "key_passes_per90",
    "pass_accuracy",

    "tackles_per90",
    "interceptions_per90",
    "recoveries_per90",

    "shot_accuracy",

    "position_stats",
    "team_name_stats"
]

# ==========================================================
# STEP 4 : CREATE X (FEATURES) AND y (TARGET)
# ==========================================================

X = ml_df[feature_columns]

y = ml_df["market_value_in_eur"]

print("\nFeatures Shape (X):", X.shape)
print("Target Shape (y):", y.shape)

print("\nTarget Preview:")
print(y.head())

# ==========================================================
# STEP 5 : ONE-HOT ENCODE CATEGORICAL FEATURES
# ==========================================================

X = pd.get_dummies(
    X,
    columns=["position_stats", "team_name_stats"],
    drop_first=True
)

print("\nEncoded Features Shape:", X.shape)

print("\nFirst 10 Encoded Columns:")
print(X.columns[:10])

# ==========================================================
# STEP 6 : SAVE ML READY DATASET
# ==========================================================

ml_ready_df = X.copy()

ml_ready_df["market_value_in_eur"] = y

OUTPUT_PATH = "data/processed/premier_league_ml_ready_dataset.csv"

ml_ready_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nML Ready Dataset Saved!")
print("Final Shape:", ml_ready_df.shape)
print("Saved to:", OUTPUT_PATH)

