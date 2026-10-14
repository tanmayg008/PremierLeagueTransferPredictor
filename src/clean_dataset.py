import os
import pandas as pd

# ==========================================================
# STEP 1 : LOAD RAW DATASET
# ==========================================================

RAW_DATA_PATH = "data/raw/premier_league_complete_stats_until35thGameDayOnSeason2025-26.csv"
PROCESSED_DATA_PATH = "data/processed/premier_league_player_stats_clean.csv"

df = pd.read_csv(RAW_DATA_PATH)

# Create a working copy
clean_df = df.copy()

print("Dataset Loaded")
print(f"Shape : {clean_df.shape}")

# ==========================================================
# STEP 2 : RENAME COLUMNS
# ==========================================================

clean_df.rename(columns={
    "minutesPlayed": "minutes",
    "expectedGoals": "xG",
    "expectedAssists": "xA",
    "goalsAssistsSum": "goal_contributions",
    "accuratePassesPercentage": "pass_accuracy"
}, inplace=True)

print("Columns renamed successfully.")

# ==========================================================
# STEP 3 : HANDLE MISSING VALUES
# ==========================================================

print("\nMissing Values Before Cleaning")
print(clean_df[["xG", "xA", "goalsPrevented"]].isnull().sum())

# xG and xA are missing for players with zero attacking actions.
clean_df["xG"] = clean_df["xG"].fillna(0)
clean_df["xA"] = clean_df["xA"].fillna(0)

print("\nMissing Values After Cleaning")
print(clean_df[["xG", "xA", "goalsPrevented"]].isnull().sum())

# We intentionally keep goalsPrevented as NaN.
# It is a goalkeeper-only statistic.

# ==========================================================
# STEP 4 : SAVE CLEAN DATASET
# ==========================================================

os.makedirs("data/processed", exist_ok=True)

clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

print("\nClean dataset saved successfully!")
print(f"Saved to : {PROCESSED_DATA_PATH}")

# ==========================================================
# STEP 5 : FINAL SUMMARY
# ==========================================================

print("\nCleaning Summary")
print(f"Rows    : {clean_df.shape[0]}")
print(f"Columns : {clean_df.shape[1]}")
print(f"Remaining Missing Values : {clean_df.isnull().sum().sum()}")

