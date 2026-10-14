import os
import pandas as pd

# ==========================================================
# STEP 1 : LOAD FEATURE DATASET
# ==========================================================

FEATURE_DATA_PATH = "data/processed/premier_league_features.csv"

df = pd.read_csv(FEATURE_DATA_PATH)

engineered_df = df.copy()

print("Feature Dataset Loaded!")
print(engineered_df.shape)

# ==========================================================
# STEP 2 : CREATE GOALS PER 90
# ==========================================================

engineered_df["goals_per90"] = (
    engineered_df["goals"] / engineered_df["minutes"]
) * 90

print(engineered_df[["player_name", "goals", "minutes", "goals_per90"]].head())


# Assists per 90
engineered_df["assists_per90"] = (
    engineered_df["assists"] / engineered_df["minutes"]
) * 90

# Goal Contributions per 90
engineered_df["goal_contributions_per90"] = (
    engineered_df["goal_contributions"] / engineered_df["minutes"]
) * 90

# xG per 90
engineered_df["xG_per90"] = (
    engineered_df["xG"] / engineered_df["minutes"]
) * 90

# xA per 90
engineered_df["xA_per90"] = (
    engineered_df["xA"] / engineered_df["minutes"]
) * 90

engineered_df["shot_accuracy"] = (
    engineered_df["shotsOnTarget"] / engineered_df["totalShots"]
)

engineered_df.rename(
    columns={
        "goalConversionPercentage": "goal_conversion"
    },
    inplace=True
)

engineered_df["big_chance_conversion"] = (
    engineered_df["goals"] /
    (engineered_df["bigChancesCreated"] + 1)
)


engineered_df["tackles_per90"] = (
    engineered_df["tackles"] / engineered_df["minutes"]
) * 90

engineered_df["interceptions_per90"] = (
    engineered_df["interceptions"] / engineered_df["minutes"]
) * 90

engineered_df["recoveries_per90"] = (
    engineered_df["ballRecovery"] / engineered_df["minutes"]
) * 90


engineered_df["passes_per90"] = (
    engineered_df["accuratePasses"] / engineered_df["minutes"]
) * 90

engineered_df["key_passes_per90"] = (
    engineered_df["keyPasses"] / engineered_df["minutes"]
) * 90

print("Players with zero minutes:")
print((engineered_df["minutes"] == 0).sum())


engineered_df.replace(
    [float("inf"), float("-inf")],
    0,
    inplace=True
)

engineered_df.fillna(0, inplace=True)

new_features = [
    "player_name",
    "goals_per90",
    "assists_per90",
    "goal_contributions_per90",
    "xG_per90",
    "passes_per90",
    "tackles_per90"
]

print(engineered_df[new_features].head(10))


# ==========================================================
# STEP 3 : SAVE ENGINEERED DATASET
# ==========================================================

OUTPUT_PATH = "data/processed/premier_league_engineered_features.csv"

engineered_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nEngineered dataset saved successfully!")
print(f"Saved to : {OUTPUT_PATH}")