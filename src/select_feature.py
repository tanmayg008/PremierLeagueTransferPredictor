#Feature Selection for Premier League Player Stats

import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/premier_league_player_stats_clean.csv")

print("Dataset Loaded!")
print(df.shape)

identity_features = [
    "player_name",
    "team_name",
    "position"
]

playing_time_features = [
    "appearances",
    "matchesStarted",
    "minutes",
    "rating"
]

attacking_features = [
    "goals",
    "assists",
    "goal_contributions",
    "xG",
    "xA",
    "totalShots",
    "shotsOnTarget",
    "bigChancesCreated",
    "bigChancesMissed",
    "keyPasses"
]


passing_features = [
    "accuratePasses",
    "pass_accuracy",
    "accurateLongBalls",
    "accurateCrosses",
    "accurateFinalThirdPasses",
    "totalPasses"
]


dribbling_features = [
    "successfulDribbles",
    "successfulDribblesPercentage",
    "touches"
]

defensive_features = [
    "tackles",
    "interceptions",
    "clearances",
    "ballRecovery",
    "aerialDuelsWon"
]

selected_columns = (
    identity_features
    + playing_time_features
    + attacking_features
    + passing_features
    + dribbling_features
    + defensive_features
)

feature_df = df[selected_columns]

print("\nSelected Dataset Shape:")
print(feature_df.shape)


print("\nColumns selected:")
print(feature_df.columns.tolist())

print("\nFirst 5 rows:")
print(feature_df.head())

import os

os.makedirs("data/processed", exist_ok=True)

feature_df.to_csv(
    "data/processed/premier_league_features.csv",
    index=False
)

print("\nFeature dataset saved!")