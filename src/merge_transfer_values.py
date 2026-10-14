# ==========================================================
# PREMIER LEAGUE TRANSFER VALUE PREDICTOR
# STEP 9 : MERGE PLAYER STATS WITH TRANSFERMARKT VALUES
# ==========================================================

import pandas as pd
from unidecode import unidecode

# ==========================================================
# STEP 1 : LOAD DATASETS
# ==========================================================

STATS_PATH = "data/processed/premier_league_engineered_features.csv"
MARKET_PATH = "data/raw/players.csv"

stats_df = pd.read_csv(STATS_PATH)
market_df = pd.read_csv(MARKET_PATH)

print("=" * 55)
print("DATASETS LOADED")
print("=" * 55)
print("Stats Dataset Shape :", stats_df.shape)
print("Market Dataset Shape:", market_df.shape)


# ==========================================================
# STEP 2 : FILTER ONLY PREMIER LEAGUE PLAYERS
# ==========================================================

pl_market_df = market_df[
    market_df["current_club_domestic_competition_id"] == "GB1"
].copy()

print("\nPremier League Rows:", pl_market_df.shape)


# ==========================================================
# STEP 3 : FILTER ONLY 2025 SEASON PLAYERS
# ==========================================================

print("\nSeason Distribution:")
print(pl_market_df["last_season"].value_counts().sort_index())

pl_market_df = pl_market_df[
    pl_market_df["last_season"] == 2025
].copy()

print("\n2025 Season Players:", pl_market_df.shape)


# ==========================================================
# STEP 4 : KEEP REQUIRED COLUMNS
# ==========================================================

pl_market_df = pl_market_df[
    [
        "name",
        "current_club_name",
        "position",
        "date_of_birth",
        "market_value_in_eur"
    ]
].copy()

print("\nRequired Columns Preview:")
print(pl_market_df.head())


# ==========================================================
# STEP 5 : CHECK DUPLICATE PLAYER NAMES
# ==========================================================

duplicates = pl_market_df["name"].duplicated().sum()

print("\nDuplicate Player Names:", duplicates)


# ==========================================================
# STEP 6 : RENAME COLUMNS
# ==========================================================

pl_market_df.rename(
    columns={
        "name": "player_name",
        "current_club_name": "team_name"
    },
    inplace=True
)

print("\nColumns renamed successfully.")


# ==========================================================
# STEP 7 : HELPER FUNCTION TO CLEAN PLAYER NAMES
# ==========================================================

def clean_player_name(name):
    """
    Convert player names into a standard format for matching.
    Example:
    Gabriel Magalhães -> gabriel magalhaes
    Saša Kalajdžić -> sasa kalajdzic
    Martín Zubimendi -> martin zubimendi
    """
    name = str(name)
    name = unidecode(name)   # Remove accents
    name = name.lower()      # Lowercase
    name = name.strip()      # Remove spaces
    return name


# ==========================================================
# STEP 8 : CREATE CLEAN NAMES IN BOTH DATASETS
# ==========================================================

stats_df["clean_name"] = stats_df["player_name"].apply(clean_player_name)

pl_market_df["clean_name"] = pl_market_df["player_name"].apply(clean_player_name)

print("\nExample Cleaned Names:")
print(stats_df[["player_name", "clean_name"]].head())


# ==========================================================
# STEP 9 : MERGE DATASETS USING CLEAN NAME
# ==========================================================

merged_df = stats_df.merge(
    pl_market_df,
    on="clean_name",
    how="left",
    suffixes=("_stats", "_tm")
)

print("\nMerged Dataset Shape:", merged_df.shape)


# ==========================================================
# STEP 10 : CHECK MERGE QUALITY
# ==========================================================

missing_market_values = merged_df["market_value_in_eur"].isnull().sum()

matched_players = len(merged_df) - missing_market_values

match_percentage = round(
    matched_players / len(merged_df) * 100,
    2
)

print("\n" + "=" * 55)
print("MERGE QUALITY REPORT")
print("=" * 55)

print("Total Players           :", len(merged_df))
print("Matched Players         :", matched_players)
print("Missing Market Values   :", missing_market_values)
print("Match Percentage        :", f"{match_percentage}%")



# ==========================================================
# STEP 11 : SHOW PLAYERS WITHOUT MARKET VALUES
# ==========================================================

unmatched_players = merged_df[
    merged_df["market_value_in_eur"].isnull()
]

print("\nPlayers without Market Values:")
print(
    unmatched_players[
        ["player_name_stats", "team_name_stats", "position_stats"]
    ].sort_values("team_name_stats")
)


# ==========================================================
# STEP 12 : SAVE FINAL ML DATASET
# ==========================================================

OUTPUT_PATH = "data/processed/premier_league_ml_dataset.csv"

if missing_market_values < 100:
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 55)
    print("DATASET SAVED SUCCESSFULLY")
    print("=" * 55)
    print("Saved to:", OUTPUT_PATH)

else:
    print("\nToo many missing market values.")
    print("Dataset NOT saved. Investigate merge first.")