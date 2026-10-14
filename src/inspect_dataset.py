import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/premier_league_complete_stats_until35thGameDayOnSeason2025-26.csv")

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing=df.isnull().sum()
missing = missing[missing > 0]
print(missing.sort_values(ascending=False))

print("\n" + "=" * 50)
print("NUMERICAL SUMMARY")
print("=" * 50)

print(df.describe())

important = [
    "player_name",
    "team_name",
    "position",
    "age",
    "minutes_played",
    "goals",
    "assists",
    "xG",
    "xA"
]

print(df[important].head(10))