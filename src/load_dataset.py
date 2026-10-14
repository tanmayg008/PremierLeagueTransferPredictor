import pandas as pd

# Read the CSV file
df = pd.read_csv(
    "data/raw/premier_league_complete_stats_until35thGameDayOnSeason2025-26.csv"
)

print("Dataset loaded successfully!")

print("\nShape:")
print(df.shape)

print("\nFirst 5 columns:")
print(df.columns[:10])

print("\nFirst 5 rows:")
print(df.head())