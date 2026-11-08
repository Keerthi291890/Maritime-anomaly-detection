import pandas as pd

df = pd.read_parquet("data/processed/ais_presence/year=2025/month=06/part-0.parquet")
# pick a vessel with a reasonable number of records
busy_vessel = df["vessel_id"].value_counts().index[0]
sample = df[df["vessel_id"] == busy_vessel].sort_values("date")
print(sample[["date", "h3_cell", "hours"]].head(20))