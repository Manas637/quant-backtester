from src.data.loader import load_market_data

data = load_market_data("data/raw/AAPL.csv")

print("Data loaded successfully!")
print("\nShape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nLast 5 rows:")
print(data.tail())

print("\nMissing values:")
print(data.isna().sum())