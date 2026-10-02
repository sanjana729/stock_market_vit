import os
import pandas as pd
import yfinance as yf


# NIFTY 50 Yahoo Finance ticker
TICKER = "^NSEI"

# Get enough history to contain at least 30 trading days
PERIOD = "60d"

# 5-minute intraday candles
INTERVAL = "5m"

OUTPUT_FILE = "data/nifty50_intraday.csv"


print("Downloading NIFTY 50 intraday data...")

df = yf.download(
    TICKER,
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=False,
    progress=False
)

if df.empty:
    print("ERROR: No data was downloaded.")
    exit()

# Handle Yahoo Finance multi-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Keep required columns
required_columns = ["Open", "High", "Low", "Close", "Volume"]

df = df[required_columns].copy()

# Remove missing rows
df.dropna(inplace=True)

# Make timestamp a normal column
df.reset_index(inplace=True)

# Rename timestamp column
if "Datetime" in df.columns:
    df.rename(columns={"Datetime": "DateTime"}, inplace=True)

# Convert timestamp
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Convert to India time if timezone information exists
if df["DateTime"].dt.tz is not None:
    df["DateTime"] = df["DateTime"].dt.tz_convert("Asia/Kolkata")

# Create separate Date and Time columns
df["Date"] = df["DateTime"].dt.date
df["Time"] = df["DateTime"].dt.strftime("%H:%M")

# Keep only the latest 30 trading days
trading_days = sorted(df["Date"].unique())

if len(trading_days) > 30:
    trading_days = trading_days[-30:]

df = df[df["Date"].isin(trading_days)]

# Save
os.makedirs("data", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSUCCESS!")
print("Number of trading days:", df["Date"].nunique())
print("Number of intraday candles:", len(df))
print("First date:", df["Date"].min())
print("Last date:", df["Date"].max())
print("\nSaved to:", OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())