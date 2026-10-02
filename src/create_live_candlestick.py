import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ==================================================
# SETTINGS
# ==================================================

INPUT_FILE = "data/nifty50_intraday.csv"

OUTPUT_FILE = "data/live_candlestick.png"

# Number of latest trading days
WINDOW_DAYS = 30


# ==================================================
# LOAD DATA
# ==================================================

print("Loading latest NIFTY 50 intraday data...")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"]
).dt.date

df["DateTime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"]
)

df = df.sort_values(
    "DateTime"
).reset_index(drop=True)


# ==================================================
# GET LATEST 30 TRADING DAYS
# ==================================================

trading_days = sorted(
    df["Date"].unique()
)

latest_days = trading_days[-WINDOW_DAYS:]

df = df[
    df["Date"].isin(latest_days)
].copy()


print("Trading days used:", len(latest_days))

print(
    "First day:",
    latest_days[0]
)

print(
    "Last day:",
    latest_days[-1]
)


# ==================================================
# RESAMPLE TO 30-MINUTE CANDLES
# ==================================================

df = df.set_index("DateTime")


candles = df.resample("30min").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last"
})


candles = candles.dropna().reset_index()


# ==================================================
# NORMALIZE USING FIRST CLOSE
# ==================================================

base_price = float(
    candles.iloc[0]["Close"]
)

candles["Open"] = (
    candles["Open"] / base_price
)

candles["High"] = (
    candles["High"] / base_price
)

candles["Low"] = (
    candles["Low"] / base_price
)

candles["Close"] = (
    candles["Close"] / base_price
)


# ==================================================
# CREATE FIGURE
# ==================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)


# ==================================================
# DRAW CANDLESTICKS
# ==================================================

for i, row in candles.iterrows():

    open_price = row["Open"]
    high_price = row["High"]
    low_price = row["Low"]
    close_price = row["Close"]


    # --------------------------------------------------
    # Wick
    # --------------------------------------------------

    ax.plot(
        [i, i],
        [low_price, high_price],
        linewidth=1
    )


    # --------------------------------------------------
    # Candle body
    # --------------------------------------------------

    body_low = min(
        open_price,
        close_price
    )

    body_height = abs(
        close_price -
        open_price
    )


    # Avoid zero-height candles
    if body_height == 0:

        body_height = 0.0001


    # Default matplotlib color
    if close_price >= open_price:

        face_color = "white"

    else:

        face_color = "black"


    rectangle = Rectangle(
        (
            i - 0.3,
            body_low
        ),
        0.6,
        body_height,
        facecolor=face_color,
        edgecolor="black",
        linewidth=1
    )

    ax.add_patch(rectangle)


# ==================================================
# CHART SETTINGS
# ==================================================

ax.set_title(
    "NIFTY 50 - Latest 30 Trading Days"
)

ax.set_xlabel(
    "30-Minute Candles"
)

ax.set_ylabel(
    "Normalized Price"
)

ax.grid(
    True,
    alpha=0.2
)


# ==================================================
# SAVE
# ==================================================

os.makedirs(
    "data",
    exist_ok=True
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=150
)

plt.close()


# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n======================================")
print("LIVE CANDLESTICK CREATED")
print("======================================")

print(
    "Trading days:",
    len(latest_days)
)

print(
    "First day:",
    latest_days[0]
)

print(
    "Last day:",
    latest_days[-1]
)

print(
    "Base price:",
    round(base_price, 2)
)

print(
    "Candles created:",
    len(candles)
)

print(
    "Saved to:",
    OUTPUT_FILE
)