import os
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms

from model_multitask import MultiTaskViT


# ==================================================
# SETTINGS
# ==================================================

IMAGE_FILE = "data/live_candlestick.png"

INTRADAY_FILE = "data/nifty50_intraday.csv"

MODEL_FILE = "models/multitask_vit_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# TIME CONVERSION
# ==================================================

def slot_to_time(slot):

    start_minutes = 9 * 60 + 15

    total_minutes = (
        start_minutes +
        int(slot) * 5
    )

    hour = total_minutes // 60
    minute = total_minutes % 60

    if hour >= 12:
        suffix = "PM"
    else:
        suffix = "AM"

    display_hour = hour

    if display_hour == 0:
        display_hour = 12
    elif display_hour > 12:
        display_hour -= 12

    return f"{display_hour:02d}:{minute:02d} {suffix}"


# ==================================================
# LOAD LIVE DATA
# ==================================================

print("Loading latest NIFTY 50 data...")

df = pd.read_csv(
    INTRADAY_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"]
).dt.date

df["DateTime"] = pd.to_datetime(
    df["Date"].astype(str)
    + " "
    + df["Time"]
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

latest_days = trading_days[-30:]


input_start = latest_days[0]
input_end = latest_days[-1]


# ==================================================
# GET LAST KNOWN PRICE
# ==================================================

latest_row = df.iloc[-1]

base_price = float(
    latest_row["Close"]
)


# ==================================================
# LOAD CANDLESTICK IMAGE
# ==================================================

if not os.path.exists(IMAGE_FILE):

    raise FileNotFoundError(
        f"Live candlestick image not found: {IMAGE_FILE}"
    )


image = Image.open(
    IMAGE_FILE
).convert("RGB")


# ==================================================
# IMAGE TRANSFORMATION
# ==================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


image = transform(image)

image = image.unsqueeze(0)

image = image.to(DEVICE)


# ==================================================
# LOAD MODEL
# ==================================================

print("\nLoading trained ViT model...")

model = MultiTaskViT()

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully!")


# ==================================================
# MAKE PREDICTION
# ==================================================

with torch.no_grad():

    output = model(image)


# ==================================================
# PRICE PREDICTIONS
# ==================================================

predicted_prices = (
    output["prices"]
    .cpu()
    .numpy()[0]
)


raw_open = (
    predicted_prices[0]
    * base_price
)

raw_high = (
    predicted_prices[1]
    * base_price
)

raw_low = (
    predicted_prices[2]
    * base_price
)

raw_close = (
    predicted_prices[3]
    * base_price
)


# ==================================================
# OHLC CONSISTENCY
# ==================================================

predicted_open = raw_open

predicted_close = raw_close

predicted_high = max(
    raw_high,
    predicted_open,
    predicted_close
)

predicted_low = min(
    raw_low,
    predicted_open,
    predicted_close
)


# ==================================================
# HIGH TIME
# ==================================================

high_time_slot = torch.argmax(
    output["high_time"],
    dim=1
).item()

high_time = slot_to_time(
    high_time_slot
)


# ==================================================
# LOW TIME
# ==================================================

low_time_slot = torch.argmax(
    output["low_time"],
    dim=1
).item()

low_time = slot_to_time(
    low_time_slot
)


# ==================================================
# TREND
# ==================================================

trend_probabilities = torch.softmax(
    output["trend"],
    dim=1
)

trend_class = torch.argmax(
    trend_probabilities,
    dim=1
).item()

trend_confidence = (
    trend_probabilities[0][trend_class]
    .item()
    * 100
)


if trend_class == 1:

    trend = "Bullish"

else:

    trend = "Bearish"


# ==================================================
# DISPLAY LIVE FORECAST
# ==================================================

print("\n")
print("==============================================")
print("          LIVE NIFTY 50 AI FORECAST")
print("==============================================")

print(
    f"Input Period : {input_start} → {input_end}"
)

print(
    f"Last Known Price : ₹{base_price:,.2f}"
)

print("----------------------------------------------")

print("        NEXT TRADING SESSION PREDICTION")

print("----------------------------------------------")

print(
    f"Open Price     : ₹{predicted_open:,.2f}"
)

print(
    f"High Price     : ₹{predicted_high:,.2f}"
)

print(
    f"High Time      : {high_time}"
)

print(
    f"Low Price      : ₹{predicted_low:,.2f}"
)

print(
    f"Low Time       : {low_time}"
)

print(
    f"Close Price    : ₹{predicted_close:,.2f}"
)

print(
    f"Trend          : {trend}"
)

print(
    f"Confidence     : {trend_confidence:.2f}%"
)

print("==============================================")


print("\nData used:")
print(
    "Latest 30 available NIFTY 50 trading days"
)

print(
    "Latest session:",
    input_end
)

print("\nNOTE:")
print(
    "These are AI model estimates, "
    "not guaranteed future prices."
)