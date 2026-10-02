import os
import sys
import subprocess

import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go

from PIL import Image
from torchvision import transforms


# ============================================================
# PROJECT SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model_multitask import MultiTaskViT


DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "nifty50_intraday.csv"
)

IMAGE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "live_candlestick.png"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "multitask_vit_best.pth"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NIFTY 50 AI Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
<style>

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.stApp {
    background:
        radial-gradient(
            circle at top left,
            #172554 0%,
            #0b1120 35%,
            #070b14 100%
        );

    color: #f8fafc;
}

.block-container {
    max-width: 1500px;

    padding-top: 2.5rem;
    padding-bottom: 3rem;

    padding-left: 4rem;
    padding-right: 4rem;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    background:
        linear-gradient(
            135deg,
            #111c35,
            #0f172a
        );

    border: 1px solid #263653;

    border-radius: 22px;

    padding: 30px 35px;

    margin-bottom: 30px;

    box-shadow:
        0 15px 40px rgba(0,0,0,0.25);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #f8fafc;
}

.hero-subtitle {
    margin-top: 8px;
    font-size: 16px;
    color: #94a3b8;
}

.hero-badge {
    display: inline-block;

    margin-top: 18px;

    padding: 7px 14px;

    border-radius: 20px;

    background: #172554;

    border: 1px solid #1d4ed8;

    color: #93c5fd;

    font-size: 13px;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 25px;
    font-weight: 750;

    color: #f8fafc;

    margin-top: 35px;
    margin-bottom: 17px;
}


/* =========================================================
   MARKET CARDS
   ========================================================= */

.metric-card {
    background:
        linear-gradient(
            145deg,
            #111827,
            #18243a
        );

    border: 1px solid #293b57;

    border-radius: 18px;

    padding: 22px;

    min-height: 135px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.22);
}

.metric-label {
    color: #94a3b8;

    font-size: 13px;

    font-weight: 600;

    text-transform: uppercase;

    letter-spacing: 0.5px;
}

.metric-value {
    color: #f8fafc;

    font-size: 29px;

    font-weight: 800;

    margin-top: 10px;
}

.metric-sub {
    color: #64748b;

    font-size: 12px;

    margin-top: 6px;
}


/* =========================================================
   FORECAST CARDS
   ========================================================= */

.forecast-card {
    background:
        linear-gradient(
            145deg,
            #111827,
            #17243a
        );

    border: 1px solid #304563;

    border-radius: 18px;

    padding: 23px;

    min-height: 145px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.25);
}

.forecast-label {
    color: #94a3b8;

    font-size: 13px;

    font-weight: 600;

    text-transform: uppercase;
}

.forecast-value {
    color: #f8fafc;

    font-size: 29px;

    font-weight: 800;

    margin-top: 10px;
}

.forecast-time {
    color: #94a3b8;

    font-size: 13px;

    margin-top: 10px;
}


/* =========================================================
   TREND
   ========================================================= */

.trend-card {
    background:
        linear-gradient(
            135deg,
            #111827,
            #17243a
        );

    border: 1px solid #304563;

    border-radius: 20px;

    padding: 28px;

    text-align: center;

    margin-top: 18px;
}

.trend-label {
    color: #94a3b8;

    font-size: 13px;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.trend-bullish {
    color: #22c55e;

    font-size: 38px;

    font-weight: 800;

    margin-top: 8px;
}

.trend-bearish {
    color: #ef4444;

    font-size: 38px;

    font-weight: 800;

    margin-top: 8px;
}

.confidence {
    color: #94a3b8;

    font-size: 14px;

    margin-top: 7px;
}


/* =========================================================
   INFO BOX
   ========================================================= */

.info-box {
    background-color: #111827;

    border: 1px solid #263449;

    border-radius: 18px;

    padding: 23px;

    color: #cbd5e1;

    line-height: 1.9;

    margin-top: 10px;
}


/* =========================================================
   INPUT WINDOW
   ========================================================= */

.window-box {
    background: #10213b;

    border: 1px solid #21436d;

    border-radius: 12px;

    padding: 13px 18px;

    color: #93c5fd;

    font-size: 14px;

    margin-bottom: 18px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;

    color: #64748b;

    font-size: 12px;

    line-height: 1.7;

    margin-top: 45px;

    padding-top: 25px;

    border-top: 1px solid #1e293b;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# TIME SLOT TO TIME
# ============================================================

def slot_to_time(slot):

    start_minutes = 9 * 60 + 15

    total_minutes = (
        start_minutes + int(slot) * 5
    )

    hour = total_minutes // 60

    minute = total_minutes % 60

    suffix = "PM" if hour >= 12 else "AM"

    display_hour = hour

    if display_hour == 0:
        display_hour = 12

    elif display_hour > 12:
        display_hour -= 12

    return f"{display_hour:02d}:{minute:02d} {suffix}"


# ============================================================
# AUTOMATIC LIVE DATA UPDATE
# ============================================================

def update_live_data():

    get_data_script = os.path.join(
        BASE_DIR,
        "src",
        "get_nifty_intraday.py"
    )

    create_image_script = os.path.join(
        BASE_DIR,
        "src",
        "create_live_candlestick.py"
    )

    python_executable = sys.executable

    try:

        data_result = subprocess.run(
            [
                python_executable,
                get_data_script
            ],

            check=True,

            capture_output=True,

            text=True
        )

        image_result = subprocess.run(
            [
                python_executable,
                create_image_script
            ],

            check=True,

            capture_output=True,

            text=True
        )

        return (
            True,
            data_result.stdout,
            image_result.stdout
        )

    except subprocess.CalledProcessError as e:

        return (
            False,
            "",
            e.stderr
        )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

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

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = MultiTaskViT()

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model, device


# ============================================================
# MODEL PREDICTION
# ============================================================

def make_prediction(df):

    model, device = load_model()

    trading_days = sorted(
        df["Date"].unique()
    )

    latest_days = trading_days[-30:]

    input_start = latest_days[0]

    input_end = latest_days[-1]

    latest_row = df.iloc[-1]

    base_price = float(
        latest_row["Close"]
    )

    image = Image.open(
        IMAGE_FILE
    ).convert("RGB")

    transform = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        output = model(image)


    # --------------------------------------------------------
    # PRICE PREDICTIONS
    # --------------------------------------------------------

    prices = (
        output["prices"]
        .cpu()
        .numpy()[0]
    )

    raw_open = (
        prices[0]
        * base_price
    )

    raw_high = (
        prices[1]
        * base_price
    )

    raw_low = (
        prices[2]
        * base_price
    )

    raw_close = (
        prices[3]
        * base_price
    )


    # --------------------------------------------------------
    # OHLC CONSISTENCY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # HIGH TIME
    # --------------------------------------------------------

    high_slot = torch.argmax(
        output["high_time"],
        dim=1
    ).item()

    high_time = slot_to_time(
        high_slot
    )


    # --------------------------------------------------------
    # LOW TIME
    # --------------------------------------------------------

    low_slot = torch.argmax(
        output["low_time"],
        dim=1
    ).item()

    low_time = slot_to_time(
        low_slot
    )


    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output["trend"],
        dim=1
    )

    trend_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[0][trend_class]
        .item()
        * 100
    )

    if trend_class == 1:

        trend = "Bullish"

    else:

        trend = "Bearish"


    return {

        "input_start":
            input_start,

        "input_end":
            input_end,

        "base_price":
            base_price,

        "open":
            predicted_open,

        "high":
            predicted_high,

        "low":
            predicted_low,

        "close":
            predicted_close,

        "high_time":
            high_time,

        "low_time":
            low_time,

        "trend":
            trend,

        "confidence":
            confidence
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f'<div class="hero">'
    f'<div class="hero-title">'
    f'📈 NIFTY 50 AI Forecast'
    f'</div>'
    f'<div class="hero-subtitle">'
    f'Vision Transformer based stock market '
    f'trend and next-session prediction'
    f'</div>'
    f'<div class="hero-badge">'
    f'● AI MODEL ACTIVE'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    st.error(
        "Trained ViT model was not found."
    )

    st.stop()


# ============================================================
# FETCH LATEST DATA
# ============================================================

with st.spinner(
    "Updating latest NIFTY 50 market data..."
):

    success, data_output, image_output = (
        update_live_data()
    )


# Clear old cached CSV

load_data.clear()


if not success:

    st.warning(
        "Live data could not be refreshed. "
        "Showing the latest available local data."
    )


# ============================================================
# CHECK DATA FILES
# ============================================================

if not os.path.exists(DATA_FILE):

    st.error(
        "NIFTY 50 data file was not found."
    )

    st.stop()


if not os.path.exists(IMAGE_FILE):

    st.error(
        "Candlestick input image was not found."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        f"Unable to load market data: {e}"
    )

    st.stop()


# ============================================================
# CURRENT MARKET DATA
# ============================================================

latest = df.iloc[-1]

latest_price = float(
    latest["Close"]
)

latest_open = float(
    latest["Open"]
)

latest_high = float(
    latest["High"]
)

latest_low = float(
    latest["Low"]
)

latest_time = latest["DateTime"]


change = (
    latest_price
    - latest_open
)

change_percent = (
    change
    / latest_open
) * 100


# ============================================================
# LIVE MARKET OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Live Market Overview'
    '</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


# NIFTY 50

with c1:

    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">'
        f'NIFTY 50'
        f'</div>'
        f'<div class="metric-value">'
        f'₹{latest_price:,.2f}'
        f'</div>'
        f'<div class="metric-sub">'
        f'Latest available value'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# SESSION CHANGE

with c2:

    color = (
        "#22c55e"
        if change >= 0
        else "#ef4444"
    )

    sign = (
        "+"
        if change >= 0
        else ""
    )

    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">'
        f'Session Change'
        f'</div>'
        f'<div style="color:{color};'
        f'font-size:29px;'
        f'font-weight:800;'
        f'margin-top:10px;">'
        f'{sign}₹{change:,.2f}'
        f'</div>'
        f'<div style="color:{color};'
        f'font-size:13px;'
        f'margin-top:6px;">'
        f'{sign}{change_percent:.2f}%'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# TODAY HIGH

with c3:

    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">'
        f"Today's High"
        f'</div>'
        f'<div class="metric-value">'
        f'₹{latest_high:,.2f}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# TODAY LOW

with c4:

    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">'
        f"Today's Low"
        f'</div>'
        f'<div class="metric-value">'
        f'₹{latest_low:,.2f}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


st.caption(
    f"Last available data: "
    f"{latest_time.strftime('%d %b %Y, %I:%M %p')}"
)


# ============================================================
# 30-DAY MARKET CHART
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 NIFTY 50 — 30 Trading Day Trend'
    '</div>',
    unsafe_allow_html=True
)


daily = (
    df.groupby("Date")
    .agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last")
    )
    .reset_index()
)


daily = daily.tail(30)


fig = go.Figure()


fig.add_trace(
    go.Candlestick(

        x=daily["Date"],

        open=daily["Open"],

        high=daily["High"],

        low=daily["Low"],

        close=daily["Close"],

        name="NIFTY 50"
    )
)


fig.update_layout(

    height=500,

    template="plotly_dark",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    margin=dict(
        l=20,
        r=20,
        t=25,
        b=20
    ),

    xaxis_rangeslider_visible=False,

    hovermode="x unified",

    font=dict(
        color="#cbd5e1"
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# AI FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🤖 AI Next Trading Session Forecast'
    '</div>',
    unsafe_allow_html=True
)


try:

    prediction = make_prediction(df)

except Exception as e:

    st.error(
        f"Prediction error: {e}"
    )

    st.stop()


# INPUT WINDOW

st.markdown(
    f'<div class="window-box">'
    f'📅 <b>Model Input Window:</b> '
    f'{prediction["input_start"]}'
    f' → '
    f'{prediction["input_end"]}'
    f'</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION CARDS
# ============================================================

p1, p2, p3, p4 = st.columns(4)


# OPEN

with p1:

    st.markdown(
        f'<div class="forecast-card">'
        f'<div class="forecast-label">'
        f'Predicted Open'
        f'</div>'
        f'<div class="forecast-value">'
        f'₹{prediction["open"]:,.2f}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# HIGH

with p2:

    st.markdown(
        f'<div class="forecast-card">'
        f'<div class="forecast-label">'
        f'Predicted High'
        f'</div>'
        f'<div class="forecast-value">'
        f'₹{prediction["high"]:,.2f}'
        f'</div>'
        f'<div class="forecast-time">'
        f'⏱ {prediction["high_time"]}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# LOW

with p3:

    st.markdown(
        f'<div class="forecast-card">'
        f'<div class="forecast-label">'
        f'Predicted Low'
        f'</div>'
        f'<div class="forecast-value">'
        f'₹{prediction["low"]:,.2f}'
        f'</div>'
        f'<div class="forecast-time">'
        f'⏱ {prediction["low_time"]}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# CLOSE

with p4:

    st.markdown(
        f'<div class="forecast-card">'
        f'<div class="forecast-label">'
        f'Predicted Close'
        f'</div>'
        f'<div class="forecast-value">'
        f'₹{prediction["close"]:,.2f}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ============================================================
# TREND
# ============================================================

if prediction["trend"] == "Bullish":

    trend_class = "trend-bullish"

else:

    trend_class = "trend-bearish"


st.markdown(
    f'<div class="trend-card">'
    f'<div class="trend-label">'
    f'Predicted Market Trend'
    f'</div>'
    f'<div class="{trend_class}">'
    f'{prediction["trend"]}'
    f'</div>'
    f'<div class="confidence">'
    f'Model confidence: '
    f'{prediction["confidence"]:.2f}%'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🧠 About the AI Model'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="info-box">'
    '<b>Model:</b> Vision Transformer (ViT)<br>'
    '<b>Architecture:</b> Multi-task Vision Transformer<br>'
    '<b>Input:</b> Latest 30 trading days of NIFTY 50 '
    '5-minute intraday data converted into candlestick '
    'image representation<br>'
    '<b>Price Outputs:</b> Open, High, Low and Close<br>'
    '<b>Time Outputs:</b> High Time and Low Time<br>'
    '<b>Trend Output:</b> Bullish / Bearish'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DATA INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📡 Data Information'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    f'<div class="info-box">'
    f'<b>Instrument:</b> NIFTY 50<br>'
    f'<b>Frequency:</b> 5-minute intraday data<br>'
    f'<b>Input Window:</b> 30 trading days<br>'
    f'<b>Latest Session:</b> '
    f'{prediction["input_end"]}<br>'
    f'<b>Latest Known Price:</b> '
    f'₹{prediction["base_price"]:,.2f}<br>'
    f'<b>Prediction:</b> Next trading session'
    f'</div>',
    unsafe_allow_html=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    '<div class="footer">'
    '<b>Academic Project Disclaimer</b><br>'
    'The values displayed are experimental AI model '
    'estimates and are not guaranteed future market values. '
    'This dashboard is developed for academic and research '
    'purposes and is not financial advice.'
    '<br><br>'
    'NIFTY 50 AI Forecast • Vision Transformer Project'
    '</div>',
    unsafe_allow_html=True
)