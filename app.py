import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd

# -------------------------------
# Load models and data
# -------------------------------
# Price prediction model (your existing pipe)
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Laptop data for dropdowns
df = pickle.load(open('df.pkl', 'rb'))

# Sentiment model (SVC) and TF-IDF vectorizer
sentiment_model = pickle.load(open('models/sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))

# -------------------------------
# Helper functions
# -------------------------------
def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def get_sentiment_details(reviews: list[str]):
    """
    reviews: list of raw review strings
    returns:
      - avg_positive (float in [0,1])
      - DataFrame with each review + its positive probability
    """
    if not reviews:
        return 0.5, pd.DataFrame(columns=["review", "positive_probability"])

    cleaned = [clean_text(r) for r in reviews]
    X = tfidf.transform(cleaned)
    probs = sentiment_model.predict_proba(X)[:, 1]  # positive class

    df_probs = pd.DataFrame({
        "review": reviews,
        "positive_probability": probs
    })

    return float(probs.mean()), df_probs

def quality_verdict(sentiment_score: float) -> str:
    """
    Product quality based only on reviews.
    """
    if sentiment_score >= 0.7:
        return "Good product 👍 (users are mostly happy)"
    elif sentiment_score >= 0.5:
        return "Mixed reviews 😐 (some like it, some don’t)"
    else:
        return "Poor product 👎 (many users are unhappy)"

def deal_verdict(fair_price: float, actual_price: float, sentiment_score: float) -> str:
    """
    Overall deal quality combining fair vs actual price + sentiment.
    """
    price_ratio = fair_price / actual_price  # >1 => cheaper than model fair price

    # Bad product
    if sentiment_score < 0.5:
        if price_ratio >= 1.1:
            return "Cheap but poorly reviewed ❌ (low quality despite low price)"
        else:
            return "Not worth it ❌ (overpriced and poorly reviewed)"

    # Mixed product
    if 0.5 <= sentiment_score < 0.7:
        if price_ratio >= 1.1:
            return "Decent deal for an average product 🙂"
        elif 0.9 <= price_ratio < 1.1:
            return "Fair deal for an average product 😐"
        else:
            return "Overpriced for what it offers ⚠️"

    # Good product
    if sentiment_score >= 0.7:
        if price_ratio >= 1.1:
            return "Excellent deal ✅ (great product, great price)"
        elif 0.95 <= price_ratio < 1.1:
            return "Good deal 👍 (great product at fair price)"
        else:
            return "Good product but overpriced 💸"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💻 Laptop Price & Review-based Recommendation")

st.markdown("### 1️⃣ Select Laptop Specifications")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type_ = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the Laptop (kg)')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160',
     '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']
)

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD / SSD
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS (processed)
os_ = st.selectbox('OS', df['os'].unique())

st.markdown("### 2️⃣ Enter Actual Price")
actual_price = st.number_input('Actual Selling Price (₹)', min_value=0.0, value=50000.0, step=1000.0)

st.markdown("### 3️⃣ Paste Reviews for this Laptop")
reviews_raw = st.text_area(
    "Paste 3–10 reviews here (one per line). You can copy them from Amazon, Flipkart, etc."
)

# split reviews
if reviews_raw.strip():
    reviews_list = [line.strip() for line in reviews_raw.split('\n') if line.strip()]
else:
    reviews_list = []

if st.button('Analyze'):
    if actual_price <= 0:
        st.error("Please enter a valid actual price greater than 0.")
    else:
        # ----------------- PRICE PART -----------------
        # convert touchscreen / ips to 0/1
        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0

        # compute PPI
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        # same order & shape as your original model expects
        query = np.array([
            company,
            type_,
            ram,
            weight,
            touchscreen_val,
            ips_val,
            ppi,
            cpu,
            hdd,
            ssd,
            gpu,
            os_
        ])

        query = query.reshape(1, 12)

        # model predicts log price, you used np.exp in original code
        fair_price = float(np.exp(pipe.predict(query)[0]))

        # ----------------- SENTIMENT PART -----------------
        sentiment_score, df_review_probs = get_sentiment_details(reviews_list)

        # ----------------- VERDICTS -----------------
        price_ratio = fair_price / actual_price if actual_price > 0 else None
        q_ver = quality_verdict(sentiment_score)
        d_ver = deal_verdict(fair_price, actual_price, sentiment_score)

        # ----------------- DISPLAY -----------------
        st.markdown("### 📊 Price Analysis")
        st.write(f"**Fair Price (model prediction):** ₹{fair_price:,.0f}")
        st.write(f"**Actual Price (user input):** ₹{actual_price:,.0f}")
        if price_ratio is not None:
            st.write(f"**Price Ratio (fair / actual):** {price_ratio:.2f}")

        st.markdown("### 🗣 Review Sentiment")
        st.write(f"**Average Positive Sentiment Score:** {sentiment_score:.3f} ({sentiment_score*100:.1f}%)")
        st.info(q_ver)

        if not df_review_probs.empty:
            st.markdown("**Per-review positive probability:**")
            df_display = df_review_probs.copy()
            df_display["positive_probability"] = df_display["positive_probability"].round(3)
            st.dataframe(df_display, use_container_width=True)

        st.markdown("### ✅ Final Deal Verdict")
        st.success(d_ver)
