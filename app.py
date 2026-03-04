# app.py

import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# CONFIG
# =========================
MAX_LEN = 200

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🎬 Sentiment Analysis App")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_sentiment_model():
    return load_model("sentiment_model.keras")

# =========================
# LOAD TOKENIZER (PICKLE)
# =========================
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_sentiment_model()
tokenizer = load_tokenizer()

# =========================
# PREDICTION FUNCTION (YOUR FORMAT)
# =========================
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    prob = model.predict(pad, verbose=0)[0][0]
    return "Positive" if prob >= 0.5 else "Negative", prob

# =========================
# UI
# =========================
user_input = st.text_area(
    "Enter a movie review:",
    placeholder="Example: Good movie with great acting!"
)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.success(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence: `{confidence:.2f}`")