import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Load sentiment pipeline once
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

st.title("📊 Sentiment Analysis Application")
st.write("Analyze single text or upload a .txt file with multiple lines.")

# --- Single text input ---
user_text = st.text_area("Enter text to analyze:")
if st.button("Analyze Text"):
    if user_text.strip():
        result = sentiment_pipeline(user_text, truncation=True, max_length=128)[0]
        sentiment = label_map.get(result['label'], result['label'])
        confidence = result['score']
        st.success(f"Sentiment: {sentiment}")
        st.info(f"Confidence: {confidence:.2f}")

# --- File upload ---
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
if uploaded_file is not None:
    texts = [line.strip() for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]
    results = sentiment_pipeline(texts, truncation=True, max_length=128)
    readable = [label_map.get(r['label'], r['label']) for r in results]

    df_out = pd.DataFrame({
        "text": texts,
        "sentiment": readable,
        "confidence": [r['score'] for r in results]
    })
    
    # 📋 Show results table
    st.subheader("Results Table")
    st.dataframe(df_out)

    # 📊 Bar chart (built-in Streamlit)
    st.subheader("Sentiment Counts (Bar Chart)")
    st.bar_chart(df_out['sentiment'].value_counts())

    # 📊 Pie chart (Matplotlib)
    st.subheader("Sentiment Distribution (Pie Chart)")
    sentiment_counts = df_out['sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=['red','gray','green'],
        startangle=90
    )
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

    # 📥 Download results
    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "sentiment_results.csv", "text/csv")

