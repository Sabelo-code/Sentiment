import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from transformers import pipeline
from firebase_config import auth,db

# -------------------------------
# Sentiment Pipeline (loaded once)
# -------------------------------
@st.cache_resource
def get_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
    )

sentiment_pipeline = get_pipeline()
label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

# -------------------------------
# Auth pages
# -------------------------------
def signup_page():
    st.title("📝 Sign Up")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if not email or not password:
            st.error("Please fill in both email and password.")
            return
        try:
            auth.create_user_with_email_and_password(email, password)
            st.success("Account created successfully! You can now log in.")
        except Exception as e:
            st.error("Registration failed. Please check your details or try again.")

def login_page():
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if not email or not password:
            st.error("Please fill in both email and password.")
            return
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user"] = user
            st.success("Logged in successfully!")
        except Exception:
            st.error("Login failed. Check your credentials and try again.")

def logout_button():
    if "user" in st.session_state:
        if st.sidebar.button("Logout"):
            st.session_state.pop("user", None)
            st.experimental_rerun()

# -------------------------------
# Helper functions
# -------------------------------
def extract_keywords(text, top_n=5):
    words = re.findall(r"[A-Za-z']+", text.lower())
    stop = set("""a an the and or but if then than this that is are was were be been being
        to of for in on at by with from as it its into up down over under out very really
        so not no yes i you he she we they them us our your their my mine ours yours theirs
        do does did doing have has had having will would should could can may might must
    """.split())
    filtered = [w for w in words if w not in stop and len(w) > 2]
    return Counter(filtered).most_common(top_n)

def color_for_sentiment(s):
    return "green" if s == "positive" else "red" if s == "negative" else "gray"

# -------------------------------
# Single text analysis
# -------------------------------
def single_text_analysis():
    st.subheader("🔎 Single Text Analysis")
    user_text = st.text_area("Enter text to analyze:", placeholder="Type a sentence or paragraph...")
    if st.button("Analyze Text"):
        if not user_text.strip():
            st.error("Please enter some text before analyzing.")
            return

        result = sentiment_pipeline(user_text, truncation=True, max_length=256)[0]
        sentiment = label_map.get(result["label"], result["label"])
        confidence = result["score"]

        st.success(f"Sentiment: {sentiment}")
        st.info(f"Confidence: {confidence:.2f}")

        # Visualization: confidence
        st.subheader("Confidence visualization")
        st.bar_chart(pd.DataFrame({sentiment: [confidence]}))

        fig, ax = plt.subplots()
        ax.pie([confidence, 1 - confidence],
               labels=[sentiment, "other"],
               autopct="%1.1f%%",
               colors=[color_for_sentiment(sentiment), "lightgray"],
               startangle=90)
        ax.set_title("Confidence distribution")
        st.pyplot(fig)

        # Keyword extraction (sentiment drivers)
        st.subheader("Top keywords driving sentiment")
        common = extract_keywords(user_text, top_n=7)
        if common:
            for word, freq in common:
                st.write(f"- {word} ({freq})")
        else:
            st.write("No significant keywords found.")

        # Explanation feature
        st.subheader("Why this sentiment?")
        st.write("The model evaluates word choice and context. Keywords like "
                 f"**{', '.join([w for w,_ in common])}** likely influenced the detected tone. "
                 "Neutral results often reflect balanced or ambiguous language.")

# -------------------------------
# Batch analysis and comparative view
# -------------------------------
def batch_analysis():
    st.subheader("📂 Batch Analysis (TXT file)")
    uploaded_file = st.file_uploader("Upload a .txt file (one text per line)", type=["txt"])
    if uploaded_file is None:
        st.info("Upload a TXT file to see batch results.")
        return

    texts = [line.strip() for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]
    if not texts:
        st.error("Your file appears to be empty. Please upload a file with at least one non-empty line.")
        return

    results = sentiment_pipeline(texts, truncation=True, max_length=256)
    readable = [label_map.get(r["label"], r["label"]) for r in results]

    df_out = pd.DataFrame({
        "text": texts,
        "sentiment": readable,
        "confidence": [r["score"] for r in results]
    })

    st.subheader("Results table")
    st.dataframe(df_out, use_container_width=True)

    # Comparative analysis: percentages per sentiment
    st.subheader("Comparative sentiment distribution (percent)")
    sentiment_percent = df_out["sentiment"].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_percent)

    fig, ax = plt.subplots()
    counts = df_out["sentiment"].value_counts()
    ax.pie(counts,
           labels=counts.index,
           autopct="%1.1f%%",
           colors=["red", "gray", "green"],
           startangle=90)
    ax.set_title("Sentiment distribution across texts")
    st.pyplot(fig)

    # Keyword extraction per sentiment (simple driver view)
    st.subheader("Keyword drivers by sentiment")
    for s in ["negative", "neutral", "positive"]:
        subset = df_out[df_out["sentiment"] == s]["text"]
        joined = " ".join(subset.tolist())
        drivers = extract_keywords(joined, top_n=10)
        st.markdown(f"**{s.capitalize()}:** " + (", ".join([w for w,_ in drivers]) if drivers else "No strong drivers"))

    # Download
    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", csv, "sentiment_results.csv", "text/csv")

# -------------------------------
# App layout and navigation
# -------------------------------
def dashboard():
    st.title("📊 Sentiment Analysis Dashboard")
    st.write("Analyze single texts or batches with visual explanations, keyword drivers, and comparisons.")
    page = st.sidebar.radio("Sections", ["Single Text Analysis", "Batch Analysis"])
    if page == "Single Text Analysis":
        single_text_analysis()
    else:
        batch_analysis()

# -------------------------------
# Main navigation (auth + app)
# -------------------------------
st.set_page_config(page_title="Sentiment App", page_icon="🧠", layout="wide")
st.sidebar.title("Menu")
logout_button()
menu = st.sidebar.selectbox("Go to", ["Login", "Sign Up", "Dashboard"])

if menu == "Login":
    login_page()
elif menu == "Sign Up":
    signup_page()
elif menu == "Dashboard":
    if "user" in st.session_state:
        dashboard()
    else:
        st.warning("Please log in first to access the dashboard.")


