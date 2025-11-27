import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from transformers import pipeline
from firebase_config import auth, db

# ----------------------------
# Sentiment Pipeline (loaded once)
# ----------------------------
@st.cache_resource
def get_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
    )

sentiment_pipeline = get_pipeline()
label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

# ----------------------------
# Helper functions
# ----------------------------
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
    return "green" if s.lower() == "positive" else "red" if s.lower() == "negative" else "gray"

# ----------------------------
# Auth Pages
# ----------------------------
def signup_page():
    st.subheader("📝 Create Account")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    if st.button("Register"):
        if not email or not password:
            st.error("Please fill in both email and password.")
            return
        try:
            auth.create_user_with_email_and_password(email, password)
            st.success("Account created successfully! Please log in.")
            st.session_state.page = "login"   # redirect to login
        except Exception as e:
            st.error("Registration failed. Please check your details or try again.")

def login_page():
    st.subheader("🔐 Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if not email or not password:
            st.error("Please fill in both email and password.")
            return
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state.user = user
            st.success("Logged in successfully!")
            st.session_state.page = "dashboard"   # redirect to dashboard
        except Exception:
            st.error("Login failed. Check your credentials and try again.")

def logout_button():
    if "user" in st.session_state:
        if st.sidebar.button("Logout"):
            st.session_state.pop("user", None)
            st.session_state.page = "login"
            st.experimental_rerun()

# ----------------------------
# Analysis Features
# ----------------------------
def single_text_analysis():
    st.subheader("✍️ Single Text Analysis")
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

        # Visualization
        st.bar_chart(pd.DataFrame({sentiment: [confidence]}))

        fig, ax = plt.subplots()
        ax.pie([confidence, 1 - confidence],
               labels=[sentiment, "other"],
               autopct="%1.1f%%",
               colors=[color_for_sentiment(sentiment), "lightgray"],
               startangle=90)
        st.pyplot(fig)

        # Keywords
        common = extract_keywords(user_text, top_n=7)
        st.subheader("Top keywords driving sentiment")
        if common:
            st.write(", ".join([w for w,_ in common]))
        else:
            st.write("No significant keywords found.")

def batch_analysis():
    st.subheader("📂 Batch Analysis (TXT file)")
    uploaded_file = st.file_uploader("Upload a .txt file (one text per line)", type=["txt"])
    if uploaded_file:
        texts = [line.strip() for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]
        if texts:
            results = sentiment_pipeline(texts, truncation=True, max_length=256)
            readable = [label_map.get(r["label"], r["label"]) for r in results]

            df_out = pd.DataFrame({
                "text": texts,
                "sentiment": readable,
                "confidence": [r["score"] for r in results]
            })

            st.dataframe(df_out, use_container_width=True)
            st.bar_chart(df_out["sentiment"].value_counts(normalize=True) * 100)

            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download results (CSV)", csv, "sentiment_results.csv", "text/csv")

# ----------------------------
# Dashboard
# ----------------------------
def dashboard():
    st.title("📊 Sentiment Analysis Dashboard")
    st.write("Welcome! Analyze single texts or batches with visual explanations, keyword drivers, and comparisons.")
    logout_button()
    page = st.sidebar.radio("Sections", ["Single Text Analysis", "Batch Analysis"])
    if page == "Single Text Analysis":
        single_text_analysis()
    else:
        batch_analysis()

# ----------------------------
# Main App Navigation
# ----------------------------
st.set_page_config(page_title="Sentiment App", page_icon="🧠", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "login"

if st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "login":
    tabs = st.tabs(["Login", "Register"])
    with tabs[0]:
        login_page()
    with tabs[1]:
        signup_page()
elif st.session_state.page == "dashboard":
    if "user" in st.session_state:
        dashboard()
    else:
        st.warning("Please log in first to access the dashboard.")
        st.session_state.page = "login"



