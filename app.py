import streamlit as st
import pandas as pd
import joblib
import nltk
import contractions

# --- Must be the very first Streamlit command in the script ---
st.set_page_config(page_title="AliExpress Review Sentiment", page_icon="🛍️")

# --- One-time NLTK downloads (cached after first run) ---
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords

# --- Recreate the exact preprocessing used during training ---
stop_words = stopwords.words("english")

negation_words = {"not", "no", "nor", "don", "don't", "won", "won't", "isn", "isn't",
                   "aren", "aren't", "wasn", "wasn't", "weren", "weren't",
                   "couldn", "couldn't", "shouldn", "shouldn't", "wouldn", "wouldn't"}

custom_stop_words = [w for w in stop_words if w not in negation_words]


def remove_stopwords(text):
    """
    Expands contractions (so negation isn't lost), tokenizes,
    removes stopwords (keeping negation words), and rejoins the text.
    """
    text = contractions.fix(text)
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in custom_stop_words]
    return " ".join(filtered_words)


# --- Load the trained artifacts (cached so this only runs once per session) ---
@st.cache_resource
def load_artifacts():
    bow_vectorizer = joblib.load("bow_vectorizer.joblib")
    mnb_bow = joblib.load("mnb_bow.joblib")
    return bow_vectorizer, mnb_bow


bow_vectorizer, mnb_bow = load_artifacts()


def inference(text):
    filtered_text = remove_stopwords(text)
    bow = bow_vectorizer.transform([filtered_text])
    prediction = mnb_bow.predict(bow)
    return prediction[0]


# --- UI ---
st.title("🛍️ AliExpress Review Sentiment Classifier")

tab_single, tab_batch = st.tabs(["Single review", "Batch (CSV upload)"])

# --- Tab 1: single review ---
with tab_single:
    st.write(
        "Paste a product review below and the model will predict whether the "
        "sentiment is **positive**, **neutral**, or **negative**."
    )

    user_input = st.text_area("Review text", height=150, placeholder="e.g. I love this product, it arrived fast and works great!")

    if st.button("Analyze sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some review text first.")
        else:
            with st.spinner("Analyzing..."):
                result = inference(user_input)
            label_style = {
                "positive": st.success,
                "negative": st.error,
                "neutral": st.info,
            }
            display_fn = label_style.get(str(result).lower(), st.write)
            display_fn(f"Predicted sentiment: **{result}**")

# --- Tab 2: batch CSV upload ---
with tab_batch:
    st.write(
        "Upload a CSV containing a column of review text and get a sentiment "
        "prediction for every row."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read that file as a CSV: {e}")
            batch_df = None

        if batch_df is not None:
            if batch_df.empty:
                st.warning("The uploaded CSV has no rows.")
            else:
                st.write("Preview of uploaded data:")
                st.dataframe(batch_df.head())

                text_column = st.selectbox(
                    "Which column contains the review text?",
                    options=batch_df.columns,
                )

                max_rows = len(batch_df)
                row_limit = st.number_input(
                    "Rows to process (limit for very large files)",
                    min_value=1,
                    max_value=max_rows,
                    value=min(max_rows, 5000),
                    step=1,
                )

                if st.button("Analyze Sentiments", type="primary"):
                    work_df = batch_df.head(int(row_limit)).copy()
                    texts = work_df[text_column].fillna("").astype(str)

                    progress = st.progress(0, text="Analyzing...")
                    predictions = []
                    total = len(texts)
                    for i, t in enumerate(texts):
                        predictions.append(inference(t) if t.strip() else "")
                        if total > 0:
                            progress.progress((i + 1) / total, text=f"Predicting... {i + 1}/{total}")
                    progress.empty()

                    work_df["predicted_sentiment"] = predictions

                    st.success(f"Done — predicted sentiment for {len(work_df)} rows.")
                    st.dataframe(work_df)

                    st.subheader("Sentiment breakdown")
                    st.bar_chart(work_df["predicted_sentiment"].value_counts())

                    csv_bytes = work_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download results as CSV",
                        data=csv_bytes,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv",
                    )

# st.caption("Model: Bag-of-Words + Multinomial Naive Bayes, trained on AliExpress electronics reviews.")
