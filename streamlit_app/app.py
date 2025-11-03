import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from utils.parser import fetch_and_parse, extract_text_from_html
from utils.features import (
    sentence_tokenize,
    word_tokenize,
    flesch_reading_ease,
    embed_tfidf,
)
from utils.scorer import load_quality_model, rule_quality_label


# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_FILE = Path(__file__).resolve().parent / "models" / "quality_model.pkl"


# -----------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------
@st.cache_data
def load_dataset():
    extracted = pd.read_csv(DATA_DIR / "extracted_content.csv")
    features = pd.read_csv(DATA_DIR / "features.csv")
    return extracted, features


@st.cache_resource
def load_embeddings(texts):
    X, vec = embed_tfidf(list(texts))
    return X, vec


@st.cache_resource
def load_model():
    return load_quality_model(MODEL_FILE)


# -----------------------------------------------------------
# Streamlit config
# -----------------------------------------------------------
st.set_page_config(
    page_title="SEO Content Quality & Duplicate Detector",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 SEO Content Quality & Duplicate Detector")
st.markdown(
    """
This dashboard builds on the Jupyter notebook pipeline to:

- Analyse the quality of a single URL in real time.
- Detect near-duplicate pages inside the dataset.
- Summarise dataset-level SEO health (thin content, readability, labels).
"""
)

# Load core artefacts
extracted_df, features_df = load_dataset()
X_tfidf, vectorizer = load_embeddings(
    extracted_df["body_text"].fillna("").astype(str)
)
model = load_model()

# Sidebar
st.sidebar.header("Settings")
sim_threshold = st.sidebar.slider(
    "Similarity threshold (cosine)",
    min_value=0.5,
    max_value=0.95,
    value=0.8,
    step=0.01,
)
st.sidebar.caption(
    f"Pairs with similarity ≥ {sim_threshold:.2f} are treated as potential duplicates."
)

tab_analyze, tab_duplicates, tab_summary = st.tabs(
    ["📍 Analyze URL", "🧬 Duplicate Map", "📊 Dataset Summary"]
)


# -----------------------------------------------------------
# Tab 1 – Analyze URL
# -----------------------------------------------------------
with tab_analyze:
    st.subheader("1. Real-time SEO analysis for a URL")

    url = st.text_input(
        "Enter a URL",
        placeholder="https://en.wikipedia.org/wiki/Data_science",
    )

    if st.button("Analyze URL"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching and analyzing page..."):
                html = fetch_and_parse(url)
                if html is None:
                    st.error("Failed to fetch page (HTTP != 200 or network error).")
                else:
                    title, body = extract_text_from_html(html)

                    wc = len(word_tokenize(body))
                    sc = len(sentence_tokenize(body))
                    fre = flesch_reading_ease(body)

                    rule_label = rule_quality_label(wc, fre)

                    # Model prediction
                    try:
                        X_new = np.array([[wc, sc, fre]])
                        model_label = model.predict(X_new)[0]
                    except Exception:
                        model_label = rule_label

                    # Similarity to dataset
                    new_vec = vectorizer.transform([body])
                    sims = cosine_similarity(new_vec, X_tfidf).ravel()
                    top_idx = sims.argsort()[::-1][:10]

                    similar_rows = []
                    for i in top_idx:
                        if sims[i] >= sim_threshold:
                            similar_rows.append(
                                {
                                    "similar_url": extracted_df.loc[i, "url"],
                                    "similarity": round(float(sims[i]), 4),
                                }
                            )
                    similar_df = pd.DataFrame(similar_rows)

                    # KPIs
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Word Count", f"{wc}")
                    c2.metric("Sentence Count", f"{sc}")
                    c3.metric("Readability (FRE)", f"{fre}")

                    st.markdown("### Quality Assessment")
                    st.write(f"- **Rule-based label:** `{rule_label}`")
                    st.write(f"- **Model prediction:** `{model_label}`")
                    st.write(f"- **Thin content?** `{wc < 500}`")

                    st.markdown("### Similar Pages in Dataset")
                    if similar_df.empty:
                        st.info(
                            f"No similar pages above similarity {sim_threshold:.2f}."
                        )
                    else:
                        st.dataframe(similar_df, use_container_width=True)

                    st.markdown("### Extracted Content Preview")
                    st.caption(f"Title: {title}")
                    st.write(body[:2000] + ("..." if len(body) > 2000 else ""))


# -----------------------------------------------------------
# Tab 2 – Duplicate Map
# -----------------------------------------------------------
with tab_duplicates:
    st.subheader("2. Near-duplicate URLs in the dataset")

    with st.spinner("Computing cosine similarities..."):
        sim_matrix = cosine_similarity(X_tfidf)
        n = sim_matrix.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if sim >= sim_threshold:
                    pairs.append(
                        {
                            "url_1": extracted_df.loc[i, "url"],
                            "url_2": extracted_df.loc[j, "url"],
                            "similarity": round(float(sim), 4),
                        }
                    )

        dup_df = pd.DataFrame(pairs)

    if dup_df.empty:
        st.info(
            f"No duplicate pairs found with similarity ≥ {sim_threshold:.2f}. "
            "Try lowering the threshold in the sidebar."
        )
    else:
        st.write(
            f"Found **{len(dup_df)}** URL pairs with similarity ≥ {sim_threshold:.2f}."
        )
        st.dataframe(dup_df, use_container_width=True)

# -----------------------------------------------------------
# Tab 3 – Dataset Summary
# -----------------------------------------------------------
with tab_summary:
    st.subheader("3. Dataset-level SEO summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pages", len(extracted_df))
    col2.metric("Avg Word Count", f"{features_df['word_count'].mean():.0f}")
    col3.metric(
        "Thin Content (%)",
        f"{(features_df['is_thin'].mean() * 100):.1f}%",
    )
    col4.metric(
        "Readable (FRE 50–70) (%)",
        f"{(features_df['flesch_reading_ease'].between(50, 70).mean() * 100):.1f}%",
    )

    st.markdown("#### Sample of engineered features")
    st.dataframe(features_df.head(20), use_container_width=True)

    st.markdown("#### Quality label distribution")
    st.bar_chart(features_df["quality_label"].value_counts())