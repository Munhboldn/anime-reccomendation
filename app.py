# app.py
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# --- Page Configuration & Custom CSS ---
st.set_page_config(
    page_title="Anime Recommender System",
    page_icon=":sparkles:",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        color: #333333;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #666666;
        margin-bottom: 1rem;
    }
    .recommendation {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    df = pd.read_csv("anime-dataset-2023.csv")
    df = df.dropna(subset=["Genres"])
    df["Genres"] = df["Genres"].str.lower()
    return df

df = load_data()

# --- Build the TF-IDF Matrix ---
@st.cache_resource
def build_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["Genres"])
    return tfidf_matrix, tfidf

tfidf_matrix, tfidf = build_tfidf_matrix(df)

# Create a reverse mapping from anime title to DataFrame index using "Name"
indices = pd.Series(df.index, index=df["Name"]).drop_duplicates()

# --- Helper Function: Franchise-Based Filtering ---
def similar_titles(title1, title2):
    """Check if two anime titles are from the same franchise."""
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio() > 0.7  # High similarity threshold

def recommend_anime(anime_title):
    """Get exactly 5 diverse recommendations without franchise duplicates."""
    if anime_title not in indices:
        return []

    idx = indices[anime_title]
    anime_vector = tfidf_matrix[idx]
    cosine_sim = cosine_similarity(anime_vector, tfidf_matrix).flatten()

    # Sort scores in descending order
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    seen_titles = set()
    query_lower = anime_title.lower()

    for i, score in sim_scores[1:]:  # Skip the original anime itself
        candidate = df["Name"].iloc[i]

        # Skip exact duplicates or very similar titles
        if candidate.lower() == query_lower or similar_titles(candidate, anime_title):
            continue

        # Ensure no duplicate franchise entries
        if candidate not in seen_titles:
            recommendations.append(candidate)
            seen_titles.add(candidate)

        if len(recommendations) >= 5:  # Stop at exactly 5 recommendations
            break

    return recommendations

# --- Sidebar Instructions ---
st.sidebar.header("How It Works")
st.sidebar.write("""
Select an anime from the dropdown, choose the number of recommendations you'd like to see, 
and the system will suggest similar anime.
""")

# --- Main App Layout ---
with st.container():
    st.markdown("<div class='title'><h1>Anime Recommender System</h1></div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'><h3>Discover your next favorite anime!</h3></div>", unsafe_allow_html=True)
    st.write("---")
    
    # Dropdown for anime titles
    anime_list = sorted(df["Name"].unique())
    selected_anime = st.selectbox("Select an Anime:", anime_list)

    if st.button("Get Recommendations"):
        recommendations = recommend_anime(selected_anime)
        if recommendations:
            st.markdown("<div class='recommendation'>", unsafe_allow_html=True)
            st.markdown(f"<h4>Because you liked <b>{selected_anime}</b>, you might also enjoy:</h4>", unsafe_allow_html=True)
            for rec in recommendations:
                st.write(f"• {rec}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Sorry, no recommendations were found. Please try a different anime.")
