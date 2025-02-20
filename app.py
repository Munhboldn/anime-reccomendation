# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    # Load the dataset using anime-dataset-2023.csv
    df = pd.read_csv("anime-dataset-2023.csv")
    # Drop rows with missing genre information and standardize the genres
    df = df.dropna(subset=["Genres"])
    df["Genres"] = df["Genres"].str.lower()
    return df

df = load_data()

# --- Build the TF-IDF Matrix and Similarity Matrix ---
@st.cache_resource
def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["Genres"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Create a reverse mapping from anime title to DataFrame index (using the "Name" column)
    indices = pd.Series(df.index, index=df["Name"]).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = build_similarity_matrix(df)

# --- Recommendation Function ---
def recommend_anime(anime_title, num_recommendations=5):
    if anime_title not in indices:
        return []
    idx = indices[anime_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort anime by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Exclude the anime itself and take the top recommendations
    sim_scores = sim_scores[1:num_recommendations+1]
    anime_indices = [i[0] for i in sim_scores]
    return df["Name"].iloc[anime_indices].tolist()

# --- Streamlit App Layout ---
st.title("Anime Recommender System")
st.write("Select an anime to receive recommendations based on genre similarity.")

# Create a dropdown for anime titles
anime_list = sorted(df["Name"].unique())
selected_anime = st.selectbox("Select an Anime:", anime_list)

if st.button("Get Recommendations"):
    recommendations = recommend_anime(selected_anime)
    if recommendations:
        st.write(f"Because you liked **{selected_anime}**, you might also enjoy:")
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("Sorry, no recommendations were found. Please try a different anime.")
