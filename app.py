import streamlit as st
import pandas as pd
from recommender import load_data, build_similarity, get_recommendations

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommender System")
st.write("Type a movie you like and get similar recommendations instantly!")

# Load data and build model (cached so it only runs once)
@st.cache_data
def setup():
    movies_data = load_data('data/movies.csv')
    similarity = build_similarity(movies_data)
    return movies_data, similarity

with st.spinner("Loading model..."):
    movies_data, similarity = setup()

st.success(f"✅ Model ready! {len(movies_data)} movies loaded.")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    movie_name = st.text_input(
        "Enter a movie name:",
        placeholder="e.g. Iron Man, Avatar, The Dark Knight..."
    )
with col2:
    n = st.slider("Number of results:", 5, 20, 10)

if st.button("🎯 Get Recommendations") and movie_name:
    matched_title, recommendations = get_recommendations(
        movie_name, movies_data, similarity, n
    )

    if recommendations is None:
        st.error("❌ Movie not found! Try a different name or check spelling.")
    else:
        st.subheader(f"Because you liked **{matched_title}**, you might also enjoy:")

        # Display recommendations in a nice table
        df_results = pd.DataFrame(recommendations)
        df_results.index = range(1, len(df_results) + 1)
        df_results.columns = ['Title', 'Genres', 'Director', 'Rating', 'Match %']

        st.dataframe(
            df_results,
            use_container_width=True,
            column_config={
                "Match %": st.column_config.ProgressColumn(
                    "Match %",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
                "Rating": st.column_config.NumberColumn(
                    "⭐ Rating",
                    format="%.1f"
                )
            }
        )

        # Show a small insight
        st.info(f"💡 Top match: **{recommendations[0]['title']}** with {recommendations[0]['similarity']}% similarity")

# Footer
st.markdown("---")
st.markdown("Built with Python · Scikit-learn · Streamlit | Dataset: TMDB 5000 Movies")
