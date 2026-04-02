import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path='data/movies.csv'):
    movies_data = pd.read_csv(path)
    features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for col in features:
        movies_data[col] = movies_data[col].fillna('')
    return movies_data

def build_similarity(movies_data):
    combined_features = (
        movies_data['genres'] + ' ' +
        movies_data['keywords'] + ' ' +
        movies_data['tagline'] + ' ' +
        movies_data['cast'] + ' ' +
        movies_data['director']
    )
    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vector)
    return similarity

def get_recommendations(movie_name, movies_data, similarity, n=10):
    list_titles = movies_data['title'].tolist()

    # Fuzzy match the movie name
    find_close_match = difflib.get_close_matches(movie_name, list_titles)
    if not find_close_match:
        return None, None

    close_match = find_close_match[0]
    index_of_movie = movies_data[movies_data['title'] == close_match]['index'].values[0]

    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended = []
    for movie in sorted_movies[1:n+1]:  # skip index 0 (the movie itself)
        idx = movie[0]
        row = movies_data[movies_data.index == idx]
        if not row.empty:
            recommended.append({
                'title': row['title'].values[0],
                'genres': row['genres'].values[0],
                'director': row['director'].values[0],
                'vote_average': row['vote_average'].values[0],
                'similarity': round(movie[1] * 100, 1)
            })

    return close_match, recommended
