# 🎬 Movie Recommender System

A content-based movie recommendation engine built using TF-IDF vectorization and cosine similarity on the TMDB 5000 Movies dataset.

## 🔗 Live Demo
[Click here to try it][(https://your-app-link.streamlit.app](https://movierecommender-e9qkwf62pk58vxunctkdvq.streamlit.app/)) ← replace after deployment

## 🛠️ Tech Stack
- Python
- Scikit-learn (TF-IDF + Cosine Similarity)
- Streamlit (UI)
- Pandas & NumPy

## 🧠 How It Works
1. Combines movie features: genres, keywords, tagline, cast, director
2. Converts them into TF-IDF vectors
3. Computes cosine similarity between all movies
4. Uses fuzzy matching (difflib) to handle typos in movie names
5. Returns top N most similar movies with match percentage

## 🚀 Run Locally
```bash
git clone https://github.com/yourusername/movie-recommender
cd movie-recommender
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
movie-recommender/
├── app.py              # Streamlit UI
├── recommender.py      # ML logic
├── data/
│   └── movies.csv      # TMDB dataset
├── requirements.txt
└── README.md
```

## 📊 Dataset
TMDB 5000 Movies Dataset — 4803 movies with genres, cast, crew, keywords and more.
Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
