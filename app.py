from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv('tmdb_movies.csv')  # Ensure the dataset is in the same directory
df = df.fillna('')
df['combined_features'] = df['Genres'] + ' ' + df['Cast'] + ' ' + df['Director'] + ' ' + df['Rating'].astype(str) + ' ' + df['Release Year'].astype(str)

# Build recommendation system
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Movie index mapping
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

def get_recommendations(movie_title):
    if movie_title not in indices:
        return []
    
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[movie_indices].tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_title = data.get("movie")
    recommendations = get_recommendations(movie_title)
    
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
