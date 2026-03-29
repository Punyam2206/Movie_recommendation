import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load movies dataframe
with open('movies.pkl', 'rb') as f:
    df = pickle.load(f)

# Load similarity matrix — handles all three methods automatically
try:
    # Method 3 — sparse matrix (.npz file)
    from scipy import sparse
    similarity = sparse.load_npz('similarity.npz').toarray()
    print("Loaded sparse similarity matrix")
except:
    try:
        # Method 1 or 2 — regular pickle (.pkl file)
        with open('similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        similarity = np.array(similarity)
        print("Loaded pickle similarity matrix")
    except Exception as e:
        print(f"Error loading similarity: {e}")

# Your exact recommendation function from Colab
def recommend(movie):
    movie_index = df[df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

    recommended_movies = []
    for i in movies_list[1:6]:
        recommended_movies.append(df.iloc[i[0]].title)
    return recommended_movies

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend_api():
    title = request.args.get('title', '').strip()
    if not title:
        return jsonify({'error': 'No title provided'}), 400
    try:
        results = recommend(title)
        return jsonify({'recommendations': results})
    except IndexError:
        return jsonify({'error': 'Movie not found in database'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
