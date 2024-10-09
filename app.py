# app.py
from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all', remove=())

# Create a TF-IDF vectorizer and fit it to the dataset
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(newsgroups.data)

# Manually perform SVD
def compute_svd(matrix, k):
    U, S, VT = np.linalg.svd(matrix.toarray(), full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    return U_k, S_k, VT_k

# Apply SVD to reduce dimensionality
k = 100  # Number of components
U_k, S_k, VT_k = compute_svd(X_tfidf, k)

# Compute the reduced representation of the documents
X_reduced = U_k @ S_k

# Normalize the reduced vectors
X_norm = X_reduced / np.linalg.norm(X_reduced, axis=1, keepdims=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        # Process the query
        query_tfidf = vectorizer.transform([query]).toarray()
        # Project the query into the reduced space
        query_reduced = query_tfidf @ VT_k.T
        query_norm = query_reduced / np.linalg.norm(query_reduced)
        # Compute cosine similarity
        similarities = X_norm @ query_norm.T
        # Get top 5 documents
        top_indices = similarities.flatten().argsort()[-5:][::-1]
        top_scores = similarities[top_indices].flatten()
        # Prepare results
        results = []
        for idx, score in zip(top_indices, top_scores):
            full_content = newsgroups.data[idx]  # Fetch full document content
            print(f"Document {idx + 1} length: {len(full_content)}")  # Log length for verification
            results.append({
                'score': float(score),
                'content': full_content  # Ensure the entire content is passed
            })
        # Return the results to the frontend
        return jsonify(results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
