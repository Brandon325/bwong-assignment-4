from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# Load the dataset without removing any parts
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

# SVD
k = 100
U_k, S_k, VT_k = compute_svd(X_tfidf, k)

# Compute the reduced representation of the documents
X_reduced = U_k @ S_k

# Normalize vectors
X_norm = X_reduced / np.linalg.norm(X_reduced, axis=1, keepdims=True)

# Inspect the first 5 documents from the dataset
print("Inspecting the first 5 documents from the dataset:")
for i in range(5):
    doc_length = len(newsgroups.data[i])
    print(f"Document {i} length: {doc_length}")
    print(newsgroups.data[i])
    print("=" * 80)

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
            # Build the content similar to the initial code snippet
            doc_length = len(newsgroups.data[idx])
            content = f"Document {idx} length: {doc_length}\n"
            content += newsgroups.data[idx]
            content += "\n" + "=" * 80
            # Append the result
            results.append({
                'score': float(score),
                'content': content  # Include the formatted content
            })
        # Return the results to the frontend
        return jsonify(results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
