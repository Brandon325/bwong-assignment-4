from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Fetch dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize vectorizer and LSA
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_tfidf = vectorizer.fit_transform(documents)

# Apply Truncated SVD (LSA)
n_components = 100  # You can adjust this number
svd_model = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd_model.fit_transform(X_tfidf)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Transform the query into the same TF-IDF space
    query_tfidf = vectorizer.transform([query])

    # Project the query into the reduced LSA space
    query_reduced = svd_model.transform(query_tfidf)

    # Compute cosine similarities
    similarities = cosine_similarity(query_reduced, X_reduced)[0]

    # Get the top 5 most similar documents
    top_indices = similarities.argsort()[-5:][::-1]
    top_similarities = similarities[top_indices]
    top_documents = [documents[i] for i in top_indices]

    return top_documents, top_similarities.tolist(), top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.get_json()['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)