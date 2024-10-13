from flask import Flask, render_template, request, jsonify
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import io
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data
stop_words = stopwords.words('english')



def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(documents)
    n_components = 12
    svd = TruncatedSVD(n_components=n_components)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    query_tfidf = vectorizer.transform([query])
    query_lsa = svd.transform(query_tfidf)
    
    similarities = cosine_similarity(query_lsa, lsa_matrix)[0]
    
    most_similar_indices = np.argsort(similarities)[::-1][:5]
    
    most_similar_documents = [documents[i] for i in most_similar_indices]
    highest_similarities = [float(similarities[i]) for i in most_similar_indices]
    return most_similar_documents, highest_similarities, most_similar_indices.tolist()
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    # Create the bar chart
    plt.figure(figsize=(10, 5))
    plt.barh(documents, similarities, color='skyblue')
    plt.xlabel('Cosine Similarity')
    plt.title('Top 5 Similar Documents')
    plt.gca().invert_yaxis()  # Invert y-axis to show the highest score on top

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
