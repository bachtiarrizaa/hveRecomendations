import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from flask import Flask, request, render_template, jsonify

nltk.download('punkt')

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_excel('dataExcel1.xlsx', header=1)
df.drop(columns=['NO', 'EQUIP NAME', 'GROUP KPI', 'MAIN COMP. PROBLEM', 'TIPE', 'ACTIVITY TYPE', 'NAMA MEKANIK', 'TAMBAHAN', 'DESC.', 'EQUIP CODE'], inplace=True)

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

df['Cleaned_Kerusakan'] = df['KERUSAKAN'].apply(preprocess_text)
df['Cleaned_Action'] = df['ACTION'].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Kerusakan'] + ' ' + df['Cleaned_Action'])

def get_recommendations(project_title, location, top_n=10):
    query = preprocess_text(project_title + ' ' + location)
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    top_indices = related_docs_indices[:top_n]
    recommendations = df.iloc[top_indices]['ACTION'].tolist()
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    project_title = request.form['project_title']
    location = ''
    title_fix = ''

    action = pd.unique(df['ACTION'])
    title_split = project_title.split()
    if len(title_split) >= 2:
        title_fix = ' '.join(title_split[:2])
    else:
        title_fix = project_title

    for prompt in title_split:
        if prompt.capitalize() in action:
            location = prompt
        else:
            title_fix = prompt

    recommendations = get_recommendations(title_fix, location)
    set_recommendations = list(set(recommendations))

    # return jsonify(set_recommendations)

    return jsonify(recommendations=[f"- {rec}" for rec in set_recommendations])

if __name__ == '__main__':
    app.run(debug=True)
