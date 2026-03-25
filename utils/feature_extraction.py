from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)


def train_tfidf(texts):
    X = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, "models/tfidf.pkl")
    return X


def load_tfidf():
    return joblib.load("models/tfidf.pkl")


def transform_tfidf(texts):
    vectorizer = load_tfidf()
    return vectorizer.transform(texts)


#  FAST BERT MODELL
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

bert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')


def bert_similarity(text1, text2):
    emb1 = bert_model.encode(text1)
    emb2 = bert_model.encode(text2)
    score = cosine_similarity([emb1], [emb2])[0][0]
    return score * 100