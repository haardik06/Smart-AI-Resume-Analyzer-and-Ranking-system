import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.feature_extraction import train_tfidf


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text


def train_from_dataset():
    df = pd.read_csv("dataset/resumes.csv")

    if "Resume_str" in df.columns:
        df.rename(columns={"Resume_str": "Resume"}, inplace=True)

    df = df.dropna(subset=["Resume", "Category"])
    df["Resume"] = df["Resume"].apply(clean_text)

    texts = df["Resume"]
    labels = df["Category"]

    X = train_tfidf(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    models = {}
    metrics = {}

    def evaluate(model, name):
        preds = model.predict(X_test)
        models[name] = model
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, preds, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_test, preds, average='weighted', zero_division=0))
        }

    evaluate(MultinomialNB().fit(X_train, y_train), "Naive Bayes")
    evaluate(RandomForestClassifier(n_estimators=200).fit(X_train, y_train), "Random Forest")
    evaluate(SVC(kernel='linear').fit(X_train, y_train), "SVM")
    evaluate(KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train), "KNN")
    evaluate(DecisionTreeClassifier().fit(X_train, y_train), "Decision Tree")

    joblib.dump(models, "models/models.pkl")
    joblib.dump(metrics, "models/metrics.pkl")

    return models, metrics


def load_models():
    return joblib.load("models/models.pkl")


def load_metrics():
    return joblib.load("models/metrics.pkl")