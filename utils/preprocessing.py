import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)

    tokens = text.split()

    tokens = [w for w in tokens if w not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)